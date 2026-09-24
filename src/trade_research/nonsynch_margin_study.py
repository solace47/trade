"""Select frozen margin-interest groups within past price non-synchronicity.

No forward outcomes enter selection. See docs/nonsynch-margin-plan.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .exchange_public_events import trading_dates
from .short_interest_study import _quintiles, select_pairs


WINDOWS = (("2024_H1", "2024-01-10", "2024-06-30"),
           ("2024_H2", "2024-07-01", "2024-12-17"),
           ("2025_H1", "2025-01-02", "2025-06-30"),
           ("2025_H2", "2025-07-01", "2025-12-17"))
TREATED = "high_margin_interest"
CONTROL = "same_day_low_margin_interest"


def _window(day: str) -> str:
    for name, start, end in WINDOWS:
        if start <= day <= end:
            return name
    raise ValueError(f"Non-synchronicity signal outside frozen windows: {day}")


def _prepare_inputs(universe_path: Path, margin_path: Path,
                    nonsynch_path: Path) -> pd.DataFrame:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute(f"CREATE TEMP VIEW margin AS SELECT * FROM read_parquet('{margin_path}')")
    source = connection.execute("""
        SELECT COUNT(*), COUNT(DISTINCT trade_date),
               COUNT(DISTINCT trade_date || code), MIN(trade_date), MAX(trade_date)
        FROM margin
    """).fetchone()
    if (source[1] != 485 or source[0] != source[2]
            or source[3] != "2024-01-02" or source[4] != "2025-12-31"):
        raise ValueError("Incomplete official margin-interest source")
    connection.execute(f"""
        CREATE TEMP VIEW nonsynch AS
        SELECT * FROM read_parquet('{nonsynch_path}')
    """)
    feature_source = connection.execute("""
        SELECT COUNT(*), COUNT(DISTINCT trade_date || code),
               MIN(n), MAX(n), MAX(span)
        FROM nonsynch
    """).fetchone()
    if (feature_source[0] != feature_source[1]
            or feature_source[2] != 120 or feature_source[3] != 120
            or feature_source[4] > 125):
        raise ValueError("Malformed past-only price non-synchronicity archive")
    frame = connection.execute(f"""
        SELECT u.* EXCLUDE (quintile), n.nonsynch, n.r_squared,
               m.balance_yuan,
               m.balance_yuan / u.float_mv AS margin_interest
        FROM read_parquet('{universe_path}') u
        JOIN nonsynch n ON n.trade_date = u.trade_date AND n.code = u.code
        JOIN margin m ON m.trade_date = u.trade_date AND m.code = u.code
        WHERE ((u.date BETWEEN '2024-01-10' AND '2024-12-17')
            OR (u.date BETWEEN '2025-01-02' AND '2025-12-17'))
          AND u.date > u.trade_date
          AND u.avg20_amount >= 100000000
          AND u.amount_1450 >= 100000000
          AND u.float_mv > 0 AND m.balance_yuan > 0
        ORDER BY u.date, u.code
    """).df()
    if (frame.empty or frame.duplicated(["date", "code"]).any()
            or not frame.date.gt(frame.trade_date).all()
            or not np.isfinite(frame.margin_interest).all()
            or not frame.nonsynch.between(0, 1).all()):
        raise ValueError("Malformed price/financing interaction inputs")
    frame["nonsynch_group"] = np.where(frame.nonsynch.ge(.5), "high", "low")
    frame["window"] = frame.date.map(_window) + "_" + frame.nonsynch_group
    return frame


def select(universe_path: Path, margin_path: Path, nonsynch_path: Path,
           calendar_path: Path, selected_path: Path,
           quintiles_path: Path, report_path: Path) -> dict:
    inputs = _prepare_inputs(universe_path, margin_path, nonsynch_path)
    days = trading_dates(calendar_path, "2024-01-01", "2025-12-31")
    session_index = {day: index for index, day in enumerate(days)}
    quintile_frames = []
    selected_frames = []
    summaries = {}
    for group in ("high", "low"):
        subset = inputs.loc[inputs.nonsynch_group.eq(group)]
        quintiles = _quintiles(subset, "margin_interest")
        selected, summary = select_pairs(
            quintiles, session_index, score_field="margin_interest",
            treated=TREATED, control=CONTROL,
            feature_caliper=("nonsynch", .10))
        high = selected.loc[selected.candidate.eq(TREATED)]
        low = selected.loc[selected.candidate.eq(CONTROL)]
        paired = high[["date", "pair_code", "margin_interest",
                       "nonsynch"]].merge(
            low[["date", "pair_code", "margin_interest", "nonsynch"]],
            on=["date", "pair_code"], suffixes=("_high", "_low"),
            validate="one_to_one")
        if (len(paired) != len(high)
                or not paired.margin_interest_high.gt(
                    paired.margin_interest_low).all()
                or not (paired.nonsynch_high - paired.nonsynch_low).abs().le(.10).all()):
            raise ValueError("Price/financing group pairing failed")
        summaries[group] = {
            "eligible_stock_days": len(subset),
            "quintile_stock_days": len(quintiles),
            "matched_by_window": high.window.value_counts().to_dict(),
            "matched_by_board": high.board.value_counts().to_dict(),
            "top10_share_by_window": {
                name: float(part.code.value_counts().head(10).sum() / len(part))
                for name, part in high.groupby("window")},
            "median_margin_interest_by_window": {
                name: {"high": float(part.margin_interest.median()),
                       "low": float(low.loc[low.window.eq(name),
                                        "margin_interest"].median())}
                for name, part in high.groupby("window")},
            "median_nonsynch_diff": float(
                (paired.nonsynch_high - paired.nonsynch_low).abs().median()),
            **summary,
        }
        quintile_frames.append(quintiles)
        selected_frames.append(selected)
    quintile_all = pd.concat(quintile_frames, ignore_index=True)
    selected_all = pd.concat(selected_frames, ignore_index=True)
    if (quintile_all.duplicated(["date", "code"]).any()
            or selected_all.duplicated(["date", "code"]).any()):
        raise ValueError("Overlapping price non-synchronicity groups")
    report = {"note": "Frozen input-only selection; no forward returns loaded",
              "eligible_by_window": inputs.window.value_counts().to_dict(),
              "groups": summaries}
    for path in (selected_path, quintiles_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    selected_all.to_parquet(selected_path, index=False, compression="zstd")
    quintile_all.to_parquet(quintiles_path, index=False, compression="zstd")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-universe", type=Path,
                        default=Path("data/research/margin_universe_quintiles.parquet"))
    parser.add_argument("--margin", type=Path,
                        default=Path("data/research/margin/margin_2024_2025.parquet"))
    parser.add_argument("--nonsynch", type=Path,
                        default=Path("data/research/price_nonsynch_2024_2025.parquet"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--selected", type=Path,
                        default=Path("data/research/nonsynch_margin_selected.parquet"))
    parser.add_argument("--quintiles", type=Path,
                        default=Path("data/research/nonsynch_margin_quintiles.parquet"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/nonsynch_margin_selection.json"))
    args = parser.parse_args()
    result = select(args.base_universe, args.margin, args.nonsynch,
                    args.calendar, args.selected, args.quintiles, args.report)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
