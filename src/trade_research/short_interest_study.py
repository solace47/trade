"""Frozen input-only selection for post-July-2024 short interest at 14:50.

The hypothesis and all thresholds precede the outcome read; see
docs/short-interest-plan.md.  Exchange day t enters session t+1 only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .exchange_public_events import trading_dates
from .margin_heterogeneity import _match_one


WINDOWS = (("2024_runoff", "2024-07-23", "2024-09-30"),
           ("2024_postrunoff", "2024-10-01", "2024-12-17"),
           ("2025_H1", "2025-01-02", "2025-06-30"),
           ("2025_H2", "2025-07-01", "2025-12-17"))
CAPACITY = 5
COOLDOWN_SESSIONS = 10
MIN_STRATUM = 25
TREATED = "high_short_interest"
CONTROL = "same_day_low_positive_short_interest"


def _window(day: str) -> str:
    for name, start, end in WINDOWS:
        if start <= day <= end:
            return name
    raise ValueError(f"Signal outside the frozen short-interest windows: {day}")


def _prepare_inputs(universe_path: Path, margin_path: Path,
                    daily_dir: Path) -> pd.DataFrame:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute(f"CREATE TEMP VIEW margin AS SELECT * FROM read_parquet('{margin_path}')")
    source = connection.execute("""
        SELECT COUNT(*) AS n, COUNT(DISTINCT trade_date) AS dates,
               COUNT(DISTINCT trade_date || code) AS keys,
               MIN(trade_date), MAX(trade_date) FROM margin
    """).fetchone()
    if (source[1] != 485 or source[0] != source[2]
            or source[3] != "2024-01-02" or source[4] != "2025-12-31"):
        raise ValueError("Missing, duplicate or unexpected official margin source")
    connection.execute(f"""
        CREATE TEMP VIEW daily AS
        SELECT date, code, close, tradestatus
        FROM read_parquet('{daily_dir}/*.parquet')
        WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
    """)
    connection.execute(f"""
        CREATE TEMP VIEW eligible AS
        SELECT u.* EXCLUDE (quintile), d.close AS source_close,
               m.short_balance_shares, m.short_balance_yuan,
               CASE WHEN u.code LIKE 'sh.%'
                    THEN m.short_balance_shares * d.close
                    ELSE m.short_balance_yuan END AS short_value
        FROM read_parquet('{universe_path}') AS u
        JOIN margin m ON m.trade_date = u.trade_date AND m.code = u.code
        JOIN daily d ON d.date = u.trade_date AND d.code = u.code
        WHERE ((u.date BETWEEN '2024-07-23' AND '2024-12-17')
            OR (u.date BETWEEN '2025-01-02' AND '2025-12-17'))
          AND u.date > u.trade_date
          AND u.avg20_amount >= 100000000
          AND u.amount_1450 >= 100000000
          AND u.float_mv > 0 AND d.close > 0 AND d.tradestatus = 1
    """)
    frame = connection.execute("""
        SELECT *, short_value / float_mv AS short_ratio
        FROM eligible WHERE short_value > 0
        ORDER BY date, code
    """).df()
    if (frame.empty or frame.duplicated(["date", "code"]).any()
            or not frame.date.gt(frame.trade_date).all()
            or not np.isfinite(frame.short_ratio).all()):
        raise ValueError("Malformed post-regulation positive-short-interest inputs")
    frame["window"] = frame.date.map(_window)
    return frame


def _quintiles(frame: pd.DataFrame,
               score_field: str = "short_ratio") -> pd.DataFrame:
    if "quintile" in frame.columns:
        raise ValueError("A prior study's quintile cannot define short-interest groups")
    if score_field not in {"short_ratio", "net_short_flow", "sell_pressure_score",
                           "buy_activity_score", "margin_interest"}:
        raise ValueError("Unknown short-interest ranking field")
    connection = duckdb.connect()
    connection.register("inputs", frame)
    result = connection.execute(f"""
        SELECT * EXCLUDE (stratum_n),
               NTILE(5) OVER (
                   PARTITION BY date, board, size_bucket
                   ORDER BY {score_field}, code
               ) AS quintile
        FROM (
            SELECT *, COUNT(*) OVER (
                PARTITION BY date, board, size_bucket
            ) AS stratum_n FROM inputs
        ) WHERE stratum_n >= {MIN_STRATUM}
        ORDER BY date, code
    """).df()
    if result.empty or result.duplicated(["date", "code"]).any():
        raise ValueError("No valid short-interest quintile strata")
    if (result.groupby(["date", "board", "size_bucket"])
            .quintile.nunique().ne(5).any()):
        raise ValueError("A short-interest stratum lacks five ranked groups")
    return result


def select_pairs(universe: pd.DataFrame, session_index: dict[str, int],
                 score_field: str = "short_ratio",
                 treated: str = TREATED, control: str = CONTROL,
                 signed_flows: bool = False,
                 prior_level_caliper: tuple[float, float] | None = None,
                 prior_level_field: str = "prior_short_interest",
                 prior5_return_caliper: float | None = None,
                 feature_caliper: tuple[str, float] | None = None,
                 ) -> tuple[pd.DataFrame, dict]:
    if score_field not in {"short_ratio", "net_short_flow", "sell_pressure_score",
                           "buy_activity_score", "margin_interest"}:
        raise ValueError("Unknown short-interest selection field")
    if prior_level_caliper is not None and prior_level_field not in {
            "prior_short_interest", "prior_financing_interest"}:
        raise ValueError("Unknown starting-position field")
    if feature_caliper is not None and (feature_caliper[0] != "nonsynch"
                                       or feature_caliper[1] < 0):
        raise ValueError("Unknown or negative matching feature caliper")
    high_rows = []
    low_rows = []
    last_kept: dict[str, int] = {}
    capacity_selected = 0
    unmatched = 0
    for day, daily in universe.groupby("date", sort=True):
        index = session_index[day]
        high = daily.loc[daily.quintile.eq(5)].sort_values(
            [score_field, "code"], ascending=[False, True]
        )
        if signed_flows:
            high = high.loc[high[score_field].gt(0)]
        high = high.loc[high.code.map(
            lambda code: index - last_kept.get(code, -1000) > COOLDOWN_SESSIONS
        )].head(CAPACITY)
        capacity_selected += len(high)
        controls = daily.loc[daily.quintile.eq(1)].copy()
        if signed_flows:
            controls = controls.loc[controls[score_field].lt(0)]
        for _, signal in high.iterrows():
            choices = controls.loc[
                controls.board.eq(signal.board)
                & controls.size_bucket.eq(signal.size_bucket)
            ]
            if prior_level_caliper is not None:
                if signal[prior_level_field] <= 0:
                    unmatched += 1
                    continue
                choices = choices.loc[
                    (choices[prior_level_field] / signal[prior_level_field])
                    .between(*prior_level_caliper)
                ]
            if prior5_return_caliper is not None:
                choices = choices.loc[
                    (choices.return5_prior_adjusted
                     - signal.return5_prior_adjusted).abs().le(
                         prior5_return_caliper)
                ]
            if feature_caliper is not None:
                field, maximum = feature_caliper
                choices = choices.loc[
                    (choices[field] - signal[field]).abs().le(maximum)
                ]
            match = _match_one(signal, choices)
            if match is None:
                unmatched += 1
                continue
            reference, distance = match
            item = signal.copy()
            item["pair_code"] = signal.code
            item["match_distance"] = distance
            item["candidate"] = treated
            high_rows.append(item)
            other = reference.copy()
            other["pair_code"] = signal.code
            other["match_distance"] = distance
            other["candidate"] = control
            low_rows.append(other)
            controls = controls.loc[controls.code.ne(reference.code)]
            last_kept[signal.code] = index
    if not high_rows:
        raise ValueError("No high short-interest stock can be paired")
    selected = pd.concat([pd.DataFrame(high_rows), pd.DataFrame(low_rows)],
                         ignore_index=True)
    counts = selected.groupby(["date", "pair_code"]).candidate.nunique()
    if (not counts.eq(2).all() or len(selected) != 2 * len(counts)
            or selected.duplicated(["date", "code", "candidate"]).any()
            or not selected.date.gt(selected.trade_date).all()):
        raise ValueError("Incomplete or future-informed short-interest pairs")
    summary = {"capacity_selected": capacity_selected,
               "matched_pairs": len(high_rows), "unmatched": unmatched,
               "matched_days": selected.date.nunique(),
               "unique_high_symbols": len({item["code"] for item in high_rows}),
               "median_match_distance": float(pd.DataFrame(low_rows).match_distance.median())}
    return selected, summary


def select(universe_path: Path, margin_path: Path, daily_dir: Path,
           calendar_path: Path, selected_path: Path,
           quintiles_path: Path, report_path: Path) -> dict:
    inputs = _prepare_inputs(universe_path, margin_path, daily_dir)
    quintiles = _quintiles(inputs)
    calendar = trading_dates(calendar_path, "2024-01-01", "2025-12-31")
    selected, summary = select_pairs(
        quintiles, {day: index for index, day in enumerate(calendar)})
    matched = selected.loc[selected.candidate.eq(TREATED)]
    report = {"note": "Input-only selection; no forward returns loaded",
              "positive_short_stock_days": len(inputs),
              "quintile_stock_days": len(quintiles),
              "positive_by_window": inputs.window.value_counts().to_dict(),
              "quintile_by_window": quintiles.window.value_counts().to_dict(),
              "matched_by_window": matched.window.value_counts().to_dict(),
              "matched_by_board": matched.board.value_counts().to_dict(),
              "top10_share_by_window": {
                  name: float(group.code.value_counts().head(10).sum() / len(group))
                  for name, group in matched.groupby("window")},
              "median_short_ratio_by_window": {
                  name: float(group.short_ratio.median())
                  for name, group in matched.groupby("window")},
              **summary}
    for path in (selected_path, quintiles_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(selected_path, index=False, compression="zstd")
    quintiles.to_parquet(quintiles_path, index=False, compression="zstd")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-universe", type=Path,
                        default=Path("data/research/margin_universe_quintiles.parquet"))
    parser.add_argument("--margin", type=Path,
                        default=Path("data/research/margin/margin_2024_2025.parquet"))
    parser.add_argument("--daily", type=Path,
                        default=Path("data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--selected", type=Path,
                        default=Path("data/research/short_interest_selected.parquet"))
    parser.add_argument("--quintiles", type=Path,
                        default=Path("data/research/short_interest_quintiles.parquet"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/short_interest_selection.json"))
    args = parser.parse_args()
    report = select(args.base_universe, args.margin, args.daily,
                    args.calendar, args.selected, args.quintiles, args.report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
