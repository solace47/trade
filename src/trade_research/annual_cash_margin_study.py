"""Input-only selection for original annual cash conversion and margin interest.

No future return or minute outcome is read. The frozen design is in
docs/annual-cash-quality-plan.md.
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


WINDOWS = (("2024_early", "2024-05-15", "2024-06-30"),
           ("2024_late", "2024-07-01", "2024-12-17"),
           ("2025_early", "2025-05-15", "2025-06-30"),
           ("2025_late", "2025-07-01", "2025-12-17"))
TREATED = "high_margin_interest"
CONTROL = "same_day_low_margin_interest"


def _window(day: str) -> str:
    for name, start, end in WINDOWS:
        if start <= day <= end:
            return name
    raise ValueError(f"Cash conversion signal outside frozen windows: {day}")


def _original_reports(index_paths: tuple[Path, Path],
                      extracted_paths: tuple[Path, Path],
                      margin_codes: set[str]) -> tuple[pd.DataFrame, dict]:
    indices = pd.concat([pd.read_parquet(path) for path in index_paths],
                        ignore_index=True)
    indices = indices.loc[indices.kind.eq("summary")
                          & indices.code.isin(margin_codes)]
    if indices.duplicated(["report_year", "code"]).any():
        raise ValueError("Original annual summary index has duplicate stock-years")
    records = pd.concat([pd.read_json(path, lines=True)
                         for path in extracted_paths], ignore_index=True)
    records = records.drop_duplicates(["report_year", "code"], keep="last")
    expected = set(zip(indices.report_year, indices.code))
    completed = set(zip(records.report_year, records.code))
    if expected != completed:
        raise ValueError(f"Original PDF extraction incomplete: "
                         f"{len(expected - completed)} missing, "
                         f"{len(completed - expected)} unexpected")
    records = records.loc[records.status.eq("ok")].copy()
    records = records.merge(
        indices[["report_year", "code", "notice_date", "pdf_url"]],
        on=["report_year", "code"], suffixes=("_record", ""),
        validate="one_to_one")
    if (not records.notice_date.eq(records.notice_date_record).all()
            or not records.pdf_url.eq(records.pdf_url_record).all()):
        raise ValueError("Extracted values lack their indexed original PDF")
    pdf_day = records.pdf_url.str.extract(r"/finalpage/(\d{4}-\d{2}-\d{2})/")[0]
    cutoff = records.report_year.map(lambda year: f"{year + 1}-04-30")
    if (pdf_day.isna().any() or not pdf_day.eq(records.notice_date).all()
            or not records.report_year.isin((2023, 2024)).all()):
        raise ValueError("Malformed original annual disclosure date")
    extracted_ok = len(records)
    records = records.loc[records.notice_date.le(cutoff)
                          & records.parent_profit_raw.gt(0)].copy()
    ratio = (records.operating_cash_raw / records.parent_profit_raw).to_numpy()
    if (not np.isfinite(ratio).all()
            or not np.allclose(ratio, records.cash_to_parent_profit.to_numpy(),
                               atol=1e-12, rtol=1e-12)):
        raise ValueError("Original annual cash conversion is malformed")
    report = {"indexed_margin_stock_years": len(indices),
              "extracted_stock_years": extracted_ok,
              "profitable_timely_stock_years": len(records),
              "note": "Original PDF values only; no later vendor imputation"}
    return records[["report_year", "code", "notice_date", "pdf_url",
                    "parent_profit_raw", "operating_cash_raw",
                    "cash_to_parent_profit"]], report


def _prepare_inputs(universe_path: Path, margin_path: Path,
                    industry_path: Path, index_paths: tuple[Path, Path],
                    extracted_paths: tuple[Path, Path]) -> tuple[pd.DataFrame, dict]:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    codes = set(connection.execute(f"SELECT DISTINCT code FROM "
                                   f"read_parquet('{universe_path}')").df().code)
    original, source_report = _original_reports(index_paths, extracted_paths,
                                                codes)
    connection.register("original", original)
    connection.execute(f"CREATE TEMP VIEW margin AS SELECT * FROM "
                       f"read_parquet('{margin_path}')")
    margin_check = connection.execute("""
        SELECT COUNT(*), COUNT(DISTINCT trade_date),
               COUNT(DISTINCT trade_date || code), MIN(trade_date), MAX(trade_date)
        FROM margin
    """).fetchone()
    if (margin_check[1] != 485 or margin_check[0] != margin_check[2]
            or margin_check[3] != "2024-01-02"
            or margin_check[4] != "2025-12-31"):
        raise ValueError("Incomplete official margin-interest source")
    connection.execute(f"CREATE TEMP VIEW industry AS SELECT * FROM "
                       f"read_parquet('{industry_path}')")
    frame = connection.execute(f"""
        SELECT u.* EXCLUDE (quintile), a.report_year, a.notice_date,
               a.pdf_url, a.parent_profit_raw, a.operating_cash_raw,
               a.cash_to_parent_profit, i.industry,
               m.balance_yuan / u.float_mv AS margin_interest
        FROM read_parquet('{universe_path}') u
        JOIN original a ON a.code = u.code
          AND a.report_year = CAST(LEFT(u.date, 4) AS INTEGER) - 1
        JOIN margin m ON m.trade_date = u.trade_date AND m.code = u.code
        JOIN industry i ON i.code = u.code
          AND i.effective_date <= u.date
          AND (i.next_effective_date > u.date OR i.next_effective_date IS NULL)
        WHERE ((u.date BETWEEN '2024-05-15' AND '2024-12-17')
            OR (u.date BETWEEN '2025-05-15' AND '2025-12-17'))
          AND a.notice_date < u.date AND u.date > u.trade_date
          AND i.industry NOT LIKE 'J%'
          AND u.avg20_amount >= 30000000
          AND u.amount_1450 >= 30000000
          AND u.float_mv > 0 AND m.balance_yuan > 0
        ORDER BY u.date, u.code
    """).df()
    if (frame.empty or frame.duplicated(["date", "code"]).any()
            or not frame.date.gt(frame.notice_date).all()
            or not frame.date.gt(frame.trade_date).all()
            or not np.isfinite(frame.margin_interest).all()):
        raise ValueError("Malformed original cash/margin inputs")
    frame["cash_group"] = np.where(frame.cash_to_parent_profit.ge(1),
                                   "cash_supported", "low_cash_conversion")
    frame["asinh_cash_conversion"] = np.arcsinh(frame.cash_to_parent_profit)
    frame["window"] = frame.date.map(_window) + "_" + frame.cash_group
    return frame, source_report


def select(universe_path: Path, margin_path: Path, industry_path: Path,
           index_paths: tuple[Path, Path], extracted_paths: tuple[Path, Path],
           calendar_path: Path, selected_path: Path,
           quintiles_path: Path, report_path: Path) -> dict:
    inputs, source_report = _prepare_inputs(
        universe_path, margin_path, industry_path, index_paths, extracted_paths)
    dates = trading_dates(calendar_path, "2024-01-01", "2025-12-31")
    session_index = {day: index for index, day in enumerate(dates)}
    lag = inputs.date.map(session_index) - inputs.trade_date.map(session_index)
    if lag.isna().any() or not lag.eq(1).all():
        raise ValueError("Margin interest is not from the previous session")
    selected_frames = []
    quintile_frames = []
    groups = {}
    for group in ("cash_supported", "low_cash_conversion"):
        subset = inputs.loc[inputs.cash_group.eq(group)]
        quintiles = _quintiles(subset, "margin_interest")
        selected, summary = select_pairs(
            quintiles, session_index, score_field="margin_interest",
            treated=TREATED, control=CONTROL,
            feature_caliper=("asinh_cash_conversion", np.log(2)))
        high = selected.loc[selected.candidate.eq(TREATED)]
        groups[group] = {"eligible_stock_days": len(subset),
                         "quintile_stock_days": len(quintiles),
                         "matched_by_window": high.window.value_counts().to_dict(),
                         "matched_by_board": high.board.value_counts().to_dict(),
                         **summary}
        selected_frames.append(selected)
        quintile_frames.append(quintiles)
    selected_all = pd.concat(selected_frames, ignore_index=True)
    quintiles_all = pd.concat(quintile_frames, ignore_index=True)
    if (selected_all.duplicated(["date", "code"]).any()
            or quintiles_all.duplicated(["date", "code"]).any()):
        raise ValueError("Overlapping original-cash quality groups")
    report = {"source": source_report,
              "note": "Frozen input-only selection; no future returns read",
              "eligible_by_window": inputs.window.value_counts().to_dict(),
              "groups": groups}
    for path in (selected_path, quintiles_path, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    selected_all.to_parquet(selected_path, index=False, compression="zstd")
    quintiles_all.to_parquet(quintiles_path, index=False, compression="zstd")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-universe", type=Path, default=Path(
        "data/research/margin_universe_quintiles.parquet"))
    parser.add_argument("--margin", type=Path, default=Path(
        "data/research/margin/margin_2024_2025.parquet"))
    parser.add_argument("--industry", type=Path, default=Path(
        "data/research/industry_intervals_warmup.parquet"))
    parser.add_argument("--annual-dir", type=Path, default=Path(
        "data/research/annual_cash_quality"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--selected", type=Path, default=Path(
        "data/research/annual_cash_margin_selected.parquet"))
    parser.add_argument("--quintiles", type=Path, default=Path(
        "data/research/annual_cash_margin_quintiles.parquet"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/annual_cash_margin_selection.json"))
    args = parser.parse_args()
    base = args.annual_dir
    result = select(
        args.base_universe, args.margin, args.industry,
        tuple(base / f"cninfo_original_{year}.parquet" for year in (2023, 2024)),
        tuple(base / f"original_cash_{year}.jsonl" for year in (2023, 2024)),
        args.calendar, args.selected, args.quintiles, args.report)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
