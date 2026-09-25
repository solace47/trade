"""Frozen 14:49 market-beta overnight gross-space diagnostic.

Ye, Jiang and Luo (Global Finance Journal, 2023,
doi:10.1016/j.gfj.2023.100827) find a positive overnight market-factor
beta premium in an older Chinese A-share sample. Here a 60-day median-market
beta is only a proxy. Rules and stop thresholds are frozen in
docs/input-gates.md before reading the next opening prices.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .market_study import _quality_keys
from .quality_period import load_period_bad_symbols


ROOT = Path("data/research")
OUTPUT = ROOT / "market_beta_overnight"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")
STRATA = ["date", "board", "day_bin", "tail_bin", "trend_bin",
          "price_bin", "amount_bin"]


def _half(dates: pd.Series) -> pd.Series:
    return dates.str[:4] + "H" + np.where(dates.str[5:7].astype(int) <= 6, "1", "2")


def _connect() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '8GB'")
    return connection


def freeze(source_file: Path = ROOT / "prior_market_beta" / "all_candidates.parquet",
           prefix_audit_file: Path = ROOT / "minute_prefix_1449" / "input_audit.json",
           output_dir: Path = OUTPUT) -> dict:
    prefix_audit = json.loads(prefix_audit_file.read_text(encoding="utf-8"))
    if not prefix_audit["input_gate_passed"]:
        raise ValueError("14:49 minute input audit failed")
    connection = _connect()
    try:
        connection.from_parquet(str(source_file)).create_view("source")
        inputs = connection.execute("""
            WITH boarded AS (
                SELECT *, CASE
                    WHEN code LIKE 'sh.60%' OR code LIKE 'sz.00%' THEN 'main'
                    WHEN code LIKE 'sz.30%' THEN 'chinext'
                    WHEN code LIKE 'sh.68%' THEN 'star'
                    ELSE NULL END AS board
                FROM source
            ), ranked AS (
                SELECT *, COUNT(*) OVER (PARTITION BY date, board) AS board_size,
                       NTILE(5) OVER (PARTITION BY date, board
                                      ORDER BY market_beta, code) AS beta_quintile
                FROM boarded WHERE board IS NOT NULL
            )
            SELECT date, code, board, price_1449, amount_1449,
                   return_1449, return_last29, return20_prior_adjusted,
                   market_beta, beta_date, prior_count, board_size,
                   CASE WHEN beta_quintile = 5 THEN 'high'
                        ELSE 'middle' END AS arm,
                   CASE WHEN return_1449 < -.01 THEN 'down'
                        WHEN return_1449 > .01 THEN 'up'
                        ELSE 'flat' END AS day_bin,
                   CASE WHEN return_last29 < -.002 THEN 'down'
                        WHEN return_last29 > .002 THEN 'up'
                        ELSE 'flat' END AS tail_bin,
                   CASE WHEN return20_prior_adjusted < 0 THEN 'down'
                        ELSE 'up' END AS trend_bin,
                   CASE WHEN price_1449 < 20 THEN 'below20'
                        ELSE 'atleast20' END AS price_bin,
                   CASE WHEN amount_1449 < 300000000 THEN 'below300m'
                        ELSE 'atleast300m' END AS amount_bin
            FROM ranked
            WHERE board_size >= 30 AND beta_quintile IN (3, 5)
            ORDER BY date, board, code
        """).df()
    finally:
        connection.close()
    if (inputs.empty or inputs.duplicated(["date", "code"]).any()
            or not inputs.date.str[:4].isin(("2024", "2025")).all()
            or not inputs.beta_date.lt(inputs.date).all()
            or inputs.prior_count.lt(50).any()
            or not np.isfinite(inputs[["price_1449", "amount_1449",
                                       "market_beta"]].to_numpy()).all()):
        raise ValueError("Invalid decision-time beta inputs")
    inputs["half"] = _half(inputs.date)
    counts = inputs.groupby(STRATA + ["arm"]).size().unstack("arm", fill_value=0)
    comparable = counts.loc[counts.high.ge(3) & counts.middle.ge(3)].reset_index()[STRATA]
    inputs = inputs.merge(comparable.assign(comparable=True), on=STRATA,
                          how="left", validate="many_to_one")
    inputs["comparable"] = inputs.comparable.fillna(False).astype(bool)
    by_half = {}
    for half in HALVES:
        rows = inputs.loc[inputs.half.eq(half)]
        high = rows.loc[rows.arm.eq("high")]
        matched = rows.loc[rows.comparable]
        matched_high = matched.loc[matched.arm.eq("high")]
        matched_middle = matched.loc[matched.arm.eq("middle")]
        by_half[half] = {
            "high_stock_days": len(high),
            "matched_high_stock_days": len(matched_high),
            "matched_middle_stock_days": len(matched_middle),
            "matched_high_fraction": len(matched_high) / len(high) if len(high) else 0,
            "comparable_dates": int(matched_high.date.nunique()),
            "mean_beta_gap": (float(matched_high.market_beta.mean()
                                    - matched_middle.market_beta.mean())
                              if len(matched_high) and len(matched_middle) else None),
        }
    gate = all(
        row["comparable_dates"] >= 90
        and row["matched_high_stock_days"] >= 3000
        and row["matched_middle_stock_days"] >= 3000
        and row["matched_high_fraction"] >= .30
        and row["mean_beta_gap"] is not None
        and row["mean_beta_gap"] >= .30
        for row in by_half.values()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs.to_parquet(output_dir / "inputs.parquet", index=False, compression="zstd")
    audit = {"cutoff": "14:49", "source": str(source_file),
             "input_count": len(inputs), "scope": list(HALVES),
             "by_half": by_half, "outcome_gate_passed": bool(gate)}
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return audit


def _weekly_interval(daily: pd.DataFrame) -> list[float]:
    if daily.empty:
        return [float("nan"), float("nan")]
    weeks = pd.to_datetime(daily.date).dt.strftime("%G-%V")
    blocks = daily.groupby(weeks).edge.agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(20260926)
    sample = blocks[rng.integers(0, len(blocks), size=(2000, len(blocks)))]
    means = sample[:, :, 0].sum(axis=1) / sample[:, :, 1].sum(axis=1)
    return [float(value) for value in np.quantile(means, [.025, .975])]


def gross(input_dir: Path = OUTPUT,
          daily_dir: Path = Path("data/baostock/market_2020_2026/daily"),
          calendar_file: Path = Path(
              "data/baostock/market_2020_2026/metadata/calendar.parquet"),
          issues_dir: Path = ROOT / "market_issues_ci",
          period_quality: Path = ROOT / "quality_period_2024_2025.json") -> dict:
    audit = json.loads((input_dir / "input_audit.json").read_text(encoding="utf-8"))
    if not audit["outcome_gate_passed"]:
        raise ValueError("Input gate failed; do not read next-day prices")
    connection = _connect()
    try:
        connection.from_parquet(str(input_dir / "inputs.parquet")).create_view("inputs")
        connection.from_parquet(str(daily_dir / "*.parquet")).create_view("daily_source")
        connection.from_parquet(str(calendar_file)).create_view("calendar_source")
        connection.register("bad_days", _quality_keys(issues_dir))
        connection.register("bad_symbols", load_period_bad_symbols(
            period_quality, "2024-01-01", "2025-12-31"))
        joined = connection.execute("""
            WITH trading AS (
                SELECT calendar_date AS date,
                       LEAD(calendar_date) OVER (ORDER BY calendar_date) AS next_date
                FROM calendar_source WHERE is_trading_day = '1'
                  AND calendar_date BETWEEN '2024-01-01' AND '2025-12-31'
            ), daily AS (
                SELECT date, code, open, close, preclose, tradestatus
                FROM daily_source WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
            )
            SELECT i.*, t.next_date,
                   CASE WHEN n.date IS NOT NULL AND d.tradestatus = 1
                         AND n.tradestatus = 1 AND d.close > 0 AND n.open > 0
                         AND ABS(n.preclose - d.close) <= .005
                         AND b.code IS NULL AND q0.code IS NULL AND q1.code IS NULL
                        THEN TRUE ELSE FALSE END AS quality_clean,
                   n.open / i.price_1449 - 1 AS gross_1449
            FROM inputs i JOIN trading t ON i.date = t.date
            LEFT JOIN daily d ON i.date = d.date AND i.code = d.code
            LEFT JOIN daily n ON t.next_date = n.date AND i.code = n.code
            LEFT JOIN bad_symbols b ON i.code = b.code
            LEFT JOIN bad_days q0 ON i.date = q0.date AND i.code = q0.code
            LEFT JOIN bad_days q1 ON t.next_date = q1.date AND i.code = q1.code
        """).df()
    finally:
        connection.close()
    if len(joined) != audit["input_count"] or joined.duplicated(["date", "code"]).any():
        raise ValueError("Incomplete or duplicated next-open join")
    clean = joined.loc[joined.quality_clean].copy()
    if not np.isfinite(clean.gross_1449.to_numpy()).all():
        raise ValueError("Nonfinite quality-clean next-open return")
    matched_input = joined.loc[joined.comparable]
    matched_clean = clean.loc[clean.comparable]
    counts = matched_clean.groupby(STRATA + ["arm"]).gross_1449.agg(
        ["mean", "count"]).unstack("arm")
    counts = counts.loc[counts[("count", "high")].ge(3)
                        & counts[("count", "middle")].ge(3)]
    strata = counts[[ ("mean", "high"), ("mean", "middle") ]].reset_index()
    strata.columns = STRATA + ["high", "middle"]
    daily = strata.groupby("date", as_index=False)[["high", "middle"]].mean()
    daily["edge"] = daily.high - daily.middle
    daily["half"] = _half(daily.date)
    daily["year"] = daily.date.str[:4]
    full_high = (clean.loc[clean.arm.eq("high")]
                 .groupby("date", as_index=False).gross_1449.mean())
    full_high["half"] = _half(full_high.date)
    by_half = {}
    for half in HALVES:
        part = daily.loc[daily.half.eq(half)]
        clean_count = int(matched_clean.half.eq(half).sum())
        input_count = int(matched_input.half.eq(half).sum())
        all_high_dates = full_high.loc[full_high.half.eq(half)]
        by_half[half] = {
            "comparable_dates": len(part),
            "clean_fraction": clean_count / input_count,
            "matched_high_gross_pp": float(part.high.mean() * 100),
            "matched_middle_gross_pp": float(part.middle.mean() * 100),
            "matched_edge_pp": float(part.edge.mean() * 100),
            "full_high_gross_pp": float(all_high_dates.gross_1449.mean() * 100),
        }
    by_year = {}
    for year in ("2024", "2025"):
        part = daily.loc[daily.year.eq(year)]
        by_year[year] = {"matched_edge_pp": float(part.edge.mean() * 100),
                         "matched_edge_week_interval_pp": [
                             value * 100 for value in _weekly_interval(part)]}
    reprice_gate = (
        all(row["clean_fraction"] >= .95
            and row["matched_high_gross_pp"] > .25
            and row["full_high_gross_pp"] > .25
            and row["matched_edge_pp"] > .10
            for row in by_half.values())
        and all(row["matched_edge_week_interval_pp"][0] > 0
                for row in by_year.values())
    )
    report = {"scope": list(HALVES), "price_anchor": "14:49 to next daily open; not a fill",
              "bad_symbol_policy": "research_period", "input_count": audit["input_count"],
              "matched_input_count": len(matched_input),
              "matched_clean_count": len(matched_clean),
              "surviving_strata": len(strata), "by_half": by_half,
              "by_year": by_year, "raw_reprice_gate_passed": bool(reprice_gate)}
    (input_dir / "gross_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "gross"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = freeze(output_dir=args.output_dir) if args.stage == "freeze" else gross(
        input_dir=args.output_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
