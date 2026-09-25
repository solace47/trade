"""Test whether very liquid stocks leave enough gross room after 14:49.

This is a frozen, two-stage diagnostic.  It is not a trading rule: the next
opening price is an outcome anchor, not an executable exit.  Only 2024–2025
stock-days may enter.  The input stage never reads the daily outcome files.

Economic hypothesis: high turnover can lower execution friction and perhaps
reflect more durable demand.  Compare >=1bn yuan turnover with 0.3–1bn yuan
within the same date, board, intraday-return bin and prior-20-day trend sign.
Before any raw-minute repricing, every half year must have high-group gross
14:49-to-next-open >0.25% and high-minus-mid >0.15 percentage points; annual
weekly bootstrap lower bounds on the edge must be positive.  Otherwise stop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .market_study import _quality_keys, _quality_symbols
from .quality_period import load_period_bad_symbols


ROOT = Path("data/research")
OUTPUT = ROOT / "high_liquidity_gross"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")
STRATA = ["date", "board", "day_bin", "trend_bin"]


def _connect() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("SET threads = 4")
    c.execute("SET memory_limit = '8GB'")
    return c


def _half(dates: pd.Series) -> pd.Series:
    return dates.str[:4] + "H" + np.where(dates.str[5:7].astype(int) <= 6, "1", "2")


def freeze(prefix_dir: Path = ROOT / "minute_prefix_1449",
           snapshot_dir: Path = ROOT / "market_snapshots_ci",
           output_dir: Path = OUTPUT) -> dict:
    audit = json.loads((prefix_dir / "input_audit.json").read_text(encoding="utf-8"))
    if not audit["input_gate_passed"]:
        raise ValueError("14:49 source audit failed")
    c = _connect()
    try:
        c.from_parquet(str(prefix_dir / "*" / "part_*.parquet")).create_view("prefix")
        c.from_parquet(str(snapshot_dir / "*.parquet")).create_view("snapshots")
        inputs = c.execute("""
            SELECT p.date, p.code, p.price_1449, p.amount_1449,
                   CASE WHEN p.amount_1449 >= 1000000000
                        THEN 'high' ELSE 'mid' END AS liquidity_group,
                   CASE WHEN p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%'
                        THEN 'main'
                        WHEN p.code LIKE 'sz.30%' THEN 'chinext'
                        WHEN p.code LIKE 'sh.68%' THEN 'star'
                        ELSE NULL END AS board,
                   CASE WHEN p.price_1449 / s.preclose - 1 < -.01 THEN 'down'
                        WHEN p.price_1449 / s.preclose - 1 > .01 THEN 'up'
                        ELSE 'flat' END AS day_bin,
                   CASE WHEN s.return20_prior_adjusted < 0 THEN 'negative'
                        ELSE 'positive' END AS trend_bin
            FROM prefix p JOIN snapshots s USING (date, code)
            WHERE p.date BETWEEN '2024-01-01' AND '2025-12-17'
              AND s.tradestatus = 1 AND s.isST = 0
              AND s.listing_age_sessions >= 20
              AND NOT s.reference_gap AND NOT p.quote_outside_traded_range
              AND s.preclose > 0 AND p.price_1449 >= 5
              AND p.amount_1449 >= 300000000
              AND s.return20_prior_adjusted BETWEEN -.10 AND .10
              AND p.return_last29 BETWEEN -.01 AND .01
              AND p.price_1449 / s.preclose - 1 BETWEEN -.03 AND .03
        """).df()
    finally:
        c.close()
    if inputs.empty or inputs.duplicated(["date", "code"]).any():
        raise ValueError("Empty or duplicate 14:49 input keys")
    if not inputs.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("Non-development year in inputs")
    if not np.isfinite(inputs[["price_1449", "amount_1449"]].to_numpy()).all():
        raise ValueError("Nonfinite decision-time input")
    inputs = inputs.loc[inputs.board.notna()].copy()
    inputs["half"] = _half(inputs.date)
    counts = inputs.groupby(STRATA + ["liquidity_group"]).size().unstack(
        "liquidity_group", fill_value=0
    )
    comparable = counts.loc[counts.high.ge(3) & counts.mid.ge(10)].reset_index()[STRATA]
    inputs = inputs.merge(comparable.assign(comparable=True), on=STRATA, how="left",
                          validate="many_to_one")
    inputs["comparable"] = inputs.comparable.fillna(False).astype(bool)
    by_half = {}
    for half in HALVES:
        segment = inputs.loc[inputs.half.eq(half)]
        high = segment.loc[segment.liquidity_group.eq("high")]
        matched = high.loc[high.comparable]
        by_half[half] = {
            "high_stock_days": len(high), "matched_high_stock_days": len(matched),
            "matched_high_fraction": len(matched) / len(high) if len(high) else 0,
            "comparable_dates": int(matched.date.nunique()),
            "matched_mid_stock_days": int((segment.comparable &
                segment.liquidity_group.eq("mid")).sum()),
        }
    gate = all(
        row["matched_high_stock_days"] >= 2500
        and row["matched_high_fraction"] >= .65
        and row["comparable_dates"] >= 100
        and row["matched_mid_stock_days"] >= 5000
        for row in by_half.values()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs.to_parquet(output_dir / "inputs.parquet", index=False, compression="zstd")
    report = {"cutoff": "14:49", "scope": list(HALVES),
              "input_count": len(inputs), "by_half": by_half,
              "outcome_gate_passed": bool(gate)}
    (output_dir / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _week_interval(daily: pd.DataFrame) -> list[float]:
    if daily.empty:
        return [float("nan"), float("nan")]
    key = pd.to_datetime(daily.date).dt.strftime("%G-%V")
    values = daily.groupby(key).edge.agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(20260926)
    sampled = values[rng.integers(0, len(values), size=(2000, len(values)))]
    means = sampled[:, :, 0].sum(axis=1) / sampled[:, :, 1].sum(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def evaluate(input_dir: Path = OUTPUT,
             daily_dir: Path = Path("data/baostock/market_2020_2026/daily"),
             calendar_file: Path = Path(
                 "data/baostock/market_2020_2026/metadata/calendar.parquet"),
             issues_dir: Path = ROOT / "market_issues_ci",
             period_quality: Path | None = None) -> dict:
    audit = json.loads((input_dir / "input_audit.json").read_text(encoding="utf-8"))
    if not audit["outcome_gate_passed"]:
        raise ValueError("Input gate failed; do not read next-day prices")
    c = _connect()
    try:
        c.from_parquet(str(input_dir / "inputs.parquet")).create_view("inputs")
        c.from_parquet(str(daily_dir / "*.parquet")).create_view("daily_source")
        c.from_parquet(str(calendar_file)).create_view("calendar_source")
        c.register("bad_days", _quality_keys(issues_dir))
        bad_symbols = (_quality_symbols(issues_dir) if period_quality is None
                       else load_period_bad_symbols(
                           period_quality, "2024-01-01", "2025-12-31"))
        c.register("bad_symbols", bad_symbols)
        joined = c.execute("""
            WITH trading AS (
                SELECT calendar_date AS date,
                       LEAD(calendar_date) OVER (ORDER BY calendar_date) AS next_date
                FROM calendar_source
                WHERE is_trading_day = '1'
                  AND calendar_date BETWEEN '2024-01-01' AND '2025-12-31'
            ), daily AS (
                SELECT date, code, open, close, preclose, tradestatus
                FROM daily_source WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
            )
            SELECT i.date, i.code, i.board, i.day_bin, i.trend_bin,
                   i.liquidity_group, i.half, i.comparable, t.next_date,
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
        c.close()
    if len(joined) != audit["input_count"] or joined.duplicated(["date", "code"]).any():
        raise ValueError("Incomplete or duplicate outcome join")
    comparable = joined.loc[joined.comparable]
    clean = comparable.loc[comparable.quality_clean].copy()
    if not np.isfinite(clean.gross_1449.to_numpy()).all():
        raise ValueError("Nonfinite quality-clean gross return")
    grouped = clean.groupby(STRATA + ["liquidity_group"]).gross_1449.agg(
        ["mean", "count"]
    ).unstack("liquidity_group")
    grouped = grouped.loc[grouped[("count", "high")].ge(3) &
                          grouped[("count", "mid")].ge(10)]
    strata = grouped[[ ("mean", "high"), ("mean", "mid") ]].reset_index()
    strata.columns = STRATA + ["high", "mid"]
    daily = strata.groupby("date", as_index=False)[["high", "mid"]].mean()
    daily["edge"] = daily.high - daily.mid
    daily["half"] = _half(daily.date)
    daily["year"] = daily.date.str[:4]
    by_half = {}
    for half in HALVES:
        part = daily.loc[daily.half.eq(half)]
        by_half[half] = {"days": len(part),
                         "high_gross_pp": float(part.high.mean() * 100),
                         "mid_gross_pp": float(part.mid.mean() * 100),
                         "edge_pp": float(part.edge.mean() * 100)}
    by_year = {}
    for year in ("2024", "2025"):
        part = daily.loc[daily.year.eq(year)]
        by_year[year] = {"days": len(part),
                         "edge_pp": float(part.edge.mean() * 100),
                         "edge_week_interval_pp": [v * 100 for v in _week_interval(part)]}
    raw_gate = (
        all(row["days"] >= 100 and row["high_gross_pp"] > .25
            and row["edge_pp"] > .15 for row in by_half.values())
        and all(row["edge_week_interval_pp"][0] > 0 for row in by_year.values())
    )
    report = {
        "scope": list(HALVES), "price_anchor": "14:49 to next daily open; not a fill",
        "bad_symbol_policy": ("full_source" if period_quality is None
                              else "research_period"),
        "matched_input_rows": len(comparable), "quality_clean_rows": len(clean),
        "quality_clean_fraction": len(clean) / len(comparable),
        "clean_matched_strata": len(strata), "clean_matched_dates": len(daily),
        "by_half": by_half, "by_year": by_year,
        "raw_reprice_gate_passed": bool(raw_gate),
    }
    (input_dir / "gross_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--period-quality", type=Path)
    args = parser.parse_args()
    report = freeze(output_dir=args.output_dir) if args.stage == "freeze" else evaluate(
        input_dir=args.output_dir, period_quality=args.period_quality)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
