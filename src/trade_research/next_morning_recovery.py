"""Frozen 14:49-to-next-10:00 gross-space check for all liquid A shares.

Motivation: Zhang, Zhang and Xue (Applied Economics, 2025,
doi:10.1080/00036846.2024.2387365) report a positive first-half-hour
component after China's typically negative overnight return.  That does not
establish profitability for a prior-afternoon buyer, so this program asks the
narrower question directly.  Its 10:00 raw-minute close is only a diagnostic
anchor, not an executable sell fill.  Do not read 2026 outcomes.

Before raw-minute buy/sell repricing, require in each 2024/2025 half year:
at least 100 clean dates, >=95% clean stock-day coverage, mean gross return
from 14:49 to 10:00 >0.25%, and 10:00 minus next-open gross gain >0.10%.
Annual weekly bootstrap lower bounds for the full gross return must be >0.
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
OUTPUT = ROOT / "next_morning_recovery"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")


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
    source = json.loads((prefix_dir / "input_audit.json").read_text(encoding="utf-8"))
    if not source["input_gate_passed"]:
        raise ValueError("14:49 source audit failed")
    c = _connect()
    try:
        c.from_parquet(str(prefix_dir / "*" / "part_*.parquet")).create_view("prefix")
        c.from_parquet(str(snapshot_dir / "*.parquet")).create_view("snapshots")
        inputs = c.execute("""
            SELECT p.date, p.code, p.price_1449,
                   CASE WHEN p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%'
                        THEN 'main'
                        WHEN p.code LIKE 'sz.30%' THEN 'chinext'
                        WHEN p.code LIKE 'sh.68%' THEN 'star'
                        ELSE NULL END AS board
            FROM prefix p JOIN snapshots s USING (date, code)
            WHERE p.date BETWEEN '2024-01-01' AND '2025-12-17'
              AND s.tradestatus = 1 AND s.isST = 0
              AND s.listing_age_sessions >= 20
              AND NOT s.reference_gap AND NOT p.quote_outside_traded_range
              AND s.preclose > 0 AND p.price_1449 >= 5
              AND p.amount_1449 BETWEEN 100000000 AND 1000000000
              AND s.return20_prior_adjusted BETWEEN -.10 AND .10
              AND p.return_last29 BETWEEN -.01 AND .01
              AND p.price_1449 / s.preclose - 1 BETWEEN -.03 AND .03
        """).df()
    finally:
        c.close()
    if inputs.empty or inputs.duplicated(["date", "code"]).any():
        raise ValueError("Empty or duplicate decision-time inputs")
    if not inputs.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("Non-development year in inputs")
    if not np.isfinite(inputs.price_1449.to_numpy()).all():
        raise ValueError("Nonfinite decision-time price")
    inputs = inputs.loc[inputs.board.notna()].copy()
    inputs["half"] = _half(inputs.date)
    by_half = {}
    for half in HALVES:
        part = inputs.loc[inputs.half.eq(half)]
        by_half[half] = {"stock_days": len(part), "dates": int(part.date.nunique()),
                         "symbols": int(part.code.nunique())}
    gate = all(row["stock_days"] >= 5000 and row["dates"] >= 100
               for row in by_half.values())
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
    values = daily.groupby(key).gross_1000.agg(["sum", "count"]).to_numpy()
    rng = np.random.default_rng(20260926)
    sampled = values[rng.integers(0, len(values), size=(2000, len(values)))]
    means = sampled[:, :, 0].sum(axis=1) / sampled[:, :, 1].sum(axis=1)
    return [float(x) for x in np.quantile(means, [.025, .975])]


def evaluate(input_dir: Path = OUTPUT,
             daily_dir: Path = Path("data/baostock/market_2020_2026/daily"),
             intraday_dir: Path = ROOT / "intraday_features",
             calendar_file: Path = Path(
                 "data/baostock/market_2020_2026/metadata/calendar.parquet"),
             issues_dir: Path = ROOT / "market_issues_ci",
             period_quality: Path | None = None) -> dict:
    audit = json.loads((input_dir / "input_audit.json").read_text(encoding="utf-8"))
    if not audit["outcome_gate_passed"]:
        raise ValueError("Input gate failed; do not read next-day minutes")
    c = _connect()
    try:
        c.from_parquet(str(input_dir / "inputs.parquet")).create_view("inputs")
        c.from_parquet(str(daily_dir / "*.parquet")).create_view("daily_source")
        c.from_parquet(str(intraday_dir / "202[45].parquet")).create_view("minute_source")
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
            SELECT i.date, i.code, i.board, i.half, t.next_date,
                   CASE WHEN n.date IS NOT NULL AND m.date IS NOT NULL
                         AND d.tradestatus = 1 AND n.tradestatus = 1
                         AND d.close > 0 AND n.open > 0 AND m.price_1000 > 0
                         AND ABS(n.preclose - d.close) <= .005
                         AND b.code IS NULL AND q0.code IS NULL AND q1.code IS NULL
                        THEN TRUE ELSE FALSE END AS quality_clean,
                   n.open / i.price_1449 - 1 AS gross_open,
                   m.price_1000 / i.price_1449 - 1 AS gross_1000
            FROM inputs i JOIN trading t ON i.date = t.date
            LEFT JOIN daily d ON i.date = d.date AND i.code = d.code
            LEFT JOIN daily n ON t.next_date = n.date AND i.code = n.code
            LEFT JOIN minute_source m ON t.next_date = m.date AND i.code = m.code
            LEFT JOIN bad_symbols b ON i.code = b.code
            LEFT JOIN bad_days q0 ON i.date = q0.date AND i.code = q0.code
            LEFT JOIN bad_days q1 ON t.next_date = q1.date AND i.code = q1.code
        """).df()
    finally:
        c.close()
    if len(joined) != audit["input_count"] or joined.duplicated(["date", "code"]).any():
        raise ValueError("Incomplete or duplicate next-day join")
    clean = joined.loc[joined.quality_clean].copy()
    if not np.isfinite(clean[["gross_open", "gross_1000"]].to_numpy()).all():
        raise ValueError("Nonfinite quality-clean next-day prices")
    clean["recovery"] = clean.gross_1000 - clean.gross_open
    daily = clean.groupby("date", as_index=False)[
        ["gross_open", "gross_1000", "recovery"]].mean()
    daily["half"] = _half(daily.date)
    daily["year"] = daily.date.str[:4]
    by_half = {}
    for half in HALVES:
        part = daily.loc[daily.half.eq(half)]
        input_count = audit["by_half"][half]["stock_days"]
        valid_count = int(clean.half.eq(half).sum())
        by_half[half] = {
            "dates": len(part), "clean_stock_days": valid_count,
            "clean_fraction": valid_count / input_count,
            "gross_open_pp": float(part.gross_open.mean() * 100),
            "gross_1000_pp": float(part.gross_1000.mean() * 100),
            "recovery_pp": float(part.recovery.mean() * 100),
        }
    by_year = {}
    for year in ("2024", "2025"):
        part = daily.loc[daily.year.eq(year)]
        by_year[year] = {
            "dates": len(part), "gross_1000_pp": float(part.gross_1000.mean() * 100),
            "gross_1000_week_interval_pp": [v * 100 for v in _week_interval(part)],
        }
    gate = (
        all(row["dates"] >= 100 and row["clean_fraction"] >= .95
            and row["gross_1000_pp"] > .25 and row["recovery_pp"] > .10
            for row in by_half.values())
        and all(row["gross_1000_week_interval_pp"][0] > 0
                for row in by_year.values())
    )
    report = {
        "scope": list(HALVES),
        "bad_symbol_policy": ("full_source" if period_quality is None
                              else "research_period"),
        "price_anchor": "14:49 to next 10:00 labeled minute; not a sell fill",
        "input_count": audit["input_count"], "quality_clean_count": len(clean),
        "by_half": by_half, "by_year": by_year,
        "raw_reprice_gate_passed": bool(gate),
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
