"""Frozen 14:49 minute-bounce execution-cost hypothesis.

Roll (1984), https://authors.library.caltech.edu/records/afwsc-6bb61,
shows why bid-ask trading can make successive observed price changes negatively
dependent. Our minute closes are not transaction directions or quotes, so rho1
is only a noisy proxy. The old strict-pair attempt stopped before outcomes
because it matched too few stocks. This new full-market design asks a narrower
question with no post-result relaxation of those failed pairing thresholds.

Economic prediction: after a negative 14:49 last-minute return, the bottom
third of 14:20--14:49 rho1 within each date and board has a higher 14:52--
14:55 buy VWAP relative to the known 14:49 price than the top third. A bounce
before the buy is an execution cost, not evidence of a profitable buy signal.

Freeze gate, before reading any 14:52+ price: in each 2024/2025 half year,
>=95% of the basic 14:49 pool has valid ACF; the down-tick pool has >=20,000
stock-days, >=100 dates, >=150 date-board groups of >=30 stocks, and >=5,000
stocks in each extreme third. High-minus-low median rho1 must be >=0.15.

Primary outcome (if input gate passes): join one unique archived T+1 entry
per selected stock-day, remove the modeled 5bp buy slippage, and measure
10,000*(VWAP_1452_1455/P_1449-1). Compare bottom minus top third within each
date-board, then date-equal and half-year-equal. Require >=99.9% outcome key
coverage, >=95% clean buy-fill coverage per third and half, bottom-minus-top
drift >5bp in all four halves, and both years' weekly bootstrap 95% lower
bounds >0. If this fails, stop before reading post-entry returns. Report
last-minute decline, tail return, day return, price, amount and variance
composition, and a same-date-board regression controlling those inputs so a
shared last-price denominator is not mistaken for a causal spread estimate.

Only if drift clears those gates: compare 10万元, T+1 close net return with
the same date-board controls. A potential buy filter additionally needs top
third net >0 in all four halves at 5bp and 15bp per-side slippage, positive
top-minus-bottom difference with annual weekly lower bounds >0, raw-minute
recalculation of a frozen stock-day sample, and independent 2026 validation.
2025 is already observed development data and never a blind test. No formula
may be published from this test alone.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("data/research")
OUTPUT = ROOT / "minute_bounce_cost"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")


def _half(dates: pd.Series) -> pd.Series:
    return dates.str[:4] + "H" + np.where(
        dates.str[5:7].astype(int) <= 6, "1", "2")


def freeze(prefix_dir: Path = ROOT / "minute_prefix_1449",
           snapshot_dir: Path = ROOT / "market_snapshots_ci",
           acf_dir: Path = ROOT / "minute_autocovariance_1449",
           output_dir: Path = OUTPUT) -> dict:
    prefix_audit = json.loads((prefix_dir / "input_audit.json").read_text())
    if not prefix_audit["input_gate_passed"]:
        raise ValueError("14:49 prefix source audit failed")
    if {p.name for p in acf_dir.glob("20??.parquet")} != {
            "2024.parquet", "2025.parquet"}:
        raise ValueError("Expected complete 2024 and 2025 ACF extractions")
    c = duckdb.connect()
    c.execute("SET threads = 4")
    c.execute("SET memory_limit = '8GB'")
    try:
        c.from_parquet(str(prefix_dir / "*" / "part_*.parquet")
                       ).create_view("prefix")
        c.from_parquet(str(snapshot_dir / "*.parquet")
                       ).create_view("snapshots")
        c.from_parquet(str(acf_dir / "20??.parquet")
                       ).create_view("acf")
        base = c.execute("""
            SELECT p.date, p.code, p.price_1449, p.amount_1449,
                   p.return_last29,
                   p.price_1449 / s.preclose - 1 AS day_return,
                   s.return20_prior_adjusted AS prior20,
                   CASE WHEN p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%'
                        THEN 'main'
                        WHEN p.code LIKE 'sz.30%' THEN 'chinext'
                        WHEN p.code LIKE 'sh.68%' THEN 'star'
                        ELSE NULL END AS board,
                   a.price_1449 AS acf_price, a.rho1,
                   a.last_minute_log_return,
                   a.realized_variance_last29
            FROM prefix p JOIN snapshots s USING(date, code)
            LEFT JOIN acf a USING(date, code)
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
    if base.empty or base.duplicated(["date", "code"]).any():
        raise ValueError("Missing or duplicated basic 14:49 stock-days")
    if not base.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("ACF inputs escaped the development years")
    base["half"] = _half(base.date)
    valid = base.loc[
        base.board.notna() & base.rho1.notna()
        & base.acf_price.sub(base.price_1449).abs().le(.005)
        & np.isfinite(base.rho1)
        & base.realized_variance_last29.gt(0)
        & base.last_minute_log_return.lt(0)
    ].copy()
    valid["date_board_count"] = valid.groupby(["date", "board"])[
        "code"].transform("size")
    valid = valid.loc[valid.date_board_count.ge(30)].copy()
    valid = valid.sort_values(["date", "board", "rho1", "code"])
    rank = valid.groupby(["date", "board"]).cumcount()
    valid["rho_third"] = np.minimum(3, (rank * 3 /
                                       valid.date_board_count).astype(int) + 1)
    if valid.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate selected ACF stock-day")
    by_half = {}
    for half in HALVES:
        b = base.loc[base.half.eq(half)]
        v = valid.loc[valid.half.eq(half)]
        low = v.loc[v.rho_third.eq(1)]
        high = v.loc[v.rho_third.eq(3)]
        by_half[half] = {
            "basic_stock_days": len(b),
            "acf_stock_days": int(b.rho1.notna().sum()),
            "acf_fraction": float(b.rho1.notna().mean()),
            "down_tick_stock_days": int((b.last_minute_log_return < 0).sum()),
            "supported_stock_days": len(v),
            "supported_dates": int(v.date.nunique()),
            "date_board_groups": int(v.groupby(["date", "board"]).ngroups),
            "low_third": len(low), "high_third": len(high),
            "rho_median_gap": float(high.rho1.median() - low.rho1.median()),
            "last_minute_median_gap": float(
                high.last_minute_log_return.median()
                - low.last_minute_log_return.median()),
        }
    gate = all(
        row["acf_fraction"] >= .95
        and row["down_tick_stock_days"] >= 20000
        and row["supported_dates"] >= 100
        and row["date_board_groups"] >= 150
        and row["low_third"] >= 5000 and row["high_third"] >= 5000
        and row["rho_median_gap"] >= .15
        for row in by_half.values()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    valid.to_parquet(output_dir / "inputs.parquet", index=False,
                     compression="zstd")
    report = {"cutoff": "14:49", "scope": list(HALVES),
              "input_only": True, "by_half": by_half,
              "input_gate_passed": bool(gate)}
    (output_dir / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(freeze(output_dir=args.output_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
