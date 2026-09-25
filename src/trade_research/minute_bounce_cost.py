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

from .market_study import _quality_keys
from .quality_period import load_period_bad_symbols


ROOT = Path("data/research")
OUTPUT = ROOT / "minute_bounce_cost"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")
ENTRY_LABEL = "1452-1455"


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


def _date_equal_drift(frame: pd.DataFrame) -> pd.DataFrame:
    """Average within board first, then weight each date equally."""
    means = frame.groupby(["half", "date", "board", "rho_third"])[
        "drift_bps"].mean().unstack("rho_third")
    if 1 not in means or 3 not in means:
        raise ValueError("Both extreme thirds are required")
    paired = means.loc[means[1].notna() & means[3].notna()].copy()
    if paired.empty:
        raise ValueError("No same-date, same-board drift controls")
    paired["low_minus_high_bps"] = paired[1] - paired[3]
    return paired.reset_index().groupby(["half", "date"], as_index=False)[
        "low_minus_high_bps"].mean()


def _yearly_week_intervals(daily: pd.DataFrame, draws: int = 2000,
                           seed: int = 20260925) -> dict[str, list[float]]:
    """Resample weeks within each half so both halves keep equal weight."""
    rng = np.random.default_rng(seed)
    intervals = {}
    rows = daily.copy()
    rows["week"] = pd.to_datetime(rows.date).dt.to_period("W-SUN").astype(str)
    for year in ("2024", "2025"):
        half_draws = []
        for half in (f"{year}H1", f"{year}H2"):
            weekly = rows.loc[rows.half.eq(half)].groupby("week")[
                "low_minus_high_bps"].agg(["sum", "count"])
            if len(weekly) < 12:
                raise ValueError(f"Too few independent weeks in {half}")
            indexes = rng.integers(0, len(weekly), (draws, len(weekly)))
            totals = weekly["sum"].to_numpy()[indexes].sum(axis=1)
            counts = weekly["count"].to_numpy()[indexes].sum(axis=1)
            half_draws.append(totals / counts)
        samples = (half_draws[0] + half_draws[1]) / 2
        intervals[year] = np.quantile(samples, [.025, .975]).tolist()
    return intervals


def _controlled_gap(frame: pd.DataFrame) -> dict[str, float | int]:
    """Within-date-board linear adjustment; descriptive, not causal."""
    extreme = frame.loc[frame.rho_third.isin((1, 3))].copy()
    extreme["low"] = extreme.rho_third.eq(1).astype(float)
    extreme["last_bps"] = extreme.last_minute_log_return * 10_000
    extreme["tail_bps"] = extreme.return_last29 * 10_000
    extreme["day_bps"] = extreme.day_return * 10_000
    extreme["log_price"] = np.log(extreme.price_1449)
    extreme["log_amount"] = np.log(extreme.amount_1449)
    extreme["log_variance"] = np.log(extreme.realized_variance_last29)
    predictors = ("low", "last_bps", "tail_bps", "day_bps", "log_price",
                  "log_amount", "log_variance")
    fields = [*predictors, "drift_bps"]
    groups = extreme.groupby(["date", "board"])
    extreme = extreme.loc[groups.rho_third.transform("nunique").eq(2)].copy()
    extreme[fields] -= extreme.groupby(["date", "board"])[
        fields].transform("mean")
    x = extreme[list(predictors)].to_numpy(dtype=float)
    y = extreme.drift_bps.to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank != len(predictors):
        raise ValueError("Controlled drift design is rank deficient")
    return {"rows": len(extreme), "date_board_groups": int(extreme.groupby(
        ["date", "board"]).ngroups),
        "adjusted_low_minus_high_bps": float(coefficients[0])}


def evaluate_entry(
    input_dir: Path = OUTPUT,
    outcomes_dir: Path = ROOT / "market_outcomes_ci",
    issues_dir: Path = ROOT / "market_issues_ci",
    period_quality: Path = ROOT / "quality_period_2024_2025.json",
) -> dict:
    audit = json.loads((input_dir / "input_audit.json").read_text())
    if not audit["input_gate_passed"] or audit["scope"] != list(HALVES):
        raise ValueError("Frozen input gate did not pass")
    inputs = pd.read_parquet(input_dir / "inputs.parquet")
    if inputs.empty or inputs.duplicated(["date", "code"]).any():
        raise ValueError("Frozen inputs are missing or nonunique")
    if not inputs.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("Outcome request escaped development years")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 4")
        connection.execute("SET memory_limit = '8GB'")
        connection.register("signals", inputs[["date", "code"]])
        connection.read_parquet(str(outcomes_dir / "*.parquet")
                                ).create_view("outcomes")
        # Intentionally project no exit field or post-entry return at this gate.
        entries = connection.execute("""
            SELECT i.date, i.code, o.entry_status, o.entry_price
            FROM signals i LEFT JOIN outcomes o
              ON i.date = o.date AND i.code = o.code
             AND o.horizon = 1 AND o.entry_label = '1452-1455'
             AND o.exit_label = '1452-1455'
        """).df()
    finally:
        connection.close()
    if entries.duplicated(["date", "code"]).any():
        raise ValueError("Nonunique archived T+1 entry")
    if len(entries) != len(inputs):
        raise ValueError("T+1 entry key cardinality changed")
    rows = inputs.merge(entries, on=["date", "code"], validate="one_to_one")
    bad_days = _quality_keys(issues_dir)
    bad_days["bad_day"] = True
    rows = rows.merge(bad_days, on=["date", "code"], how="left",
                      validate="one_to_one")
    bad_symbols = set(load_period_bad_symbols(
        period_quality, "2024-01-01", "2025-12-17").code.dropna())
    rows["clean_fill"] = (
        rows.entry_status.eq("filled") & rows.entry_price.gt(0)
        & rows.bad_day.isna() & ~rows.code.isin(bad_symbols)
    )
    rows["drift_bps"] = np.where(
        rows.clean_fill,
        10_000 * ((rows.entry_price / 1.0005) / rows.price_1449 - 1),
        np.nan,
    )
    composition = {}
    by_half = {}
    for half in HALVES:
        group = rows.loc[rows.half.eq(half)]
        by_third = {}
        for third in (1, 2, 3):
            part = group.loc[group.rho_third.eq(third)]
            by_third[str(third)] = {
                "signals": len(part),
                "clean_fills": int(part.clean_fill.sum()),
                "clean_fill_fraction": float(part.clean_fill.mean()),
                "mean_drift_bps": float(part.drift_bps.mean()),
            }
        by_half[half] = {"thirds": by_third}
        composition[half] = {
            str(third): {
                "last_minute_bps": float(part.last_minute_log_return.median() * 10_000),
                "tail29_bps": float(part.return_last29.median() * 10_000),
                "day_bps": float(part.day_return.median() * 10_000),
                "price": float(part.price_1449.median()),
                "amount_yuan": float(part.amount_1449.median()),
                "variance": float(part.realized_variance_last29.median()),
            }
            for third in (1, 3)
            for part in [group.loc[group.rho_third.eq(third)]]
        }
    clean = rows.loc[rows.clean_fill].copy()
    daily = _date_equal_drift(clean)
    for half in HALVES:
        part = daily.loc[daily.half.eq(half)]
        by_half[half]["paired_dates"] = len(part)
        by_half[half]["date_equal_low_minus_high_bps"] = float(
            part.low_minus_high_bps.mean())
        by_half[half]["adjusted"] = _controlled_gap(
            clean.loc[clean.half.eq(half)])
    intervals = _yearly_week_intervals(daily)
    key_fraction = float(rows.entry_status.notna().mean())
    gate = (
        key_fraction >= .999
        and all(by_half[h]["thirds"][str(t)]["clean_fill_fraction"] >= .95
                for h in HALVES for t in (1, 3))
        and all(by_half[h]["date_equal_low_minus_high_bps"] > 5
                for h in HALVES)
        and all(intervals[y][0] > 0 for y in ("2024", "2025"))
    )
    report = {
        "input_cutoff": "14:49", "entry_window": ENTRY_LABEL,
        "read_post_entry_returns": False,
        "outcome_key_fraction": key_fraction,
        "by_half": by_half,
        "composition_medians": composition,
        "annual_week_bootstrap_95_bps": intervals,
        "post_entry_return_gate_passed": bool(gate),
        "note": "Entry VWAP inferred by removing fixed 5bp model slippage; minute VWAP cannot prove order-book fills.",
    }
    (input_dir / "entry_drift_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("freeze", "entry"),
                        default="freeze")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = (freeze(output_dir=args.output_dir) if args.stage == "freeze"
              else evaluate_entry(input_dir=args.output_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
