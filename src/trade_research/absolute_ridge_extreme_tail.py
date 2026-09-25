"""Diagnose the frozen main-board ridge model's most extreme scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .absolute_ridge_calibration import HALVES
from .annual_cash_direct_eval import _month_bootstrap
from .late_variance_risk_eval import _archive_check
from .market_study import _quality_keys, _quality_symbols
from .strategy_scan import _stressed_returns


ROOT = Path("data/research")
OUTPUT = ROOT / "absolute_ridge_extreme_tail"
GROUPS = ("raised", "high", "extreme")


def assign_groups(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank within the signal day; never use an outcome to define the groups."""
    if scores.empty or scores.duplicated(["date", "code"]).any():
        raise ValueError("Missing or duplicated model scores")
    ranked = scores.sort_values(["date", "score", "code"]).copy()
    fraction = (ranked.groupby("date").cumcount()
                / ranked.groupby("date").code.transform("size"))
    ranked["group"] = np.select(
        [fraction >= .995, fraction >= .95, fraction >= .8],
        ["extreme", "high", "raised"], default="other")
    return ranked.loc[ranked.group.ne("other")].reset_index(drop=True)


def freeze_inputs(output_dir: Path = OUTPUT) -> dict:
    source = pd.read_parquet(ROOT / "absolute_ridge_calibration" / "scores.parquet")
    grouped = assign_groups(source)
    if (not grouped.half.isin(HALVES).all()
            or grouped.duplicated(["date", "code"]).any()):
        raise ValueError("The frozen score universe changed")
    counts = grouped.groupby(["half", "date", "group"]).size().unstack()
    expected = set(HALVES) == set(counts.index.get_level_values("half"))
    expected = expected and all(group in counts for group in GROUPS)
    expected = expected and counts.notna().all().all()
    minimum_extreme = int(counts.extreme.min()) if expected else 0
    half_days = grouped.groupby("half").date.nunique().to_dict()
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.register("grouped", grouped[["date", "code"]])
    connection.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                            ).create_view("outcomes")
    archive_rows, matched = connection.execute("""
        SELECT COUNT(*), COUNT(o.code) FROM grouped g
        LEFT JOIN outcomes o ON g.date = o.date AND g.code = o.code
                            AND o.horizon = 5
    """).fetchone()
    gate = (expected and minimum_extreme >= 3
            and all(half_days.get(half, 0) >= 80 for half in HALVES)
            and archive_rows == matched == len(grouped))
    extreme_signals = pd.DataFrame()
    if gate:
        connection.register("extreme_keys", grouped.loc[
            grouped.group.eq("extreme"), ["date", "code"]])
        connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                                ).create_view("snapshots")
        extreme_signals = connection.execute("""
            SELECT s.* FROM extreme_keys e JOIN snapshots s USING (date, code)
        """).df()
        if (len(extreme_signals) != int(grouped.group.eq("extreme").sum())
                or extreme_signals.duplicated(["date", "code"]).any()):
            raise ValueError("An extreme-score stock lacks a unique snapshot")
    audit = {
        "grouped_rows": len(grouped), "half_days": half_days,
        "minimum_daily_extreme": minimum_extreme,
        "archive_rows": archive_rows, "archive_matched": matched,
        "extreme_repricing_signals": len(extreme_signals),
        "outcome_gate_passed": bool(gate),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("report.json", "repriced.parquet", "repricing_signals.parquet"):
        (output_dir / stale).unlink(missing_ok=True)
    grouped.to_parquet(output_dir / "groups.parquet", index=False,
                       compression="zstd")
    if gate:
        extreme_signals.to_parquet(
            output_dir / "repricing_signals.parquet", index=False,
            compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return audit


def summarize_daily(daily: pd.DataFrame) -> dict:
    if daily.empty or daily.groupby("date").group.nunique().ne(3).any():
        raise ValueError("Every test date must have all three frozen groups")
    wide = daily.pivot(index="date", columns="group", values="cash")
    stressed = daily.pivot(index="date", columns="group", values="stress15")
    main_diff = wide.extreme - wide.raised
    high_diff = wide.extreme - wide.high
    dates = pd.Series(wide.index)
    aggregates = daily.groupby("group").agg(
        stocks=("stocks", "sum"), cash=("cash", "mean"),
        stress15=("stress15", "mean"), entry=("entry", "mean"),
        clean_exit=("clean_exit", "mean"), score=("score", "mean"),
    )
    return {
        "days": len(wide),
        "groups": {str(k): {name: float(value) for name, value in row.items()}
                   for k, row in aggregates.to_dict(orient="index").items()},
        "extreme_minus_raised": float(main_diff.mean()),
        "extreme_minus_raised_month_ci": _month_bootstrap(
            main_diff.reset_index(drop=True), dates, 3171),
        "extreme_minus_high": float(high_diff.mean()),
        "extreme_minus_high_month_ci": _month_bootstrap(
            high_diff.reset_index(drop=True), dates, 3173),
        "extreme_minus_raised_stress15": float(
            (stressed.extreme - stressed.raised).mean()),
    }


def evaluate(output_dir: Path = OUTPUT) -> dict:
    audit = json.loads((output_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if not audit["outcome_gate_passed"]:
        raise ValueError("Frozen extreme-tail input gate failed")
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(output_dir / "groups.parquet")
                            ).create_view("groups")
    connection.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                            ).create_view("outcomes")
    connection.register("bad_days", _quality_keys(ROOT / "market_issues_ci"))
    connection.register("bad_symbols", _quality_symbols(ROOT / "market_issues_ci"))
    rows = connection.execute("""
        SELECT g.date, g.code, g.half, g."group", g.score, o.horizon,
               o.entry_status, o.entry_price, o.shares,
               o.exit_status, o.exit_date, o.exit_delay_sessions,
               o.exit_price, o.net_return,
               o.exit_status = 'filled' AND o.exit_delay_sessions = 0
               AND NOT EXISTS (SELECT 1 FROM bad_symbols b
                               WHERE b.code = g.code)
               AND NOT EXISTS (
                   SELECT 1 FROM bad_days q WHERE q.code = g.code
                     AND q.date >= g.date AND q.date <= o.exit_date
               ) AS valid
        FROM groups g JOIN outcomes o USING (date, code)
        WHERE o.horizon IN (1, 5)
    """).df()
    if (len(rows) != 2 * audit["grouped_rows"]
            or rows.duplicated(["date", "code", "horizon"]).any()
            or rows.loc[rows.valid, "net_return"].isna().any()):
        raise ValueError("Missing or duplicated archived minute outcomes")
    rows["cash"] = rows.net_return.where(rows.valid, 0.0)
    rows["stress15"] = 0.0
    rows.loc[rows.valid, "stress15"] = _stressed_returns(
        rows.loc[rows.valid], 15)
    rows["entry"] = rows.entry_status.eq("filled").astype(float)
    rows["clean_exit"] = rows.valid.astype(float)
    daily = rows.groupby(["horizon", "date", "half", "group"],
                         sort=True).agg(
        stocks=("code", "size"), cash=("cash", "mean"),
        stress15=("stress15", "mean"), score=("score", "mean"),
        entry=("entry", "mean"), clean_exit=("clean_exit", "mean"),
    ).reset_index()
    report = {"input_audit": audit, "results": {}}
    repriced_path = output_dir / "repriced.parquet"
    if repriced_path.exists():
        repriced = pd.read_parquet(repriced_path)
        expected = 2 * audit["extreme_repricing_signals"]
        if (len(repriced) != expected
                or set(repriced.horizon) != {1, 5}
                or set(repriced.target_notional) != {100000}
                or set(repriced.exit_window) != {"close"}
                or repriced.duplicated(["date", "code", "horizon"]).any()):
            raise ValueError("Incomplete extreme-score raw-minute repricing")
        report["extreme_raw_minute_check"] = _archive_check(repriced)
    for horizon in (5, 1):
        selected = daily.loc[daily.horizon.eq(horizon)]
        report["results"][str(horizon)] = {
            half: summarize_daily(selected.loc[selected.half.eq(half)])
            for half in HALVES
        }
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.output) if args.evaluate else freeze_inputs(args.output)
    print({"report": str(args.output / "report.json")}
          if args.evaluate else result)


if __name__ == "__main__":
    main()
