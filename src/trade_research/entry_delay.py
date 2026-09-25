"""Audit and evaluate a pre-scheduled one-minute tail-entry delay."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .annual_cash_direct_eval import _month_bootstrap
from .market_study import _week_bootstrap
from .strategy_scan import _stressed_returns


ROOT = Path("data/research")
SIGNALS = ROOT / "late_variance_risk" / "selections.parquet"
MINUTES = Path("data/hf/pilot/data/stock_1m")
OUTPUT = ROOT / "entry_delay"
REQUIRED = ("1452", "1453", "1454", "1455", "1456")
REPRICED = OUTPUT / "repriced.parquet"


def _period(date: str) -> str:
    return date[:4] + ("H1" if int(date[5:7]) <= 6 else "H2")


def audit_coverage(signals_path: Path = SIGNALS, minute_root: Path = MINUTES,
                   output_dir: Path = OUTPUT, workers: int = 4) -> dict:
    if workers < 1:
        raise ValueError("Workers must be positive")
    signals = pd.read_parquet(signals_path, columns=["date", "code"])
    if (signals.empty or signals.duplicated(["date", "code"]).any()
            or not signals.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Invalid frozen input list")

    def one(item: tuple[str, str]) -> dict:
        date, code = item
        exchange, symbol = code.split(".")
        path = minute_root / exchange.upper() / f"{symbol}.parquet"
        timestamps = pd.read_parquet(
            path, columns=["timestamp"],
            filters=[("timestamp", ">=", pd.Timestamp(date + " 14:52")),
                     ("timestamp", "<=", pd.Timestamp(date + " 14:56"))],
        ).timestamp
        labels = tuple(timestamps.dt.strftime("%H%M").tolist())
        return {"date": date, "code": code, "complete": labels == REQUIRED}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(one, signals.itertuples(index=False, name=None)))
    frame = pd.DataFrame(rows)
    frame["period"] = frame.date.map(_period)
    by_half = frame.groupby("period").complete.agg(["size", "sum"]).reset_index()
    if set(by_half.period) != {"2024H1", "2024H2", "2025H1", "2025H2"}:
        raise ValueError("Coverage audit lacks a half-year")
    periods = {row.period: {"stockdays": int(row.size),
                            "complete": int(row.sum),
                            "rate": float(row.sum / row.size)}
               for row in by_half.itertuples(index=False)}
    report = {
        "signals": len(signals),
        "complete": int(frame.complete.sum()),
        "by_half": periods,
        "outcome_gate_passed": all(item["rate"] >= .99
                                   for item in periods.values()),
        "note": "Five labels 14:52-14:56 must each appear exactly once",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "coverage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _archive_check(repriced: pd.DataFrame) -> dict:
    connection = duckdb.connect()
    connection.register("new", repriced)
    connection.read_parquet(str(ROOT / "late_variance_risk" / "repriced.parquet")
                            ).create_view("old")
    result = connection.execute("""
        SELECT COUNT(*) AS rows,
               COUNT(*) FILTER (WHERE o.code IS NULL) AS missing,
               COUNT(*) FILTER (
                   WHERE n.entry_status IS DISTINCT FROM o.entry_status
                      OR n.exit_status IS DISTINCT FROM o.exit_status
                      OR n.exit_date IS DISTINCT FROM o.exit_date
                      OR n.shares IS DISTINCT FROM o.shares
                      OR n.entry_price IS DISTINCT FROM o.entry_price
                      OR n.exit_price IS DISTINCT FROM o.exit_price
                      OR n.net_return IS DISTINCT FROM o.net_return
                      OR n.quality_clean_exit
                          IS DISTINCT FROM o.quality_clean_exit
               ) AS mismatches
        FROM new n LEFT JOIN old o
        USING (date, code, horizon, target_notional)
        WHERE n.entry_window = 'baseline'
    """).df().iloc[0].to_dict()
    if (result["rows"] != len(repriced) // 2
            or result["missing"] or result["mismatches"]):
        raise ValueError("Baseline minute repricing differs from frozen archive")
    return {key: int(value) for key, value in result.items()}


def _intervals(values: pd.Series, dates: pd.Series, seed: int) -> dict:
    return {
        "week_ci": _week_bootstrap(values.reset_index(drop=True),
                                   dates.reset_index(drop=True), seed),
        "month_ci": _month_bootstrap(values.reset_index(drop=True),
                                     dates.reset_index(drop=True), seed + 1),
    }


def _summary(rows: pd.DataFrame) -> dict:
    baseline = rows.loc[rows.entry_window.eq("baseline")]
    delayed = rows.loc[rows.entry_window.eq("delay_one_minute")]
    both = baseline.merge(
        delayed, on=["date", "code", "candidate", "pair_id",
                     "target_notional", "horizon"],
        suffixes=("_baseline", "_delayed"), validate="one_to_one",
    )
    if len(both) != len(baseline) or len(both) != len(delayed):
        raise ValueError("An entry arm lacks a frozen stock-day")
    comparable = both.loc[
        both.valid_baseline & both.valid_delayed
        & both.exit_date_baseline.eq(both.exit_date_delayed)
    ].copy()
    comparable["delta"] = (comparable.net_return_delayed
                           - comparable.net_return_baseline)
    common_low = comparable.loc[
        comparable.candidate.eq("normal_variance_control")
    ]
    common_high = comparable.loc[
        comparable.candidate.eq("high_variance")
    ]
    low_daily = common_low.groupby("date", sort=True).delta.mean()
    high_daily = common_high.groupby("date", sort=True).delta.mean()
    timing_pairs = common_high[["date", "pair_id", "delta"]].merge(
        common_low[["date", "pair_id", "delta"]],
        on=["date", "pair_id"], suffixes=("_high", "_low"),
        validate="one_to_one",
    )
    if common_low.empty or common_high.empty or timing_pairs.empty:
        raise ValueError("No comparable delayed entries in a period")
    timing_pairs["interaction"] = (timing_pairs.delta_low
                                   - timing_pairs.delta_high)
    interaction_daily = timing_pairs.groupby("date", sort=True).interaction.mean()

    delayed_high = delayed.loc[
        delayed.candidate.eq("high_variance")
    ]
    delayed_low = delayed.loc[
        delayed.candidate.eq("normal_variance_control")
    ]
    selected_pairs = delayed_high.merge(
        delayed_low, on=["date", "pair_id"],
        suffixes=("_high", "_low"), validate="one_to_one",
    )
    if len(selected_pairs) != len(delayed) // 2:
        raise ValueError("Delayed treatment lacks a same-day control")
    daily = selected_pairs.groupby("date", sort=True).agg(
        high_cash=("cash_high", "mean"), low_cash=("cash_low", "mean"),
        low_stress15=("stress15_low", "mean"),
        high_stress15=("stress15_high", "mean"),
    )
    daily["edge"] = daily.low_cash - daily.high_cash
    daily["stress_edge"] = daily.low_stress15 - daily.high_stress15
    dates = pd.Series(daily.index.tolist())
    result = {
        "signals_per_arm": len(delayed),
        "pairs": len(selected_pairs), "days": len(daily),
        "low_entry_rate": float(delayed_low.entry_status.eq("filled").mean()),
        "low_clean_exit_rate": float(delayed_low.valid.mean()),
        "high_entry_rate": float(delayed_high.entry_status.eq("filled").mean()),
        "high_clean_exit_rate": float(delayed_high.valid.mean()),
        "low_cash": float(daily.low_cash.mean()),
        "high_cash": float(daily.high_cash.mean()),
        "low_minus_high": float(daily.edge.mean()),
        "low_cash_stress15": float(daily.low_stress15.mean()),
        "low_minus_high_stress15": float(daily.stress_edge.mean()),
        "control_intervals": _intervals(daily.edge, dates, 936),
        "common_low_trades": len(common_low),
        "common_low_days": len(low_daily),
        "low_delay_minus_baseline_common": float(low_daily.mean()),
        "high_delay_minus_baseline_common": float(high_daily.mean()),
        "timing_interaction_pairs": len(timing_pairs),
        "low_minus_high_timing_interaction": float(interaction_daily.mean()),
    }
    if len(low_daily):
        result["timing_intervals"] = _intervals(
            low_daily, pd.Series(low_daily.index.tolist()), 937,
        )
    if len(interaction_daily):
        result["timing_interaction_intervals"] = _intervals(
            interaction_daily, pd.Series(interaction_daily.index.tolist()), 938,
        )
    low = both.loc[both.candidate.eq("normal_variance_control")]
    both_entries = low.loc[low.entry_status_baseline.eq("filled")
                           & low.entry_status_delayed.eq("filled")]
    result["both_low_entries"] = len(both_entries)
    if len(both_entries):
        result["low_entry_price_improvement"] = float((
            1 - both_entries.entry_price_delayed
            / both_entries.entry_price_baseline
        ).mean())
    return result


def evaluate(signals_path: Path = SIGNALS, repriced_path: Path = REPRICED,
             output_dir: Path = OUTPUT) -> dict:
    coverage = json.loads((output_dir / "coverage.json").read_text(
        encoding="utf-8"))
    if not coverage["outcome_gate_passed"]:
        raise ValueError("Entry-minute coverage gate failed")
    signals = pd.read_parquet(signals_path)
    repriced = pd.read_parquet(repriced_path)
    if (len(repriced) != len(signals) * 8
            or repriced.duplicated(["date", "code", "target_notional",
                                    "horizon", "entry_window"]).any()
            or set(repriced.entry_window) != {"baseline", "delay_one_minute"}
            or set(repriced.horizon) != {1, 5}
            or set(repriced.target_notional) != {20000, 100000}):
        raise ValueError("Incomplete or duplicated entry-delay repricing")
    archive_check = _archive_check(repriced)
    labelled = repriced.merge(
        signals[["date", "code", "candidate", "pair_id"]],
        on=["date", "code"], validate="many_to_one",
    )
    if len(labelled) != len(repriced) or labelled.candidate.isna().any():
        raise ValueError("An entry arm lacks a frozen candidate label")
    labelled["valid"] = (labelled.exit_status.eq("filled")
                         & labelled.quality_clean_exit)
    if not np.allclose(_stressed_returns(labelled.loc[labelled.valid], 5),
                       labelled.loc[labelled.valid, "net_return"],
                       atol=1e-12):
        raise ValueError("Repricing disagrees with the stored fee model")
    labelled["cash"] = labelled.net_return.where(labelled.valid, 0)
    labelled["stress15"] = 0.0
    labelled.loc[labelled.valid, "stress15"] = _stressed_returns(
        labelled.loc[labelled.valid], 15,
    )
    report = {
        "archive_check": archive_check,
        "coverage": coverage,
        "note": "2025 not blind; delayed entry pre-scheduled at 14:50; 2026 unused",
        "results": {},
    }
    for amount in (20000, 100000):
        report["results"][str(amount)] = {}
        for horizon in (1, 5):
            selected = labelled.loc[
                labelled.target_notional.eq(amount)
                & labelled.horizon.eq(horizon)
            ]
            periods = {}
            for year in ("2024", "2025"):
                annual = selected.loc[selected.date.str.startswith(year)]
                for period, frame in (
                    (f"{year}H1", annual.loc[
                        annual.date.str[5:7].astype(int).le(6)]),
                    (f"{year}H2", annual.loc[
                        annual.date.str[5:7].astype(int).gt(6)]),
                    (year, annual),
                ):
                    periods[period] = _summary(frame)
            report["results"][str(amount)][str(horizon)] = periods
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--signals", type=Path, default=SIGNALS)
    parser.add_argument("--minute-root", type=Path, default=MINUTES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        report = evaluate(args.signals,
                          args.output_dir / "repriced.parquet", args.output_dir)
        print({"archive_check": report["archive_check"],
               "report": str(args.output_dir / "report.json")})
    else:
        print(audit_coverage(args.signals, args.minute_root, args.output_dir,
                             args.workers))


if __name__ == "__main__":
    main()
