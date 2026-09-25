"""Evaluate frozen discounted-block pairs with original minute executions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .annual_cash_direct_eval import _month_bootstrap
from .block_trade_inputs import CONTROL, EVENT
from .buyback_reprice import _exact
from .hf_outcomes import Assumptions
from .market_study import _quality_keys, _quality_symbols
from .strategy_scan import _stressed_returns


def _check_membership(pairs: pd.DataFrame, audit: dict) -> None:
    scope = audit.get("pair_scope")
    if (pairs.empty or pairs.duplicated(["date", "code"]).any()
            or scope not in {"capacity", "all_eligible"}
            or set(pairs.candidate) != {EVENT, CONTROL}
            or len(pairs) != 2 * sum(audit["year"][year]["matched_pairs"]
                                      for year in ("2024", "2025"))
            or not pairs.date.str[:4].isin(("2024", "2025")).all()
            or not pairs.date.gt(pairs.trade_date).all()):
        raise ValueError("Block trade pairs differ from frozen input audit")
    groups = pairs.groupby(["date", "pair_code"]).candidate.agg(
        ["size", "nunique"])
    if not groups["size"].eq(2).all() or not groups["nunique"].eq(2).all():
        raise ValueError("Block trade event lost its same-day control")
    events = pairs.loc[pairs.candidate.eq(EVENT)]
    controls = pairs.loc[pairs.candidate.eq(CONTROL)]
    compared = events[["date", "pair_code", "board", "industry"]].merge(
        controls[["date", "pair_code", "board", "industry"]],
        on=["date", "pair_code"], suffixes=("_event", "_control"),
        validate="one_to_one")
    if ((scope == "capacity" and events.groupby("date").size().gt(5).any())
            or len(compared) != len(events)
            or not compared.board_event.eq(compared.board_control).all()
            or not compared.industry_event.eq(compared.industry_control).all()):
        raise ValueError("Block trade capacity, board or industry changed")
    for year in ("2024", "2025"):
        section = events.loc[events.date.str.startswith(year)]
        expected = audit["year"][year]
        if (len(section) != expected["matched_pairs"]
                or section.date.nunique() != expected["matched_days"]):
            raise ValueError("Block trade yearly pair count changed")


def _stored_100k(pairs: pd.DataFrame, outcome_dir: Path,
                 issues_dir: Path) -> pd.DataFrame:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.register("pairs", pairs[["date", "code"]])
    connection.read_parquet(str(outcome_dir / "*.parquet")).create_view(
        "outcomes")
    connection.register("bad_days", _quality_keys(issues_dir))
    connection.register("bad_symbols", _quality_symbols(issues_dir))
    frame = connection.execute("""
        SELECT o.*, o.exit_status = 'filled'
          AND NOT EXISTS (SELECT 1 FROM bad_symbols b
                          WHERE b.code = o.code)
          AND NOT EXISTS (SELECT 1 FROM bad_days b
                          WHERE b.code = o.code
                            AND b.date >= o.date
                            AND b.date <= o.exit_date)
          AS quality_clean_exit
        FROM pairs p JOIN outcomes o USING (date, code)
        WHERE o.horizon IN (1, 5)
    """).df()
    if (len(frame) != len(pairs) * 2
            or frame.duplicated(["date", "code", "horizon"]).any()):
        raise ValueError("Stored 100k block trades lack market outcome rows")
    return frame


def _score_raw(raw: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    expected = len(pairs) * 4
    if (len(raw) != expected
            or set(raw.target_notional) != {20_000.0, 100_000.0}
            or set(raw.horizon) != {1, 5}
            or not raw.exit_window.eq("close").all()
            or raw.duplicated(["date", "code", "target_notional",
                               "horizon"]).any()
            or not raw.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Original-minute block trade repricing is incomplete")
    result = raw.merge(pairs[["date", "code", "candidate", "pair_code",
                              "trade_date", "industry", "board"]],
                       on=["date", "code"], validate="many_to_one")
    if len(result) != len(raw) or result.quality_clean_exit.isna().any():
        raise ValueError("Repriced block trade left the frozen pair list")
    clean = result.quality_clean_exit
    baseline = Assumptions().slippage_bps_each_side
    if not np.allclose(_stressed_returns(result.loc[clean], baseline),
                       result.loc[clean, "net_return"], atol=1e-12):
        raise ValueError("Block trade raw-minute return disagrees with fees")
    result["cash_return"] = np.where(clean, result.net_return, 0.0)
    result["cash_stress10"] = 0.0
    result.loc[clean, "cash_stress10"] = _stressed_returns(
        result.loc[clean], baseline + 10)
    if not np.isfinite(result[["cash_return", "cash_stress10"]].to_numpy()).all():
        raise ValueError("Nonfinite block trade net return")
    return result


def _summary(rows: pd.DataFrame, year: str, segment: str,
             peak_month: str) -> dict:
    frame = rows.loc[rows.date.str.startswith(year)].copy()
    if frame.empty:
        return {"pairs": 0, "days": 0}
    if segment == "H1":
        frame = frame.loc[frame.date.str[5:7].astype(int).le(6)]
    elif segment == "H2":
        frame = frame.loc[frame.date.str[5:7].astype(int).ge(7)]
    elif segment == "without_peak_month":
        frame = frame.loc[~frame.date.str.startswith(peak_month)]
    elif segment != "full":
        raise ValueError("Unknown block trade segment")
    if frame.empty:
        return {"pairs": 0, "days": 0}
    wide = frame.groupby(["date", "candidate"])[
        ["cash_return", "cash_stress10"]].mean().unstack("candidate")
    if wide.isna().any().any():
        raise ValueError("Block trade signal day lost an event or control")
    event = wide[("cash_return", EVENT)]
    control = wide[("cash_return", CONTROL)]
    edge = event - control
    stress = (wide[("cash_stress10", EVENT)]
              - wide[("cash_stress10", CONTROL)])
    event_rows = frame.loc[frame.candidate.eq(EVENT)]
    control_rows = frame.loc[frame.candidate.eq(CONTROL)]
    return {"pairs": len(event_rows), "days": len(wide),
            "event_cash_mean": float(event.mean()),
            "control_cash_mean": float(control.mean()),
            "edge_mean": float(edge.mean()),
            "edge_month_ci": _month_bootstrap(
                edge.reset_index(drop=True),
                pd.Series(edge.index.to_list()), 718),
            "stress10_edge_mean": float(stress.mean()),
            "event_entry_rate": float(event_rows.entry_status.eq("filled").mean()),
            "control_entry_rate": float(control_rows.entry_status.eq("filled").mean()),
            "event_clean_exit_rate": float(event_rows.quality_clean_exit.mean()),
            "control_clean_exit_rate": float(control_rows.quality_clean_exit.mean()),
            "event_ontime_exit_rate": float((
                event_rows.quality_clean_exit
                & event_rows.exit_delay_sessions.eq(0)).mean())}


def evaluate(pair_dir: Path, raw_path: Path, outcome_dir: Path,
             issues_dir: Path, report_path: Path) -> dict:
    audit = json.loads((pair_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if (audit.get("note")
            != "Input-only; no future return or execution files read"):
        raise ValueError("Block trade input audit is absent or not frozen")
    pairs = pd.read_parquet(pair_dir / "pairs.parquet")
    _check_membership(pairs, audit)
    raw = pd.read_parquet(raw_path)
    exact = _exact(_stored_100k(pairs, outcome_dir, issues_dir),
                   raw.loc[raw.target_notional.eq(100_000)])
    rows = _score_raw(raw, pairs)
    report = {"exact_100k_rows": exact, "results": {},
              "note": "Exploratory 2024/2025, 2025 not blind, 2026 untouched"}
    for size in (20_000, 100_000):
        report["results"][str(size)] = {}
        for horizon in (1, 5):
            sample = rows.loc[rows.target_notional.eq(size)
                              & rows.horizon.eq(horizon)]
            report["results"][str(size)][str(horizon)] = {
                year: {
                    segment: _summary(sample, year, segment,
                                      audit["year"][year]["peak_selected_month"])
                    for segment in ("full", "H1", "H2", "without_peak_month")
                } for year in ("2024", "2025")
            }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("data/research/block_trade"))
    parser.add_argument("--raw", type=Path, default=Path(
        "data/research/block_trade/repriced.parquet"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/block_trade/report.json"))
    args = parser.parse_args()
    report = evaluate(args.source, args.raw, args.outcomes,
                      args.issues, args.report)
    print({"exact_100k_rows": report["exact_100k_rows"],
           "main_20k_t5": report["results"]["20000"]["5"]})


if __name__ == "__main__":
    main()
