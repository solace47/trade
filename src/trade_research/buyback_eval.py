"""Evaluate frozen repurchase-plan pairs using 2024/2025 minute outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .annual_cash_direct_eval import _month_bootstrap
from .buyback_inputs import CONTROL, EVENT
from .market_study import _quality_keys, _quality_symbols, _week_bootstrap
from .strategy_scan import _stressed_returns


def _check_pairs(pairs: pd.DataFrame, expected: dict) -> None:
    if (pairs.empty or len(pairs) != 2 * expected["matched_pairs"]
            or pairs.duplicated(["date", "code"]).any()
            or set(pairs.candidate) != {EVENT, CONTROL}
            or not pairs.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Incomplete frozen buyback event/control membership")
    sizes = pairs.groupby(["date", "pair_code"]).candidate.agg(
        ["size", "nunique"])
    if not (sizes["size"].eq(2).all() and sizes["nunique"].eq(2).all()
            and len(sizes) == expected["matched_pairs"]):
        raise ValueError("Buyback events lack one-to-one same-day controls")
    events = pairs.loc[pairs.candidate.eq(EVENT)]
    if (events.notice_date.isna().any()
            or not events.date.gt(events.notice_date).all()
            or events.groupby("date").size().gt(5).any()
            or events.date.nunique() != expected["matched_days"]):
        raise ValueError("Buyback signal violates disclosure or capacity timing")
    for year in ("2024", "2025"):
        group = events.loc[events.date.str.startswith(year)]
        if (len(group) != expected["by_year"][year]["pairs"]
                or group.date.nunique() != expected["by_year"][year]["days"]):
            raise ValueError("Buyback pair list changed after input audit")


def _score(pairs: pd.DataFrame, outcome_dir: Path,
           issues_dir: Path, horizon: int) -> pd.DataFrame:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.register("selected", pairs)
    connection.read_parquet(str(outcome_dir / "*.parquet")).create_view("outcomes")
    connection.register("bad_days", _quality_keys(issues_dir))
    connection.register("bad_symbols", _quality_symbols(issues_dir))
    rows = connection.execute("""
        SELECT s.date, s.code, s.candidate, s.pair_code, s.industry,
               s.board, s.notice_date, o.* EXCLUDE (date, code),
               o.exit_status = 'filled'
                 AND NOT EXISTS (SELECT 1 FROM bad_symbols b
                                 WHERE b.code = s.code)
                 AND NOT EXISTS (SELECT 1 FROM bad_days b
                                 WHERE b.code = s.code
                                   AND b.date >= s.date
                                   AND b.date <= o.exit_date)
                 AS quality_clean_exit
        FROM selected s JOIN outcomes o USING (date, code)
        WHERE o.horizon = ?
          AND (o.exit_date IS NULL
               OR LEFT(o.exit_date, 4) = LEFT(o.date, 4))
    """, [horizon]).df()
    if (len(rows) != len(pairs)
            or rows.duplicated(["date", "code"]).any()
            or rows.quality_clean_exit.isna().any()):
        raise ValueError("Frozen buyback pair lacks same-year minute outcome")
    rows["cash_return"] = np.where(rows.quality_clean_exit,
                                   rows.net_return, 0.0)
    rows["cash_stress10"] = 0.0
    clean = rows.quality_clean_exit
    rows.loc[clean, "cash_stress10"] = _stressed_returns(rows.loc[clean], 10)
    if (not np.isfinite(rows.cash_return).all()
            or not np.isfinite(rows.cash_stress10).all()):
        raise ValueError("Buyback execution return is nonfinite")
    return rows


def _summary(rows: pd.DataFrame, year: str, segment: str) -> dict:
    frame = rows.loc[rows.date.str.startswith(year)]
    if segment == "H1":
        frame = frame.loc[frame.date.str[5:7].astype(int).le(6)]
    elif segment == "H2":
        frame = frame.loc[frame.date.str[5:7].astype(int).gt(6)]
    elif segment == "without_peak_month":
        peak = "2024-02" if year == "2024" else "2025-04"
        frame = frame.loc[~frame.date.str.startswith(peak)]
    elif segment != "full":
        raise ValueError("Unknown frozen buyback segment")
    if frame.empty:
        return {"pairs": 0, "days": 0}
    days = sorted(frame.date.unique())
    daily = frame.groupby(["date", "candidate"])[
        ["cash_return", "cash_stress10"]].mean().unstack("candidate")
    if len(daily) != len(days) or daily.isna().any().any():
        raise ValueError("Buyback control is absent from a signal day")
    event = daily[("cash_return", EVENT)].reset_index(drop=True)
    peer = daily[("cash_return", CONTROL)].reset_index(drop=True)
    edge = event - peer
    stress = (daily[("cash_stress10", EVENT)]
              - daily[("cash_stress10", CONTROL)]).reset_index(drop=True)
    date_series = pd.Series(days)
    treated = frame.loc[frame.candidate.eq(EVENT)]
    control = frame.loc[frame.candidate.eq(CONTROL)]
    return {
        "pairs": len(treated), "days": len(days),
        "event_cash_mean": float(event.mean()),
        "control_cash_mean": float(peer.mean()),
        "edge_mean": float(edge.mean()),
        "edge_month_ci": _month_bootstrap(edge, date_series, 141),
        "edge_week_ci": _week_bootstrap(edge, date_series, 142),
        "stress10_edge_mean": float(stress.mean()),
        "event_entry_rate": float(treated.entry_status.eq("filled").mean()),
        "control_entry_rate": float(control.entry_status.eq("filled").mean()),
        "event_clean_exit_rate": float(treated.quality_clean_exit.mean()),
        "control_clean_exit_rate": float(control.quality_clean_exit.mean()),
        "event_ontime_exit_rate": float(
            (treated.quality_clean_exit
             & treated.exit_delay_sessions.eq(0)).mean()),
    }


def evaluate(pair_dir: Path, outcome_dir: Path, issues_dir: Path,
             report_path: Path, trades_path: Path) -> dict:
    audit = json.loads((pair_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if audit.get("note") != "Inputs only; no future returns opened":
        raise ValueError("Buyback source was not frozen before outcome read")
    main = pd.read_parquet(pair_dir / "pairs.parquet")
    industry = pd.read_parquet(pair_dir / "industry_pairs.parquet")
    _check_pairs(main, audit["main"])
    _check_pairs(industry, audit["same_industry"])
    scored = _score(main, outcome_dir, issues_dir, 5)
    industry_scored = _score(industry, outcome_dir, issues_dir, 5)
    report = {"main": {}, "same_industry": {},
              "note": "Exploratory 2024/2025, 2025 not blind, 2026 not read"}
    for year in ("2024", "2025"):
        report["main"][year] = {
            segment: _summary(scored, year, segment)
            for segment in ("full", "H1", "H2", "without_peak_month")
        }
        report["same_industry"][year] = _summary(
            industry_scored, year, "full")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    scored.to_parquet(trades_path, index=False, compression="zstd")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path(
        "data/research/buyback"))
    parser.add_argument("--outcomes", type=Path, default=Path(
        "data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path, default=Path(
        "data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/buyback/report.json"))
    parser.add_argument("--trades", type=Path, default=Path(
        "data/research/buyback/trades.parquet"))
    args = parser.parse_args()
    report = evaluate(args.pairs, args.outcomes, args.issues,
                      args.report, args.trades)
    print({year: report["main"][year]["full"] for year in ("2024", "2025")})


if __name__ == "__main__":
    main()
