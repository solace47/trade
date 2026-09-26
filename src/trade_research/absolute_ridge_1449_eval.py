"""Evaluate the predeclared 14:49 ridge cutoff sensitivity on raw minutes."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from .absolute_ridge_1449 import OUTPUT, ROOT
from .absolute_ridge_eval import _period, summarize
from .market_study import _quality_keys
from .quality_period import load_period_bad_symbols
from .strategy_scan import _stressed_returns


HALVES = ("2024H2", "2025H1", "2025H2")
PERIOD_QUALITY = ROOT / "quality_period_2024_2025.json"


def apply_period_quality(repriced: pd.DataFrame, *, report_path: Path = PERIOD_QUALITY,
                         first_date: str = "2024-01-01", last_date: str = "2025-12-31") -> tuple[pd.DataFrame, int]:
    """Use faults in the research years, while retaining every bad stock-day."""
    connection = duckdb.connect()
    connection.register("raw", repriced.drop(columns="quality_clean_exit"))
    connection.register("bad_days", _quality_keys(ROOT / "market_issues_ci"))
    bad_symbols = load_period_bad_symbols(
        report_path, first_date, last_date,
    )
    connection.register("bad_symbols", bad_symbols)
    qualified = connection.execute("""
        SELECT r.*, r.exit_status = 'filled'
        AND NOT EXISTS (SELECT 1 FROM bad_symbols b WHERE b.code = r.code)
        AND NOT EXISTS (
            SELECT 1 FROM bad_days q WHERE q.code = r.code
              AND q.date >= r.date AND q.date <= r.exit_date
        ) AS quality_clean_exit
        FROM raw r
    """).df()
    connection.close()
    joined = qualified.merge(
        repriced[["date", "code", "target_notional", "entry_window",
                  "exit_window", "horizon", "quality_clean_exit"]],
        on=["date", "code", "target_notional", "entry_window",
            "exit_window", "horizon"], validate="one_to_one",
        suffixes=("_period", "_full_source"),
    )
    released = int((joined.quality_clean_exit_period
                    & ~joined.quality_clean_exit_full_source).sum())
    if len(qualified) != len(repriced):
        raise ValueError("Period quality changed the trade grid")
    return qualified, released


def archive_comparison(repriced: pd.DataFrame) -> dict:
    """Check archived prices, allowing the predeclared share-sizing change."""
    connection = duckdb.connect()
    comparison = repriced.loc[
        repriced.target_notional.eq(100000)
        & repriced.exit_window.eq("close")
    ]
    connection.register("repriced", comparison)
    connection.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                            ).create_view("archived")
    row = connection.execute("""
        SELECT COUNT(*) AS rows,
               COUNT(*) FILTER (WHERE r.shares IS DISTINCT FROM o.shares)
                   AS share_differences,
               COUNT(*) FILTER (WHERE r.entry_status IS DISTINCT FROM o.entry_status)
                   AS entry_status_differences,
               COUNT(*) FILTER (WHERE r.exit_status IS DISTINCT FROM o.exit_status
                                   OR r.exit_date IS DISTINCT FROM o.exit_date)
                   AS exit_status_or_date_differences,
               COUNT(*) FILTER (WHERE r.entry_status = 'filled'
                                   AND o.entry_status = 'filled'
                                   AND ABS(r.entry_price - o.entry_price) > 1e-9)
                   AS joint_entry_price_differences,
               COUNT(*) FILTER (WHERE r.exit_status = 'filled'
                                   AND o.exit_status = 'filled'
                                   AND r.exit_date = o.exit_date
                                   AND ABS(r.exit_price - o.exit_price) > 1e-9)
                   AS joint_exit_price_differences,
               COUNT(*) FILTER (WHERE r.shares = o.shares
                                   AND r.exit_status = 'filled'
                                   AND o.exit_status = 'filled'
                                   AND ABS(r.net_return - o.net_return) > 1e-9)
                   AS same_share_return_differences
        FROM repriced r JOIN archived o USING (date, code, horizon)
    """).fetchone()
    fields = ("rows", "share_differences", "entry_status_differences",
              "exit_status_or_date_differences", "joint_entry_price_differences",
              "joint_exit_price_differences", "same_share_return_differences")
    result = dict(zip(fields, map(int, row), strict=True))
    if (result["rows"] != len(comparison)
            or result["joint_entry_price_differences"]
            or result["joint_exit_price_differences"]
            or result["same_share_return_differences"]):
        raise ValueError("Raw-minute prices or equal-share returns disagree with archive")
    return result


def failure_diagnostics(rows: pd.DataFrame) -> dict:
    """Describe the already-rejected primary result without changing its gate."""
    primary = rows.loc[
        rows.target_notional.eq(20000) & rows.horizon.eq(5)
        & rows.exit_window.eq("close")
    ]
    diagnostics = {}
    for half in HALVES:
        period = primary.loc[primary.period.eq(half)]
        model = period.loc[period.candidate.eq("absolute_model")]
        control = period.loc[period.candidate.eq("same_day_control")]
        own_daily = model.groupby("date", sort=True).stress15.mean()
        pairs = model.merge(control, on=["date", "pair_id"],
                            suffixes=("_model", "_control"),
                            validate="one_to_one")
        if len(pairs) != len(control):
            raise ValueError("An existing same-day control is unmatched")
        edge_daily = pairs.assign(
            edge=pairs.stress15_model - pairs.stress15_control,
        ).groupby("date", sort=True).edge.mean()
        quarter = own_daily.index.str[:4] + "Q" + (
            (own_daily.index.str[5:7].astype(int) - 1) // 3 + 1
        ).astype(str)
        diagnostics[half] = {
            "own_signal_days": len(own_daily),
            "own_negative_days": int(own_daily.lt(0).sum()),
            "own_median_stress15": float(own_daily.median()),
            "own_quarter_means_stress15": {
                name: float(value) for name, value in
                own_daily.groupby(quarter).mean().items()
            },
            "matched_days": len(edge_daily),
            "matched_negative_edge_days": int(edge_daily.lt(0).sum()),
            "matched_median_edge_stress15": float(edge_daily.median()),
        }
    return diagnostics


def evaluate(output_dir: Path = OUTPUT) -> dict:
    audit = json.loads((output_dir / "input_audit.json").read_text(encoding="utf-8"))
    if (not audit["outcome_gate_passed"] or audit["cutoff_label"] != "1449"
            or audit["decision_price_column"] != "price_1449"):
        raise ValueError("The 14:49 model did not pass its frozen input gate")
    membership = pd.read_parquet(output_dir / "selections.parquet")
    repriced = pd.read_parquet(output_dir / "repriced.parquet")
    keys = ["date", "code", "target_notional", "entry_window",
            "exit_window", "horizon"]
    if (len(repriced) != len(membership) * 8
            or repriced.duplicated(keys).any()
            or set(repriced.target_notional) != {20000, 100000}
            or set(repriced.entry_window) != {"baseline"}
            or set(repriced.exit_window) != {"morning", "close"}
            or set(repriced.horizon) != {1, 5}):
        raise ValueError("Incomplete frozen raw-minute trade grid")
    archive = archive_comparison(repriced)
    period_repriced, released = apply_period_quality(repriced)
    rows = period_repriced.merge(
        membership[["date", "code", "candidate", "pair_id"]],
        on=["date", "code"], validate="many_to_one",
    )
    if len(rows) != len(repriced):
        raise ValueError("A raw-minute result lacks a frozen selection")
    rows["valid"] = (
        rows.exit_status.eq("filled") & rows.quality_clean_exit
        & rows.exit_delay_sessions.eq(0)
    )
    rows["cash"] = rows.net_return.where(rows.valid, 0.0)
    rows["stress15"] = 0.0
    rows.loc[rows.valid, "stress15"] = _stressed_returns(
        rows.loc[rows.valid], 15,
    )
    rows["period"] = rows.date.map(_period)
    result = {"input_audit": audit, "archive_comparison": archive,
              "bad_symbol_policy": "2024-2025 period quality",
              "quality_released_trade_rows": released,
              "slippage_bps_each_side": 15, "results": {}}
    for notional in (20000, 100000):
        per_amount = {}
        for horizon in (1, 5):
            per_horizon = {}
            for window in ("morning", "close"):
                group = rows.loc[
                    rows.target_notional.eq(notional)
                    & rows.horizon.eq(horizon)
                    & rows.exit_window.eq(window)
                ]
                model = group.loc[group.candidate.eq("absolute_model")]
                control = group.loc[group.candidate.eq("same_day_control")]
                if (len(model) != audit["signals"]
                        or len(control) != audit["controls"]):
                    raise ValueError("A treatment or control arm is incomplete")
                summaries = {}
                for period in (*HALVES, "2025"):
                    use = (group.date.str.startswith("2025") if period == "2025"
                           else group.period.eq(period))
                    summaries[period] = summarize(
                        model.loc[use.loc[model.index]],
                        control.loc[use.loc[control.index]],
                        primary=notional == 20000 and horizon == 5
                        and window == "close",
                    )
                per_horizon[window] = summaries
            per_amount[str(horizon)] = per_horizon
        result["results"][str(notional)] = per_amount
    primary = result["results"]["20000"]["5"]["close"]
    result["failure_diagnostics"] = failure_diagnostics(rows)
    for half in HALVES:
        if (result["failure_diagnostics"][half]["own_signal_days"]
                != primary[half]["signal_days"]):
            raise ValueError("Failure diagnostics changed primary signal days")
    result["release_gate_passed"] = bool(
        all(primary[half]["own_all_cash_stress15"] > 0
            and primary[half]["matched_edge_stress15"] > 0
            and primary[half]["own_entry_rate"] >= .90
            and primary[half]["own_clean_ontime_exit_rate"] >= .90
            and primary[half]["control_entry_rate"] >= .90
            and primary[half]["control_clean_ontime_exit_rate"] >= .90
            for half in HALVES)
        and primary["2024H2"]["edge_stress_intervals"]["week_ci"][0] > 0
        and primary["2025"]["edge_stress_intervals"]["week_ci"][0] > 0
    )
    (output_dir / "report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


if __name__ == "__main__":
    report = evaluate()
    print({"archive_comparison": report["archive_comparison"],
           "release_gate_passed": report["release_gate_passed"]})
