"""Evaluate the two predeclared short horizons on one institutional buy list."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .first_limitup_overnight_eval import daily_statistics, trade_distribution
from .lhb_institutional_short import ROOT, RULE_COMMIT
from .price_limit_queue_audit import audit, conservative_returns
from .reference_gain_accounting import evaluate as account
from .reference_gain_eval import reprice
from .risk_removal_eval import evaluate as tick
from .shallow_tree_continuation import continue_model


def execute(output: Path = ROOT, *, horizons: tuple[int, int] = (1, 3),
            rule_commit: str = RULE_COMMIT) -> dict:
    selection = json.loads((output / "input_report.json").read_text())
    checks = json.loads((output / "independent_input_checks.json").read_text())
    expected = selection["signals_sha256"]
    if sha(output / "signals.parquet") != expected or checks["signals_sha256"] != expected:
        raise ValueError("The independently checked list changed")
    if not (output / "execution_report.json").exists():
        reprice(output, expected_signal_sha=expected, rule_commit=rule_commit, horizons=horizons)
    continued = output / "continued"
    if not (continued / "execution_report.json").exists():
        continue_model(output, continued)
    if not (continued / "execution_queue_report.json").exists():
        audit(continued)
    if not (continued / "catalog_scenario_report.json").exists():
        account(continued)
    if not (continued / "tick_report.json").exists():
        tick(continued, primary_horizon=1)
    return json.loads((continued / "execution_report.json").read_text())


def compare(output: Path = ROOT, *, horizons: tuple[int, int] = (1, 3),
            rule_commit: str = RULE_COMMIT) -> dict:
    selection = json.loads((output / "input_report.json").read_text())
    folder = output / "continued"
    report = json.loads((folder / "tick_report.json").read_text())
    queue_report = json.loads((folder / "execution_queue_report.json").read_text())
    if (sha(folder / "signals.parquet") != selection["signals_sha256"]
            or sha(folder / "tick_cost_scenario.parquet") != report["tick_scenario_sha256"]
            or sha(folder / "execution_queue_audit.parquet") != queue_report["output_sha256"]):
        raise ValueError("A frozen input or execution ledger changed")
    rows = pd.read_parquet(folder / "tick_cost_scenario.parquet")
    queue = pd.read_parquet(folder / "execution_queue_audit.parquet")
    if len(horizons) != 2 or horizons[0] != 1 or set(rows.horizon) != set(horizons):
        raise ValueError("Both predeclared short horizons are required")
    one, other = (rows.loc[rows.horizon.eq(h)].set_index(["date", "code"]).sort_index() for h in horizons)
    same = ["arm", "pair_id", "shares", "entry_status", "entry_price"]
    pd.testing.assert_frame_equal(one[same], other[same], check_exact=True)
    for bps in (5, 15):
        rows[f"queue_checked_return{bps}"] = conservative_returns(rows, queue, f"tick_return{bps}")
    periods = [("full", "2024-01-01", "2025-12-31")]
    for year in ("2024", "2025"):
        periods.extend([(year, year+"-01-01", year+"-12-31"),
            (year+"H1", year+"-01-01", year+"-06-30"), (year+"H2", year+"-07-01", year+"-12-31")])
    metrics, distributions = [], []
    for horizon, part in rows.groupby("horizon"):
        high, low = part.loc[part.arm.eq("high")], part.loc[part.arm.eq("low")]
        paired = high.merge(low, on=["date", "pair_id"], validate="one_to_one", suffixes=("_high", "_low"))
        for bps in (5, 15):
            col = f"tick_return{bps}"
            own = high[["date", col]].rename(columns={col: "value"})
            edge = paired[["date"]].assign(value=paired[col+"_high"]-paired[col+"_low"])
            unknown = high[["date", f"queue_checked_return{bps}"]].rename(columns={f"queue_checked_return{bps}": "value"})
            for metric, frame in (("own", own), ("same_day_edge", edge), ("own_with_queue_unknown_retained", unknown)):
                daily = frame.groupby("date").value.agg(lambda x: x.mean() if x.notna().all() else np.nan)
                for label, first, last in periods:
                    metrics.append({"horizon": int(horizon), "bps": bps, "metric": metric, "period": label,
                        **daily_statistics(daily.loc[(daily.index >= first) & (daily.index <= last)])})
            for label, first, last in periods:
                distributions.append({"horizon": int(horizon), "bps": bps, "period": label,
                    **trade_distribution(high.loc[high.date.between(first, last)], col)})
    result = {"rule_commit": rule_commit, "signals_sha256": selection["signals_sha256"],
        "interpretation": "exposed_2024_2025_catalogue_and_recorded_fill_scenario_not_portfolio_return",
        "primary": "T1_tail_max_15bps_or_half_cent_per_leg", "secondary": f"T{horizons[1]}_same_tail_window",
        "same_buy_orders_verified": True, "metrics": metrics, "trade_distributions": distributions,
        "queue_report": queue_report, "tick_report_sha256": sha(folder / "tick_report.json"),
        "new_2026_holding_prices_read": False, "publishable_formula": False}
    save_json(output / "comparison_report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["execute", "compare"])
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](), ensure_ascii=False, indent=2))
