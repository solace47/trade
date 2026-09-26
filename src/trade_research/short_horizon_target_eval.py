"""Trade both training-target selections with the same next-day exit policy."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .first_limitup_overnight_eval import daily_statistics, trade_distribution
from .price_limit_queue_audit import audit, conservative_returns
from .reference_gain_accounting import evaluate as account
from .reference_gain_eval import reprice
from .risk_removal_eval import evaluate as tick
from .shallow_tree_continuation import continue_model
from .short_horizon_labels import ROOT, RULE_COMMIT


def execute(output: Path = ROOT, *, model_names: tuple[str, str] = ("t1_target", "t5_target"),
            rule_commit: str = RULE_COMMIT) -> dict:
    inputs = json.loads((output / "input_report.json").read_text())
    checks = json.loads((output / "independent_input_checks.json").read_text())
    labels = json.loads((output / "independent_label_checks.json").read_text())
    if labels["label_report_sha256"] != sha(output / "label_report.json"):
        raise ValueError("The independently checked training labels changed")
    results = {}
    for model in model_names:
        folder = output / model; expected = inputs["models"][model]["signals_sha256"]
        if sha(folder / "signals.parquet") != expected or checks["models"][model]["signals_sha256"] != expected:
            raise ValueError("An independently checked model list changed")
        if inputs["models"][model]["candidates"] == 0:
            results[model] = {"no_positive_selections": True}
            continue
        if not (folder / "execution_report.json").exists():
            reprice(folder, expected_signal_sha=expected, rule_commit=rule_commit, horizons=(1,))
        continued = folder / "continued"
        if not (continued / "execution_report.json").exists():
            continue_model(folder, continued)
        if not (continued / "execution_queue_report.json").exists():
            audit(continued)
        if not (continued / "catalog_scenario_report.json").exists():
            account(continued)
        if not (continued / "tick_report.json").exists():
            tick(continued, primary_horizon=1)
        results[model] = json.loads((continued / "execution_report.json").read_text())
        print(model, {k: results[model][k] for k in ("continued_rows", "remaining_unresolved")}, flush=True)
    return results


def compare(output: Path = ROOT, *, model_names: tuple[str, str] = ("t1_target", "t5_target"),
            rule_commit: str = RULE_COMMIT,
            contrast_key: str = "common_date_t1_target_minus_t5_target",
            preserve_comparison_models: tuple[str, ...] = ()) -> dict:
    inputs = json.loads((output / "input_report.json").read_text())
    periods = [("full", "2024-01-01", "2025-12-31")]
    for year in ("2024", "2025"):
        periods.extend([(year, year+"-01-01", year+"-12-31"), (year+"H1", year+"-01-01", year+"-06-30"),
                        (year+"H2", year+"-07-01", year+"-12-31")])
    model_reports, daily_models = {}, {}
    for model in model_names:
        root = output / model; folder = root / "continued"
        if inputs["models"][model]["candidates"] == 0:
            model_reports[model] = {"no_positive_selections": True}
            continue
        tick_report = json.loads((folder / "tick_report.json").read_text())
        queue_report = json.loads((folder / "execution_queue_report.json").read_text())
        if (sha(folder / "signals.parquet") != inputs["models"][model]["signals_sha256"]
                or sha(folder / "tick_cost_scenario.parquet") != tick_report["tick_scenario_sha256"]
                or sha(folder / "execution_queue_audit.parquet") != queue_report["output_sha256"]):
            raise ValueError("A fixed model, execution or queue ledger changed")
        rows = pd.read_parquet(folder / "tick_cost_scenario.parquet")
        queues = pd.read_parquet(folder / "execution_queue_audit.parquet")
        if set(rows.horizon) != {1}:
            raise ValueError("Both models must be evaluated with T+1 exits")
        for bps in (5, 15):
            rows[f"queue_checked_return{bps}"] = conservative_returns(rows, queues, f"tick_return{bps}")
        high, low = rows.loc[rows.arm.eq("high")], rows.loc[rows.arm.eq("low")]
        paired = high.merge(low, on=["date", "pair_id"], validate="one_to_one", suffixes=("_high", "_low"))
        metrics, trades = [], []
        for bps in (5, 15):
            column = f"tick_return{bps}"
            own = high[["date", column]].rename(columns={column: "value"})
            unknown = high[["date", f"queue_checked_return{bps}"]].rename(columns={f"queue_checked_return{bps}": "value"})
            edge = paired[["date"]].assign(value=paired[column+"_high"]-paired[column+"_low"])
            for metric, frame in (("own", own), ("same_day_edge", edge), ("own_with_queue_unknown_retained", unknown)):
                daily = frame.groupby("date").value.agg(lambda x: x.mean() if x.notna().all() else np.nan)
                if metric == "own":
                    daily_models[model, bps] = daily
                for label, first, last in periods:
                    metrics.append({"horizon": 1, "bps": bps, "metric": metric, "period": label,
                        **daily_statistics(daily.loc[(daily.index >= first) & (daily.index <= last)])})
            for label, first, last in periods:
                trades.append({"horizon": 1, "bps": bps, "period": label,
                    **trade_distribution(high.loc[high.date.between(first, last)], column)})
        result = {"rule_commit": rule_commit, "model": model, "metrics": metrics,
            "trade_distributions": trades, "queue_report": queue_report,
            "tick_report_sha256": sha(folder / "tick_report.json"),
            "interpretation": "exposed_2024_2025_conditional_recorded_fill_scenario_not_portfolio_return"}
        if model in preserve_comparison_models:
            previous = json.loads((root / "comparison_report.json").read_text())
            result["rule_commit"] = previous["rule_commit"]
            if result != previous:
                raise ValueError("The reused baseline statistics changed")
        else:
            save_json(root / "comparison_report.json", result)
        model_reports[model] = {"comparison_report_sha256": sha(root / "comparison_report.json")}
    differences = []
    primary, comparator = model_names
    for bps in (5, 15):
        if (primary, bps) not in daily_models or (comparator, bps) not in daily_models:
            continue
        pair = pd.concat([daily_models[primary, bps].rename("t1"), daily_models[comparator, bps].rename("t5")],
            axis=1, join="inner")
        daily = pair.t1-pair.t5
        for label, first, last in periods:
            differences.append({"bps": bps, "period": label,
                **daily_statistics(daily.loc[(daily.index >= first) & (daily.index <= last)])})
    result = {"rule_commit": rule_commit, "models": model_reports,
        "both_models_evaluated_at_T1": True, "primary_model": primary,
        contrast_key: differences,
        "input_report_sha256": sha(output / "input_report.json"),
        "new_2026_prices_read": False, "publishable_formula": False}
    save_json(output / "comparison_report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["execute", "compare"])
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](), ensure_ascii=False, indent=2))
