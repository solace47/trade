"""Compare fixed pool sampling with each unchanged highest-score strategy."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .alpha158_eval import daily_candidates
from .corporate_cash import save_json, sha
from .fixed_capital_1449 import ROOT as CAPITAL_ROOT, SOURCES, prepare as prepare_capital, evaluate as capital
from .positive_pool_1449 import ROOT, RULE_COMMIT
from .reference_gain_accounting import weekly_interval
from .risk_removal_eval import add_cost_scenarios


def compare(output: Path = ROOT) -> dict:
    selection = json.loads((output / "input_report.json").read_text())
    old_capital_manifest = json.loads((CAPITAL_ROOT / "manifest.json").read_text())
    common_tables, halves, annual, sources = [], [], [], {}
    for model, old_folder in SOURCES.items():
        new_folder = output / "continued" / model
        new_report = json.loads((new_folder / "tick_report.json").read_text())
        old_path = old_folder / "catalog_scenario.parquet"
        if (sha(old_path) != old_capital_manifest["source_sha256"][model][str(old_path)]
                or sha(new_folder / "signals.parquet") != selection["models"][model]["signals_sha256"]
                or sha(new_folder / "tick_cost_scenario.parquet") != new_report["tick_scenario_sha256"]):
            raise ValueError("An old or new fixed comparison ledger changed")
        original = add_cost_scenarios(pd.read_parquet(old_path))
        sampled = pd.read_parquet(new_folder / "tick_cost_scenario.parquet")
        sources[model] = {"original_catalogue_sha256": sha(old_path), "pool_tick_report_sha256": sha(new_folder / "tick_report.json")}
        for bps in (5, 15):
            original_daily = daily_candidates(original, f"tick_return{bps}")
            pool_daily = daily_candidates(sampled, f"tick_return{bps}")
            common = original_daily.merge(pool_daily, on=["date", "horizon"], how="inner",
                validate="one_to_one", suffixes=("_highest", "_pool"))
            common["difference"] = common.value_pool - common.value_highest
            common["year"] = common.date.str[:4]
            common["half"] = common.year + np.where(common.date.str[5:7].le("06"), "H1", "H2")
            common["model"], common["bps"] = model, bps
            common_tables.append(common)
            for (half, horizon), group in common.groupby(["half", "horizon"]):
                halves.append({"model": model, "half": half, "horizon": int(horizon), "bps": bps,
                    "dates": len(group), "unknown_dates": int(group.difference.isna().sum()),
                    "mean": None if group.difference.isna().any() else float(group.difference.mean())})
            for (year, horizon), group in common.groupby(["year", "horizon"]):
                daily = group.set_index("date").difference.sort_index()
                annual.append({"model": model, "year": year, "horizon": int(horizon), "bps": bps,
                    "dates": len(group), "weeks": pd.to_datetime(daily.index).to_period("W-SUN").nunique(),
                    "unknown_dates": int(daily.isna().sum()),
                    "mean": None if daily.isna().any() else float(daily.mean()), "weekly_interval": weekly_interval(daily)})
    pd.concat(common_tables, ignore_index=True).to_parquet(output / "common_dates.parquet", index=False, compression="zstd")
    cash_output = output / "capital"
    if not (cash_output / "report.json").exists():
        prepare_capital(cash_output, sources={name: output / "continued" / name for name in SOURCES})
        capital(cash_output)
    new_cash = json.loads((cash_output / "report.json").read_text())
    old_cash = json.loads((CAPITAL_ROOT / "report.json").read_text())
    for folder, report in ((cash_output, new_cash), (CAPITAL_ROOT, old_cash)):
        if (sha(folder / "manifest.json") != report["manifest_sha256"]
                or sha(folder / "cash_ledger.parquet") != report["ledger_sha256"]
                or sha(folder / "order_decisions.parquet") != report["decisions_sha256"]):
            raise ValueError("A completed fixed-capital comparison changed")
    cash_comparison = []
    for name in SOURCES:
        for bps in (5, 15):
            old = next(x for x in old_cash["books"] if x["model"] == name and x["arm"] == "high" and x["bps"] == bps)
            new = next(x for x in new_cash["books"] if x["model"] == name and x["arm"] == "high" and x["bps"] == bps)
            left, right = old["conditional_capital_return"], new["conditional_capital_return"]
            cash_comparison.append({"model": name, "bps": bps, "highest_score_return": left, "pool_return": right,
                "difference": None if left is None or right is None else right - left,
                "highest_capital_rejected": old["capital_rejected"], "pool_capital_rejected": new["capital_rejected"]})
    result = {"interpretation": "fixed_hash_positive_pool_minus_original_highest_scores_exposed_development_sample",
        "rule_commit": RULE_COMMIT, "primary": "T5_max_15bps_or_half_cent_with_same_500000_yuan_cash_policy",
        "by_half": halves, "annual": annual, "cash_comparison": cash_comparison, "source_hashes": sources,
        "common_dates_sha256": sha(output / "common_dates.parquet"),
        "old_capital_report_sha256": sha(CAPITAL_ROOT / "report.json"),
        "pool_capital_report_sha256": sha(cash_output / "report.json"), "holdout_prices_read": False}
    save_json(output / "comparison_report.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(compare(), ensure_ascii=False, indent=2))
