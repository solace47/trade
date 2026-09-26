"""Economic scenarios and same-date comparison for the frozen two-model study."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .corporate_cash import save_json, sha
from .reference_gain_accounting import evaluate, weekly_interval
from .shallow_tree_1449 import ROOT


def daily_returns(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame.groupby("date")[column].agg(
        lambda values: values.mean() if values.notna().all() else float("nan"))


def summarize(output: Path = ROOT) -> dict:
    inputs = json.loads((output / "input_report.json").read_text())
    models, own_annual, comparisons, terminal_stress = {}, [], [], []
    economic = {}
    for name in ("ridge", "tree"):
        folder = output / name
        frozen = inputs["models"][name]
        if sha(folder / "signals.parquet") != frozen["signals_sha256"]:
            raise ValueError("The fixed model list changed")
        result = evaluate(folder)
        rows = pd.read_parquet(folder / "catalog_scenario.parquet")
        high = rows.loc[rows.arm.eq("high")].copy()
        economic[name] = high
        models[name] = {"candidates": frozen["candidates"],
            "controls": frozen["controls"], "scenario": result}
        for (half, horizon), part in high.groupby(["half", "horizon"]):
            pending = part.entry_status.eq("filled") & part.exit_price.isna()
            for slip in (5, 15):
                field = f"catalog_scenario_return{slip}"
                stress = part[["date", field]].copy()
                stress.loc[pending, field] = -1.
                daily = daily_returns(stress, field)
                terminal_stress.append({"model": name, "half": half, "horizon": int(horizon),
                    "slip": slip, "unresolved_positions": int(pending.sum()),
                    "full_loss_scenario": None if daily.isna().any() else float(daily.mean())})
        for (year, horizon), part in high.groupby([high.date.str[:4], "horizon"]):
            for slip in (5, 15):
                daily = daily_returns(part, f"catalog_scenario_return{slip}")
                own_annual.append({"model": name, "year": year, "horizon": int(horizon),
                    "slip": slip, "days": len(daily),
                    "return": None if daily.isna().any() else float(daily.mean()),
                    "weekly_interval": weekly_interval(daily)})
    for horizon in (1, 5):
        for slip in (5, 15):
            series = {name: daily_returns(rows.loc[rows.horizon.eq(horizon)],
                f"catalog_scenario_return{slip}") for name, rows in economic.items()}
            together = pd.concat(series, axis=1, join="inner")
            together["difference"] = together.tree - together.ridge
            together["half"] = together.index.str[:4] + together.index.map(
                lambda date: "H1" if date[5:7] <= "06" else "H2")
            groups = list(together.groupby("half")) + list(together.groupby(together.index.str[:4]))
            for period, frame in groups:
                differences = frame.difference
                comparisons.append({"period": period, "horizon": horizon, "slip": slip,
                    "common_dates": len(frame),
                    "tree_minus_ridge": None if differences.isna().any() else float(differences.mean()),
                    "weekly_interval": weekly_interval(differences)})
    result = {"rule_commit": "491a012", "list_commit": "4f8bb7c",
        "interpretation": "signal_day_means_conditional_on_catalogue_and_recorded_fills_not_portfolio_returns",
        "models": models, "own_annual": own_annual, "same_date_comparisons": comparisons,
        "accounting_scope": "continued_beyond_original_five_day_exit_delay" if output.name == "continued" else "original_five_day_exit_delay",
        "unresolved_full_loss_stress_not_actual_returns": terminal_stress,
        "holdout_read": False}
    save_json(output / "comparison_report.json", result)
    return result


if __name__ == "__main__":
    report = summarize()
    for name, model in report["models"].items():
        print(name, json.dumps([row for row in model["scenario"]["by_half"]
            if row["arm"] == "high" and row["horizon"] == 5], ensure_ascii=False, indent=2))
    print(json.dumps([row for row in report["same_date_comparisons"]
        if row["horizon"] == 5 and row["slip"] == 15], ensure_ascii=False, indent=2))
