"""Compare the frozen feature libraries on common dates, preserving unknown P&L."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .alpha158_inputs import ROOT
from .corporate_cash import save_json, sha
from .reference_gain_accounting import weekly_interval

MODELS = ("robust18", "alpha158")


def daily_candidates(rows: pd.DataFrame, column: str) -> pd.DataFrame:
    if rows.duplicated(["date", "code", "horizon"]).any():
        raise ValueError("Duplicate signals would change the fixed daily denominator")
    high = rows.loc[rows.arm.eq("high")]
    daily = high.groupby(["date", "horizon"])[column].agg(["size", "count", "mean"])
    daily.loc[daily["count"].ne(daily["size"]), "mean"] = np.nan
    return daily.reset_index().rename(columns={"mean": "value"})


def compare_rows(left: pd.DataFrame, right: pd.DataFrame, column: str) -> pd.DataFrame:
    common = daily_candidates(left, column).merge(daily_candidates(right, column),
        on=["date", "horizon"], how="inner", validate="one_to_one", suffixes=("_robust18", "_alpha158"))
    common["difference"] = common.value_alpha158 - common.value_robust18
    common["year"] = common.date.str[:4]
    common["half"] = common.year + np.where(common.date.str[5:7].le("06"), "H1", "H2")
    return common


def compare(output: Path = ROOT) -> dict:
    original = json.loads((output / "input_report.json").read_text())
    ledgers, sources = {}, {}
    for name in MODELS:
        folder = output / "continued" / name
        report = json.loads((folder / "tick_report.json").read_text())
        if (sha(folder / "signals.parquet") != original["models"][name]["signals_sha256"]
                or sha(folder / "tick_cost_scenario.parquet") != report["tick_scenario_sha256"]):
            raise ValueError("A frozen feature-library ledger changed")
        ledgers[name] = pd.read_parquet(folder / "tick_cost_scenario.parquet")
        sources[name] = sha(folder / "tick_report.json")
    differences, cells, annual = [], [], []
    for bps in (5, 15):
        common = compare_rows(ledgers["robust18"], ledgers["alpha158"], f"tick_return{bps}")
        common["bps"] = bps
        differences.append(common)
        for (half, horizon), part in common.groupby(["half", "horizon"]):
            cells.append({"half": half, "horizon": int(horizon), "bps": bps, "dates": len(part),
                "unknown_dates": int(part.difference.isna().sum()),
                "mean": None if part.difference.isna().any() else float(part.difference.mean())})
        for (year, horizon), part in common.groupby(["year", "horizon"]):
            daily = part.set_index("date").difference.sort_index()
            annual.append({"year": year, "horizon": int(horizon), "bps": bps, "dates": len(part),
                "weeks": pd.to_datetime(daily.index).to_period("W-SUN").nunique(),
                "unknown_dates": int(daily.isna().sum()),
                "mean": None if daily.isna().any() else float(daily.mean()), "weekly_interval": weekly_interval(daily)})
    pd.concat(differences, ignore_index=True).to_parquet(output / "common_dates.parquet", index=False, compression="zstd")
    result = {"interpretation": "alpha158_minus_same_normalization_18_features_on_common_signal_dates",
        "primary": "T5_max_15bps_or_half_cent_per_leg_conditional_catalogue_and_recorded_fills",
        "by_half": cells, "annual": annual, "source_tick_reports_sha256": sources,
        "common_dates_sha256": sha(output / "common_dates.parquet"), "holdout_prices_read": False}
    save_json(output / "comparison_report.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(compare(), ensure_ascii=False, indent=2))
