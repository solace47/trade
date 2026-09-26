"""Rebuild the target-model contrast from independently verified per-trade cash flows."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha

root = Path("data/research/short_horizon_target")
report = json.loads((root / "comparison_report.json").read_text())
frames = {}
for model, item in report["models"].items():
    if item.get("no_positive_selections"):
        assert pd.read_parquet(root/model/"signals.parquet").empty
        continue
    checks = json.loads((root/model/"independent_economic_checks.json").read_text())
    assert checks["comparison_report_sha256"] == item["comparison_report_sha256"] == sha(root/model/"comparison_report.json")
    rows = pd.read_parquet(root/model/"continued/tick_cost_scenario.parquet")
    assert set(rows.horizon) == {1}
    rows = rows.loc[rows.arm.eq("high")]
    for bps in (5, 15):
        frames[model, bps] = rows.groupby("date")[f"tick_return{bps}"].agg(
            lambda x: np.nan if x.isna().any() else sum(x)/len(x))
for item in report["common_date_t1_target_minus_t5_target"]:
    high, low = frames["t1_target", item["bps"]], frames["t5_target", item["bps"]]
    dates = sorted(set(high.index) & set(low.index))
    if item["period"] != "full":
        dates = [d for d in dates if d.startswith(item["period"][:4]) and
            (len(item["period"]) == 4 or (d[5:7] <= "06") == item["period"].endswith("H1"))]
    values = pd.Series([high.loc[d]-low.loc[d] for d in dates], index=dates, dtype=float)
    assert len(values) == item["signal_dates"] and int(values.isna().sum()) == item["unknown_dates"]
    if values.empty or values.isna().any():
        assert item["daily_mean"] is None and item["weekly_interval"] is None
        continue
    assert abs(values.mean()-item["daily_mean"]) < 1e-12
    weeks = pd.to_datetime(values.index).to_period("W-SUN")
    blocks = [g.to_numpy() for _, g in values.groupby(weeks)]
    choices = np.random.default_rng(20260926).integers(len(blocks), size=(10000, len(blocks)))
    weights = np.stack([np.bincount(x, minlength=len(blocks)) for x in choices])
    sums, counts = np.array([x.sum() for x in blocks]), np.array([len(x) for x in blocks])
    ci = np.quantile((weights@sums)/(weights@counts), [.025, .975])
    assert np.max(abs(ci-item["weekly_interval"])) < 1e-12
result = {"models": list(report["models"]), "same_T1_exit_horizon_verified": True,
    "common_date_statistics": len(report["common_date_t1_target_minus_t5_target"]),
    "comparison_report_sha256": sha(root / "comparison_report.json")}
save_json(root / "independent_comparison_checks.json", result)
print(json.dumps(result, indent=2))
