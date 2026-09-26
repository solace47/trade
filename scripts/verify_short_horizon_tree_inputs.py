"""Traverse every tree without library predict, then rebuild ranks and matching."""
import json
from pathlib import Path

import duckdb
import joblib
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha

root = Path("data/research/short_horizon_tree")
base_root = Path("data/research/short_horizon_target")
report = json.loads((root / "input_report.json").read_text())
protocol = json.loads(Path("config/short_horizon_tree_protocol.json").read_text())
base = json.loads(Path(protocol["base_protocol"]).read_text())
original_fits = json.loads((base_root / "input_report.json").read_text())["training"]
assert sha(Path(protocol["label_file"])) == protocol["label_sha256"]
labels = pd.read_parquet(protocol["label_file"])
frame = pd.read_parquet(base["features_source"])
assert sha(Path(base["features_source"])) == base["features_sha256"]
frame["price_1449"] = np.floor(frame.price_1449*100+.5)/100
frame["price_signal"] = frame.price_1449
raw_cal = pd.read_parquet("data/baostock/market_2020_2026/metadata/calendar.parquet")
calendar = sorted(raw_cal.loc[raw_cal.is_trading_day.eq("1") &
    raw_cal.calendar_date.between("2022-01-01", "2025-12-31"), "calendar_date"])
position = {date: i for i, date in enumerate(calendar)}
folder = root / "t1_tree"; info = report["models"]["t1_tree"]
assert sha(folder / "signals.parquet") == info["signals_sha256"]
assert sha(folder / "all_scores.parquet") == info["all_scores_sha256"]
signals, scores = pd.read_parquet(folder / "signals.parquet"), pd.read_parquet(folder / "all_scores.parquet")
picks, model_checks = [], []
c = duckdb.connect(); c.execute("SET threads=4")
for fold, audit in zip(base["folds"], report["training"], strict=True):
    path = root / ("tree_"+fold["name"]+".joblib")
    assert sha(path) == audit["model_sha256"]
    bundle = joblib.load(path); model = bundle["model"]; columns = list(bundle["features"])
    assert all(model.get_params()[k] == v for k, v in protocol["tree_parameters"].items())
    original = next(a for a in original_fits if a["model"] == "t1_target" and a["fold"] == fold["name"])
    for field in ("median", "scale"):
        assert np.max(abs(bundle[field].loc[columns].to_numpy()-np.array([original[field][k] for k in columns]))) == 0
    train = labels.loc[labels.date.between(fold["train_first"], fold["train_last"])]
    assert len(train) == audit["training_rows"]
    assert calendar[position[train.date.max()]+10] < fold["test_first"]
    assert train.exit_date.dropna().lt(fold["test_first"]).all()
    baseline = float(model._baseline_prediction.item())
    assert abs(baseline-train.downside_score.clip(-.15, .15).mean()) < 1e-12
    test = frame.loc[frame.date.between(fold["test_first"], fold["test_last"])].merge(
        scores, on=["date", "code"], validate="one_to_one")
    x = np.clip((test[columns].to_numpy()-bundle["median"].loc[columns].to_numpy())/
        bundle["scale"].loc[columns].to_numpy(), -5, 5)
    assert np.isfinite(x).all()
    predicted = np.full(len(test), baseline)
    assert len(model._predictors) == 100
    for iteration in model._predictors:
        assert len(iteration) == 1
        nodes = iteration[0].nodes
        assert nodes["count"][0] == len(train) and not nodes["is_categorical"].any()
        leaf = nodes["is_leaf"].astype(bool)
        assert leaf.sum() <= 7 and nodes["depth"].max() <= 3 and nodes["count"][leaf].min() >= 500
        node_id = np.zeros(len(test), dtype=int)
        for _ in range(3):
            active = ~leaf[node_id]
            if not active.any():
                break
            ids = np.flatnonzero(active); current = node_id[active]
            left = x[ids, nodes["feature_idx"][current]] <= nodes["num_threshold"][current]
            node_id[active] = np.where(left, nodes["left"][current], nodes["right"][current])
        assert leaf[node_id].all()
        predicted += nodes["value"][node_id]
    error = float(np.max(abs(predicted-test.score)))
    assert error < 1e-12
    c.register("scores", test)
    order = c.sql("SELECT date,code FROM scores WHERE score>0 ORDER BY date,score DESC,code").df()
    last, counts = {}, {}
    for date, code in order.itertuples(index=False, name=None):
        if counts.get(date, 0) == 5 or position[date]-last.get(code, -1000) <= 5:
            continue
        counts[date] = counts.get(date, 0)+1; last[code] = position[date]
        picks.append((date, code, counts[date]))
    model_checks.append({"fold": fold["name"], "training_rows": len(train), "manual_predictions": len(test),
        "trees": len(model._predictors), "prediction_max_error": error})
high, low = signals.loc[signals.arm.eq("high")], signals.loc[signals.arm.eq("low")]
assert set(picks) == set(zip(high.date, high.code, high.daily_rank))
pool = frame.loc[frame.date.ge("2024-01-01")]; c.register("pool", pool); c.register("high", high)
edges = c.sql("""WITH x AS(SELECT h.date,h.code AS event,h.daily_rank,p.code AS peer,
 abs(h.return20_prior_adjusted-p.return20_prior_adjusted) AS prior_gap,
 abs(h.return_1450-p.return_1450) AS day_gap,p.amount_signal/h.amount_signal AS ar,
 p.price_signal/h.price_signal AS pr FROM high h JOIN pool p ON h.date=p.date AND h.board=p.board
 WHERE NOT EXISTS(SELECT 1 FROM high a WHERE a.date=p.date AND a.code=p.code))
 SELECT *,prior_gap/.05+day_gap/.02+abs(ln(ar))/ln(2)+abs(ln(pr))/ln(2) AS distance
 FROM x WHERE prior_gap<=.05 AND day_gap<=.02 AND ar BETWEEN .5 AND 2 AND pr BETWEEN .5 AND 2
 ORDER BY date,daily_rank,distance,peer""").df()
matched, assigned, used = [], set(), set()
for row in edges.itertuples():
    if (row.date, row.event) in assigned or (row.date, row.peer) in used:
        continue
    assigned.add((row.date, row.event)); used.add((row.date, row.peer))
    matched.append((row.date, row.peer, row.date+":"+row.event))
assert set(matched) == set(zip(low.date, low.code, low.pair_id))
coverage = pd.read_parquet("data/research/cash_dividend_catalog/query_coverage.parquet")
needed = pd.DataFrame(sorted({(r.code, str(year)) for r in signals.itertuples()
    for year in range(int(r.date[:4]), int(calendar[position[r.date]+10][:4])+1)}), columns=["code", "year"])
assert needed.merge(coverage, on=["code", "year"], how="left", indicator=True)._merge.eq("both").all()
for name, expected in report["baseline_reused_sha256"].items():
    assert sha(root/"t1_target"/name) == expected == sha(Path(protocol["linear_baseline"])/name)
tree_check = {"fits": model_checks, "candidates": len(high), "controls": len(low), "all_matching_edges": len(edges),
    "catalogue_code_years": len(needed), "signals_sha256": info["signals_sha256"]}
baseline_check = json.loads((base_root / "independent_input_checks.json").read_text())["models"]["t1_target"]
save_json(folder / "independent_input_checks.json", tree_check)
result = {"models": {"t1_tree": tree_check, "t1_target": baseline_check},
    "baseline_execution_reused_unchanged": True, "input_report_sha256": sha(root / "input_report.json"),
    "new_tree_test_returns_read": False}
save_json(root / "independent_input_checks.json", result)
print(json.dumps(result, indent=2))
