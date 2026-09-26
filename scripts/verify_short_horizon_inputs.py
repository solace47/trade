"""Independent normal equations, date boundaries, selection and full matching graph."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from trade_research.corporate_cash import save_json, sha

root = Path("data/research/short_horizon_target")
protocol = json.loads(Path("config/short_horizon_target_protocol.json").read_text())
report = json.loads((root / "input_report.json").read_text())
columns = ["return_1450", "return_1450_sq", "position_1450", "log_amount", "log_volume_ratio",
    "return5_prior_adjusted", "return20_prior_adjusted", "distance_ma20_adjusted", "intraday_range",
    "overnight_gap", "abs_overnight_gap", "return_last30", "abs_return_last30", "volume_share_last30",
    "premium_to_last30_vwap", "log_price", "star", "chinext"]
frame = pd.read_parquet(protocol["features_source"])
assert sha(Path(protocol["features_source"])) == protocol["features_sha256"]
frame["price_1449"] = np.floor(frame.price_1449*100+.5)/100
frame["price_signal"] = frame.price_1449
raw_cal = pd.read_parquet("data/baostock/market_2020_2026/metadata/calendar.parquet")
calendar = sorted(raw_cal.loc[raw_cal.is_trading_day.eq("1") &
    raw_cal.calendar_date.between("2022-01-01", "2025-12-31"), "calendar_date"])
position = {d: i for i, d in enumerate(calendar)}
coverage = pd.read_parquet("data/research/cash_dividend_catalog/query_coverage.parquet")
pool = frame.loc[frame.date.ge("2024-01-01")]
c = duckdb.connect(); c.execute("SET threads=4"); c.register("pool", pool)
checks = {}
for name in ("t1_target", "t5_target"):
    folder = root / name
    info = report["models"][name]
    assert sha(folder / "signals.parquet") == info["signals_sha256"]
    assert sha(folder / "all_scores.parquet") == info["all_scores_sha256"]
    labels = pd.read_parquet(root / (name[:2]+"_labels.parquet"))
    signals = pd.read_parquet(folder / "signals.parquet")
    scores = pd.read_parquet(folder / "all_scores.parquet")
    picks, fit_checks = [], []
    for fold in protocol["folds"]:
        audit = next(x for x in report["training"] if x["model"] == name and x["fold"] == fold["name"])
        train = frame.loc[frame.date.between(fold["train_first"], fold["train_last"])].merge(
            labels[["date", "code", "downside_score", "target_exit_date", "exit_date"]],
            on=["date", "code"], validate="one_to_one").sort_values(["date", "code"])
        assert len(train) == audit["rows"]
        assert calendar[position[train.date.max()]+10] < fold["test_first"]
        assert train.target_exit_date.lt(fold["test_first"]).all() and train.exit_date.dropna().lt(fold["test_first"]).all()
        x = train[columns].to_numpy()
        q1, median, q3 = np.quantile(x, [.25, .5, .75], axis=0)
        scale = np.maximum(q3-q1, .01)
        assert np.max(abs(median-np.array([audit["median"][x] for x in columns]))) < 1e-12
        assert np.max(abs(scale-np.array([audit["scale"][x] for x in columns]))) < 1e-12
        x = np.maximum(-5, np.minimum(5, (x-median)/scale))
        y = np.maximum(-.15, np.minimum(.15, train.downside_score.to_numpy()))
        center, target = x.mean(axis=0), y.mean()
        xc = x-center
        with threadpool_limits(limits=4):
            beta = np.linalg.lstsq(xc.T@xc+len(x)*.05*np.eye(len(columns)), xc.T@(y-target), rcond=None)[0]
        intercept = float(target-center@beta)
        error = max(np.max(abs(beta-np.array([audit["coefficients"][x] for x in columns]))), abs(intercept-audit["intercept"]))
        assert error < 1e-12
        test = pool.loc[pool.date.between(fold["test_first"], fold["test_last"])]
        test = test.merge(scores, on=["date", "code"], validate="one_to_one")
        standardized = np.maximum(-5, np.minimum(5, (test[columns].to_numpy()-median)/scale))
        predicted = np.full(len(test), intercept)
        for i, coefficient in enumerate(beta):
            predicted += standardized[:, i]*coefficient
        assert np.max(abs(test.score-predicted)) < 1e-12
        c.register("scored", test)
        ordered = c.sql("SELECT date,code,score FROM scored WHERE score>0 ORDER BY date,score DESC,code").df()
        last, counts = {}, {}
        for row in ordered.itertuples():
            if counts.get(row.date, 0) == 5 or position[row.date]-last.get(row.code, -1000) <= 5:
                continue
            counts[row.date] = counts.get(row.date, 0)+1; last[row.code] = position[row.date]
            picks.append((row.date, row.code, counts[row.date]))
        fit_checks.append({"fold": fold["name"], "training_rows": len(train), "scores": len(test),
            "coefficient_max_error": float(error), "prediction_max_error": float(np.max(abs(test.score-predicted)))})
    high, low = signals.loc[signals.arm.eq("high")], signals.loc[signals.arm.eq("low")]
    assert set(picks) == set(zip(high.date, high.code, high.daily_rank))
    c.register("high", high)
    edges = c.sql("""WITH x AS(SELECT h.date,h.code AS event,h.daily_rank,p.code AS peer,
      abs(h.return20_prior_adjusted-p.return20_prior_adjusted) AS prior_gap,
      abs(h.return_1450-p.return_1450) AS day_gap,p.amount_signal/h.amount_signal AS ar,
      p.price_signal/h.price_signal AS pr FROM high h JOIN pool p ON h.date=p.date AND h.board=p.board
      WHERE NOT EXISTS(SELECT 1 FROM high a WHERE a.date=p.date AND a.code=p.code))
      SELECT *,prior_gap/.05+day_gap/.02+abs(ln(ar))/ln(2)+abs(ln(pr))/ln(2) AS distance
      FROM x WHERE prior_gap<=.05 AND day_gap<=.02 AND ar BETWEEN .5 AND 2 AND pr BETWEEN .5 AND 2
      ORDER BY date,daily_rank,distance,peer""").df()
    matched, used, assigned = [], set(), set()
    for row in edges.itertuples():
        if (row.date, row.event) in assigned or (row.date, row.peer) in used:
            continue
        assigned.add((row.date, row.event)); used.add((row.date, row.peer))
        matched.append((row.date, row.peer, row.date+":"+row.event))
    assert set(matched) == set(zip(low.date, low.code, low.pair_id))
    needed = pd.DataFrame(sorted({(r.code, str(year)) for r in signals.itertuples()
        for year in range(int(r.date[:4]), int(calendar[position[r.date]+10][:4])+1)}), columns=["code", "year"])
    assert needed.merge(coverage, on=["code", "year"], how="left", indicator=True)._merge.eq("both").all()
    checks[name] = {"fits": fit_checks, "candidates": len(high), "controls": len(low),
        "all_matching_edges": len(edges), "catalogue_code_years": len(needed), "signals_sha256": info["signals_sha256"]}
    save_json(folder / "independent_input_checks.json", checks[name])
save_json(root / "independent_input_checks.json", {"models": checks,
    "input_report_sha256": sha(root / "input_report.json"), "new_selected_test_returns_read": False})
print(json.dumps(checks, indent=2))
