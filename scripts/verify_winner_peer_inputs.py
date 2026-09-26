"""Rebuild daily histories, sampled complete peer rankings and every visible peer mean."""
import gc
import hashlib
import json
from pathlib import Path
import warnings

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import DAILY, save_json, sha
from trade_research.turnover_reference import CALENDAR

root = Path("data/research/winner_peer_timing")
cohort = Path("data/research/next_day_winner")
report = json.loads((root / "graph_report.json").read_text())
features_report = json.loads((root / "feature_report.json").read_text())
manifest = json.loads((root / "source_manifest.json").read_text())
assert sha(root / "source_manifest.json") == report["source_manifest_sha256"]
assert sha(root / "graph_report.json") == features_report["graph_report_sha256"]
assert sha(root / "history_returns.parquet") == report["history_sha256"]
assert sha(root / "features.parquet") == features_report["features_sha256"]
for path, digest in manifest["sha256"].items():
    assert sha(Path(path)) == digest
calendar = pd.read_parquet(CALENDAR)
days = calendar.loc[calendar.is_trading_day.eq("1"), "calendar_date"].sort_values().tolist()
required = set()
for item in report["months"]:
    first = next(day for day in days if day.startswith(item["month"]))
    offset = days.index(first)
    window = days[offset - 60:offset]
    assert len(window) == 60 and window[0] == item["first"] and window[-1] == item["last"] < item["asof"] == first
    required.update(window)
c = duckdb.connect()
c.execute("SET threads=4")
raw = c.execute("""SELECT date,code,open,high,low,close,preclose,volume,adjustflag,tradestatus,isST
    FROM read_parquet(?) WHERE date BETWEEN ? AND ?""",
    [str(DAILY / "*.parquet"), min(required), max(required)]).df()
c.close()
valid = (raw.date.isin(required) & raw.code.str.match(r"(?:sh\.60|sz\.00|sz\.30|sh\.68)\d{4}$")
    & raw.adjustflag.eq(3) & raw.tradestatus.eq(1) & raw.isST.eq(0) & raw.volume.gt(0)
    & raw[["open", "close", "low", "preclose"]].gt(0).all(axis=1)
    & raw.high.ge(raw[["open", "close", "low"]].max(axis=1))
    & raw.low.le(raw[["open", "close", "high"]].min(axis=1)))
history = raw.loc[valid, ["date", "code"]].copy()
history["board"] = np.select([history.code.str.startswith("sz.30"), history.code.str.startswith("sh.68")],
                              ["chinext", "star"], default="main")
history["daily_return"] = raw.loc[valid, "close"] / raw.loc[valid, "preclose"] - 1
history["board_return"] = history.groupby(["date", "board"]).daily_return.transform("mean")
history = history.sort_values(["date", "code"]).reset_index(drop=True)
stored = pd.read_parquet(root / "history_returns.parquet").sort_values(["date", "code"]).reset_index(drop=True)
pd.testing.assert_frame_equal(history, stored, atol=1e-13, rtol=1e-13)
del raw, valid, stored
gc.collect()
graphs = {}
sampled_anchors = 0
max_correlation_error = 0.0
for item in report["months"]:
    path = root / "graphs" / (item["month"] + ".parquet")
    assert sha(path) == item["sha256"]
    g = pd.read_parquet(path)
    assert g.month.eq(item["month"]).all() and g.history_last.eq(item["last"]).all()
    assert g.history_first.eq(item["first"]).all() and g["asof"].eq(item["asof"]).all()
    assert not g.code.eq(g.peer_code).any() and not g.duplicated(["code", "peer_code"]).any()
    assert g.groupby("code").size().eq(10).all() and g.correlation.gt(0).all()
    assert g.groupby("code")["rank"].apply(list).map(lambda values: values == list(range(1, 11))).all()
    h = history.loc[history.date.between(item["first"], item["last"])]
    y = h.pivot(index="date", columns="code", values="daily_return").dropna(axis=1)
    benchmarks = h.groupby(["date", "board"]).board_return.first().unstack().reindex(y.index)
    boards = h[["code", "board"]].drop_duplicates().set_index("code").board.reindex(y.columns)
    residual = np.empty(y.shape)
    for board in ("main", "chinext", "star"):
        mask = boards.eq(board).to_numpy()
        x = np.column_stack([np.ones(len(y)), benchmarks[board]])
        target = y.to_numpy()[:, mask]
        coefficients = np.linalg.lstsq(x, target, rcond=None)[0]
        residual[:, mask] = target - x @ coefficients
    residual -= residual.mean(axis=0)
    norm = np.linalg.norm(residual, axis=0)
    keep = norm > 1e-12
    codes = y.columns.to_numpy()[keep]
    z = residual[:, keep] / norm[keep]
    assert set(g.code) <= set(codes) and set(g.peer_code) <= set(codes)
    for board in ("main", "chinext", "star"):
        candidates = [code for code in g.code.unique() if boards.loc[code] == board]
        anchors = sorted(candidates, key=lambda code: hashlib.sha256((item["month"] + code).encode()).hexdigest())[:2]
        for code in anchors:
            index = np.flatnonzero(codes == code)[0]
            corr = z.T @ z[:, index]
            ordered = sorted([(float(corr[j]), str(peer)) for j, peer in enumerate(codes)
                              if peer != code and corr[j] > 0], key=lambda pair: (-pair[0], pair[1]))[:10]
            selected = g.loc[g.code.eq(code)].sort_values("rank")
            assert selected.peer_code.tolist() == [peer for _, peer in ordered]
            error = np.max(np.abs(selected.correlation.to_numpy() - np.array([value for value, _ in ordered])))
            max_correlation_error = max(max_correlation_error, float(error))
            assert error < 1e-12
            sampled_anchors += 1
    graphs[item["month"]] = g
    print(json.dumps({"graph_checked": item["month"]}), flush=True)
del history, h, residual, y, z
gc.collect()
base = pd.read_parquet(cohort / "visible_base.parquet")
raw = pd.concat([pd.read_parquet(p, columns=["date", "code", "price_1000"])
                for p in sorted((cohort / "raw_features").glob("*.parquet"))], ignore_index=True)
base = base.merge(raw, on=["date", "code"], how="left", validate="one_to_one")
all_graphs = pd.concat(graphs.values())
all_codes = sorted(set(base.code) | set(all_graphs.code) | set(all_graphs.peer_code))
indexer = {code: index for index, code in enumerate(all_codes)}
sentinel = len(all_codes)
stored = pd.read_parquet(root / "features.parquet").set_index(["date", "code"])
assert len(base) == len(stored)
names = ["peer_return_1000", "peer_return_1420", "peer_return_1449",
         "peer_tail_return", "peer_tail_breadth", "peer_volume_ratio"]
maximum_feature_error = 0.0
checked_rows = 0
for month, g in graphs.items():
    neighbors = np.full((len(all_codes) + 1, 10), sentinel, dtype="int64")
    for code, group in g.groupby("code"):
        neighbors[indexer[code]] = group.sort_values("rank").peer_code.map(indexer).to_numpy()
    for day, part in base.loc[base.date.str.startswith(month)].groupby("date", sort=True):
        ids = part.code.map(indexer).to_numpy()
        table = np.full((len(all_codes) + 1, 6), np.nan)
        table[ids] = np.column_stack([part.price_1000 / part.preclose - 1,
            part.price_1420 / part.preclose - 1, part.return_1449, part.return_tail29,
            part.return_tail29.gt(0).astype(float), part.volume_ratio_prior5])
        values = table[neighbors[ids]]
        counts = np.isfinite(values).sum(axis=1)
        sums = np.nansum(values, axis=1)
        expected = np.divide(sums, counts, out=np.full_like(sums, np.nan), where=counts > 0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            expected[:, 5] = np.nanmedian(values[:, :, 5], axis=1)
        expected[counts[:, 0] < 8] = np.nan
        expected[counts[:, 5] < 8, 5] = np.nan
        current = stored.loc[pd.MultiIndex.from_arrays([part.date, part.code])]
        assert np.array_equal(counts[:, 0], current.peer_available)
        assert np.array_equal(neighbors[ids, 0] != sentinel, current.graph_present)
        actual = current[names].to_numpy()
        assert np.array_equal(np.isnan(expected), np.isnan(actual))
        error = float(np.nanmax(np.abs(expected - actual)))
        maximum_feature_error = max(maximum_feature_error, error)
        assert error < 1e-12
        checked_rows += len(part)
    print(json.dumps({"features_checked": month, "rows": checked_rows}), flush=True)
result = {"history_rows_checked": report["history_rows"], "monthly_graphs": len(graphs),
    "graph_edges": sum(len(g) for g in graphs.values()), "anchors_full_ranking_checked": sampled_anchors,
    "maximum_correlation_error": max_correlation_error, "feature_rows_checked": checked_rows,
    "feature_values_checked": checked_rows * len(names), "maximum_feature_error": maximum_feature_error,
    "feature_report_sha256": sha(root / "feature_report.json"), "new_2026_prices_read": False}
save_json(root / "independent_input_checks.json", result)
print(json.dumps(result, indent=2))
