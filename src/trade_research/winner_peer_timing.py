"""Build monthly price-comovement peers strictly from the preceding 60 sessions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .corporate_cash import DAILY, save_json, sha
from .next_day_winner import ROOT as COHORT, connection
from .turnover_reference import CALENDAR

ROOT = Path("data/research/winner_peer_timing")
PROTOCOL = Path("config/winner_peer_timing_protocol.json")
RULE_COMMIT = "a40ec60"
FEATURES = tuple(json.loads(PROTOCOL.read_text())["features"])
BASE_COLUMNS = ["date", "code", "board", "half", "industry", "necessary_tradeable",
    "price_1449", "price_1420", "preclose", "amount_1449", "return_1449",
    "return20_prior_adjusted", "return_tail29", "high_1449", "low_1449",
    "volume_ratio_prior5", "industry_return"]


def schedule() -> pd.DataFrame:
    table = pd.read_parquet(CALENDAR)
    days = sorted(table.loc[table.is_trading_day.eq("1"), "calendar_date"])
    first = {}
    for day in days:
        if "2024-01-01" <= day <= "2025-12-31":
            first.setdefault(day[:7], day)
    rows = []
    for month, asof in first.items():
        index = days.index(asof)
        history = days[index-60:index]
        assert len(history) == 60 and history[-1] < asof
        rows.append({"month": month, "asof": asof, "first": history[0],
                     "last": history[-1], "days": history})
    return pd.DataFrame(rows)


def graph() -> dict:
    if (ROOT / "graph_report.json").exists():
        raise ValueError("Do not replace a completed graph")
    ROOT.mkdir(parents=True, exist_ok=True)
    folder = ROOT / "graphs"
    folder.mkdir(exist_ok=True)
    plan = schedule()
    c = connection()
    c.read_parquet(str(DAILY / "*.parquet")).create_view("daily")
    c.register("sessions", pd.DataFrame({"date": sorted(set(sum(plan.days.tolist(), [])))}))
    history = c.sql("""WITH good AS (
        SELECT d.date,d.code,
            CASE WHEN d.code LIKE 'sh.60%' OR d.code LIKE 'sz.00%' THEN 'main'
                 WHEN d.code LIKE 'sz.30%' THEN 'chinext' ELSE 'star' END AS board,
            d.close/d.preclose-1 AS daily_return
        FROM daily d JOIN sessions s USING(date)
        WHERE (d.code LIKE 'sh.60%' OR d.code LIKE 'sz.00%' OR d.code LIKE 'sz.30%' OR d.code LIKE 'sh.68%')
        AND d.adjustflag=3 AND d.tradestatus=1 AND d.isST=0 AND d.volume>0
        AND d.open>0 AND d.close>0 AND d.low>0 AND d.preclose>0
        AND d.high>=greatest(d.open,d.close,d.low) AND d.low<=least(d.open,d.close,d.high)
    ) SELECT *,avg(daily_return) OVER(PARTITION BY date,board) AS board_return
    FROM good ORDER BY date,code""").df()
    c.close()
    if history.duplicated(["date", "code"]).any() or not np.isfinite(history[["daily_return", "board_return"]]).all().all():
        raise ValueError("Invalid daily histories")
    history.to_parquet(ROOT / "history_returns.parquet", index=False, compression="zstd")
    sources = [PROTOCOL, CALENDAR, *sorted(DAILY.glob("*.parquet"))]
    save_json(ROOT / "source_manifest.json", {"rule_commit": RULE_COMMIT,
        "sha256": {str(p): sha(p) for p in sources}, "history_first": history.date.min(),
        "history_last": history.date.max(), "new_2026_prices_read": False})
    boards = history[["code", "board"]].drop_duplicates().set_index("code").board
    reports = []
    for row in plan.itertuples(index=False):
        window = history.loc[history.date.isin(row.days)]
        prices = window.pivot(index="date", columns="code", values="daily_return").reindex(row.days).dropna(axis=1)
        matrix = prices.to_numpy(dtype="float64")
        board = boards.reindex(prices.columns).to_numpy()
        residual = np.empty_like(matrix)
        benchmarks = window.groupby(["date", "board"]).board_return.first().unstack().reindex(row.days)
        for group in ("main", "chinext", "star"):
            mask = board == group
            target = matrix[:, mask] - matrix[:, mask].mean(axis=0)
            market = benchmarks[group].to_numpy()
            market = market - market.mean()
            variance = market @ market
            if variance <= 0:
                raise ValueError("Constant board benchmark")
            beta = market @ target / variance
            residual[:, mask] = target - market[:, None] * beta[None, :]
        norms = np.sqrt((residual * residual).sum(axis=0))
        good = np.isfinite(norms) & (norms > 1e-12)
        codes = prices.columns.to_numpy()[good]
        normalized = (residual[:, good] / norms[good]).T.copy()
        with threadpool_limits(limits=4):
            correlations = normalized @ normalized.T
        correlations = np.clip(correlations, -1, 1)
        np.fill_diagonal(correlations, -np.inf)
        thresholds = np.partition(correlations, -10, axis=1)[:, -10]
        records = []
        for index, code in enumerate(codes):
            if thresholds[index] <= 0:
                continue
            values = correlations[index]
            candidates = np.flatnonzero(values >= thresholds[index])
            order = np.lexsort((codes[candidates], -values[candidates]))[:10]
            for rank, peer in enumerate(candidates[order], 1):
                records.append((row.month, row.asof, row.first, row.last, code,
                                codes[peer], rank, float(values[peer])))
        edges = pd.DataFrame(records, columns=["month", "asof", "history_first", "history_last",
            "code", "peer_code", "rank", "correlation"])
        path = folder / (row.month + ".parquet")
        edges.to_parquet(path, index=False, compression="zstd")
        stats = {"month": row.month, "asof": row.asof, "first": row.first, "last": row.last,
            "sessions": 60, "complete_histories": len(prices.columns), "nonconstant_residuals": len(codes),
            "anchors": int(edges.code.nunique()), "edges": len(edges),
            "minimum_selected_correlation": float(edges.correlation.min()),
            "median_selected_correlation": float(edges.correlation.median()), "sha256": sha(path)}
        reports.append(stats)
        print(json.dumps(stats), flush=True)
    result = {"rule_commit": RULE_COMMIT, "protocol_sha256": sha(PROTOCOL),
        "source_manifest_sha256": sha(ROOT / "source_manifest.json"),
        "history_sha256": sha(ROOT / "history_returns.parquet"), "history_rows": len(history),
        "months": reports, "new_2026_prices_read": False}
    save_json(ROOT / "graph_report.json", result)
    return result


def visible_base() -> pd.DataFrame:
    report = json.loads((COHORT / "base_report.json").read_text())
    if sha(COHORT / "visible_base.parquet") != report["base_sha256"]:
        raise ValueError("The original visible universe changed")
    base = pd.read_parquet(COHORT / "visible_base.parquet", columns=BASE_COLUMNS)
    raw = json.loads((COHORT / "raw_report.json").read_text())
    pieces = []
    for path, digest in raw["batch_manifests_sha256"].items():
        meta = Path(path)
        if sha(meta) != digest or sha(meta.with_suffix(".parquet")) != json.loads(meta.read_text())["sha256"]:
            raise ValueError("Historical minute extraction changed")
        pieces.append(pd.read_parquet(meta.with_suffix(".parquet"), columns=["date", "code", "price_1000"]))
    base = base.merge(pd.concat(pieces, ignore_index=True), on=["date", "code"], how="left", validate="one_to_one")
    if not np.isfinite(base[["price_1000", "price_1420", "price_1449", "preclose"]]).all().all():
        raise ValueError("Incomplete visible prices")
    return base


def features() -> dict:
    if (ROOT / "feature_report.json").exists():
        raise ValueError("Do not replace frozen features")
    report = json.loads((ROOT / "graph_report.json").read_text())
    if report["protocol_sha256"] != sha(PROTOCOL):
        raise ValueError("Peer protocol changed")
    base = visible_base()
    c = connection()
    c.register("base", base)
    c.execute("""CREATE VIEW visible AS SELECT *,date[:7] AS month,
        price_1000/preclose-1 AS return_1000,price_1420/preclose-1 AS return_1420
        FROM base""")
    pieces = []
    for month in report["months"]:
        path = ROOT / "graphs" / (month["month"] + ".parquet")
        if sha(path) != month["sha256"]:
            raise ValueError("A monthly graph changed")
        c.read_parquet(str(path)).create_view("edges", replace=True)
        piece = c.execute("""SELECT b.date,b.code,count(p.code) AS peer_available,
            count(g.peer_code)>0 AS graph_present,
            CASE WHEN count(p.code)>=8 THEN avg(p.return_1000) END AS peer_return_1000,
            CASE WHEN count(p.code)>=8 THEN avg(p.return_1420) END AS peer_return_1420,
            CASE WHEN count(p.code)>=8 THEN avg(p.return_1449) END AS peer_return_1449,
            CASE WHEN count(p.code)>=8 THEN avg(p.return_tail29) END AS peer_tail_return,
            CASE WHEN count(p.code)>=8 THEN avg((p.return_tail29>0)::DOUBLE) END AS peer_tail_breadth,
            CASE WHEN count(p.code)>=8 AND count(p.volume_ratio_prior5)>=8
                THEN median(p.volume_ratio_prior5) END AS peer_volume_ratio
            FROM visible b LEFT JOIN edges g ON b.code=g.code
            LEFT JOIN visible p ON b.date=p.date AND g.peer_code=p.code
            WHERE b.month=? GROUP BY b.date,b.code ORDER BY b.date,b.code""", [month["month"]]).df()
        pieces.append(piece)
        print(json.dumps({"features_month": month["month"], "rows": len(piece),
                          "complete_peers": int(piece.peer_available.ge(8).sum())}), flush=True)
    c.close()
    frame = base.merge(pd.concat(pieces, ignore_index=True), on=["date", "code"], validate="one_to_one")
    if len(frame) != len(base) or frame.peer_available.gt(10).any():
        raise ValueError("Peer join changed the visible universe")
    frame.to_parquet(ROOT / "features.parquet", index=False, compression="zstd")
    coverage = frame.groupby(["half", "board"]).agg(rows=("code", "size"),
        graph_present=("graph_present", "sum"), features_present=("peer_tail_return", "count"),
        necessary=("necessary_tradeable", "sum")).reset_index()
    result = {"rule_commit": RULE_COMMIT, "protocol_sha256": sha(PROTOCOL),
        "graph_report_sha256": sha(ROOT / "graph_report.json"), "rows": len(frame),
        "base_sha256": sha(COHORT / "visible_base.parquet"),
        "raw_report_sha256": sha(COHORT / "raw_report.json"), "features_sha256": sha(ROOT / "features.parquet"),
        "coverage": coverage.to_dict("records"), "new_2026_prices_read": False,
        "future_labels_read_in_this_stage": False}
    save_json(ROOT / "feature_report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["graph", "features"])
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](), ensure_ascii=False, indent=2))
