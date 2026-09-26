"""Both tails and timing interactions for the frozen historical peer groups."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .reference_gain_accounting import weekly_interval
from .winner_peer_timing import ROOT, COHORT, PROTOCOL, FEATURES

METRICS = ("up", "down", "balance", "reference_gain")


def half(date: pd.Series) -> pd.Series:
    return date.str[:4] + np.where(date.str[5:7].le("06"), "H1", "H2")


def bands(rank: pd.Series) -> np.ndarray:
    return np.select([rank.isna(), rank.le(.2), rank.ge(.8)],
                     ["missing", "low20", "high20"], default="middle60")


def tables(frame: pd.DataFrame, *, group_keys: list[str], reference_keys: list[str],
           reference: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = frame.groupby(["date", *group_keys]).agg(
        rows=("code", "size"), known=("known_label", "sum"),
        up=("up", "mean"), down=("down", "mean"), balance=("balance", "mean"),
        reference_gain=("reference_gain", "mean"), winners=("up", "sum"), losers=("down", "sum")).reset_index()
    daily["half"] = half(daily.date)
    daily["unknown"] = daily.rows - daily.known
    joined = daily.merge(reference, on=["date", *reference_keys], how="left", validate="many_to_one", suffixes=("", "_baseline"))
    for metric in METRICS:
        joined[metric + "_difference"] = joined[metric] - joined[metric + "_baseline"]
    summaries = []
    for key, part in joined.groupby(["half", *group_keys], dropna=False):
        row = dict(zip(["half", *group_keys], key))
        row.update({name: int(part[name].sum()) for name in ("rows", "known", "unknown", "winners", "losers")})
        row["days"] = int(part.known.gt(0).sum())
        for metric in METRICS:
            values = part.set_index("date")[metric].dropna()
            delta = part.set_index("date")[metric + "_difference"].dropna()
            row[metric] = float(values.mean()) if len(values) else None
            row[metric + "_week_ci"] = weekly_interval(values)
            row[metric + "_difference"] = float(delta.mean()) if len(delta) else None
            row[metric + "_difference_week_ci"] = weekly_interval(delta)
        summaries.append(row)
    return joined, pd.DataFrame(summaries)


def evaluate() -> dict:
    if (ROOT / "analysis_report.json").exists():
        raise ValueError("Do not overwrite inspected peer results")
    report = json.loads((ROOT / "feature_report.json").read_text())
    check = json.loads((ROOT / "independent_input_checks.json").read_text())
    if sha(ROOT / "features.parquet") != report["features_sha256"] or sha(ROOT / "feature_report.json") != check["feature_report_sha256"]:
        raise ValueError("Peer features have not been independently verified")
    base_report = json.loads((COHORT / "base_report.json").read_text())
    if sha(COHORT / "labels.parquet") != base_report["labels_sha256"]:
        raise ValueError("Frozen outcome identities changed")
    f = pd.read_parquet(ROOT / "features.parquet")
    labels = pd.read_parquet(COHORT / "labels.parquet", columns=["date", "code", "known_label",
        "next_close", "next_preclose", "next_gain", "cohort"])
    close = np.rint(labels.next_close.fillna(0) * 100).astype("int64")
    reference = np.rint(labels.next_preclose.fillna(0) * 100).astype("int64")
    labels["up"] = (close * 100 >= reference * 105).where(labels.known_label).astype(float)
    labels["down"] = (close * 100 <= reference * 95).where(labels.known_label).astype(float)
    labels["balance"] = labels.up - labels.down
    labels["reference_gain"] = labels.next_gain
    f = f.merge(labels[["date", "code", "known_label", "cohort", *METRICS]], on=["date", "code"], validate="one_to_one")
    ranks = f.groupby(["date", "board"])[list(FEATURES)].rank(method="average", pct=True)
    f["own_tail_band"] = np.select([f.return_tail29.gt(.005), f.return_tail29.lt(-.005)],
                                    ["rising", "falling"], default="flat")
    f["day_range"] = (f.high_1449 - f.low_1449) / f.preclose
    baseline_frames, baseline_summaries = [], []
    for scope, mask in (("all", np.ones(len(f), dtype=bool)), ("necessary", f.necessary_tradeable)):
        p = f.loc[mask].copy()
        b = p.groupby(["date", "board"])[list(METRICS)].mean().reset_index()
        b["scope"] = scope
        baseline_frames.append(b)
        b["half"] = half(b.date)
        baseline_summaries.extend(b.groupby(["half", "board", "scope"])[list(METRICS)].mean().reset_index().to_dict("records"))
    baseline = pd.concat(baseline_frames, ignore_index=True).drop(columns="half", errors="ignore")
    band_daily, band_summary, portraits, matched = [], [], [], []
    rank_file = f[["date", "code"]].copy()
    rank_file[list(FEATURES)] = ranks
    rank_file.to_parquet(ROOT / "ranks.parquet", index=False, compression="zstd")
    for feature in FEATURES:
        for scope, mask in (("all", np.ones(len(f), dtype=bool)), ("necessary", f.necessary_tradeable)):
            p = f.loc[mask, ["date", "code", "board", "half", "cohort", "known_label", feature, *METRICS]].copy()
            p["band"] = bands(ranks.loc[p.index, feature])
            p["rank"] = ranks.loc[p.index, feature]
            daily, summary = tables(p, group_keys=["board", "band"], reference_keys=["board"],
                reference=baseline.loc[baseline.scope.eq(scope), ["date", "board", *METRICS]])
            daily["feature"], daily["scope"] = feature, scope
            summary["feature"], summary["scope"] = feature, scope
            band_daily.append(daily)
            band_summary.append(summary)
            desc = p.groupby(["half", "board", "cohort"]).agg(rows=("code", "size"),
                observed=(feature, "count"), mean_value=(feature, "mean"), median_value=(feature, "median"), mean_rank=("rank", "mean")).reset_index()
            desc["feature"], desc["scope"] = feature, scope
            portraits.append(desc)
        mask = f.necessary_tradeable & f.known_label & f.return20_prior_adjusted.notna() & ranks[feature].notna()
        p = f.loc[mask].copy()
        p["rank"] = ranks.loc[p.index, feature]
        dimensions = {"day_bin": p.return_1449 / .02, "prior_bin": p.return20_prior_adjusted / .1,
            "price_bin": np.log2(p.price_1449 / 5), "amount_bin": np.log2(p.amount_1449 / 3e7),
            "range_bin": p.day_range / .02, "tail_bin": p.return_tail29 / .01}
        for name, values in dimensions.items():
            p[name] = np.floor(values).astype("int64")
        keys = ["date", "board", *dimensions]
        cells = p.groupby(keys + ["up"]).agg(mean_rank=("rank", "mean"), n=("rank", "size")).unstack("up").dropna()
        day = pd.DataFrame({"weight": cells["n", 1.],
            "weighted_difference": (cells["mean_rank", 1.] - cells["mean_rank", 0.]) * cells["n", 1.]}).reset_index()
        day = day.groupby(["date", "board"])[["weight", "weighted_difference"]].sum().reset_index()
        day["rank_difference"] = day.weighted_difference / day.weight
        day["feature"] = feature
        matched.append(day)
        print(json.dumps({"peer_feature_analyzed": feature}), flush=True)
    matched = pd.concat(matched, ignore_index=True)
    matched["half"] = half(matched.date)
    summaries = []
    for (feature, period, board), part in matched.groupby(["feature", "half", "board"]):
        all_winners = int(f.loc[f.necessary_tradeable & f.half.eq(period) & f.board.eq(board), "up"].sum())
        summaries.append({"feature": feature, "half": period, "board": board, "days": len(part),
            "matched_winners": int(part.weight.sum()), "all_winners": all_winners,
            "coverage": float(part.weight.sum() / all_winners) if all_winners else None,
            "rank_difference": float(part.rank_difference.mean()),
            "rank_difference_week_ci": weekly_interval(part.set_index("date").rank_difference)})
    p = f.loc[f.necessary_tradeable].copy()
    p["band"] = bands(ranks.loc[p.index, "peer_tail_return"])
    reference = p.groupby(["date", "board", "own_tail_band"])[list(METRICS)].mean().reset_index()
    interaction_daily, interaction_summary = tables(p, group_keys=["board", "own_tail_band", "band"],
        reference_keys=["board", "own_tail_band"], reference=reference)
    outputs = {"band_daily": pd.concat(band_daily, ignore_index=True),
        "band_summary": pd.concat(band_summary, ignore_index=True),
        "portraits": pd.concat(portraits, ignore_index=True), "matched_daily": matched,
        "matched_summary": pd.DataFrame(summaries), "interaction_daily": interaction_daily,
        "interaction_summary": interaction_summary, "baseline_daily": baseline}
    for name, table in outputs.items():
        table.to_parquet(ROOT / (name + ".parquet"), index=False, compression="zstd")
    result = {"protocol_sha256": sha(PROTOCOL), "feature_report_sha256": sha(ROOT / "feature_report.json"),
        "labels_sha256": sha(COHORT / "labels.parquet"), "rows": len(f),
        "necessary_rows": int(f.necessary_tradeable.sum()), "known_labels": int(f.known_label.sum()),
        "winners": int(f.up.sum()), "losers": int(f.down.sum()), "baseline": baseline_summaries,
        "outputs_sha256": {name: sha(ROOT / (name + ".parquet")) for name in [*outputs, "ranks"]},
        "trading_returns_computed": False, "new_2026_prices_read": False,
        "multiple_exploratory_features_not_independent_confirmations": True}
    save_json(ROOT / "analysis_report.json", result)
    return {key: value for key, value in result.items() if key not in ("baseline", "outputs_sha256")}


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
