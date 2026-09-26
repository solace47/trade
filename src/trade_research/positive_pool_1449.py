"""One fixed hash ordering within four existing positive-prediction pools."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .absolute_ridge import PERIODS, choose, match_controls
from .corporate_cash import save_json, sha
from .long_history_inputs import ROOT as HISTORY
from .quote_precision import quote_cents, fixed_quote_shares
from .turnover_reference import CALENDAR

ROOT = Path("data/research/positive_pool_1449")
RULE_COMMIT = "112e710"
SCORE_SOURCES = {
    "long_ridge": Path("data/research/long_history_ridge_1449/long"),
    "long_tree": Path("data/research/long_history_tree_1449/tree"),
    "robust18": Path("data/research/alpha158_1449/robust18"),
    "alpha158": Path("data/research/alpha158_1449/alpha158"),
}
SALT = "positive-score-pool-v1"


def choose_pool(scores: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    if scores.duplicated(["date", "code"]).any() or not np.isfinite(scores.score).all():
        raise ValueError("Positive-pool selection needs unique finite frozen predictions")
    positions, last, selected = {d: i for i, d in enumerate(calendar)}, {}, []
    pool = scores.loc[scores.score.gt(0), ["date", "code", "score"]].copy()
    pool["selection_key"] = [hashlib.sha256((SALT + date + code).encode()).hexdigest()
                             for date, code in zip(pool.date, pool.code)]
    for date, group in pool.groupby("date", sort=True):
        rank = 0
        for row in group.sort_values(["selection_key", "code"]).to_dict("records"):
            if positions[date] - last.get(row["code"], -1000) <= 5:
                continue
            rank += 1
            row["daily_rank"] = rank
            selected.append(row)
            last[row["code"]] = positions[date]
            if rank == 5:
                break
    return pd.DataFrame(selected, columns=[*pool.columns, "daily_rank"])


def freeze(output: Path = ROOT) -> dict:
    if any((output / name / "repriced.parquet").exists() for name in SCORE_SOURCES):
        raise ValueError("Do not change a pool list after its execution outcomes exist")
    output.mkdir(parents=True, exist_ok=True)
    feature_report = json.loads((HISTORY / "feature_report.json").read_text())
    if sha(HISTORY / "features.parquet") != feature_report["features_sha256"]:
        raise ValueError("The common long-history feature pool changed")
    base = pd.read_parquet(HISTORY / "features.parquet")
    include = pd.Series(False, index=base.index)
    for _, first, last, _ in PERIODS:
        include |= base.date.between(first, last)
    eligible = base.loc[include].copy()
    eligible["price_1449"] = eligible.price_1449.map(lambda p: quote_cents(p) / 100)
    eligible["price_signal"] = eligible.price_1449
    expected = eligible[["date", "code"]].sort_values(["date", "code"]).reset_index(drop=True)
    table = pd.read_parquet(CALENDAR)
    calendar = sorted(table.loc[table.is_trading_day.eq("1")
        & table.calendar_date.between("2024-01-01", "2025-12-31"), "calendar_date"])
    models, sources = {}, {}
    for name, source in SCORE_SOURCES.items():
        report = json.loads((source / "input_report.json").read_text())
        if sha(source / "signals.parquet") != report["signals_sha256"]:
            raise ValueError("An original highest-score comparison list changed")
        scores = pd.read_parquet(source / "all_scores.parquet").sort_values(["date", "code"]).reset_index(drop=True)
        pd.testing.assert_frame_equal(scores[["date", "code"]], expected)
        original = pd.read_parquet(source / "signals.parquet")
        high_original = original.loc[original.arm.eq("high"), ["date", "code", "daily_rank", "score"]]
        reproduced = choose(scores, calendar)
        pd.testing.assert_frame_equal(reproduced.sort_values(["date", "code"]).reset_index(drop=True),
            high_original.sort_values(["date", "code"]).reset_index(drop=True), check_dtype=False, atol=1e-12, rtol=0)
        high = choose_pool(scores, calendar)
        low = match_controls(high, eligible)
        high["arm"], high["pair_id"] = "high", high.code
        low["arm"] = "low"
        members = pd.concat([high, low], ignore_index=True)
        members["pair_id"] = members.date + ":" + members.pair_id
        signals = members.merge(eligible, on=["date", "code"], validate="one_to_one")
        signals["isST"], signals["reference_gap"], signals["listing_age_sessions"] = 0, False, 20
        signals["half"] = signals.date.str[:4] + np.where(signals.date.str[5:7].le("06"), "H1", "H2")
        signals["decision_shares"] = [fixed_quote_shares(r.code, r.price_1449, 20000) for r in signals.itertuples()]
        signals = signals.sort_values(["date", "arm", "daily_rank", "code"]).reset_index(drop=True)
        if len(signals) != len(members) or signals.duplicated(["date", "code"]).any():
            raise ValueError("A new pool selection or comparison identity changed")
        folder = output / name
        folder.mkdir(exist_ok=True)
        signals.to_parquet(folder / "signals.parquet", index=False, compression="zstd")
        model_report = {"candidates": len(high), "controls": len(low),
            "positive_prediction_pool": int(scores.score.gt(0).sum()),
            "original_highest_score_candidates": len(high_original),
            "candidate_overlap_with_original": len(high.merge(high_original, on=["date", "code"])),
            "by_half": signals.groupby(["half", "arm"]).agg(rows=("code", "size"), days=("date", "nunique"))
                .reset_index().to_dict("records"), "signals_sha256": sha(folder / "signals.parquet")}
        save_json(folder / "input_report.json", model_report)
        models[name] = model_report
        sources[name] = {str(source / path): sha(source / path)
            for path in ("all_scores.parquet", "signals.parquet", "input_report.json")}
        print(name, {key: model_report[key] for key in ("candidates", "controls", "positive_prediction_pool")}, flush=True)
    result = {"rule_commit": RULE_COMMIT, "hash_salt": SALT, "models": models,
        "score_source_sha256": sources, "features_sha256": feature_report["features_sha256"],
        "calendar_sha256": sha(CALENDAR), "new_rule_holding_results_computed": False,
        "prior_development_outcomes_already_exposed": True, "holdout_prices_read": False}
    save_json(output / "input_report.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(freeze()["models"], ensure_ascii=False, indent=2))
