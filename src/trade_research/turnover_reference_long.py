"""Rebuild the fixed 2024–2025 reference inputs with available 2019+ history."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import pandas as pd

from .corporate_cash import save_json, sha
from .reference_gain_inputs import BARS, classify, assess_pairs
from .cash_ex_matching import match_candidates
from .turnover_reference import (
    BASE, BASE_SHA, CALENDAR, ROOT as OLD_ROOT, attach_references,
    history_states, summarize,
)

ROOT = Path("data/research/turnover_reference_long")
RULE_COMMIT = "c17b6a3"


def build(output: Path = ROOT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if sha(BASE / "base.parquet") != BASE_SHA:
        raise ValueError("Fixed base changed")
    base = pd.read_parquet(BASE / "base.parquet")
    bounds = base.groupby("code", as_index=False).agg(history_end=("previous_traded_date", "max"))
    old_manifest = json.loads((OLD_ROOT / "manifest.json").read_text())
    sources = old_manifest["daily_sha256"]
    for name, digest in sources.items():
        if sha(Path(name)) != digest:
            raise ValueError(f"Daily source changed: {name}")
    manifest = {"rule_commit": RULE_COMMIT, "base_sha256": BASE_SHA,
        "daily_sha256": sources, "calendar_sha256": sha(CALENDAR),
        "history_start": "2019-01-01", "reset_after_unknown": True,
        "signal_years": [2024, 2025], "entry_prices_read": False,
        "holding_returns_read": False, "holdout_read": False}
    if (output / "manifest.json").exists() and json.loads((output / "manifest.json").read_text()) != manifest:
        raise ValueError("Cannot overwrite frozen long-history inputs")
    save_json(output / "manifest.json", manifest)
    c = duckdb.connect()
    c.execute("SET threads=4")
    c.register("bounds", bounds)
    c.read_parquet(list(sources)).create_view("daily")
    c.read_parquet(str(CALENDAR)).create_view("calendar_source")
    history = c.execute("""
        SELECT d.date,d.code,d.close,d.preclose,d.turn,d.volume,d.tradestatus,d.adjustflag
        FROM daily d JOIN bounds b USING(code)
        WHERE d.date BETWEEN '2019-01-01' AND '2025-12-31' AND d.date<=b.history_end
        ORDER BY code,date
    """).df()
    calendar = c.execute("""SELECT calendar_date FROM calendar_source
        WHERE is_trading_day='1' AND calendar_date BETWEEN '2019-01-01' AND '2025-12-31'
        ORDER BY calendar_date""").fetchnumpy()["calendar_date"].tolist()
    c.register("history", history)
    counts = c.execute("""
        SELECT code,date previous_traded_date,count(*) OVER w all_prior_sessions,
          count(*) FILTER(WHERE date>='2024-01-01') OVER w prior_sessions_2024_check,
          count(turn) OVER w20 prior20_turnover_count,avg(turn) OVER w20 prior20_turnover_pct
        FROM history WHERE tradestatus=1
        WINDOW w AS(PARTITION BY code ORDER BY date ROWS UNBOUNDED PRECEDING),
          w20 AS(PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
    """).df()
    base = base.merge(counts, on=["code", "previous_traded_date"], how="left", validate="many_to_one")
    if not base.prior_sessions.eq(base.prior_sessions_2024_check).all():
        raise ValueError("2024-era history no longer agrees with the fixed base")
    base["prior_sessions_2024"] = base.prior_sessions
    base["prior_sessions"] = base.all_prior_sessions
    needs = base.groupby("code").previous_traded_date.agg(set).to_dict()
    frames = []
    for i, (code, group) in enumerate(history.groupby("code", sort=False), 1):
        states = history_states(group, calendar, history_start="2019-01-01", reset_after_unknown=True)
        frames.append(states.loc[states.source_date.isin(needs[code])])
        if i % 500 == 0 or i == len(needs):
            print(f"Long-history references {i}/{len(needs)}", flush=True)
    states = pd.concat(frames, ignore_index=True)
    references = attach_references(base, states)
    c.register("reference_inputs", references)
    c.read_parquet([str(p) for p in sorted(BARS.glob("part_*.parquet"))]).create_view("bars")
    joined = c.execute("""SELECT r.*,b.n_1449,b.raw_price_1449,b.valid_1449
        FROM reference_inputs r LEFT JOIN bars b USING(code,date) ORDER BY r.date,r.code""").df()
    c.close()
    if len(joined) != 328895 or joined.duplicated(["date", "code"]).any():
        raise ValueError("Signal denominator changed")
    screened, candidates, controls = classify(joined)
    attempts = candidates.loc[candidates.daily_rank.le(5)].copy()
    pairs, missed = match_candidates(attempts, controls,
        extra_ratio_limits={"prior20_turnover_pct": 2.0})
    diag = ["screen_gain_low", "screen_gain_high", "screen_width"]
    pairs = pairs.merge(attempts[["date", "code", *diag]], on=["date", "code"], validate="one_to_one")
    pairs = pairs.merge(controls[["date", "code", *diag]].rename(columns={
        "code": "control_code", **{k: "control_" + k for k in diag}}),
        on=["date", "control_code"], validate="one_to_one")
    tables = {"history_inputs": history, "reference_states": states,
        "input_references": joined, "screened_inputs": screened,
        "old_rule_attempts": attempts, "old_rule_pairs": pairs, "old_rule_unmatched": missed}
    for name, table in tables.items():
        table.to_parquet(output / f"{name}.parquet", index=False, compression="zstd")
    report = {"rule_commit": RULE_COMMIT, "history_rows": len(history),
        "history_first_date": history.date.min(), "signal_rows": len(joined),
        "by_half": summarize(joined), "old_rule_matching": assess_pairs(attempts, pairs),
        "reference_valid_rows": int(joined.reference_valid.sum()),
        "signals_after_history_reset": int(joined.history_resets.gt(0).sum()),
        "output_sha256": {k: sha(output / f"{k}.parquet") for k in tables},
        "entry_prices_read": False, "holding_returns_read": False, "holdout_read": False}
    save_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    result = build()
    print(json.dumps({"history_rows": result["history_rows"],
        "reference_valid_rows": result["reference_valid_rows"],
        "signals_after_history_reset": result["signals_after_history_reset"],
        "stable_coverage": [{"half": r["half"], "rows": r["width_le_1pp_rows"],
            "fraction": r["width_le_1pp_fraction_of_all"]} for r in result["by_half"]],
        "old_rule_matching": result["old_rule_matching"]}, ensure_ascii=False, indent=2))
