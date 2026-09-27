"""Frozen prior-state x theme-context groups against existing strict T1 cash."""
import json
from pathlib import Path

import duckdb
import pandas as pd

from .corporate_cash import save_json, sha
from .economic_winner_analysis import METRICS
from .prior_theme_broken import ROOT
from .tick_flow_winner_analysis import aggregate, in_period, summary, PERIODS

LABELS = Path("data/research/economic_winner/period_quality")
STATES = ["broken_limit", "closed_limit", "other", "unknown"]
CONTEXTS = ["strong", "neutral", "weak", "unknown"]


def evaluate(ROOT=ROOT, LABELS=LABELS, states=STATES, label_proof="analysis"):
    if (ROOT / "analysis_report.json").exists():
        raise ValueError("Do not overwrite inspected theme results")
    check = json.loads((ROOT / "input_verification.json").read_text())
    assert check["passed"] and check["input_report_sha256"] == sha(ROOT / "input_report.json")
    inputs = json.loads((ROOT / "input_report.json").read_text())
    for name, digest in inputs["output_sha256"].items():
        assert sha(ROOT / name) == digest
    assert label_proof in ["analysis", "label"]
    proof_file = LABELS / (label_proof + "_verification.json")
    proof = json.loads(proof_file.read_text())
    assert proof["passed"] and proof[label_proof + "_report_sha256"] == sha(LABELS / (label_proof + "_report.json"))
    labels = json.loads((LABELS / "label_report.json").read_text())
    assert labels["labels_sha256"] == sha(LABELS / "labels.parquet")
    c = duckdb.connect()
    mismatch = c.execute("""SELECT count(*) FROM read_parquet(?) f
        JOIN read_parquet(?) l USING(date,code) WHERE f.necessary_tradeable
        AND (f.decision_shares IS DISTINCT FROM l.decision_shares OR NOT l.necessary_tradeable)""",
        [str(ROOT / "features.parquet"), str(LABELS / "labels.parquet")]).fetchone()[0]
    assert mismatch == 0, "Economic labels must use exactly the same visible orders"
    frame = c.execute("""SELECT f.date,f.code,f.half,f.prior_status,f.context,f.primary,f.source_valid,
        f.decision_shares,l.label5,l.label15,l.net_return5,l.net_return15
        FROM read_parquet(?) f JOIN read_parquet(?) l USING(date,code)
        WHERE f.necessary_tradeable ORDER BY f.date,f.code""",
        [str(ROOT / "features.parquet"), str(LABELS / "labels.parquet")]).df()
    manifest = json.loads((ROOT / "manifest.json").read_text())
    assert len(frame) == manifest["necessary_stock_days"] and not frame.duplicated(["date", "code"]).any()
    frame.to_parquet(ROOT / "joined.parquet", index=False, compression="zstd")
    tables, summaries, reverse = [], [], []
    for cost in [5, 15]:
        p = frame[["date", "code", "half", "prior_status", "context", "primary", "source_valid"]].copy()
        p["cost_bps"] = cost
        p["label"] = frame[f"label{cost}"]
        p["net_return"] = frame[f"net_return{cost}"]
        p["known"] = p.net_return.notna()
        p["unknown"] = p.label.eq("unknown")
        p["no_trade"] = p.label.eq("no_trade")
        p["winner"] = p.label.eq("economic_winner")
        p["loser"] = p.label.eq("economic_loser")
        p["positive"] = p.net_return.gt(0)
        assert (p.known.astype(int) + p.unknown.astype(int) + p.no_trade.astype(int)).eq(1).all()
        assert p.winner.equals(p.net_return.ge(.01)) and p.loser.equals(p.net_return.le(-.01))
        p.to_parquet(ROOT / f"scenarios_{cost}.parquet", index=False, compression="zstd")
        baseline = aggregate(p, ["date", "half", "prior_status"])
        grouped = aggregate(p, ["date", "half", "prior_status", "context"])
        grouped = grouped.merge(baseline[["date", "prior_status", *METRICS]],
                                on=["date", "prior_status"], validate="many_to_one", suffixes=("", "_baseline"))
        for name in ["winner", "loser", "positive"]:
            grouped[name + "_lower_delta"] = grouped[name + "_lower"] - grouped[name + "_upper_baseline"]
            grouped[name + "_upper_delta"] = grouped[name + "_upper"] - grouped[name + "_lower_baseline"]
        grouped["net_mean_delta"] = grouped.net_mean - grouped.net_mean_baseline
        baseline["context"] = "all"
        for data in [baseline, grouped]:
            data["cost_bps"] = cost
            tables.append(data)
        for period in PERIODS:
            daily_period = in_period(pd.concat([baseline, grouped], ignore_index=True), period)
            rows = in_period(p, period)
            for state in states:
                for context in ["all", *CONTEXTS]:
                    days = daily_period.loc[daily_period.prior_status.eq(state) & daily_period.context.eq(context)]
                    raw = rows.loc[rows.prior_status.eq(state)]
                    if context != "all":
                        raw = raw.loc[raw.context.eq(context)]
                    # Baselines have no paired-delta fields; avoid meaningless
                    # all-null paired intervals introduced by table concatenation.
                    if context == "all":
                        days = days.drop(columns=[k for k in days if k.endswith("_delta")])
                    summaries.append(summary(days, raw, dict(cost_bps=cost, period=period,
                                                             prior_status=state, context=context)))
            print(json.dumps(dict(cost_bps=cost, period=period, summaries=len(summaries))), flush=True)
        if cost == 15:
            # Descriptive reverse portrait retains the missing-context category.
            for period in PERIODS:
                rows = in_period(p, period)
                rows = rows.loc[rows.prior_status.eq("broken_limit")]
                for label, group in rows.groupby("label"):
                    counts = group.groupby(["date", "context"]).size().unstack(fill_value=0).reindex(columns=CONTEXTS, fill_value=0)
                    fractions = counts.div(counts.sum(axis=1), axis=0)
                    reverse.append(dict(period=period, label=label, stock_days=len(group), dates=len(counts),
                                        counts=group.context.value_counts().to_dict(),
                                        mean_daily_context_fraction=fractions.mean().to_dict()))
    pd.concat(tables, ignore_index=True).to_parquet(ROOT / "groups_daily.parquet", index=False, compression="zstd")
    outputs = ["joined.parquet", "scenarios_5.parquet", "scenarios_15.parquet", "groups_daily.parquet"]
    report = dict(interpretation="Exploratory supplier-archive-date assumption; independent small cash scenarios, not a portfolio or investor identity",
                  input_report_sha256=sha(ROOT / "input_report.json"),
                  input_verification_sha256=sha(ROOT / "input_verification.json"),
                  label_report_sha256=sha(LABELS / "label_report.json"),
                  label_proof=label_proof,
                  rows=len(frame), primary=int(frame.primary.sum()), groups=summaries, reverse=reverse,
                  output_sha256={name: sha(ROOT / name) for name in outputs},
                  prices_2026_read=False, precise_first_publication_verified=False, new_strategy_selected=False)
    report["label_analysis_verification_sha256" if label_proof == "analysis" else "label_verification_sha256"] = sha(proof_file)
    save_json(ROOT / "analysis_report.json", report)
    return dict(rows=len(frame), summaries=len(summaries), primary=int(frame.primary.sum()))


if __name__ == "__main__":
    print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
