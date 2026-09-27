"""Same historical-theme cohort and buy orders, one predeclared morning exit."""
import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from .corporate_cash import save_json, sha
from .economic_winner_analysis import number
from .prior_theme_broken_analysis import evaluate
from .reference_gain_accounting import weekly_interval
from .tick_flow_winner_analysis import in_period, PERIODS
from .tick_morning_exit import classify, raw as extract_raw, labels as calculate_labels

ROOT = Path("data/research/prior_theme_morning")
SOURCE = Path("data/research/prior_theme_broken")
LABELS = Path("data/research/economic_winner/period_quality")
PROTOCOL = Path("config/prior_theme_morning_protocol.json")
COUNTS = ["both_known", "morning_only", "tail_only", "both_no_trade", "neither_known"]
VALUES = ["morning_paired_mean", "tail_paired_mean", "net_difference"]


def freeze():
    if (ROOT / "input_report.json").exists():
        raise ValueError("Do not replace frozen morning theme inputs")
    rule = json.loads(PROTOCOL.read_text())
    assert sha(SOURCE / "input_report.json") == rule["parent_input_sha256"]
    assert sha(SOURCE / "analysis_report.json") == rule["parent_analysis_sha256"]
    for kind in ["input", "analysis"]:
        check = json.loads((SOURCE / (kind + "_verification.json")).read_text())
        assert check["passed"] and check[kind + "_report_sha256"] == sha(SOURCE / (kind + "_report.json"))
    inputs = json.loads((SOURCE / "input_report.json").read_text())
    assert sha(SOURCE / "features.parquet") == inputs["output_sha256"]["features.parquet"]
    check = json.loads((LABELS / "analysis_verification.json").read_text())
    assert check["passed"] and check["analysis_report_sha256"] == sha(LABELS / "analysis_report.json")
    assert sha(LABELS / "labels.parquet") == json.loads((LABELS / "label_report.json").read_text())["labels_sha256"]
    features = pd.read_parquet(SOURCE / "features.parquet")
    features = features.loc[features.necessary_tradeable & features.prior_status.eq("broken_limit")].reset_index(drop=True)
    assert len(features) == 13593 and features.primary.sum() == 4866
    c = duckdb.connect()
    expected = c.execute("""SELECT * FROM read_parquet(?) WHERE necessary_tradeable AND prior_status='broken_limit'
        ORDER BY date,code""", [str(SOURCE / "features.parquet")]).df()
    pd.testing.assert_frame_equal(features, expected, check_dtype=False, check_exact=True)
    c.register("keys", features[["date", "code"]])
    old = c.execute("""SELECT l.* FROM read_parquet(?) l JOIN keys k USING(date,code)
        ORDER BY date,code""", [str(LABELS / "labels.parquet")]).df()
    assert len(old) == len(features)
    replay = classify(old)
    pd.testing.assert_frame_equal(old, replay[old.columns], check_dtype=False, atol=1e-10, rtol=0)
    pd.testing.assert_series_equal(old.decision_shares, features.decision_shares, check_dtype=False)
    assert old.next_date.gt(old.date).all() and old.next_date.le("2025-12-31").all()
    keys = old[["next_date", "code"]].rename(columns={"next_date": "date"}).sort_values(["date", "code"])
    assert not keys.duplicated().any()
    ROOT.mkdir(parents=True, exist_ok=True)
    for name, data in [("features", features), ("old_tail_labels", old), ("window_keys", keys)]:
        data.to_parquet(ROOT / (name + ".parquet"), index=False, compression="zstd")
    save_json(ROOT / "manifest.json", dict(necessary_stock_days=len(features), main_stock_days=len(features),
                                          primary=int(features.primary.sum()), parent_input_sha256=rule["parent_input_sha256"]))
    report = dict(protocol_sha256=sha(PROTOCOL), source_input_report_sha256=sha(SOURCE / "input_report.json"),
                  source_analysis_report_sha256=sha(SOURCE / "analysis_report.json"),
                  old_label_report_sha256=sha(LABELS / "label_report.json"),
                  economic_manifest_sha256=sha(Path("data/research/economic_winner/input_manifest.json")),
                  output_sha256={name: sha(ROOT / name) for name in ["features.parquet", "old_tail_labels.parquet", "window_keys.parquet"]},
                  rows=len(features), primary=int(features.primary.sum()), tail_reproduction_columns=len(old.columns),
                  new_2026_prices_read=False, new_morning_outcomes_read=False)
    save_json(ROOT / "input_report.json", report)
    save_json(ROOT / "input_verification.json", dict(passed=True, input_report_sha256=sha(ROOT / "input_report.json"),
                                                     parent_features_and_all_orders_unchanged=True, rows=len(features),
                                                     old_tail_all_columns_reproduced=len(old.columns)))
    return report


def raw():
    return extract_raw(ROOT=ROOT, PROTOCOL=PROTOCOL)


def labels():
    return calculate_labels(ROOT=ROOT)


def analyze():
    return evaluate(ROOT=ROOT, LABELS=ROOT, states=["broken_limit"], label_proof="label")


def pairs():
    if (ROOT / "paired_report.json").exists():
        raise ValueError("Do not overwrite inspected morning-tail comparisons")
    check = json.loads((ROOT / "label_verification.json").read_text())
    assert check["passed"] and check["label_report_sha256"] == sha(ROOT / "label_report.json")
    analysis = json.loads((ROOT / "analysis_report.json").read_text())
    old = pd.read_parquet(ROOT / "old_tail_labels.parquet", columns=["date", "code", "label5", "label15", "net_return5", "net_return15"])
    old = old.rename(columns={x: "tail_" + x for x in old if x not in ["date", "code"]})
    all_pairs, all_days, results = [], [], []
    for cost in [5, 15]:
        name = f"scenarios_{cost}.parquet"
        assert sha(ROOT / name) == analysis["output_sha256"][name]
        rows = pd.read_parquet(ROOT / name).merge(old, on=["date", "code"], validate="one_to_one")
        rows["tail_label"] = rows[f"tail_label{cost}"]
        rows["tail_exit_net"] = rows[f"tail_net_return{cost}"]
        morning = rows.net_return.notna()
        tail = rows.tail_exit_net.notna()
        rows["both_known"] = morning & tail
        rows["morning_only"] = morning & ~tail
        rows["tail_only"] = ~morning & tail
        rows["both_no_trade"] = rows.label.eq("no_trade") & rows.tail_label.eq("no_trade")
        rows["neither_known"] = ~morning & ~tail & ~rows.both_no_trade
        assert rows[COUNTS].sum(axis=1).eq(1).all()
        rows["morning_paired_mean"] = rows.net_return.where(rows.both_known)
        rows["tail_paired_mean"] = rows.tail_exit_net.where(rows.both_known)
        rows["net_difference"] = rows.morning_paired_mean - rows.tail_paired_mean
        all_pairs.append(rows)
        for context in ["all", "strong", "neutral", "weak", "unknown"]:
            part = rows if context == "all" else rows.loc[rows.context.eq(context)]
            day = part.groupby(["date", "half"], dropna=False).agg(n=("code", "size"),
                **{x: (x, "sum") for x in COUNTS}, **{x: (x, "mean") for x in VALUES}).reset_index()
            day["context"] = context
            day["cost_bps"] = cost
            all_days.append(day)
            for period in PERIODS:
                d = in_period(day, period)
                item = dict(cost_bps=cost, context=context, period=period, stock_days=int(d.n.sum()),
                            dates=len(d), paired_dates=int(d.net_difference.notna().sum()),
                            **{x: int(d[x].sum()) for x in COUNTS})
                for x in VALUES:
                    values = d.set_index("date")[x].sort_index()
                    item[x] = number(values.mean())
                    item[x + "_week_interval"] = weekly_interval(values)
                results.append(item)
    pd.concat(all_pairs, ignore_index=True).to_parquet(ROOT / "exit_pairs.parquet", index=False, compression="zstd")
    pd.concat(all_days, ignore_index=True).to_parquet(ROOT / "exit_pairs_daily.parquet", index=False, compression="zstd")
    report = dict(analysis_report_sha256=sha(ROOT / "analysis_report.json"),
                  old_tail_sha256=sha(ROOT / "old_tail_labels.parquet"),
                  paired_interpretation="Both exits known only; all one-sided unknowns retained, never zero-filled",
                  groups=results, output_sha256={n: sha(ROOT / n) for n in ["exit_pairs.parquet", "exit_pairs_daily.parquet"]},
                  prices_2026_read=False)
    save_json(ROOT / "paired_report.json", report)
    return dict(groups=len(results), rows=sum(len(x) for x in all_pairs))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["freeze", "raw", "labels", "analyze", "pairs"])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
