"""Report every frozen tick group and matched morning-minus-tail exits."""
import json

import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .economic_winner_analysis import number
from .reference_gain_accounting import weekly_interval
from .tick_flow_winner import FEATURES as DIRECTION
from .tick_flow_winner_analysis import evaluate, in_period, PERIODS
from .tick_absorption_winner_analysis import BANDS as JOINT_BANDS
from .tick_morning_exit import ROOT

FEATURES = [*DIRECTION, "pressure_response"]
BANDS = {feature: ["low", "middle", "high", "unknown"] for feature in DIRECTION}
BANDS["pressure_response"] = JOINT_BANDS
COUNTS = ["both_known", "morning_only", "tail_only", "both_no_trade", "neither_known"]
VALUES = ["morning_paired_mean", "tail_paired_mean", "net_difference"]


def run():
    evaluate(ROOT=ROOT, FEATURES=FEATURES, LABELS=ROOT, label_proof="label",
             bands_by_feature=BANDS, include_inverse=False)
    report = json.loads((ROOT / "analysis_report.json").read_text())
    rows = pd.read_parquet(ROOT / "evaluation_rows.parquet")
    old = pd.read_parquet(ROOT / "old_tail_labels.parquet", columns=["date", "code", "label5", "label15", "net_return5", "net_return15"])
    old = old.rename(columns={name: "tail_" + name for name in old if name not in ["date", "code"]})
    pairs = rows.merge(old, on=["date", "code"], validate="many_to_one")
    pairs["tail_label"] = np.where(pairs.cost_bps.eq(5), pairs.tail_label5, pairs.tail_label15)
    pairs["tail_exit_net"] = np.where(pairs.cost_bps.eq(5), pairs.tail_net_return5, pairs.tail_net_return15)
    uncertain = pairs.quality.eq("tick_quality_sensitivity") & ~pairs.source_valid & ~pairs.tail_label.eq("no_trade")
    pairs.loc[uncertain, "tail_label"] = "unknown"
    pairs.loc[uncertain, "tail_exit_net"] = np.nan
    morning = pairs.net_return.notna()
    tail = pairs.tail_exit_net.notna()
    pairs["both_known"] = morning & tail
    pairs["morning_only"] = morning & ~tail
    pairs["tail_only"] = ~morning & tail
    pairs["both_no_trade"] = pairs.label.eq("no_trade") & pairs.tail_label.eq("no_trade")
    pairs["neither_known"] = ~morning & ~tail & ~pairs.both_no_trade
    assert pairs[COUNTS].sum(axis=1).eq(1).all()
    pairs["morning_paired_mean"] = pairs.net_return.where(pairs.both_known)
    pairs["tail_paired_mean"] = pairs.tail_exit_net.where(pairs.both_known)
    pairs["net_difference"] = pairs.morning_paired_mean - pairs.tail_paired_mean
    pairs.to_parquet(ROOT / "exit_pairs.parquet", index=False, compression="zstd")
    tables = []
    summaries = []
    for feature in ["baseline", *FEATURES]:
        p = pairs.copy()
        p["band"] = "all" if feature == "baseline" else p[feature + "_group"]
        daily = p.groupby(["quality", "cost_bps", "date", "half", "band"], dropna=False).agg(
            n=("code", "size"), **{name: (name, "sum") for name in COUNTS},
            **{name: (name, "mean") for name in VALUES}).reset_index()
        daily["feature"] = feature
        tables.append(daily)
        for quality in ["original_labels", "tick_quality_sensitivity"]:
            for cost in [5, 15]:
                for period in PERIODS:
                    for band in ["all"] if feature == "baseline" else BANDS[feature]:
                        d = in_period(daily.loc[daily.quality.eq(quality) & daily.cost_bps.eq(cost) & daily.band.eq(band)], period)
                        r = dict(quality=quality, cost_bps=cost, period=period, feature=feature, band=band,
                                 stock_days=int(d.n.sum()), dates=len(d), paired_dates=int(d.net_difference.notna().sum()),
                                 **{name: int(d[name].sum()) for name in COUNTS})
                        for name in VALUES:
                            values = d.set_index("date")[name].sort_index()
                            r[name] = number(values.mean())
                            r[name + "_week_interval"] = weekly_interval(values)
                        summaries.append(r)
    pd.concat(tables, ignore_index=True).to_parquet(ROOT / "exit_pairs_daily.parquet", index=False, compression="zstd")
    report.update(primary="pressure_response:sell:flat", exit_window="T1 09:35–09:38", paired_exits=summaries,
                  paired_interpretation="Same buy, both exits known only; missing exits are not zero and remain counted")
    for name in ["exit_pairs", "exit_pairs_daily"]:
        report["outputs_sha256"][name] = sha(ROOT / (name + ".parquet"))
    save_json(ROOT / "analysis_report.json", report)
    return dict(groups=len(report["groups"]), paired_summaries=len(summaries), selected_new_strategy=False)


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
