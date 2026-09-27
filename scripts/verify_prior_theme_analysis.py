"""Independent SQL accounting-label joins and every reported theme statistic."""
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha
import verify_tick_flow_analysis as numeric

ROOT = Path("data/research/prior_theme_broken")
LABELS = Path("data/research/economic_winner/period_quality")
METRICS = ["winner_lower", "winner_upper", "loser_lower", "loser_upper", "positive_lower", "positive_upper", "net_mean"]


def check(ROOT=ROOT, LABELS=LABELS):
    report = json.loads((ROOT / "analysis_report.json").read_text())
    assert report["input_report_sha256"] == sha(ROOT / "input_report.json")
    assert report["input_verification_sha256"] == sha(ROOT / "input_verification.json")
    assert report["label_report_sha256"] == sha(LABELS / "label_report.json")
    proof = report.get("label_proof", "analysis")
    assert proof in ["analysis", "label"]
    proof_file = LABELS / (proof + "_verification.json")
    proof_key = "label_analysis_verification_sha256" if proof == "analysis" else "label_verification_sha256"
    assert report[proof_key] == sha(proof_file)
    label_check = json.loads(proof_file.read_text())
    assert label_check["passed"] and label_check[proof + "_report_sha256"] == sha(LABELS / (proof + "_report.json"))
    assert json.loads((ROOT / "input_verification.json").read_text())["passed"]
    assert sha(LABELS / "labels.parquet") == json.loads((LABELS / "label_report.json").read_text())["labels_sha256"]
    for name, digest in report["output_sha256"].items():
        assert sha(ROOT / name) == digest
    c = duckdb.connect()
    c.read_parquet(str(ROOT / "features.parquet")).create_view("features")
    c.read_parquet(str(LABELS / "labels.parquet")).create_view("labels")
    assert c.sql("""SELECT count(*) FROM features f JOIN labels l USING(date,code)
        WHERE f.necessary_tradeable AND (f.decision_shares IS DISTINCT FROM l.decision_shares
        OR NOT l.necessary_tradeable)""").fetchone()[0] == 0
    c.execute("""CREATE VIEW joined AS SELECT f.date,f.code,f.half,f.prior_status,f.context,f.primary,f.source_valid,
        f.decision_shares,l.label5,l.label15,l.net_return5,l.net_return15
        FROM features f JOIN labels l USING(date,code) WHERE f.necessary_tradeable""")
    joined = c.sql("SELECT * FROM joined ORDER BY date,code").df()
    saved = pd.read_parquet(ROOT / "joined.parquet")
    pd.testing.assert_frame_equal(saved, joined, check_dtype=False, check_exact=True)
    del saved
    expected_n = json.loads((ROOT / "manifest.json").read_text())["necessary_stock_days"]
    assert len(joined) == expected_n and report["rows"] == expected_n
    assert report["primary"] == int(joined.primary.sum())
    c.execute("""CREATE VIEW scenarios AS SELECT j.date,j.code,j.half,j.prior_status,j.context,j.primary,j.source_valid,k.cost_bps,
        CASE WHEN k.cost_bps=5 THEN label5 ELSE label15 END AS label,
        CASE WHEN k.cost_bps=5 THEN net_return5 ELSE net_return15 END AS net_return
        FROM joined j CROSS JOIN (VALUES (5),(15)) k(cost_bps)""")
    c.execute("""CREATE VIEW arms AS SELECT * FROM scenarios UNION ALL
        SELECT * REPLACE('all' AS context) FROM scenarios""")
    c.execute("""CREATE VIEW counted AS SELECT date,half,prior_status,context,cost_bps,count(*) AS n,
        count(net_return) AS known,count(*) FILTER(WHERE label='unknown') AS unknown,
        count(*) FILTER(WHERE label='no_trade') AS no_trade,
        count(*) FILTER(WHERE net_return>=.01) AS winner_count,
        count(*) FILTER(WHERE net_return<=-.01) AS loser_count,
        count(*) FILTER(WHERE net_return>0) AS positive_count,avg(net_return) AS net_mean
        FROM arms GROUP BY date,half,prior_status,context,cost_bps""")
    daily = c.sql("""SELECT *,winner_count::DOUBLE/n AS winner_lower,(winner_count+unknown)::DOUBLE/n AS winner_upper,
        loser_count::DOUBLE/n AS loser_lower,(loser_count+unknown)::DOUBLE/n AS loser_upper,
        positive_count::DOUBLE/n AS positive_lower,(positive_count+unknown)::DOUBLE/n AS positive_upper
        FROM counted ORDER BY cost_bps,prior_status,context,date""").df()
    base = daily.loc[daily.context.eq("all"), ["date", "prior_status", "cost_bps", *METRICS]]
    daily = daily.merge(base, on=["date", "prior_status", "cost_bps"], suffixes=("", "_baseline"), validate="many_to_one")
    for label in ["winner", "loser", "positive"]:
        daily[label + "_lower_delta"] = daily[label + "_lower"] - daily[label + "_upper_baseline"]
        daily[label + "_upper_delta"] = daily[label + "_upper"] - daily[label + "_lower_baseline"]
    daily["net_mean_delta"] = daily.net_mean - daily.net_mean_baseline
    extra = [n for n in daily if n.endswith("_baseline") or n.endswith("_delta")]
    daily.loc[daily.context.eq("all"), extra] = np.nan
    order = ["cost_bps", "prior_status", "context", "date"]
    daily = daily.sort_values(order).reset_index(drop=True)
    actual = pd.read_parquet(ROOT / "groups_daily.parquet").sort_values(order).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[daily.columns], daily, check_dtype=False, atol=2e-12, rtol=0)
    for cost in [5, 15]:
        scenario = c.sql(f"SELECT * FROM scenarios WHERE cost_bps={cost} ORDER BY date,code").df()
        actual = pd.read_parquet(ROOT / f"scenarios_{cost}.parquet")
        pd.testing.assert_frame_equal(actual[scenario.columns], scenario, check_dtype=False, check_exact=True)
        for name, condition in {"known": scenario.net_return.notna(), "unknown": scenario.label.eq("unknown"),
                                "no_trade": scenario.label.eq("no_trade"), "winner": scenario.net_return.ge(.01),
                                "loser": scenario.net_return.le(-.01), "positive": scenario.net_return.gt(0)}.items():
            np.testing.assert_array_equal(actual[name], condition)
        assert len(scenario) == expected_n
        del actual
        raw_groups = {key: frame for key, frame in scenario.groupby(["prior_status", "context"])}
        raw_groups.update({(key, "all"): frame for key, frame in scenario.groupby("prior_status")})
        empty = scenario.iloc[:0]
        for item in [r for r in report["groups"] if r["cost_bps"] == cost]:
            identity = (cost, item["prior_status"], item["context"], item["period"])
            p = daily.loc[daily.cost_bps.eq(cost) & daily.prior_status.eq(item["prior_status"]) & daily.context.eq(item["context"])]
            p = numeric.period(p, item["period"])
            rows = numeric.period(raw_groups.get((item["prior_status"], item["context"]), empty), item["period"])
            counters = dict(stock_days=int(p.n.sum()), dates=len(p), known=int(p.known.sum()), unknown=int(p.unknown.sum()),
                            no_trade=int(p.no_trade.sum()), winner_cases=int(p.winner_count.sum()),
                            loser_cases=int(p.loser_count.sum()), positive_cases=int(p.positive_count.sum()),
                            valid_net_dates=int(p.net_mean.notna().sum()))
            for name, value in counters.items():
                numeric.eq(item[name], value, (identity, name))
            metrics = METRICS + [name + "_delta" for name in METRICS if name + "_delta" in item]
            for name in metrics:
                numeric.eq(item[name], numeric.clean(p[name].mean()), (identity, name))
                numeric.eq(item[name + "_week_interval"], numeric.interval(p, name), (identity, name, "interval"))
            values = rows.net_return.dropna().to_numpy()
            wins, losses = values[values > 0], values[values < 0]
            stats = dict(conditional_win_rate=numeric.clean(np.mean(values > 0)) if len(values) else None,
                         median=numeric.clean(np.median(values)) if len(values) else None,
                         mean_win=numeric.clean(np.mean(wins)) if len(wins) else None,
                         mean_loss=numeric.clean(np.mean(losses)) if len(losses) else None,
                         payoff_ratio=numeric.clean(np.mean(wins) / -np.mean(losses)) if len(wins) and len(losses) else None,
                         worst_five_percent_mean=numeric.clean(np.sort(values)[:max(1, math.ceil(len(values) * .05))].mean()) if len(values) else None)
            for name, value in stats.items():
                numeric.eq(item[name], value, (identity, name))
        if cost == 15:
            for item in report["reverse"]:
                rows = numeric.period(scenario, item["period"])
                rows = rows.loc[rows.prior_status.eq("broken_limit") & rows.label.eq(item["label"])]
                counts = rows.context.value_counts().to_dict()
                numeric.eq(item["stock_days"], len(rows), ("reverse", item["period"], item["label"], "rows"))
                numeric.eq(item["dates"], rows.date.nunique(), ("reverse", item["period"], item["label"], "dates"))
                assert counts == item["counts"]
                all_daily = rows.groupby("date").size()
                for context in ["strong", "neutral", "weak", "unknown"]:
                    count = rows.loc[rows.context.eq(context)].groupby("date").size().reindex(all_daily.index, fill_value=0)
                    numeric.eq(item["mean_daily_context_fraction"][context], float((count / all_daily).mean()),
                               ("reverse", item["period"], item["label"], context))
        print(json.dumps(dict(cost_bps=cost, statistics_checked=numeric.checked)), flush=True)
        del raw_groups, scenario
    result = dict(passed=True, analysis_report_sha256=sha(ROOT / "analysis_report.json"), label_joins=len(joined),
                  scenarios=expected_n * 2, group_days=len(daily), groups=len(report["groups"]),
                  statistics_checked=numeric.checked, reverse_summaries=len(report["reverse"]), prices_2026_read=False)
    save_json(ROOT / "analysis_verification.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False, indent=2))
