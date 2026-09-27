"""Verify all new morning summaries and same-order exit comparisons."""
import json
from pathlib import Path

import duckdb
import pandas as pd

import verify_tick_flow_analysis as common
from trade_research.corporate_cash import save_json, sha

ROOT = Path("data/research/tick_morning_exit")
FEATURES = [*common.FEATURES, "pressure_response"]
COUNTS = ["both_known", "morning_only", "tail_only", "both_no_trade", "neither_known"]
VALUES = ["morning_paired_mean", "tail_paired_mean", "net_difference"]


def main():
    result = common.main(ROOT=ROOT, FEATURES=FEATURES, write_report=False, labels=str(ROOT / "labels.parquet"))
    report = json.loads((ROOT / "analysis_report.json").read_text())
    c = duckdb.connect()
    c.read_parquet(str(ROOT / "evaluation_rows.parquet")).create_view("morning")
    c.read_parquet(str(ROOT / "old_tail_labels.parquet")).create_view("tail")
    c.execute("""create view pair0 as select m.*,
        case when m.cost_bps=5 then t.label5 else t.label15 end as old_label,
        case when m.cost_bps=5 then t.net_return5 else t.net_return15 end as old_net
        from morning m join tail t using(date,code)""")
    c.execute("""create view pair1 as select *,
        case when quality='tick_quality_sensitivity' and not source_valid and old_label!='no_trade' then 'unknown' else old_label end as tail_label,
        case when quality='tick_quality_sensitivity' and not source_valid and old_label!='no_trade' then NULL else old_net end as tail_exit_net
        from pair0""")
    c.execute("""create view pair2 as select *,net_return is not null and tail_exit_net is not null as both_known,
        net_return is not null and tail_exit_net is null as morning_only,
        net_return is null and tail_exit_net is not null as tail_only,
        label='no_trade' and tail_label='no_trade' as both_no_trade,
        net_return is null and tail_exit_net is null and not(label='no_trade' and tail_label='no_trade') as neither_known
        from pair1""")
    c.execute("""create view pairs as select *,case when both_known then net_return else NULL end as morning_paired_mean,
        case when both_known then tail_exit_net else NULL end as tail_paired_mean,
        case when both_known then net_return-tail_exit_net else NULL end as net_difference from pair2""")
    order = ["quality", "cost_bps", "date", "code"]
    want = c.sql("select * from pairs order by " + ",".join(order)).df()
    actual = pd.read_parquet(ROOT / "exit_pairs.parquet").sort_values(order).reset_index(drop=True)
    cols = [*order, "tail_label", "tail_exit_net", *COUNTS, *VALUES]
    pd.testing.assert_frame_equal(actual[cols], want[cols], check_dtype=False, atol=2e-12, rtol=0)
    arms = ["select *,'baseline' as feature,'all' as band from pairs"]
    arms.extend(f"select *,'{f}' as feature,{f}_group as band from pairs" for f in FEATURES)
    c.execute("create view arms as " + " union all ".join(arms))
    expressions = [f"sum({n}::int) as {n}" for n in COUNTS] + [f"avg({n}) as {n}" for n in VALUES]
    query = "select quality,cost_bps,date,half,feature,band,count(*) as n," + ",".join(expressions)
    query += " from arms group by quality,cost_bps,date,half,feature,band order by quality,cost_bps,feature,band,date"
    daily = c.sql(query).df()
    actual_daily = pd.read_parquet(ROOT / "exit_pairs_daily.parquet").sort_values(["quality", "cost_bps", "feature", "band", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_daily[daily.columns], daily, check_dtype=False, atol=2e-12, rtol=0)
    for r in report["paired_exits"]:
        d = daily.loc[daily.quality.eq(r["quality"]) & daily.cost_bps.eq(r["cost_bps"])
                      & daily.feature.eq(r["feature"]) & daily.band.eq(r["band"])]
        d = common.period(d, r["period"])
        expected = dict(stock_days=int(d.n.sum()), dates=len(d), paired_dates=int(d.net_difference.notna().sum()),
                        **{n: int(d[n].sum()) for n in COUNTS})
        for name, value in expected.items():
            common.eq(r[name], value, (r["period"], r["feature"], r["band"], name))
        for name in VALUES:
            common.eq(r[name], common.clean(d[name].mean()), (r["period"], name))
            common.eq(r[name + "_week_interval"], common.interval(d, name), (r["period"], name, "interval"))
    result.update(analysis_report_sha256=sha(ROOT / "analysis_report.json"), paired_rows=len(want),
                  paired_daily_rows=len(daily), paired_summaries=len(report["paired_exits"]), statistics_checked=common.checked)
    save_json(ROOT / "analysis_verification.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, indent=2))
