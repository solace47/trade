"""Check all morning groups and independently paired identical-buy exits."""
import json
from pathlib import Path

import duckdb
import pandas as pd

from trade_research.corporate_cash import save_json, sha
from verify_prior_theme_analysis import check as verify_groups
import verify_tick_flow_analysis as numeric

ROOT = Path("data/research/prior_theme_morning")
COUNTS = ["both_known", "morning_only", "tail_only", "both_no_trade", "neither_known"]
VALUES = ["morning_paired_mean", "tail_paired_mean", "net_difference"]


def check():
    base = verify_groups(ROOT=ROOT, LABELS=ROOT)
    report = json.loads((ROOT / "paired_report.json").read_text())
    assert report["analysis_report_sha256"] == sha(ROOT / "analysis_report.json")
    assert report["old_tail_sha256"] == sha(ROOT / "old_tail_labels.parquet")
    for name, digest in report["output_sha256"].items():
        assert sha(ROOT / name) == digest
    c = duckdb.connect()
    c.read_parquet([str(ROOT / "scenarios_5.parquet"), str(ROOT / "scenarios_15.parquet")]).create_view("scenarios")
    c.read_parquet(str(ROOT / "old_tail_labels.parquet")).create_view("old")
    c.execute("""CREATE VIEW combined AS SELECT s.date,s.code,s.half,s.context,s.cost_bps,s.net_return,s.label,
        CASE WHEN s.cost_bps=5 THEN o.net_return5 ELSE o.net_return15 END AS tail_exit_net,
        CASE WHEN s.cost_bps=5 THEN o.label5 ELSE o.label15 END AS tail_label
        FROM scenarios s JOIN old o USING(date,code)""")
    c.execute("""CREATE VIEW status AS SELECT *,net_return IS NOT NULL AND tail_exit_net IS NOT NULL AS both_known,
        net_return IS NOT NULL AND tail_exit_net IS NULL AS morning_only,
        net_return IS NULL AND tail_exit_net IS NOT NULL AS tail_only,
        label='no_trade' AND tail_label='no_trade' AS both_no_trade,
        net_return IS NULL AND tail_exit_net IS NULL AND NOT(label='no_trade' AND tail_label='no_trade') AS neither_known
        FROM combined""")
    c.execute("""CREATE VIEW pairs AS SELECT *,CASE WHEN both_known THEN net_return END AS morning_paired_mean,
        CASE WHEN both_known THEN tail_exit_net END AS tail_paired_mean,
        CASE WHEN both_known THEN net_return-tail_exit_net END AS net_difference FROM status""")
    rows = c.sql("SELECT * FROM pairs ORDER BY cost_bps,date,code").df()
    actual = pd.read_parquet(ROOT / "exit_pairs.parquet").sort_values(["cost_bps", "date", "code"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[rows.columns], rows, check_dtype=False, atol=2e-12, rtol=0)
    assert rows[COUNTS].sum(axis=1).eq(1).all()
    c.execute("CREATE VIEW arms AS SELECT * FROM pairs UNION ALL SELECT * REPLACE('all' AS context) FROM pairs")
    daily = c.sql("""SELECT cost_bps,context,date,half,count(*) AS n,
        sum(both_known::INT) AS both_known,sum(morning_only::INT) AS morning_only,
        sum(tail_only::INT) AS tail_only,sum(both_no_trade::INT) AS both_no_trade,sum(neither_known::INT) AS neither_known,
        avg(morning_paired_mean) AS morning_paired_mean,avg(tail_paired_mean) AS tail_paired_mean,
        avg(net_difference) AS net_difference FROM arms GROUP BY cost_bps,context,date,half
        ORDER BY cost_bps,context,date""").df()
    actual = pd.read_parquet(ROOT / "exit_pairs_daily.parquet").sort_values(["cost_bps", "context", "date"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[daily.columns], daily, check_dtype=False, atol=2e-12, rtol=0)
    before = numeric.checked
    for item in report["groups"]:
        identity = (item["cost_bps"], item["context"], item["period"])
        d = daily.loc[daily.cost_bps.eq(item["cost_bps"]) & daily.context.eq(item["context"])]
        d = numeric.period(d, item["period"])
        counts = dict(stock_days=int(d.n.sum()), dates=len(d), paired_dates=int(d.net_difference.notna().sum()),
                      **{x: int(d[x].sum()) for x in COUNTS})
        for key, value in counts.items():
            numeric.eq(item[key], value, (identity, key))
        for key in VALUES:
            numeric.eq(item[key], numeric.clean(d[key].mean()), (identity, key))
            numeric.eq(item[key + "_week_interval"], numeric.interval(d, key), (identity, key, "interval"))
    result = dict(passed=True, paired_report_sha256=sha(ROOT / "paired_report.json"),
                  analysis_report_sha256=sha(ROOT / "analysis_report.json"),
                  analysis_verification_sha256=sha(ROOT / "analysis_verification.json"),
                  rows=len(rows), group_days=len(daily), paired_summaries=len(report["groups"]),
                  paired_statistics_checked=numeric.checked - before, total_statistics_checked=numeric.checked,
                  prices_2026_read=False)
    save_json(ROOT / "paired_verification.json", result)
    return dict(base=base, paired=result)


if __name__ == "__main__":
    print(json.dumps(check(), ensure_ascii=False, indent=2))
