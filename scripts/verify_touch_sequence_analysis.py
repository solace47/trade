"""Independent SQL scenarios, complete pair denominators and report statistics."""
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha
import verify_tick_flow_analysis as numeric

ROOT = Path('data/research/touch_sequence')
METRICS = ['winner_lower','winner_upper','loser_lower','loser_upper','positive_lower','positive_upper','net_mean']
PAIR_COUNTS = ['no_match','both_known','selected_only','control_only','both_no_trade','neither_known']
PAIR_VALUES = ['selected_paired_mean','control_paired_mean','net_difference']
GROUPS = ['early_open_rising','early_open_not_rising','late_first_touch_open','tail_retouch_open','unknown_source']


def check():
    report = json.loads((ROOT/'analysis_report.json').read_text())
    assert report['input_report_sha256'] == sha(ROOT/'input_report.json')
    assert report['input_verification_sha256'] == sha(ROOT/'input_verification.json')
    assert json.loads((ROOT/'input_verification.json').read_text())['passed']
    assert report['morning_label_verification_sha256'] == sha(ROOT/'morning/label_verification.json')
    assert json.loads((ROOT/'morning/label_verification.json').read_text())['passed']
    assert report['morning_labels_sha256'] == sha(ROOT/'morning/labels.parquet')
    assert report['tail_labels_sha256'] == sha(ROOT/'morning/old_tail_labels.parquet')
    for name,h in report['output_sha256'].items():
        assert sha(ROOT/name) == h
    c = duckdb.connect()
    c.read_parquet(str(ROOT/'features.parquet')).create_view('features')
    c.read_parquet(str(ROOT/'primary_pairs.parquet')).create_view('fixed_pairs')
    c.read_parquet(str(ROOT/'morning/labels.parquet')).create_view('morning')
    c.read_parquet(str(ROOT/'morning/old_tail_labels.parquet')).create_view('tail')
    c.execute("""CREATE VIEW exits AS SELECT 'morning' AS exit,* EXCLUDE(intended_buy_cash5,intended_buy_cash15) FROM morning
        UNION ALL BY NAME SELECT 'tail' AS exit,* FROM tail""")
    assert c.sql("""SELECT count(*) FROM features f JOIN exits e USING(date,code)
        WHERE f.decision_shares<>e.decision_shares OR NOT e.necessary_tradeable""").fetchone()[0] == 0
    c.execute("""CREATE VIEW originals AS SELECT f.date,f.code,f.half,f."group",f.primary,f.touched,f.source_valid,e.exit,k.cost_bps,
        CASE WHEN k.cost_bps=5 THEN label5 ELSE label15 END AS original_label,
        CASE WHEN k.cost_bps=5 THEN net_return5 ELSE net_return15 END AS original_net_return
        FROM features f JOIN exits e USING(date,code) CROSS JOIN (VALUES(5),(15)) k(cost_bps)""")
    c.execute("""CREATE VIEW masked AS SELECT *,
        CASE WHEN source_valid OR original_label='no_trade' THEN original_label ELSE 'unknown' END AS label,
        CASE WHEN source_valid THEN original_net_return END AS net_return FROM originals""")
    c.execute("""CREATE VIEW scenarios AS SELECT *,net_return IS NOT NULL AS known,label='unknown' AS unknown,
        label='no_trade' AS no_trade,coalesce(net_return>=.01,false) AS winner,
        coalesce(net_return<=-.01,false) AS loser,coalesce(net_return>0,false) AS positive FROM masked""")
    order = ['exit','cost_bps','date','code']
    expected = c.sql('SELECT * FROM scenarios ORDER BY exit,cost_bps,date,code').df()
    actual = pd.read_parquet(ROOT/'scenarios.parquet').sort_values(order).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[expected.columns],expected,check_dtype=False,check_exact=True)
    assert len(expected) == report['rows']*4
    c.execute("""CREATE VIEW arms AS SELECT * FROM scenarios WHERE touched UNION ALL
        SELECT * REPLACE('all' AS "group") FROM scenarios WHERE touched""")
    c.execute("""CREATE VIEW counts AS SELECT date,half,"group",exit,cost_bps,count(*) AS n,
        count(net_return) AS known,count(*) FILTER(WHERE label='unknown') AS unknown,
        count(*) FILTER(WHERE label='no_trade') AS no_trade,count(*) FILTER(WHERE net_return>=.01) AS winner_count,
        count(*) FILTER(WHERE net_return<=-.01) AS loser_count,count(*) FILTER(WHERE net_return>0) AS positive_count,
        avg(net_return) AS net_mean FROM arms GROUP BY date,half,"group",exit,cost_bps""")
    daily = c.sql("""SELECT *,winner_count::DOUBLE/n AS winner_lower,(winner_count+unknown)::DOUBLE/n AS winner_upper,
        loser_count::DOUBLE/n AS loser_lower,(loser_count+unknown)::DOUBLE/n AS loser_upper,
        positive_count::DOUBLE/n AS positive_lower,(positive_count+unknown)::DOUBLE/n AS positive_upper FROM counts
        ORDER BY exit,cost_bps,"group",date""").df()
    actual = pd.read_parquet(ROOT/'groups_daily.parquet').sort_values(['exit','cost_bps','group','date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[daily.columns],daily,check_dtype=False,atol=2e-12,rtol=0)
    for item in report['groups']:
        identity = (item['exit'],item['cost_bps'],item['group'],item['period'])
        rows = expected.loc[expected.exit.eq(item['exit']) & expected.cost_bps.eq(item['cost_bps']) & expected.touched]
        if item['group'] != 'all':
            rows = rows.loc[rows.group.eq(item['group'])]
        rows = numeric.period(rows,item['period'])
        d = numeric.period(daily.loc[daily.exit.eq(item['exit']) & daily.cost_bps.eq(item['cost_bps'])
            & daily.group.eq(item['group'])],item['period'])
        counters = dict(stock_days=int(d.n.sum()),dates=len(d),known=int(d.known.sum()),unknown=int(d.unknown.sum()),
            no_trade=int(d.no_trade.sum()),winner_cases=int(d.winner_count.sum()),loser_cases=int(d.loser_count.sum()),
            positive_cases=int(d.positive_count.sum()),valid_net_dates=int(d.net_mean.notna().sum()))
        for name,value in counters.items():
            numeric.eq(item[name],value,(identity,name))
        for name in METRICS:
            numeric.eq(item[name],numeric.clean(d[name].mean()),(identity,name))
            numeric.eq(item[name+'_week_interval'],numeric.interval(d,name),(identity,name,'interval'))
        values = rows.net_return.dropna().to_numpy()
        wins,losses = values[values>0],values[values<0]
        stats = dict(conditional_win_rate=numeric.clean(np.mean(values>0)) if len(values) else None,
            median=numeric.clean(np.median(values)) if len(values) else None,
            mean_win=numeric.clean(np.mean(wins)) if len(wins) else None,
            mean_loss=numeric.clean(np.mean(losses)) if len(losses) else None,
            payoff_ratio=numeric.clean(np.mean(wins)/-np.mean(losses)) if len(wins) and len(losses) else None,
            worst_five_percent_mean=numeric.clean(np.sort(values)[:max(1,math.ceil(len(values)*.05))].mean()) if len(values) else None)
        for name,value in stats.items():
            numeric.eq(item[name],value,(identity,name))
    c.execute("""CREATE VIEW joined_pairs AS SELECT p.* EXCLUDE(half),
        substr(p.date,1,4)||CASE WHEN month(p.date::DATE)<=6 THEN 'H1' ELSE 'H2' END AS half,
        s.exit,s.cost_bps,s.label,s.net_return,
        b.label AS control_label,b.net_return AS control_return,p.control_code IS NULL AS no_match,
        s.net_return IS NOT NULL AND b.net_return IS NOT NULL AS both_known
        FROM fixed_pairs p JOIN scenarios s ON p.date=s.date AND p.code=s.code
        LEFT JOIN scenarios b ON p.date=b.date AND p.control_code=b.code AND s.exit=b.exit AND s.cost_bps=b.cost_bps""")
    c.execute("""CREATE VIEW pairs AS SELECT *,
        NOT no_match AND net_return IS NOT NULL AND control_return IS NULL AS selected_only,
        NOT no_match AND net_return IS NULL AND control_return IS NOT NULL AS control_only,
        coalesce(NOT no_match AND label='no_trade' AND control_label='no_trade',false) AS both_no_trade,
        NOT no_match AND net_return IS NULL AND control_return IS NULL AND NOT both_no_trade AS neither_known,
        CASE WHEN both_known THEN net_return END AS selected_paired_mean,
        CASE WHEN both_known THEN control_return END AS control_paired_mean,
        CASE WHEN both_known THEN net_return-control_return END AS net_difference FROM joined_pairs""")
    q = c.sql('SELECT * FROM pairs ORDER BY exit,cost_bps,date,code').df()
    actual = pd.read_parquet(ROOT/'pairs.parquet').sort_values(order).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[q.columns],q,check_dtype=False,atol=2e-12,rtol=0)
    sumcols = ','.join('sum('+n+'::INTEGER) AS '+n for n in PAIR_COUNTS)
    avgcols = ','.join('avg('+n+') AS '+n for n in PAIR_VALUES)
    qd = c.sql(f'SELECT date,half,exit,cost_bps,count(*) AS n,{sumcols},{avgcols} FROM pairs GROUP BY date,half,exit,cost_bps ORDER BY exit,cost_bps,date').df()
    actual = pd.read_parquet(ROOT/'pairs_daily.parquet').sort_values(['exit','cost_bps','date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[qd.columns],qd,check_dtype=False,atol=2e-12,rtol=0)
    for item in report['pair_groups']:
        d = numeric.period(qd.loc[qd.exit.eq(item['exit']) & qd.cost_bps.eq(item['cost_bps'])],item['period'])
        counters = dict(stock_days=int(d.n.sum()),dates=len(d),paired_dates=int(d.net_difference.notna().sum()),
            **{n:int(d[n].sum()) for n in PAIR_COUNTS})
        for name,value in counters.items():
            numeric.eq(item[name],value,('pair',item['exit'],item['cost_bps'],item['period'],name))
        for name in PAIR_VALUES:
            numeric.eq(item[name],numeric.clean(d[name].mean()),('pair',item['period'],name))
            numeric.eq(item[name+'_week_interval'],numeric.interval(d,name),('pair',item['period'],name,'interval'))
    for item in report['reverse']:
        rows = numeric.period(expected.loc[expected.touched & expected.exit.eq(item['exit']) & expected.cost_bps.eq(15)],item['period'])
        if item['label']=='big_winner':
            rows = rows.loc[rows.net_return.ge(.03)]
        elif item['label']=='big_loser':
            rows = rows.loc[rows.net_return.le(-.03)]
        else:
            rows = rows.loc[rows.label.eq(item['label'])]
        numeric.eq(item['rows'],len(rows),('reverse',item['exit'],item['period'],item['label'],'rows'))
        numeric.eq(item['dates'],rows.date.nunique(),('reverse',item['exit'],item['period'],item['label'],'dates'))
        assert item['counts'] == rows.group.value_counts().to_dict()
        denominators = rows.groupby('date').size()
        for group in GROUPS:
            counts = rows.loc[rows.group.eq(group)].groupby('date').size().reindex(denominators.index,fill_value=0)
            numeric.eq(item['mean_daily_group_fraction'][group],numeric.clean((counts/denominators).mean()),
                ('reverse',item['exit'],item['period'],item['label'],group))
    result = dict(passed=True,analysis_report_sha256=sha(ROOT/'analysis_report.json'),scenarios=len(expected),
        group_days=len(daily),pair_scenarios=len(q),statistics_checked=numeric.checked,
        groups=len(report['groups']),pair_groups=len(report['pair_groups']),reverse=len(report['reverse']),
        new_2026_prices_read=False)
    save_json(ROOT/'analysis_verification.json',result)
    return result


if __name__=='__main__':
    print(json.dumps(check(),ensure_ascii=False,indent=2))
