"""Independent complete-universe joins, composition and strategy accounting."""
import argparse
import json
import math
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha
import verify_tick_flow_analysis as numeric
from verify_touch_sequence_analysis import check as strategy_check

ROOT=Path('data/research/opening_cash_history')
STRATEGY=ROOT/'strategy'
CONTRACT=Path('config/opening_cash_history_analysis_contract.json')
LABELS=Path('data/research/economic_winner/period_quality')
GROUPS=['high_held','normal_held','high_lost','normal_lost','high_not_up','normal_not_up','unknown_source']
PERIODS=['2024H1','2024H2','2025H1','2025H2','2024','2025']
METRICS=['winner_lower','winner_upper','loser_lower','loser_upper','positive_lower','positive_upper','net_mean']


def strategy_inputs():
    report=json.loads((STRATEGY/'input_report.json').read_text())
    assert report['parent_input_report_sha256']==sha(ROOT/'input_report.json')
    assert report['parent_input_verification_sha256']==sha(ROOT/'input_verification.json')
    assert json.loads((ROOT/'input_verification.json').read_text())['passed']
    assert report['contract_sha256']==sha(CONTRACT)
    assert report['protocol_sha256']==sha(Path('config/opening_cash_history_protocol.json'))
    parent=json.loads((ROOT/'input_report.json').read_text())
    for dest,original in [('features.parquet','strategy_features.parquet'),('primary_pairs.parquet','primary_pairs.parquet')]:
        assert sha(STRATEGY/dest)==sha(ROOT/original)==parent['output_sha256'][original]==report['output_sha256'][dest]
    c=duckdb.connect()
    rows,primary,invalid=c.execute('SELECT count(*),sum("primary"::INT),count(*) FILTER(WHERE NOT source_valid OR event<>"primary") FROM read_parquet(?)',
        [str(STRATEGY/'features.parquet')]).fetchone()
    assert rows==report['rows'] and primary==report['primary'] and invalid==0
    result=dict(passed=True,input_report_sha256=sha(STRATEGY/'input_report.json'),copied_inputs_unchanged=True,
        rows=rows,primary=primary,paired=parent['paired'],outcomes_read=False,new_2026_prices_read=False)
    save_json(STRATEGY/'input_verification.json',result)
    return result


def group_check(item,rows,d):
    where=(item['quality'],item['cost_bps'],item['group'],item['period'])
    counts=dict(stock_days=int(d.n.sum()),dates=len(d),known=int(d.known.sum()),unknown=int(d.unknown.sum()),
        no_trade=int(d.no_trade.sum()),winner_cases=int(d.winner_count.sum()),loser_cases=int(d.loser_count.sum()),
        positive_cases=int(d.positive_count.sum()),valid_net_dates=int(d.net_mean.notna().sum()))
    for name,value in counts.items():
        numeric.eq(item[name],value,(where,name))
    for name in METRICS:
        numeric.eq(item[name],numeric.clean(d[name].mean()),(where,name))
        numeric.eq(item[name+'_week_interval'],numeric.interval(d,name),(where,name,'interval'))
    values=rows.net_return.dropna().to_numpy()
    wins,losses=values[values>0],values[values<0]
    stats=dict(conditional_win_rate=numeric.clean(np.mean(values>0)) if len(values) else None,
        median=numeric.clean(np.median(values)) if len(values) else None,
        mean_win=numeric.clean(np.mean(wins)) if len(wins) else None,
        mean_loss=numeric.clean(np.mean(losses)) if len(losses) else None,
        payoff_ratio=numeric.clean(np.mean(wins)/-np.mean(losses)) if len(wins) and len(losses) else None,
        worst_five_percent_mean=numeric.clean(np.sort(values)[:max(1,math.ceil(len(values)*.05))].mean()) if len(values) else None)
    for name,value in stats.items():
        numeric.eq(item[name],value,(where,name))


def portrait():
    report=json.loads((ROOT/'portrait_report.json').read_text())
    assert report['input_report_sha256']==sha(ROOT/'input_report.json')
    assert report['input_verification_sha256']==sha(ROOT/'input_verification.json')
    assert json.loads((ROOT/'input_verification.json').read_text())['passed']
    assert report['contract_sha256']==sha(CONTRACT)
    assert report['label_report_sha256']==sha(LABELS/'label_report.json')
    assert report['labels_sha256']==sha(LABELS/'labels.parquet')
    for name,h in report['output_sha256'].items():
        assert sha(ROOT/name)==h
    c=duckdb.connect()
    c.execute('SET threads=4')
    c.execute("SET memory_limit='5GB'")
    c.read_parquet(str(ROOT/'features.parquet')).create_view('features')
    c.read_parquet(str(LABELS/'labels.parquet')).create_view('labels')
    c.execute('''CREATE VIEW joined AS SELECT f.date,f.code,f.half,f."group",f.primary,f.source_valid,f.decision_shares,
        l.necessary_tradeable,l.decision_shares AS decision_shares_label,l.label5,l.label15,l.net_return5,l.net_return15
        FROM features f JOIN labels l USING(date,code)''')
    expected=c.sql('SELECT * FROM joined ORDER BY date,code').df()
    actual=pd.read_parquet(ROOT/'portrait_joined.parquet').sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[expected.columns],expected,check_dtype=False,check_exact=True)
    assert len(expected)==report['rows'] and expected.necessary_tradeable.all()
    assert expected.decision_shares.eq(expected.decision_shares_label).all()
    del expected,actual
    identities={(r['quality'],r['cost_bps'],r['group'],r['period']) for r in report['groups']}
    expected_identities={(q,k,g,p) for q in ['original_labels','input_quality_sensitivity'] for k in [5,15] for g in ['all',*GROUPS] for p in PERIODS}
    assert identities==expected_identities and len(identities)==len(report['groups'])==192
    assert len(report['reverse'])==84
    all_daily=pd.read_parquet(ROOT/'portrait_daily.parquet')
    total_scenarios=group_days=0
    for quality in ['original_labels','input_quality_sensitivity']:
        for cost in [5,15]:
            c.execute(f'''CREATE OR REPLACE VIEW original AS SELECT date,code,half,"group","primary",source_valid,
                label{cost} AS original_label,net_return{cost} AS original_net_return,
                '{quality}' AS quality,{cost} AS cost_bps FROM joined''')
            c.execute('''CREATE OR REPLACE VIEW masked AS SELECT *,
                CASE WHEN quality='input_quality_sensitivity' AND NOT source_valid AND original_label<>'no_trade'
                    THEN 'unknown' ELSE original_label END AS label,
                CASE WHEN quality='input_quality_sensitivity' AND NOT source_valid THEN NULL ELSE original_net_return END AS net_return
                FROM original''')
            c.execute('''CREATE OR REPLACE VIEW scenarios AS SELECT *,net_return IS NOT NULL AS known,label='unknown' AS unknown,
                label='no_trade' AS no_trade,coalesce(net_return>=.01,false) AS winner,
                coalesce(net_return<=-.01,false) AS loser,coalesce(net_return>0,false) AS positive FROM masked''')
            expected=c.sql('SELECT * FROM scenarios ORDER BY date,code').df()
            actual=pd.read_parquet(ROOT/f'portrait_{quality}_{cost}.parquet').sort_values(['date','code']).reset_index(drop=True)
            pd.testing.assert_frame_equal(actual[expected.columns],expected,check_dtype=False,check_exact=True)
            assert len(expected)==report['rows']
            total_scenarios+=len(expected)
            del actual
            c.execute('''CREATE OR REPLACE VIEW arms AS SELECT * FROM scenarios UNION ALL
                SELECT * REPLACE('all' AS "group") FROM scenarios''')
            c.execute('''CREATE OR REPLACE VIEW counts AS SELECT date,half,"group",quality,cost_bps,count(*) AS n,
                count(net_return) AS known,count(*) FILTER(WHERE label='unknown') AS unknown,
                count(*) FILTER(WHERE label='no_trade') AS no_trade,count(*) FILTER(WHERE net_return>=.01) AS winner_count,
                count(*) FILTER(WHERE net_return<=-.01) AS loser_count,count(*) FILTER(WHERE net_return>0) AS positive_count,
                avg(net_return) AS net_mean FROM arms GROUP BY date,half,"group",quality,cost_bps''')
            daily=c.sql('''SELECT *,winner_count::DOUBLE/n AS winner_lower,(winner_count+unknown)::DOUBLE/n AS winner_upper,
                loser_count::DOUBLE/n AS loser_lower,(loser_count+unknown)::DOUBLE/n AS loser_upper,
                positive_count::DOUBLE/n AS positive_lower,(positive_count+unknown)::DOUBLE/n AS positive_upper
                FROM counts ORDER BY "group",date''').df()
            actual=all_daily.loc[all_daily.quality.eq(quality)&all_daily.cost_bps.eq(cost)].sort_values(['group','date']).reset_index(drop=True)
            pd.testing.assert_frame_equal(actual[daily.columns],daily,check_dtype=False,atol=2e-12,rtol=0)
            group_days+=len(daily)
            for item in report['groups']:
                if item['quality']!=quality or item['cost_bps']!=cost:
                    continue
                rows=expected if item['group']=='all' else expected.loc[expected.group.eq(item['group'])]
                rows=numeric.period(rows,item['period'])
                d=numeric.period(daily.loc[daily.group.eq(item['group'])],item['period'])
                group_check(item,rows,d)
            if cost==15:
                for item in report['reverse']:
                    if item['quality']!=quality:
                        continue
                    rows=numeric.period(expected,item['period'])
                    if item['label']=='big_winner':
                        rows=rows.loc[rows.net_return.ge(.03)]
                    elif item['label']=='big_loser':
                        rows=rows.loc[rows.net_return.le(-.03)]
                    else:
                        rows=rows.loc[rows.label.eq(item['label'])]
                    key=(quality,item['period'],item['label'])
                    numeric.eq(item['rows'],len(rows),(*key,'rows'))
                    numeric.eq(item['dates'],rows.date.nunique(),(*key,'dates'))
                    assert item['counts']==rows.group.value_counts().to_dict()
                    denominators=rows.groupby('date').size()
                    for group in GROUPS:
                        counts=rows.loc[rows.group.eq(group)].groupby('date').size().reindex(denominators.index,fill_value=0)
                        numeric.eq(item['mean_daily_group_fraction'][group],numeric.clean((counts/denominators).mean()),(*key,group))
            print(json.dumps(dict(quality=quality,cost_bps=cost,verified_rows=total_scenarios)),flush=True)
    result=dict(passed=True,portrait_report_sha256=sha(ROOT/'portrait_report.json'),scenarios=total_scenarios,
        group_days=group_days,statistics_checked=numeric.checked,groups=len(report['groups']),reverse=len(report['reverse']),
        feature_missing_not_conflated_with_profit_missing=True,new_2026_prices_read=False)
    save_json(ROOT/'portrait_verification.json',result)
    return result


def analyze():
    original=json.loads((ROOT/'portrait_report.json').read_text())
    assert json.loads((ROOT/'portrait_verification.json').read_text())['passed']
    strategy=json.loads((STRATEGY/'analysis_report.json').read_text())
    for item in strategy['groups']:
        if item['exit']!='tail':
            continue
        baseline=next(r for r in original['groups'] if r['quality']=='input_quality_sensitivity' and r['group']=='high_held'
            and r['cost_bps']==item['cost_bps'] and r['period']==item['period'])
        for name,value in baseline.items():
            if name not in ['quality','group']:
                numeric.eq(item[name],value,('strategy_portrait',item['period'],item['cost_bps'],name))
    result=strategy_check(ROOT=STRATEGY,GROUPS=['high_held'],event_column='event')
    result.update(portrait_verification_sha256=sha(ROOT/'portrait_verification.json'),full_portrait_primary_tail_identical=True)
    save_json(STRATEGY/'analysis_verification.json',result)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['strategy_inputs','portrait','analyze'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
