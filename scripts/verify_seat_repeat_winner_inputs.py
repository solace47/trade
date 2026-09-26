"""Rebuild the seat history with normalized raw entries and independent SQL."""
from collections import Counter,defaultdict
from decimal import Decimal
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json,sha

root=Path('data/research/seat_repeat_winner');r=json.loads((root/'input_report.json').read_text())
for key,path in [('protocol_sha256','config/seat_repeat_winner_protocol.json'),('calendar_sha256','data/baostock/market_2020_2026/metadata/calendar.parquet'),
    ('cohort_sha256','data/research/next_day_winner/cohort.parquet'),('prior_source_report_sha256','data/research/lhb_institutional_short/input_report.json'),
    ('prior_source_manifest_sha256','data/research/lhb_institutional_short/source_manifest.json'),
    ('float_input_report_sha256','data/research/float_turnover_winner/input_report.json'),('float_input_verification_sha256','data/research/float_turnover_winner/input_verification.json')]:assert r[key]==sha(Path(path))
for name,digest in r['outputs_sha256'].items():assert sha(root/(name+'.parquet'))==digest
for field in ['disclosure_files_sha256','daily_files_sha256']:
    for name,digest in r[field].items():assert sha(Path(name))==digest
cal=pd.read_parquet('data/baostock/market_2020_2026/metadata/calendar.parquet')
days=sorted(cal.loc[cal.is_trading_day.eq('1')&cal.calendar_date.between('2024-01-01','2025-12-31'),'calendar_date'])
pos={day:i for i,day in enumerate(days)};schedule={day:days[pos[day]+1] for day in days[:-11]}
assert len(r['disclosure_files_sha256'])==len(schedule)==474
source_rows=pd.read_parquet(root/'source_rows.parquet').set_index(['trade_date','code','reason'])
raw_groups=defaultdict(list);raw_count=0


def cents(value):
    parsed=Decimal(value)*100
    assert parsed.is_finite() and parsed>=0 and parsed==parsed.to_integral_value()
    return int(parsed)


for path in r['disclosure_files_sha256']:
    raw=json.loads(Path(path).read_text());date=raw['trade_date'];assert date in schedule
    for item in raw['main']:
        if item.get('secType')!='A' or not item['secCode'].startswith('60') or item['refType'] not in {'11','12','13','14'}:continue
        assert item['tradeDate']==item['abnormalStart']==item['abnormalEnd']==date.replace('-','')
        code='sh.'+item['secCode'];total=cents(item['secTxAmount']);assert total>0
        sides={}
        for side in ['B','S']:
            names=item['branchName'+side].split(',');amount=item['branchTxAmt'+side].split(',')
            assert 1<=len(names)<=5 and len(names)==len(amount)
            sides[side]=[(name.strip(),cents(value)) for name,value in zip(names,amount)]
        saved=source_rows.loc[(date,code,item['refType'])]
        assert saved.date==schedule[date] and saved.total_cents==total
        for field,side in [('buy_json','B'),('sell_json','S')]:assert [tuple(x) for x in json.loads(saved[field])]==sides[side]
        assert saved.buy_excess_cents==max(0,sum(v for _,v in sides['B'])-total)
        assert saved.sell_excess_cents==max(0,sum(v for _,v in sides['S'])-total)
        signature=json.loads(saved.signature)
        assert signature[0]==total and Counter(map(tuple,signature[1]))==Counter(sides['B']) and Counter(map(tuple,signature[2]))==Counter(sides['S'])
        raw_groups[(date,code)].append((item['refType'],total,sides))
        raw_count+=1
assert raw_count==r['raw_reason_rows']==len(source_rows)
facts=[];entries=[]
for (date,code),items in sorted(raw_groups.items()):
    identical=all(total==items[0][1] and all(Counter(sides[s])==Counter(items[0][2][s]) for s in ['B','S']) for _,total,sides in items)
    reason,total,sides=items[0]
    buy_excess=max(0,sum(v for _,v in sides['B'])-total) if identical else None
    sell_excess=max(0,sum(v for _,v in sides['S'])-total) if identical else None
    facts.append({'date':schedule[date],'trade_date':date,'trade_index':pos[date],'code':code,'reasons':','.join(sorted(x[0] for x in items)),
        'reason_rows':len(items),'current_ambiguous':not identical,'total_cents':total if identical else None,
        'buy_excess_cents':buy_excess,'sell_excess_cents':sell_excess,'amount_totals_consistent':identical and buy_excess==sell_excess==0})
    if identical:
        for side in ['B','S']:
            for rank,(name,value) in enumerate(sides[side],start=1):
                entries.append({'date':schedule[date],'trade_date':date,'trade_index':pos[date],'code':code,'side':side,'rank':rank,'name':name,'cents':value})
c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='6GB'")
c.register('facts',pd.DataFrame(facts));c.register('entries',pd.DataFrame(entries))
c.execute('''CREATE TABLE named AS SELECT date,trade_date,trade_index,code,name,sum(cents)::BIGINT AS current_buy_cents
    FROM entries WHERE side='B' AND name LIKE '%证券%' AND(name LIKE '%营业部%' OR name LIKE '%分公司%') GROUP BY ALL''')
evidence=c.sql('''SELECT a.date,a.trade_date,a.code,a.name,a.current_buy_cents,
    to_json(coalesce(list(DISTINCT b.trade_date ORDER BY b.trade_date) FILTER(WHERE b.name IS NOT NULL),[])) AS observed_prior_dates_json,
    a.trade_index>=5 AS history_covered,
    EXISTS(SELECT 1 FROM facts f WHERE f.code=a.code AND f.trade_index BETWEEN a.trade_index-5 AND a.trade_index-1 AND f.current_ambiguous) AS history_ambiguous,
    count(b.name)>0 AS repeat_observed,
    EXISTS(SELECT 1 FROM entries e WHERE e.code=a.code AND e.trade_date=a.trade_date AND e.side='S' AND e.name=a.name) AS also_current_sell
    FROM named a LEFT JOIN named b ON a.code=b.code AND a.name=b.name AND b.trade_index BETWEEN a.trade_index-5 AND a.trade_index-1
    GROUP BY a.date,a.trade_date,a.trade_index,a.code,a.name,a.current_buy_cents''').df()
stored=pd.read_parquet(root/'evidence.parquet')
for frame in [stored,evidence]:frame['observed_prior_dates_json']=frame.observed_prior_dates_json.map(lambda x:json.dumps(json.loads(x)))
keys=['date','code','name'];pd.testing.assert_frame_equal(evidence.set_index(keys).sort_index()[stored.set_index(keys).columns],stored.set_index(keys).sort_index(),check_dtype=False)
c.register('evidence',evidence)
events=c.sql('''WITH a AS(SELECT f.* EXCLUDE(trade_index),f.trade_index>=5 AS history_covered,
    EXISTS(SELECT 1 FROM facts p WHERE p.code=f.code AND p.trade_index BETWEEN f.trade_index-5 AND f.trade_index-1 AND p.current_ambiguous) AS history_ambiguous,
    (SELECT count(*) FROM evidence e WHERE e.code=f.code AND e.date=f.date) AS named_buyers,
    (SELECT count(*) FROM evidence e WHERE e.code=f.code AND e.date=f.date AND e.repeat_observed) AS repeat_buyers,
    (SELECT count(*) FROM evidence e WHERE e.code=f.code AND e.date=f.date AND e.repeat_observed AND e.also_current_sell) AS repeat_also_sell,
    CASE WHEN NOT f.current_ambiguous THEN coalesce((SELECT sum(e.current_buy_cents) FROM evidence e WHERE e.code=f.code AND e.date=f.date AND e.repeat_observed),0) END AS repeat_buy_cents,
    (SELECT sum(e.cents) FROM entries e WHERE e.code=f.code AND e.date=f.date AND e.side='B') AS buy_cents,
    (SELECT sum(e.cents) FROM entries e WHERE e.code=f.code AND e.date=f.date AND e.side='S') AS sell_cents FROM facts f)
    SELECT * EXCLUDE(buy_cents,sell_cents),CASE WHEN current_ambiguous THEN 'current_ambiguous'
      WHEN NOT history_covered OR history_ambiguous THEN 'history_unknown' WHEN named_buyers=0 THEN 'no_named_buyers'
      WHEN repeat_buyers=0 THEN 'fresh_named_buyers' WHEN repeat_also_sell>0 THEN 'repeat_and_sell' ELSE 'repeat_buy_list_only' END AS seat_class,
      CASE WHEN amount_totals_consistent THEN repeat_buy_cents*1./total_cents END AS repeat_fraction,
      CASE WHEN amount_totals_consistent THEN buy_cents*1./total_cents END AS buy_fraction,
      CASE WHEN amount_totals_consistent THEN sell_cents*1./total_cents END AS sell_fraction FROM a''').df()
stored=pd.read_parquet(root/'events.parquet');keys=['date','code']
pd.testing.assert_frame_equal(events.set_index(keys).sort_index()[stored.set_index(keys).columns],stored.set_index(keys).sort_index(),check_dtype=False,atol=1e-14,rtol=0)
assert events.seat_class.value_counts().to_dict()==r['event_class_counts']
c.register('events',events);c.register('schedule',pd.DataFrame({'date':list(schedule.values()),'prior':list(schedule)}))
c.read_parquet('data/research/next_day_winner/cohort.parquet').create_view('cohort')
c.read_parquet('data/research/float_turnover_winner/features.parquet').create_view('floats')
c.read_parquet(list(r['daily_files_sha256'])).create_view('daily')
base=c.sql('''SELECT b.date,b.code,b.board,b.half,b.necessary_tradeable,b.return_1449,b.return20_prior_adjusted,
    b.price_1449,b.amount_1449,b.decision_shares,f.float_cap_proxy,
    CASE WHEN d.tradestatus=1 AND d.adjustflag=3 AND d.close>0 AND d.preclose>0 THEN d.close/d.preclose-1 END AS prior_day_return
    FROM cohort b JOIN schedule s USING(date) JOIN floats f USING(date,code)
    LEFT JOIN daily d ON d.code=b.code AND d.date=s.prior AND d.date BETWEEN '2024-01-02' AND '2025-12-16'
    WHERE b.code LIKE 'sh.60%' ''').df()
pool=base.merge(events,on=['date','code'],how='left',validate='one_to_one');pool['seat_class']=pool.seat_class.fillna('prior_day_unlisted')
stored=pd.read_parquet(root/'pool.parquet')
rebuilt=pool.set_index(keys).sort_index();saved=stored.set_index(keys).sort_index()
assert set(rebuilt.columns)==set(saved.columns)
for name in saved.columns:
    # Preserve every missing position; Arrow nullable booleans may return None
    # while the independent SQL left join produces NaN for the same absent fact.
    pd.testing.assert_series_equal(rebuilt[name].isna(),saved[name].isna())
    observed=saved[name].notna()
    pd.testing.assert_series_equal(rebuilt.loc[observed,name],saved.loc[observed,name],check_dtype=False,atol=1e-14,rtol=0)
assert len(pool)==r['visible_rows'] and pool.necessary_tradeable.sum()==r['necessary_tradeable']
assert pool.loc[pool.necessary_tradeable,'seat_class'].value_counts().to_dict()==r['necessary_class_counts']
c.register('pool',pool)
distance='''pow((h.return_1449-l.return_1449)/.02,2)+pow((h.prior_day_return-l.prior_day_return)/.02,2)
  +pow((h.return20_prior_adjusted-l.return20_prior_adjusted)/.10,2)
  +pow(ln(h.float_cap_proxy/l.float_cap_proxy)/ln(2),2)+pow(ln(h.amount_1449/l.amount_1449)/ln(2),2)'''
c.execute(f'''CREATE TABLE edges AS SELECT h.date,h.code,l.code AS control_code,h.half,{distance} AS distance
    FROM pool h JOIN pool l ON h.date=l.date AND h.reasons=l.reasons
    WHERE h.necessary_tradeable AND h.seat_class='repeat_buy_list_only' AND l.necessary_tradeable AND l.seat_class='fresh_named_buyers'
    AND h.float_cap_proxy>0 AND l.float_cap_proxy>0 AND h.amount_1449>0 AND l.amount_1449>0''')
chosen=c.sql('''SELECT * FROM edges WHERE isfinite(distance) QUALIFY row_number() OVER(PARTITION BY date,code ORDER BY distance,control_code)=1''').df()
pairs=pd.read_parquet(root/'pairs.parquet')
pd.testing.assert_frame_equal(chosen.set_index(keys).sort_index(),pairs[chosen.columns].set_index(keys).sort_index(),check_dtype=False,atol=1e-10,rtol=0)
aligned=pairs.merge(pool[['date','code',*['return_1449','prior_day_return','return20_prior_adjusted','float_cap_proxy','amount_1449']]],on=keys,validate='one_to_one').merge(pool[['date','code','return_1449','prior_day_return','return20_prior_adjusted','float_cap_proxy','amount_1449']].rename(columns={'code':'control_code'}),on=['date','control_code'],suffixes=('_h','_l'),validate='many_to_one')
for col in ['return_1449','prior_day_return','return20_prior_adjusted','float_cap_proxy','amount_1449']:
    np.testing.assert_allclose(aligned[col+'_gap'],aligned[col+'_h']-aligned[col+'_l'],atol=1e-10,rtol=0)
unmatched=pd.read_parquet(root/'unmatched.parquet')
high=pool.loc[pool.necessary_tradeable&pool.seat_class.eq('repeat_buy_list_only')]
assert len(high)==r['primary_rows'] and len(pairs)==r['pairs'] and len(unmatched)==r['unmatched']
assert set(map(tuple,high[keys].to_numpy()))==set(map(tuple,pairs[keys].to_numpy()))|set(map(tuple,unmatched[keys].to_numpy()))
assert not set(map(tuple,pairs[keys].to_numpy()))&set(map(tuple,unmatched[keys].to_numpy()))
result={'passed':True,'input_report_sha256':sha(root/'input_report.json'),'raw_disclosure_rows':raw_count,'events':len(events),
    'named_branch_evidence':len(evidence),'visible_rows':len(pool),'all_matching_edges':c.sql('SELECT count(*) FROM edges').fetchone()[0],
    'primary_rows':len(high),'pairs':len(pairs),'unmatched_primary_retained':len(unmatched),'new_2026_prices_read':False,'outcomes_read':False}
save_json(root/'input_verification.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
