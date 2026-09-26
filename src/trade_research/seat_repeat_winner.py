"""Freeze observable repeat appearances in SSE daily top-buying-branch lists."""
from collections import defaultdict
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import DAILY,save_json,sha
from .exchange_public_events import _check_saved_sse,trading_dates
from .lhb_institutional_seats import DAILY_REASONS,_seat_amount
from .lhb_institutional_short import money_cents
from .turnover_reference import CALENDAR

ROOT=Path('data/research/seat_repeat_winner')
PROTOCOL=Path('config/seat_repeat_winner_protocol.json')
ARCHIVE=Path('data/research/lhb/sse_daily')
COHORT=Path('data/research/next_day_winner/cohort.parquet')
FLOAT=Path('data/research/float_turnover_winner')
COVARIATES=['return_1449','prior_day_return','return20_prior_adjusted','float_cap_proxy','amount_1449']


def named_branch(name):return '证券' in name and ('营业部' in name or '分公司' in name)


def vector(p):
    return np.column_stack([p.return_1449/.02,p.prior_day_return/.02,p.return20_prior_adjusted/.10,
        np.log(p.float_cap_proxy)/np.log(2),np.log(p.amount_1449)/np.log(2)])


def freeze():
    if (ROOT/'input_report.json').exists():raise ValueError('Do not overwrite repeat-seat inputs')
    old=Path('data/research/lhb_institutional_short/input_report.json')
    old_report=json.loads(old.read_text())
    prior_manifest=Path('data/research/lhb_institutional_short/source_manifest.json')
    assert old_report['source_manifest_sha256']==sha(prior_manifest)
    prior_hashes=json.loads(prior_manifest.read_text())['source_sha256']
    calendar=trading_dates(CALENDAR,'2024-01-01','2025-12-31');position={d:i for i,d in enumerate(calendar)}
    decision=dict(zip(calendar[:-11],calendar[1:-10]))
    paths={str(ARCHIVE/('sse_'+date.replace('-','')+'.json')):date for date in decision}
    assert set(paths).issubset(prior_hashes) and len(paths)==old_report['source']['archived_trade_days']==474
    disclosure_hashes={path:prior_hashes[path] for path in paths}
    for path,digest in disclosure_hashes.items():assert sha(Path(path))==digest
    float_input=json.loads((FLOAT/'input_report.json').read_text())
    float_check=json.loads((FLOAT/'input_verification.json').read_text())
    assert float_check['passed'] and float_check['input_report_sha256']==sha(FLOAT/'input_report.json')
    assert float_input['features_sha256']==sha(FLOAT/'features.parquet')
    portrait=json.loads(Path('data/research/next_day_winner/analysis_report.json').read_text())
    assert sha(COHORT)==portrait['cohort_sha256']
    records=[];facts={};by_key=defaultdict(list)
    for path,date in paths.items():
        day=date.replace('-','');archived=_check_saved_sse(Path(path),day)
        for item in archived['main']:
            if item.get('secType')!='A' or not item['secCode'].startswith('60') or item['refType'] not in DAILY_REASONS:continue
            assert item['abnormalStart']==item['abnormalEnd']==day
            total=money_cents(item['secTxAmount']);assert total>0
            lists={}
            for side in ['B','S']:
                _seat_amount(item,side)
                lists[side]=[(name.strip(),money_cents(amount)) for name,amount in zip(item['branchName'+side].split(','),item['branchTxAmt'+side].split(','))]
            signature=json.dumps([total,sorted(lists['B']),sorted(lists['S'])],ensure_ascii=False,separators=(',',':'))
            row={'date':decision[date],'trade_date':date,'code':'sh.'+item['secCode'],'reason':item['refType'],
                'total_cents':total,'buy_excess_cents':max(0,sum(value for _,value in lists['B'])-total),
                'sell_excess_cents':max(0,sum(value for _,value in lists['S'])-total),
                'buy_json':json.dumps(lists['B'],ensure_ascii=False),'sell_json':json.dumps(lists['S'],ensure_ascii=False),
                'signature':signature}
            records.append(row);by_key[(date,row['code'])].append(row)
    for key,group in by_key.items():
        first=group[0];consistent=len({x['signature'] for x in group})==1
        buys=json.loads(first['buy_json']) if consistent else []
        sells=json.loads(first['sell_json']) if consistent else []
        facts[key]={'date':first['date'],'trade_date':key[0],'code':key[1],
            'reasons':','.join(sorted(x['reason'] for x in group)),'reason_rows':len(group),'current_ambiguous':not consistent,
            'total_cents':first['total_cents'] if consistent else None,'buys':buys,'sells':sells,
            'buy_excess_cents':first['buy_excess_cents'] if consistent else None,
            'sell_excess_cents':first['sell_excess_cents'] if consistent else None,
            'amount_totals_consistent':consistent and first['buy_excess_cents']==first['sell_excess_cents']==0,
            'named_buyers':{name for name,_ in buys if named_branch(name)}}
    events=[];evidence=[]
    for (date,code),fact in sorted(facts.items()):
        index=position[date];history=calendar[max(0,index-5):index]
        covered=len(history)==5 and all(day in decision for day in history)
        ambiguous=any(facts.get((day,code),{}).get('current_ambiguous',False) for day in history)
        repeat=set();sell_names={name for name,_ in fact['sells']}
        for name in sorted(fact['named_buyers']):
            seen=[day for day in history if name in facts.get((day,code),{}).get('named_buyers',set())]
            if seen:repeat.add(name)
            evidence.append({'date':fact['date'],'trade_date':date,'code':code,'name':name,
                'current_buy_cents':sum(value for branch,value in fact['buys'] if branch==name),
                'observed_prior_dates_json':json.dumps(seen),'history_covered':covered,'history_ambiguous':ambiguous,
                'repeat_observed':bool(seen),'also_current_sell':name in sell_names})
        if fact['current_ambiguous']:category='current_ambiguous'
        elif not covered or ambiguous:category='history_unknown'
        elif not fact['named_buyers']:category='no_named_buyers'
        elif not repeat:category='fresh_named_buyers'
        elif repeat&sell_names:category='repeat_and_sell'
        else:category='repeat_buy_list_only'
        total=fact['total_cents']
        amount_ok=bool(total) and fact['amount_totals_consistent']
        events.append({k:fact[k] for k in ['date','trade_date','code','reasons','reason_rows','current_ambiguous','total_cents',
            'buy_excess_cents','sell_excess_cents','amount_totals_consistent']}|{
            'seat_class':category,'history_covered':covered,'history_ambiguous':ambiguous,
            'named_buyers':len(fact['named_buyers']),'repeat_buyers':len(repeat),'repeat_also_sell':len(repeat&sell_names),
            'repeat_buy_cents':sum(value for name,value in fact['buys'] if name in repeat) if total else None,
            'repeat_fraction':sum(value for name,value in fact['buys'] if name in repeat)/total if amount_ok else None,
            'buy_fraction':sum(value for _,value in fact['buys'])/total if amount_ok else None,
            'sell_fraction':sum(value for _,value in fact['sells'])/total if amount_ok else None})
    events=pd.DataFrame(events);assert len(events)==len(by_key) and not events.duplicated(['date','code']).any()
    c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='6GB'")
    c.register('schedule',pd.DataFrame({'date':list(decision.values()),'trade_date':list(decision)}))
    c.read_parquet(str(COHORT)).create_view('cohort');c.read_parquet(str(FLOAT/'features.parquet')).create_view('float_features')
    c.read_parquet(str(DAILY/'*.parquet')).create_view('daily')
    pool=c.sql('''SELECT b.date,b.code,b.board,b.half,b.necessary_tradeable,b.return_1449,b.return20_prior_adjusted,
        b.price_1449,b.amount_1449,b.decision_shares,f.float_cap_proxy,
        CASE WHEN d.tradestatus=1 AND d.adjustflag=3 AND d.preclose>0 AND d.close>0 THEN d.close/d.preclose-1 END AS prior_day_return
      FROM cohort b JOIN schedule s USING(date) JOIN float_features f USING(date,code)
      LEFT JOIN daily d ON d.code=b.code AND d.date=s.trade_date AND d.date BETWEEN '2024-01-02' AND '2025-12-16'
      WHERE b.code LIKE 'sh.60%' ORDER BY b.date,b.code''').df();c.close()
    pool=pool.merge(events,on=['date','code'],how='left',validate='one_to_one')
    pool['seat_class']=pool.seat_class.fillna('prior_day_unlisted')
    high=pool.loc[pool.necessary_tradeable&pool.seat_class.eq('repeat_buy_list_only')]
    control=pool.loc[pool.necessary_tradeable&pool.seat_class.eq('fresh_named_buyers')]
    controls={(date,reasons):p.sort_values('code') for (date,reasons),p in control.groupby(['date','reasons'])}
    pairs=[];unmatched=[]
    for row in high.itertuples():
        part=controls.get((row.date,row.reasons))
        target=vector(pd.DataFrame([row._asdict()]))[0]
        if not np.isfinite(target).all():unmatched.append({'date':row.date,'code':row.code,'reason':'target_covariate_unknown'});continue
        if part is None:unmatched.append({'date':row.date,'code':row.code,'reason':'no_same_day_fresh_reason'});continue
        values=vector(part);valid=np.isfinite(values).all(axis=1)
        if not valid.any():unmatched.append({'date':row.date,'code':row.code,'reason':'control_covariate_unknown'});continue
        choices=part.loc[valid];distances=((values[valid]-target)**2).sum(axis=1)
        j=int(np.argmin(distances));peer=choices.iloc[j]
        pairs.append({'date':row.date,'code':row.code,'control_code':peer.code,'half':row.half,'distance':float(distances[j]),
            **{field+'_gap':float(getattr(row,field)-peer[field]) for field in COVARIATES}})
    pairs=pd.DataFrame(pairs,columns=['date','code','control_code','half','distance',*[field+'_gap' for field in COVARIATES]])
    missing=pd.DataFrame(unmatched,columns=['date','code','reason'])
    assert len(pairs)+len(missing)==len(high)
    ROOT.mkdir(exist_ok=True)
    for name,frame in [('source_rows',pd.DataFrame(records)),('events',events),('evidence',pd.DataFrame(evidence)),('pool',pool),('pairs',pairs),('unmatched',missing)]:
        frame.to_parquet(ROOT/(name+'.parquet'),index=False,compression='zstd')
    codes=set(pool.code);daily_hashes={name:digest for name,digest in float_input['daily_files_sha256'].items() if Path(name).stem.replace('_','.') in codes}
    assert len(daily_hashes)==len(codes)
    for name,digest in daily_hashes.items():assert sha(Path(name))==digest
    report={'protocol_sha256':sha(PROTOCOL),'calendar_sha256':sha(CALENDAR),'cohort_sha256':sha(COHORT),
        'prior_source_report_sha256':sha(old),'prior_source_manifest_sha256':sha(prior_manifest),'disclosure_files_sha256':disclosure_hashes,
        'float_input_report_sha256':sha(FLOAT/'input_report.json'),'float_input_verification_sha256':sha(FLOAT/'input_verification.json'),
        'daily_files_sha256':daily_hashes,'raw_reason_rows':len(records),'source_events':len(events),'visible_rows':len(pool),
        'necessary_tradeable':int(pool.necessary_tradeable.sum()),'amount_inconsistent_events':int((~events.amount_totals_consistent).sum()),
        'event_class_counts':events.seat_class.value_counts().to_dict(),
        'necessary_class_counts':pool.loc[pool.necessary_tradeable,'seat_class'].value_counts().to_dict(),
        'primary_rows':len(high),'pairs':len(pairs),'unmatched':len(missing),
        'primary_by_half':high.groupby('half').size().to_dict(),'pairs_by_half':pairs.groupby('half').size().to_dict(),
        'outputs_sha256':{name:sha(ROOT/(name+'.parquet')) for name in ['source_rows','events','evidence','pool','pairs','unmatched']},
        'outcomes_read':False,'new_2026_prices_read':False}
    save_json(ROOT/'input_report.json',report)
    return {k:report[k] for k in ['raw_reason_rows','source_events','visible_rows','necessary_tradeable','event_class_counts','necessary_class_counts','primary_rows','pairs','unmatched','primary_by_half','pairs_by_half']}


if __name__=='__main__':print(json.dumps(freeze(),ensure_ascii=False,indent=2))
