"""Adapt an externally specified complete tail screen to historical 14:49 inputs."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import DAILY,MINUTES,save_json,sha
from .turnover_reference import CALENDAR

ROOT=Path('data/research/external_tail_combo')
PROTOCOL=Path('config/external_tail_combo_protocol.json')
PORTRAIT=Path('data/research/next_day_winner')
FLOAT=Path('data/research/float_turnover_winner')
EXTERNAL=Path('data/research/external_tail_review/manifest.json')
GATES=['gain_gate','volume_gate','turnover_gate','size_gate','market_gate']


def labels():
    expected=[x.strftime('%H%M') for a,b in [('09:30','11:30'),('13:01','14:49')]
        for x in pd.date_range('2024-01-02 '+a,'2024-01-02 '+b,freq='min')]
    observe=[x.strftime('%H%M') for a,b in [('09:35','11:30'),('13:05','14:45')]
        for x in pd.date_range('2024-01-02 '+a,'2024-01-02 '+b,freq='5min')]
    assert len(expected)==230 and len(observe)==45
    return expected,observe


def all_gates(p):
    # A known failure makes the conjunction false even if another term is unknown.
    result=pd.Series(pd.NA,index=p.index,dtype='boolean')
    result.loc[p.eq(False).any(axis=1)]=False
    result.loc[p.eq(True).all(axis=1)&p.notna().all(axis=1)]=True
    return result


def freeze():
    if (ROOT/'input_report.json').exists():raise ValueError('Do not replace the external complete screen')
    ROOT.mkdir(exist_ok=True)
    prior=json.loads((PORTRAIT/'analysis_report.json').read_text())
    assert prior['cohort_sha256']==sha(PORTRAIT/'cohort.parquet')
    floating=json.loads((FLOAT/'input_report.json').read_text());check=json.loads((FLOAT/'input_verification.json').read_text())
    assert check['passed'] and check['input_report_sha256']==sha(FLOAT/'input_report.json')
    assert floating['features_sha256']==sha(FLOAT/'features.parquet')
    external=json.loads(EXTERNAL.read_text())
    for item in external['sources']:
        assert sha(EXTERNAL.parent/item['repo'].split('/')[1]/item['file'])==item['sha256']
    columns=['date','code','half','board','necessary_tradeable','price_1449','preclose','volume_1449','amount_1449',
        'return_1449','market_return','decision_shares','return20_prior_adjusted']
    pool=pd.read_parquet(PORTRAIT/'cohort.parquet',columns=columns)
    pool=pool.loc[pool.code.str.startswith(('sz.002','sz.003'))].copy()
    floats=pd.read_parquet(FLOAT/'features.parquet',columns=['date','code','float_cap_proxy','turnover_1449_proxy'])
    pool=pool.merge(floats,on=['date','code'],how='left',validate='one_to_one')
    dates=pd.read_parquet(CALENDAR);days=sorted(dates.loc[dates.is_trading_day.eq('1'),'calendar_date']);positions={d:i for i,d in enumerate(days)}
    schedule=pd.DataFrame([{'date':d,'prior20_start':days[positions[d]-20],'prior5_start':days[positions[d]-5],
        'prior_date':days[positions[d]-1]} for d in sorted(pool.date.unique())])
    daily_hashes={str(DAILY/(code.replace('.','_')+'.parquet')):floating['daily_files_sha256'][str(DAILY/(code.replace('.','_')+'.parquet'))] for code in pool.code.unique()}
    for name,digest in daily_hashes.items():assert sha(Path(name))==digest
    c=duckdb.connect();c.execute('SET threads=4');c.execute("SET memory_limit='6GB'")
    c.register('keys',pool[['date','code']]);c.register('schedule',schedule)
    c.read_parquet(list(daily_hashes)).create_view('daily')
    history=c.sql('''SELECT k.date,k.code,s.prior20_start,s.prior5_start,s.prior_date,
        count(d.date) AS history_rows,count(DISTINCT d.date) AS history_dates,
        count(*) FILTER(WHERE d.adjustflag=3 AND(d.tradestatus=0 OR(d.tradestatus=1 AND d.close>0 AND d.preclose>0 AND isfinite(d.pctChg)))) AS valid_history_rows,
        count(*) FILTER(WHERE d.tradestatus=1 AND d.pctChg>=9.9) AS big_up_days,
        count(*) FILTER(WHERE d.date>=s.prior5_start AND d.adjustflag=3 AND d.volume>=0 AND isfinite(d.volume)) AS volume5_rows,
        sum(d.volume) FILTER(WHERE d.date>=s.prior5_start) AS volume5_sum
      FROM keys k JOIN schedule s USING(date) LEFT JOIN daily d ON d.code=k.code AND d.date BETWEEN s.prior20_start AND s.prior_date
        AND d.date BETWEEN '2023-11-01' AND '2025-12-29'
      GROUP BY ALL ORDER BY k.date,k.code''').df()
    history['history_valid']=history.history_rows.eq(20)&history.history_dates.eq(20)&history.valid_history_rows.eq(20)
    history['recent_activity']=history.big_up_days.gt(0).astype('boolean').where(history.history_valid,pd.NA)
    history['simple_volume_ratio']=(pool.set_index(['date','code']).volume_1449.reindex(pd.MultiIndex.from_frame(history[['date','code']])).to_numpy()*5/history.volume5_sum).where(history.volume5_rows.eq(5)&history.volume5_sum.gt(0))
    history.to_parquet(ROOT/'history.parquet',index=False,compression='zstd')
    pool=pool.merge(history,on=['date','code'],validate='one_to_one')
    price_cents=np.rint(pool.price_1449*100).astype('int64');reference_cents=np.rint(pool.preclose*100).astype('int64')
    pool['gain_gate']=((price_cents*100>=reference_cents*103)&(price_cents*100<=reference_cents*105)).astype('boolean')
    pool['volume_gate']=(pool.volume_1449*5>pool.volume5_sum).astype('boolean').where(pool.simple_volume_ratio.notna(),pd.NA)
    pool['turnover_gate']=pool.turnover_1449_proxy.between(5,10).astype('boolean').where(pool.turnover_1449_proxy.notna(),pd.NA)
    pool['size_gate']=pool.float_cap_proxy.between(5e9,2e10).astype('boolean').where(pool.float_cap_proxy.notna(),pd.NA)
    pool['market_gate']=pool.return_1449.gt(pool.market_return).astype('boolean').where(pool.market_return.notna(),pd.NA)
    pool['base_combo']=all_gates(pool[GATES])
    target=pool.loc[pool.base_combo.fillna(False),['date','code']]
    source_hashes={}
    raw_meta=json.loads((PORTRAIT/'raw_report.json').read_text())
    for path,digest in raw_meta['batch_manifests_sha256'].items():
        assert sha(Path(path))==digest
        source_hashes.update(json.loads(Path(path).read_text())['source_sha256'])
    raw_parts=[];selected_hashes={}
    codes=sorted(target.code.unique())
    for start in range(0,len(codes),64):
        subset=codes[start:start+64];paths=[MINUTES/code[:2].upper()/(code[3:]+'.parquet') for code in subset]
        for path in paths:
            digest=source_hashes[str(path)];assert sha(path)==digest;selected_hashes[str(path)]=digest
        c.read_parquet([str(p) for p in paths]).create_view('minutes',replace=True)
        c.register('target',target.loc[target.code.isin(subset)])
        raw=c.sql('''SELECT t.date,t.code,strftime(m.timestamp,'%H%M') AS label,m.open,m.high,m.low,m.close,m.volume,m.turnover
          FROM minutes m JOIN target t ON lower(m.exchange)||'.'||m.symbol=t.code AND strftime(m.timestamp,'%Y-%m-%d')=t.date
          WHERE m.timestamp>=TIMESTAMP '2024-01-01' AND m.timestamp<TIMESTAMP '2026-01-01'
            AND(strftime(m.timestamp,'%H%M') BETWEEN '0930' AND '1130' OR strftime(m.timestamp,'%H%M') BETWEEN '1301' AND '1449')
          ORDER BY t.date,t.code,m.timestamp''').df()
        raw_parts.append(raw)
        print(json.dumps({'processed_codes':min(start+64,len(codes)),'total_codes':len(codes),'raw_rows':sum(len(p) for p in raw_parts)}),flush=True)
    raw=pd.concat(raw_parts,ignore_index=True) if raw_parts else pd.DataFrame(columns=['date','code','label','open','high','low','close','volume','turnover'])
    raw.to_parquet(ROOT/'raw_prefix.parquet',index=False,compression='zstd')
    expected,observe=labels();late=set(observe[-15:]);records=[]
    by_key={key:part for key,part in raw.groupby(['date','code'])}
    for row in pool.loc[pool.base_combo.fillna(False)].itertuples():
        part=by_key.get((row.date,row.code),raw.iloc[:0]).sort_values('label')
        complete=part.label.tolist()==expected
        numeric=part[['open','high','low','close','volume','turnover']].to_numpy(dtype=float)
        valid=np.isfinite(numeric).all(axis=1)&(numeric[:,:4]>0).all(axis=1)&(numeric[:,4:]>=0).all(axis=1)
        valid &= part.high.ge(part[['open','low','close']].max(axis=1)).to_numpy()&part.low.le(part[['open','high','close']].min(axis=1)).to_numpy()
        valid &= (np.abs(numeric[:,:4]-np.round(numeric[:,:4],2))<=.0001).all(axis=1)
        valid &= part.volume.eq(0).eq(part.turnover.eq(0)).to_numpy()
        vwap=part.turnover.div(part.volume.where(part.volume.gt(0)))
        amount_bad=part.volume.gt(0)&(~vwap.between(part.low-.0101,part.high+.0101)|~np.isfinite(vwap))
        valid &= ~amount_bad.to_numpy()
        cumulative_volume=part.volume.cumsum();cumulative_amount=part.turnover.cumsum()
        cumulative_price=cumulative_amount/cumulative_volume.where(cumulative_volume.gt(0))
        above=part.close.round(2).ge(cumulative_price)
        sample=part.label.isin(observe);last=part.label.isin(late)
        valid_samples=int((sample&cumulative_volume.gt(0)).sum())
        source_ok=complete and bool(valid.all()) and valid_samples==45
        if complete:
            assert int(part.volume.sum())==row.volume_1449
            assert abs(part.turnover.sum()-row.amount_1449)<=.001+abs(row.amount_1449)*1e-12
            assert abs(float(part.close.iloc[-1])-row.price_1449)<=.0001
        records.append({'date':row.date,'code':row.code,'bars':len(part),'complete_labels':complete,
            'bad_bars':int((~valid).sum()),'amount_bad_bars':int(amount_bad.sum()),'valid_samples':valid_samples,
            'above_samples':int(above.loc[sample].sum()),'late_above_samples':int(above.loc[last].sum()),
            'current_above':bool(above.iloc[-1]) if len(part) else None,'path_source_valid':source_ok,
            'vwap_support':bool(above.loc[sample].sum()>=34 and above.loc[last].all() and above.iloc[-1]) if source_ok else None})
    path=pd.DataFrame(records);path['vwap_support']=path.vwap_support.astype('boolean')
    path.to_parquet(ROOT/'paths.parquet',index=False,compression='zstd')
    pool=pool.merge(path,on=['date','code'],how='left',validate='one_to_one')
    pool['group']='outside_base';unknown=pool.base_combo.isna();pool.loc[unknown,'group']='base_unknown'
    base=pool.base_combo.fillna(False);known=base&pool.recent_activity.notna()&pool.vwap_support.notna()
    pool.loc[base&~known,'group']='detail_unknown'
    pool.loc[known,'group']=pool.loc[known,'recent_activity'].astype(int).astype(str)+':'+pool.loc[known,'vwap_support'].astype(int).astype(str)
    pool['primary']=pool.necessary_tradeable&pool['group'].eq('1:1')
    pool.to_parquet(ROOT/'pool.parquet',index=False,compression='zstd')
    result={'protocol_sha256':sha(PROTOCOL),'cohort_sha256':sha(PORTRAIT/'cohort.parquet'),'calendar_sha256':sha(CALENDAR),
        'float_input_report_sha256':sha(FLOAT/'input_report.json'),'float_input_verification_sha256':sha(FLOAT/'input_verification.json'),
        'external_manifest_sha256':sha(EXTERNAL),'daily_files_sha256':daily_hashes,'minute_files_sha256':selected_hashes,
        'source_raw_report_sha256':sha(PORTRAIT/'raw_report.json'),'visible_rows':len(pool),'necessary_rows':int(pool.necessary_tradeable.sum()),
        'base_rows':int(base.sum()),'necessary_base_rows':int((base&pool.necessary_tradeable).sum()),'primary_rows':int(pool.primary.sum()),
        'groups':pool.loc[pool.necessary_tradeable].groupby(['half','group']).size().rename('n').reset_index().to_dict('records'),
        'path_source_unknown':int((~path.path_source_valid).sum()),'raw_bars':len(raw),
        'outputs_sha256':{name:sha(ROOT/(name+'.parquet')) for name in ['pool','history','paths','raw_prefix']},
        'outcomes_read':False,'new_2026_prices_read':False}
    save_json(ROOT/'input_report.json',result);c.close()
    return {key:result[key] for key in ['visible_rows','necessary_rows','base_rows','necessary_base_rows','primary_rows','groups','path_source_unknown','raw_bars']}


if __name__=='__main__':print(json.dumps(freeze(),ensure_ascii=False,indent=2))
