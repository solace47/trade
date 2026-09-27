"""Independent pandas reconstruction of every frozen earlier trade window."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trade_research import tail_formula_additive as base
from trade_research.corporate_cash import MINUTES, save_json, sha
from trade_research.tail_formula_long48_inputs import OUT
from trade_research.tail_formula_long48_observations import LABELS, checked_keys


def main():
    p,m,keys = checked_keys()
    report=json.loads((LABELS/'window_report.json').read_text()); raw_report=json.loads((LABELS/'raw_report.json').read_text())
    assert report['raw_report_sha256']==sha(LABELS/'raw_report.json')
    sources=json.loads((OUT/'source_manifest.json').read_text())
    entries=[]; mornings=[]; count=0
    for name,digest in raw_report['parts_sha256'].items():
        assert sha(Path(name))==digest
        r=pd.read_parquet(name); count+=len(r)
        assert r.date.between(p['signal_first'],p['signal_last']).all()
        assert r.source_date.le(p['observation_last']).all()
        assert ((r.kind.eq('entry')&r.source_date.eq(r.date)&r.clock.between('14:52','14:55'))
            | (r.kind.eq('morning')&r.source_date.gt(r.date)&r.clock.between('09:31','10:00'))).all()
        assert r.clock.eq(r.timestamp.dt.strftime('%H:%M')).all() and r.source_date.eq(r.timestamp.dt.strftime('%Y-%m-%d')).all()
        assert not r.duplicated(['date','code','kind','timestamp']).any()
        a=r[['open','high','low','close','volume','amount']].to_numpy(dtype=float)
        o,h,l,cl,vol,amt=a.T
        price=amt/np.where(vol>0,vol,np.nan)
        r['good']=(np.isfinite(a).all(axis=1)&(a[:,:4].min(axis=1)>0)
            &(h+.0001>=np.maximum.reduce([o,l,cl]))&(l-.0001<=np.minimum(o,cl))
            &(vol>=0)&(amt>=0)&((vol==0)==(amt==0))
            &((vol==0)|((price>=l-.0101)&(price<=h+.0101)))&r.timestamp.eq(r.timestamp.dt.floor('min')))
        r['bounds']=np.isfinite(h)&np.isfinite(l)&(np.abs(h-np.rint(h*100)/100)<=.0001)&(np.abs(l-np.rint(l*100)/100)<=.0001)
        r['cents']=np.isfinite(a[:,:4]).all(axis=1)&(np.abs(a[:,:4]-np.rint(a[:,:4]*100)/100)<=.0001).all(axis=1)
        r['positive']=vol>0; r['invalid_bounds']=r.positive&~r.bounds
        r['positive_low']=r.low.where(r.positive);r['positive_high']=r.high.where(r.positive)
        e=r.loc[r.kind.eq('entry')].copy()
        e=e.groupby(['date','code']).agg(entry_bars=('timestamp','size'),entry_labels=('clock','nunique'),
            valid_bars=('good','sum'),positive_bars=('positive','sum'),invalid_bounds=('invalid_bounds','sum'),
            entry_volume=('volume','sum'),amount=('amount','sum'),entry_low=('positive_low','min'),entry_high=('positive_high','max')).reset_index()
        e['entry_vwap']=e.amount/e.entry_volume.where(e.entry_volume.gt(0));e=e.drop(columns='amount');entries.append(e)
        z=r.loc[r.kind.eq('morning')].sort_values(['date','code','timestamp']).reset_index(drop=True)
        z['morning_valid']=z.good&z.cents;z['active']=z.morning_valid&z.positive
        same=z.date.eq(z.date.shift(2))&z.code.eq(z.code.shift(2))
        triple=same&z.active.astype(int).rolling(3,min_periods=3).sum().eq(3)&z.timestamp.sub(z.timestamp.shift(2)).eq(pd.Timedelta(minutes=2))
        z['three']=z.close.rolling(3,min_periods=3).min().where(triple)
        z['active_close']=z.close.where(z.active);z['active_low']=z.low.where(z.active)
        z['at1000']=z.close.where(z.active&z.clock.eq('10:00'))
        g=z.groupby(['date','code']).agg(bars=('timestamp','size'),labels=('clock','nunique'),valid_bars=('morning_valid','sum'),
            active_minutes=('active','sum'),max_close=('active_close','max'),sustained_close=('three','max'),
            min_low=('active_low','min'),price_1000=('at1000','max')).reset_index();mornings.append(g)
    assert count==raw_report['raw_rows']
    e=keys[['date','code']].merge(pd.concat(entries,ignore_index=True),on=['date','code'],how='left',validate='one_to_one')
    e['entry_source_valid']=e.entry_bars.eq(4)&e.entry_labels.eq(4)&e.valid_bars.eq(4)
    e['entry_bounds_valid']=e.positive_bars.gt(0)&e.invalid_bounds.eq(0)
    o=keys[['date','code','next_date']].merge(pd.concat(mornings,ignore_index=True),on=['date','code'],how='left',validate='one_to_one')
    o['source_valid']=o.bars.eq(30)&o.labels.eq(30)&o.valid_bars.eq(30)
    for frame,stem in [(e,'entry'),(o,'morning')]:
        path=LABELS/f'{stem}_windows.parquet';assert report[f'{stem}_windows_sha256']==sha(path)
        actual=pd.read_parquet(path)
        pd.testing.assert_frame_equal(actual,frame[actual.columns],check_dtype=False,rtol=1e-13,atol=2e-9)
        frame[actual.columns].to_parquet(LABELS/f'independent_{stem}.parquet',index=False,compression='zstd')
    sample=pd.concat([e[['date','code','entry_source_valid']].rename(columns={'entry_source_valid':'valid'}).assign(kind='entry'),
        o[['date','code','source_valid']].rename(columns={'source_valid':'valid'}).assign(kind='morning')],ignore_index=True)
    sample['month']=sample.date.str[:7]
    sample['hash']=[hashlib.sha256(('long48-window-v1|'+d+'|'+code+'|'+k).encode()).hexdigest() for d,code,k in zip(sample.date,sample.code,sample.kind)]
    sample=sample.sort_values('hash').groupby(['month','kind','valid']).head(4)
    next_dates=keys.set_index(['date','code']).next_date
    c=base.conn();c.read_parquet(list(raw_report['parts_sha256'])).create_view('raw')
    checks=0
    for s in sample.itertuples():
        day=s.date if s.kind=='entry' else next_dates.loc[(s.date,s.code)]
        first,last=('14:52','14:56') if s.kind=='entry' else ('09:31','10:01')
        path=MINUTES/s.code[:2].upper()/(s.code[3:]+'.parquet');assert sha(path)==sources['minute_sha256'][str(path)]
        raw=pq.read_table(path,columns=['timestamp','open','high','low','close','volume','turnover'],
            filters=[('timestamp','>=',pd.Timestamp(day+' '+first).to_pydatetime()),('timestamp','<',pd.Timestamp(day+' '+last).to_pydatetime())]).to_pandas()
        raw=raw.rename(columns={'turnover':'amount'}).sort_values('timestamp').reset_index(drop=True)
        stored=c.execute('SELECT timestamp,open,high,low,close,volume,amount FROM raw WHERE date=? AND code=? AND kind=? ORDER BY timestamp',
            [s.date,s.code,s.kind]).df()
        pd.testing.assert_frame_equal(raw,stored,check_dtype=False,check_exact=True);checks+=len(raw)
    c.close()
    proof=dict(passed=True,window_report_sha256=sha(LABELS/'window_report.json'),raw_rows=count,
        rows=len(keys),all_buy_and_morning_aggregates_independently_rebuilt=True,
        independent_entry_sha256=sha(LABELS/'independent_entry.parquet'),independent_morning_sha256=sha(LABELS/'independent_morning.parquet'),
        fixed_source_windows=len(sample),original_minutes_rechecked=checks,sample_keys=sample[['date','code','kind']].to_dict('records'),
        new_2026_prices_read=False,last_observation_only_morning_prices_read=True,no_exit_rules=True)
    save_json(LABELS/'window_verification.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k!='sample_keys'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
