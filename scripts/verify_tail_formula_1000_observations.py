"""Check fixed identities and independently rebuild raw morning samples."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from trade_research.corporate_cash import MINUTES, save_json, sha

ROOT=Path('data/research/tail_formula_1000')


def rebuild(raw):
    p=raw[['open','high','low','close']].astype(float)
    v=raw.volume.astype(float)
    a=raw.turnover.astype(float)
    t=pd.to_datetime(raw.timestamp)
    good=(np.isfinite(p).all(axis=1)&p.gt(0).all(axis=1)&np.isfinite(v)&np.isfinite(a)
        &p.high.add(.0001).ge(p.max(axis=1))&p.low.sub(.0001).le(p.min(axis=1))
        &(p-p.round(2)).abs().le(.0001).all(axis=1)&v.ge(0)&a.ge(0)&v.eq(0).eq(a.eq(0))
        &(v.eq(0)|(a/v).between(p.low-.0101,p.high+.0101))&t.eq(t.dt.floor('min')))
    active=good&v.gt(0)
    lows=[]
    for i in range(2,len(raw)):
        if active.iloc[i-2:i+1].all() and t.iloc[i]-t.iloc[i-2]==pd.Timedelta(minutes=2):
            lows.append(p.close.iloc[i-2:i+1].min())
    return dict(bars=len(raw),labels=t.dt.strftime('%H:%M').nunique(),valid_bars=int(good.sum()),
        active_minutes=int(active.sum()),max_close=p.close[active].max(),
        sustained_close=max(lows) if lows else np.nan,min_low=p.low[active].min(),
        price_1000=p.close[active&t.dt.strftime('%H:%M').eq('10:00')].max(),
        source_valid=len(raw)==30 and t.dt.strftime('%H:%M').nunique()==30 and good.all())


def check():
    report=json.loads((ROOT/'observation_report.json').read_text())
    manifest=json.loads((ROOT/'observation_manifest.json').read_text())
    assert report['manifest_sha256']==sha(ROOT/'observation_manifest.json')
    assert manifest['protocol_sha256']==sha(Path('config/tail_formula_1000_protocol.json'))
    assert manifest['keys_sha256']==sha(ROOT/'observation_keys.parquet')
    assert report['observations_sha256']==sha(ROOT/'observations.parquet')
    parts=[]
    for path,digest in report['parts_sha256'].items():
        assert sha(Path(path))==digest
        parts.append(pd.read_parquet(path))
    keys=pd.read_parquet(ROOT/'observation_keys.parquet')
    expected=keys.merge(pd.concat(parts,ignore_index=True),on=['date','code','next_date'],how='left',validate='one_to_one')
    expected['source_valid']=expected.source_valid.fillna(False)
    expected=expected.sort_values(['date','code']).reset_index(drop=True)
    actual=pd.read_parquet(ROOT/'observations.parquet')
    pd.testing.assert_frame_equal(actual,expected,check_dtype=False,check_exact=True)
    assert actual.next_date.gt(actual.date).all() and actual.next_date.le('2025-12-31').all()
    assert not actual.code.str[3:].str.startswith(('92','688','300','301')).any()
    actual['key_hash']=[hashlib.sha256((d+'|'+c).encode()).hexdigest() for d,c in zip(actual.date,actual.code)]
    samples=actual.sort_values('key_hash').groupby('half',sort=True).head(24)
    # Include incomplete / invalid windows as well; selection uses quality only.
    extra=actual.loc[~actual.source_valid].sort_values('key_hash').groupby('half',sort=True).head(4)
    samples=pd.concat([samples,extra]).drop_duplicates(['date','code']).sort_values(['date','code'])
    source_manifest=Path('data/research/economic_winner/input_manifest.json')
    assert sha(source_manifest)==manifest['source_manifest_sha256']
    hashes=json.loads(source_manifest.read_text())['source_sha256']
    checks=0
    raw_rows=0
    for code,group in samples.groupby('code'):
        source=MINUTES/code[:2].upper()/(code[3:]+'.parquet')
        assert sha(source)==hashes[str(source)]
        clauses=[[('timestamp','>=',pd.Timestamp(d+' 09:31:00').to_pydatetime()),
                  ('timestamp','<',pd.Timestamp(d+' 10:01:00').to_pydatetime())] for d in group.next_date]
        raw=pq.read_table(source,columns=['timestamp','open','high','low','close','volume','turnover'],filters=clauses).to_pandas()
        raw=raw.sort_values('timestamp').reset_index(drop=True)
        for row in group.itertuples():
            r=raw.loc[pd.to_datetime(raw.timestamp).dt.strftime('%Y-%m-%d').eq(row.next_date)].reset_index(drop=True)
            check=rebuild(r)
            raw_rows+=len(r)
            for name,value in check.items():
                got=getattr(row,name)
                # SQL has no aggregate row for an absent window.
                if len(r)==0 and name!='source_valid':
                    assert pd.isna(got),(row.date,row.code,name,got)
                else:
                    np.testing.assert_allclose(float(got),float(value),atol=2e-10,rtol=0,equal_nan=True,
                        err_msg=str((row.date,row.code,name)))
                checks+=1
    result=dict(passed=True,observation_report_sha256=sha(ROOT/'observation_report.json'),
        rows=len(actual),complete=int(actual.source_valid.sum()),sample_windows=len(samples),sample_raw_rows=raw_rows,
        sample_checks=checks,all_identities_rebuilt=True,profit_labels_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'observation_verification.json',result)
    return result


if __name__=='__main__':
    print(json.dumps(check(),ensure_ascii=False,indent=2))
