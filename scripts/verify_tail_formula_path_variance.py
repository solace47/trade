"""Rebuild all minute-path fields in SQL and inspect frozen raw minute samples."""
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_path_variance as study
from trade_research.corporate_cash import save_json, sha


def raw_window(raw,date):
    timestamps=raw.timestamp
    minute=timestamps.dt.hour*100+timestamps.dt.minute
    mask=timestamps.dt.strftime('%Y-%m-%d').eq(date)&minute.between(1420,1449)
    x=raw.loc[mask].sort_values('timestamp').copy()
    labels=x.timestamp.dt.hour*100+x.timestamp.dt.minute
    prices=x.close.astype(float)
    rounded=np.copysign(np.floor(np.abs(prices)*100+.5)/100,prices)
    good=(x.timestamp.eq(x.timestamp.dt.floor('min'))&np.isfinite(prices)&prices.gt(0)
        &(prices-rounded).abs().le(.0001))
    points=[]
    for label in range(1420,1450):
        values=rounded[labels.eq(label)]
        points.append(float(values.max()) if len(values) else np.nan)
    return dict(pv_bars=len(x),pv_clocks=labels.nunique(),pv_good_bars=int(good.sum())),np.array(points)


def main():
    p,old,source_hashes=study.checked_source();root=study.ROOT
    r=json.loads((root/'feature_report.json').read_text())
    for key,path in [('protocol_sha256',study.PROTOCOL),('previous_feature_report_sha256',study.previous.ROOT/'feature_report.json'),
        ('window_report_sha256',root/'window_report.json'),('features_sha256',root/'features.parquet')]:
        assert r[key]==sha(path)
    windows=json.loads((root/'window_report.json').read_text())
    assert windows['extractor_sha256']==sha(Path(study.__file__)) and windows['protocol_sha256']==sha(study.PROTOCOL)
    assert windows['minute_manifest_sha256']==sha(study.MANIFEST)
    for path,digest in windows['parts_sha256'].items():assert sha(Path(path))==digest
    for path,digest in windows['source_sha256'].items():
        assert digest==source_hashes[path] and sha(Path(path))==digest
    expected_paths={str(study.MINUTES/code[:2].upper()/(code[3:]+'.parquet')) for code in old.code.unique()}
    assert expected_paths==set(windows['source_sha256'])
    f=pd.read_parquet(root/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    assert f.prior_formula_input_valid.equals(old.formula_input_valid)
    c=study.base.conn();c.register('original',old)
    c.read_parquet(list(windows['parts_sha256'])).create_view('wide')
    rebuilt_windows=c.sql('SELECT o.date,o.code,w.* EXCLUDE(date,code) FROM original o LEFT JOIN wide w USING(date,code) ORDER BY date,code').df()
    pd.testing.assert_frame_equal(f[rebuilt_windows.columns],rebuilt_windows,check_exact=True)
    returns=','.join(f'CASE WHEN pv_c{n}>0 AND pv_c{n-1}>0 THEN 100*ln(pv_c{n}/pv_c{n-1}) END AS d{n}' for n in range(21,50))
    c.execute('CREATE VIEW changes AS SELECT date,code,'+returns+' FROM wide')
    square=lambda n:f'd{n}*d{n}'
    total='+'.join(square(n) for n in range(21,50))
    down='+'.join(f'CASE WHEN d{n}<0 THEN {square(n)} ELSE 0. END' for n in range(21,50))
    tail='+'.join(square(n) for n in range(36,50));prior='+'.join(square(n) for n in range(22,36))
    c.execute(f'CREATE VIEW aggregates AS SELECT date,code,{total} AS pv_sum29,{down} AS pv_down29,{tail} AS pv_tail14,{prior} AS pv_prior14 FROM changes')
    all_prices=' AND '.join(f'isfinite(pv_c{n}) AND pv_c{n}>0' for n in range(20,50))
    expected=c.sql(f'''WITH j AS(SELECT o.*,w.* EXCLUDE(date,code),a.* EXCLUDE(date,code),
        coalesce(pv_bars=30 AND pv_clocks=30 AND pv_good_bars=30 AND {all_prices},false) AS path_variance_valid
        FROM original o LEFT JOIN wide w USING(date,code) LEFT JOIN aggregates a USING(date,code))
        SELECT date,code,pv_sum29,pv_down29,pv_tail14,pv_prior14,path_variance_valid,
        CASE WHEN path_variance_valid THEN sqrt(pv_sum29/29)/V01 END AS Z01,
        CASE WHEN path_variance_valid THEN pv_down29/greatest(pv_sum29,1e-12) END AS Z02,
        CASE WHEN path_variance_valid THEN (pv_tail14-pv_prior14)/greatest(pv_tail14+pv_prior14,1e-12) END AS Z03,
        formula_input_valid AND path_variance_valid AND isfinite(Z01) AND isfinite(Z02) AND isfinite(Z03) AS formula_input_valid
        FROM j ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(f[expected.columns],expected,check_dtype=False,rtol=0,atol=2e-10)
    names=list(study.NEW_EXPRESSIONS)
    for n in names:
        a=f.loc[f.formula_input_valid,n].to_numpy();b=expected.loc[expected.formula_input_valid,n].to_numpy()
        np.testing.assert_array_equal(np.floor(np.clip(100*a+10000+.000001,0,999999)),np.floor(np.clip(100*b+10000+.000001,0,999999)))
    valid=f.formula_input_valid
    assert f.loc[valid,'Z01'].ge(0).all() and f.loc[valid,'Z02'].between(0,1).all()
    assert f.loc[valid,'Z03'].between(-1,1).all()
    assert int(valid.sum())==r['valid'] and int((old.formula_input_valid&~valid).sum())==r['newly_invalid']
    assert int((valid&f.pv_sum29.eq(0)).sum())==r['flat_valid_windows']
    pd.testing.assert_series_equal(f.loc[valid,'pv_c49'],f.loc[valid,'price_1449'],check_names=False,check_exact=True)
    pd.testing.assert_series_equal(f.loc[valid,'pv_c20'],f.loc[valid,'p20'],check_names=False,check_exact=True)
    assert r['expressions']==study.EXPRESSIONS and r['native_header']==study.HEADER
    symbols=re.findall(r'(?m)^([A-Za-z][A-Za-z0-9]*):=',study.HEADER)+list(study.EXPRESSIONS)
    assert len(symbols)==len(set(symbols))
    # Four nonflat valid, two flat valid and two invalid examples per half.
    # The hash ordering and strata are fixed without observing any outcomes.
    samples=[]
    for half in ['2024H1','2024H2','2025H1','2025H2']:
        for name,mask,count in [('valid_nonflat',valid&f.pv_sum29.gt(0),4),
            ('valid_flat',valid&f.pv_sum29.eq(0),2),('invalid',~valid,2)]:
            pool=f.loc[f.half.eq(half)&mask].copy()
            pool['sample_key']=[hashlib.sha256(f'path-variance-v1|{d}|{code}'.encode()).hexdigest() for d,code in zip(pool.date,pool.code)]
            for _,row in pool.sort_values('sample_key').head(count).iterrows():
                source=study.MINUTES/row.code[:2].upper()/(row.code[3:]+'.parquet')
                raw=pd.read_parquet(source,columns=['timestamp','close'],filters=[('timestamp','>=',pd.Timestamp(row.date+' 14:20:00')),
                    ('timestamp','<',pd.Timestamp(row.date+' 14:50:00'))])
                counts,prices=raw_window(raw,row.date)
                for key,value in counts.items():assert row[key]==value
                np.testing.assert_allclose(prices,row[study.PRICE_COLUMNS].to_numpy(dtype=float),rtol=0,atol=1e-12,equal_nan=True)
                contaminated=pd.concat([raw,pd.DataFrame({'timestamp':[pd.Timestamp(row.date+' 14:50:00'),pd.Timestamp(row.date+' 15:00:00')],
                    'close':[999999.,.01]})],ignore_index=True)
                counts2,prices2=raw_window(contaminated,row.date)
                assert counts==counts2;np.testing.assert_array_equal(prices,prices2)
                if row.formula_input_valid:
                    changes=[100*math.log(prices[i]/prices[i-1]) for i in range(1,30)]
                    squares=[x*x for x in changes];s=sum(squares)
                    x=[math.sqrt(s/29)/row.V01,sum(v for v,z in zip(squares,changes) if z<0)/max(s,1e-12),
                        (sum(squares[15:])-sum(squares[1:15]))/max(sum(squares[1:]),1e-12)]
                    np.testing.assert_allclose(x,row[names].to_numpy(dtype=float),rtol=0,atol=2e-10)
                samples.append(dict(date=row.date,code=row.code,stratum=name,raw_minutes=len(raw)))
    assert len(samples)==32
    proof=dict(passed=True,feature_report_sha256=sha(root/'feature_report.json'),rows=len(f),valid=int(valid.sum()),
        all_previous_keys_and_48_values_unchanged=True,all_window_links_changes_aggregates_inputs_and_encodings_rebuilt=True,
        sampled_raw_windows=samples,sampled_raw_minutes=sum(s['raw_minutes'] for s in samples),
        future_minute_contamination_does_not_change_inputs=True,all_native_variable_names_unique=True,
        outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'feature_verification.json',proof)
    return {k:v for k,v in proof.items() if k!='sampled_raw_windows'}


if __name__=='__main__':
    print(json.dumps(main(),ensure_ascii=False,indent=2))
