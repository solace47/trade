"""Rebuild index history from archived JSON, SQL windows and scalar formulas."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from trade_research import tail_formula_index_volatility as study
from trade_research.corporate_cash import save_json, sha


def main():
    p,old,days,indices=study.checked_source();root=study.ROOT
    r=json.loads((root/'feature_report.json').read_text())
    for key,path in [('protocol_sha256',study.PROTOCOL),('previous_feature_report_sha256',study.previous.ROOT/'feature_report.json'),
        ('index_source_report_sha256',study.INDEX/'index_source_report.json'),
        ('stock_day_manifest_sha256',study.DAYS/'input_manifest.json'),('stock_day_verification_sha256',study.DAYS/'feature_verification.json'),
        ('history_sha256',root/'history.parquet'),('features_sha256',root/'features.parquet')]:
        assert r[key]==sha(path)
    sources=json.loads((study.INDEX/'index_source_report.json').read_text())
    frames=[]
    for path,digest in sources['raw_files_sha256'].items():
        assert sha(Path(path))==digest
        response=json.loads(Path(path).read_text())
        frame=pd.DataFrame(response['records'])
        assert frame.code.eq(response['code']).all()
        assert response['start']==p['history_first'] and response['end']==p['signal_last']
        for column in ['open','high','low','close']:
            frame[column]=pd.to_numeric(frame[column],errors='raise')
        assert frame[['open','high','low','close']].gt(0).all().all()
        frames.append(frame)
    raw_index=pd.concat(frames,ignore_index=True).sort_values(['code','date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(indices,raw_index,check_exact=True,check_dtype=False)
    c=study.base.conn();c.register('days',days);c.register('indices',raw_index)
    c.execute('''CREATE VIEW aligned AS WITH i AS(SELECT *,row_number() OVER(PARTITION BY code ORDER BY date) AS session FROM indices)
        SELECT s.date,s.code,i.close AS ic,i.session FROM days s LEFT JOIN i
        ON s.date=i.date AND i.code=CASE WHEN s.code LIKE 'sh.%' THEN 'sh.000001' ELSE 'sz.399001' END''')
    c.execute('''CREATE VIEW history AS WITH a AS(SELECT *,
        CASE WHEN isfinite(ic) AND ic>0 AND lag(ic) OVER w>0 THEN 100*(ic/lag(ic) OVER w-1) END AS change,
        session-lag(session) OVER w AS gap FROM aligned WINDOW w AS(PARTITION BY code ORDER BY date))
        SELECT date,code,lag(date,21) OVER w AS index_vol_first_date,lag(date) OVER w AS index_vol_last_date,
        CASE WHEN count(*) OVER v=20 THEN count(change) OVER v END AS index_vol_count,
        CASE WHEN count(change) OVER v=20 THEN sqrt(avg(change*change) OVER v) END AS index_rms20,
        CASE WHEN count(*) OVER v=20 THEN sum(CAST(coalesce(gap>1,false) AS INTEGER)) OVER v END AS index_gap_count20,
        CASE WHEN count(gap) OVER v=20 THEN max(gap) OVER v END AS index_max_gap20
        FROM a WINDOW w AS(PARTITION BY code ORDER BY date),
        v AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)''')
    h=c.sql('SELECT * FROM history ORDER BY date,code').df()
    # SQL nullable integer counts and Pandas rolling counts represent the same
    # missing warmup values with different scalar types; compare as floats.
    for name in ['index_vol_count','index_gap_count20','index_max_gap20']:
        h[name]=h[name].astype('float64')
    pd.testing.assert_frame_equal(pd.read_parquet(root/'history.parquet'),h,check_dtype=False,rtol=0,atol=2e-12)
    f=pd.read_parquet(root/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    assert f.prior_formula_input_valid.equals(old.formula_input_valid)
    c.register('original',old)
    expected=c.sql('''WITH joined AS(SELECT o.*,h.* EXCLUDE(date,code),
        coalesce(index_vol_count=20 AND index_vol_first_date IS NOT NULL AND index_vol_last_date<o.date
            AND isfinite(index_rms20) AND index_rms20>0,false) AS index_volatility_valid
        FROM original o LEFT JOIN history h USING(date,code))
        SELECT date,code,index_vol_first_date,index_vol_last_date,index_vol_count,index_rms20,index_gap_count20,index_max_gap20,
            index_volatility_valid,CASE WHEN index_volatility_valid THEN index_rms20 END AS M01,
            CASE WHEN index_volatility_valid THEN J01/index_rms20 END AS M02,
            CASE WHEN index_volatility_valid THEN J02/index_rms20 END AS M03,
            CASE WHEN index_volatility_valid THEN J03/index_rms20 END AS M04,
            CASE WHEN index_volatility_valid THEN J04/index_rms20 END AS M05
        FROM joined ORDER BY date,code''').df()
    pd.testing.assert_frame_equal(f[expected.columns],expected,check_dtype=False,rtol=0,atol=2e-12)
    # Confirm that the historical stock-date alignment is exactly the one used
    # by the existing four index context inputs, including suspension gaps.
    index_context=c.sql('''WITH h AS(SELECT date,code,lag(ic,1) OVER w AS p1,lag(ic,2) OVER w AS p2,
        lag(ic,6) OVER w AS p6,lag(ic,21) OVER w AS p21,avg(ic) OVER v AS m20 FROM aligned
        WINDOW w AS(PARTITION BY code ORDER BY date),v AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING))
        SELECT o.date,o.code,100*(p1/p2-1) AS I01,100*(p1/p6-1) AS I02,100*(p1/p21-1) AS I03,100*(p1/m20-1) AS I04
        FROM original o JOIN h USING(date,code) ORDER BY date,code''').df()
    pd.testing.assert_frame_equal(old[index_context.columns],index_context,check_dtype=False,rtol=0,atol=2e-12)
    names=[f'M{i:02d}' for i in range(1,6)]
    valid=old.formula_input_valid & expected.index_volatility_valid & np.isfinite(expected[names]).all(axis=1)
    assert valid.equals(f.formula_input_valid)
    def encode(values):
        return np.floor(np.clip(100*values+10000+.000001,0,999999)).astype('int32')
    np.testing.assert_array_equal(encode(f.loc[valid,names].to_numpy()),encode(expected.loc[valid,names].to_numpy()))
    assert len(f)==r['rows'] and int(valid.sum())==r['valid']
    assert int((old.formula_input_valid&~valid).sum())==r['newly_invalid']
    assert int((valid&expected.index_gap_count20.gt(0)).sum())==r['valid_with_index_gap']
    assert float(expected.loc[valid,'index_max_gap20'].max())==r['maximum_market_gap_in_valid_history']
    assert list(r['expressions'].items())==list(study.EXPRESSIONS.items()) and r['native_header']==study.HEADER
    assert len(study.EXPRESSIONS)==53
    native_names=re.findall(r'\b([A-Z][A-Z0-9]*):=',study.HEADER)
    assert len(native_names)==len(set(native_names))
    for i in range(20):
        assert f'IR{i+1:02d}:=100*(REF(INDEXC,B{i})/REF(INDEXC,B{i+1})-1);' in study.HEADER
    assert 'IVOL:=SQRT(('+ '+'.join(f'IR{i:02d}*IR{i:02d}' for i in range(1,21))+')/20);' in study.HEADER
    sample=f.loc[valid,['date','code','half',*names,'index_gap_count20','index_max_gap20']].copy()
    sample['key_hash']=[hashlib.sha256(('index-volatility-native/'+d+'/'+k).encode()).hexdigest() for d,k in zip(sample.date,sample.code)]
    sample=sample.sort_values('key_hash').groupby('half',sort=True).head(8)
    source_indices={code:q.set_index('date').close.to_dict() for code,q in raw_index.groupby('code')}
    ranks={d:i for i,d in enumerate(sorted(raw_index.date.unique()))}
    cases=[]
    for row in sample.sort_values(['date','code']).itertuples():
        dates=days.loc[days.code.eq(row.code)&days.date.lt(row.date),'date'].tail(21).tolist()
        assert len(dates)==21 and max(dates)<row.date
        code='sh.000001' if row.code.startswith('sh.') else 'sz.399001'
        prices=[source_indices[code][d] for d in reversed(dates)]
        changes=[100*(prices[i]/prices[i+1]-1) for i in range(20)]
        rms=(sum(x*x for x in changes)/20)**.5
        source=old.loc[old.date.eq(row.date)&old.code.eq(row.code)].iloc[0]
        values=[rms,*[source[f'J{i:02d}']/rms for i in range(1,5)]]
        np.testing.assert_allclose(values,[getattr(row,n) for n in names],rtol=0,atol=2e-12)
        np.testing.assert_array_equal(encode(np.array(values)),encode(np.array([getattr(row,n) for n in names])))
        gaps=np.diff([ranks[d] for d in dates])
        assert int((gaps>1).sum())==row.index_gap_count20 and int(gaps.max())==row.index_max_gap20
        cases.append(dict(date=row.date,code=row.code,index_code=code,first_source=dates[0],last_source=dates[-1],
            prior_prices=21,rms20=rms,gapped_intervals=int((gaps>1).sum())))
    c.close()
    proof=dict(passed=True,feature_report_sha256=sha(root/'feature_report.json'),rows=len(f),valid=int(valid.sum()),
        all_index_rows_rebuilt_from_original_json=len(raw_index),all_rolling_histories_and_five_inputs_rebuilt=True,
        original_48_values_keys_and_prior_index_alignment_unchanged=True,all_integer_encodings_rebuilt=True,
        fixed_native_cases=cases,native_case_count=len(cases),prior_raw_index_prices=len(cases)*21,
        native_source_parity_verified=False,outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'feature_verification.json',proof)
    return {k:v for k,v in proof.items() if k!='fixed_native_cases'}


if __name__=='__main__':
    print(json.dumps(main(),ensure_ascii=False,indent=2))
