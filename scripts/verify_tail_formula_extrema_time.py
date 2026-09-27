"""Verify extreme-close timing from SQL, native clauses and fixed raw examples."""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_extrema_time as study
from trade_research.corporate_cash import save_json, sha


def main():
    p,old,prices=study.checked_source();root=study.ROOT
    r=json.loads((root/'feature_report.json').read_text())
    for key,path in [('protocol_sha256',study.PROTOCOL),('previous_feature_report_sha256',study.previous.ROOT/'feature_report.json'),
        ('window_feature_report_sha256',study.window.ROOT/'feature_report.json'),
        ('window_feature_verification_sha256',study.window.ROOT/'feature_verification.json'),
        ('features_sha256',root/'features.parquet')]:assert r[key]==sha(path)
    f=pd.read_parquet(root/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    assert f.prior_formula_input_valid.equals(old.formula_input_valid)
    assert all(z not in study.EXPRESSIONS for z in ['Z01','Z02','Z03'])
    c=study.base.conn();c.register('original',old);c.register('prices',prices)
    columns=','.join(study.PRICE_COLUMNS)
    conditions=' AND '.join(f'isfinite({n}) AND {n}>0' for n in study.PRICE_COLUMNS)
    high='CASE '+' '.join(f'WHEN pv_c{49-i}=highest THEN {i}' for i in range(29))+' END'
    low='CASE '+' '.join(f'WHEN pv_c{49-i}=lowest THEN {i}' for i in range(29))+' END'
    expected=c.sql(f'''WITH values_ AS(SELECT *,greatest({columns}) AS highest,least({columns}) AS lowest,
        path_variance_valid AND {conditions} AS extrema_valid FROM prices),
        ages AS(SELECT date,code,extrema_valid,CASE WHEN extrema_valid THEN {high} END::DOUBLE AS latest_high_age,
        CASE WHEN extrema_valid THEN {low} END::DOUBLE AS latest_low_age FROM values_)
        SELECT a.*,100*latest_high_age/28 AS E01,100*latest_low_age/28 AS E02,
        o.formula_input_valid AND extrema_valid AND isfinite(E01) AND isfinite(E02) AS formula_input_valid
        FROM ages a JOIN original o USING(date,code) ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(f[expected.columns],expected,check_dtype=False,check_exact=True)
    for name in ['E01','E02']:
        a=f.loc[f.formula_input_valid,name].to_numpy();b=expected.loc[expected.formula_input_valid,name].to_numpy()
        np.testing.assert_array_equal(np.floor(np.clip(100*a+10000+.000001,0,999999)),np.floor(np.clip(100*b+10000+.000001,0,999999)))
        assert f.loc[f.formula_input_valid,name].between(0,100).all()
    assert r['valid']==int(f.formula_input_valid.sum())
    assert r['newly_invalid']==int((old.formula_input_valid&~f.formula_input_valid).sum())
    wide=prices[study.PRICE_COLUMNS]
    flat=(wide.max(axis=1)==wide.min(axis=1))&f.formula_input_valid
    assert int(flat.sum())==r['flat_valid_windows'] and f.loc[flat,['E01','E02']].eq(0).all().all()
    # Check the expanded native formula's window positions and most-recent tie order.
    assert r['expressions']==study.EXPRESSIONS and r['native_header']==study.HEADER
    symbols=re.findall(r'(?m)^([A-Za-z][A-Za-z0-9]*):=',study.HEADER)+list(study.EXPRESSIONS)
    assert len(symbols)==len(set(symbols))
    header=study.HEADER.splitlines()
    assert 'PH29:=VALUEWHEN(TIME=1449,HHV(C,29));' in header
    assert 'PL29:=VALUEWHEN(TIME=1449,LLV(C,29));' in header
    for index in range(29):
        rhs='C' if index==0 else f'REF(C,{index})'
        assert f'PE{index:02d}:=VALUEWHEN(TIME=1449,{rhs});' in header
    a=wide.to_numpy(dtype=float)[:,::-1];mask=f.extrema_valid
    for name,peak in [('E01','PH29'),('E02','PL29')]:
        expression=study.NEW_EXPRESSIONS[name]
        conditions=re.findall(r'IF\(PE(\d{2})='+peak+r',(\d+),',expression)
        assert conditions==[(f'{i:02d}',str(i)) for i in range(28)]
        assert expression.startswith('100*') and expression.endswith('28'+')'*28+'/28')
        extreme=np.max(a,axis=1) if name=='E01' else np.min(a,axis=1)
        result=np.full(len(f),28.)
        for pe,age in reversed(conditions):
            result=np.where(a[:,int(pe)]==extreme,float(age),result)
        np.testing.assert_array_equal(100*result[mask]/28,f.loc[mask,name].to_numpy())
    samples=json.loads((study.window.ROOT/'feature_verification.json').read_text())['sampled_raw_windows']
    indexed=f.set_index(['date','code']);raw_minutes=0
    for item in samples:
        date,code=item['date'],item['code'];row=indexed.loc[(date,code)]
        path=study.window.MINUTES/code[:2].upper()/(code[3:]+'.parquet')
        raw=pd.read_parquet(path,columns=['timestamp','close'],filters=[('timestamp','>=',pd.Timestamp(date+' 14:21:00')),
            ('timestamp','<',pd.Timestamp(date+' 14:50:00'))]).sort_values('timestamp')
        raw_minutes+=len(raw)
        if row.extrema_valid:
            assert len(raw)==29 and raw.timestamp.nunique()==29
            x=[round(float(value),2) for value in raw.close]
            maximum,minimum=max(x),min(x)
            h=28-max(i for i,value in enumerate(x) if value==maximum)
            l=28-max(i for i,value in enumerate(x) if value==minimum)
            assert h==row.latest_high_age and l==row.latest_low_age
            np.testing.assert_array_equal([100*h/28,100*l/28],row[['E01','E02']].to_numpy(dtype=float))
    assert len(samples)==32
    proof=dict(passed=True,feature_report_sha256=sha(root/'feature_report.json'),rows=len(f),valid=int(f.formula_input_valid.sum()),
        all_previous_keys_and_48_inputs_unchanged=True,all_extreme_ages_ties_flags_and_encodings_independently_rebuilt=True,
        all_native_window_offsets_and_nested_clauses_rebuilt=True,raw_sample_windows=len(samples),raw_sample_minutes=raw_minutes,
        no_new_raw_history_downloads=True,outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'feature_verification.json',proof);return proof


if __name__=='__main__':
    print(json.dumps(main(),ensure_ascii=False,indent=2))
