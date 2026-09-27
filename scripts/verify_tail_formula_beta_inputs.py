"""Independent rolling covariance and explicit native-history reconstruction."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_beta as study
from trade_research.corporate_cash import save_json, sha


def main():
    root=study.ROOT; files=study.source_files()
    r=json.loads((root/'feature_report.json').read_text())
    for key,path in [('protocol_sha256',study.PROTOCOL),
        ('previous_feature_report_sha256',study.previous.ROOT/'feature_report.json'),
        ('daily_feature_report_sha256',study.base.SOURCE/'feature_report.json'),
        ('index_source_report_sha256',study.INDEX/'index_source_report.json'),
        ('index_feature_report_sha256',study.INDEX/'feature_report.json'),
        ('index_verification_sha256',study.INDEX/'feature_verification.json'),
        ('indices_sha256',study.INDEX/'indices.parquet'),('features_sha256',root/'features.parquet')]:
        assert r[key]==sha(path)
    assert list(r['native_expressions'].items())==list(study.EXPRESSIONS.items())
    assert r['native_header']==study.HEADER and len(study.EXPRESSIONS)==49
    names=re.findall(r'\b([A-Z][A-Z0-9]*):=',study.HEADER)
    assert len(names)==len(set(names))
    assert all('ICP'+str(i) in names and 'DCP'+str(i) in names for i in range(1,22))
    assert not any(k in study.EXPRESSIONS for k in ['R01','R02','R03','R04'])
    old=pd.read_parquet(study.previous.ROOT/'features.parquet'); f=pd.read_parquet(root/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    assert f.prior_formula_input_valid.equals(old.formula_input_valid)
    raw=json.loads((study.INDEX/'index_source_report.json').read_text())
    indices={}
    for path in raw['raw_files_sha256']:
        item=json.loads(Path(path).read_text())
        idx=pd.DataFrame(item['records']); assert not idx.date.duplicated().any()
        indices[item['code']]=idx.set_index('date').close.astype(float)
    pieces=[]; sources={}
    for path in files:
        d=pd.read_parquet(path,columns=['date','code','close','preclose','adjustflag','tradestatus'],
            filters=[('date','>=','2023-06-01'),('date','<=','2025-12-30')])
        d=d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True)
        assert len(d) and not d.date.duplicated().any() and d.code.nunique()==1
        code=d.code.iloc[0]; sources[code]=path
        ic=d.date.map(indices['sh.000001' if code.startswith('sh.') else 'sz.399001'])
        pc=d.close.shift(); pic=ic.shift()
        good=(d.close.gt(0)&pc.gt(0)&ic.gt(0)&pic.gt(0)&d.adjustflag.eq(3)&d.adjustflag.shift().eq(3)
            &np.isfinite(d.close)&np.isfinite(pc)&np.isfinite(ic)&np.isfinite(pic))
        sr=((d.close-pc)/pc).where(good); ir=((ic-pic)/pic).where(good)
        out=d[['date','code']].copy()
        out['beta_history_rows']=pd.Series(np.ones(len(d))).rolling(20,min_periods=1).sum().shift()
        out['beta_history_good']=good.astype(int).rolling(20,min_periods=1).sum().shift()
        out['beta_history_start']=d.date.shift(20); out['beta_history_end']=d.date.shift()
        out['beta_history_reference_start']=d.date.shift(21)
        out['beta_stock_mean']=sr.rolling(20,min_periods=1).mean().shift()
        out['beta_index_mean']=ir.rolling(20,min_periods=1).mean().shift()
        out['beta_product_mean']=(sr*ir).rolling(20,min_periods=1).mean().shift()
        out['beta_index_square_mean']=(ir*ir).rolling(20,min_periods=1).mean().shift()
        # Pandas central-moment rolling implementation is independent of SQL raw moments.
        out['beta_index_variance']=ir.rolling(20,min_periods=1).var(ddof=0).shift()
        out['beta_covariance']=sr.rolling(20,min_periods=1).cov(ir,ddof=0).shift()
        out['beta_history_reference_breaks']=(d.preclose-pc).abs().gt(.005).astype(int).rolling(20,min_periods=1).sum().shift()
        pieces.append(out.loc[out.date.ge('2024-01-01')])
    a=old[['date','code']].merge(pd.concat(pieces,ignore_index=True),on=['date','code'],how='left',validate='one_to_one')
    full=a.beta_history_rows.eq(20)
    for name in ['beta_history_start','beta_history_end','beta_history_reference_start']:
        pd.testing.assert_series_equal(f.loc[full,name],a.loc[full,name],check_names=False)
    numeric=['beta_history_rows','beta_history_good','beta_stock_mean','beta_index_mean',
        'beta_product_mean','beta_index_square_mean','beta_history_reference_breaks']
    np.testing.assert_allclose(f[numeric],a[numeric],rtol=0,atol=3e-15,equal_nan=True)
    np.testing.assert_allclose(f.loc[full,['beta_index_variance','beta_covariance']],
        a.loc[full,['beta_index_variance','beta_covariance']],rtol=0,atol=3e-15,equal_nan=True)
    valid_source=(a.beta_history_rows.eq(20)&a.beta_history_good.eq(20)&a.beta_history_end.lt(a.date)
        &a.beta_history_reference_start.lt(a.beta_history_start)&a.beta_index_variance.gt(0)&old.V01.gt(0)
        &np.isfinite(a[['beta_index_variance','beta_covariance']]).all(axis=1)&np.isfinite(old.V01))
    assert valid_source.equals(f.beta_source_valid)
    expected=pd.DataFrame({'KB01':a.beta_covariance/a.beta_index_variance}).where(valid_source,axis=0)
    for name,stock,index in [('RB01','A01','J01'),('RB02','A05','J02'),('RB03','A06','J03'),('RB04','A07','J04')]:
        expected[name]=((old[stock]-expected.KB01*old[index])/old.V01).where(valid_source)
    np.testing.assert_allclose(f[list(expected)],expected,rtol=0,atol=2e-10,equal_nan=True)
    valid=old.formula_input_valid&valid_source&np.isfinite(f[list(study.EXPRESSIONS)]).all(axis=1)
    assert valid.equals(f.formula_input_valid)
    def encode(x):
        return np.floor(np.clip(x.to_numpy()*100+10000+.000001,0,999999)).astype('int32')
    np.testing.assert_array_equal(encode(expected.loc[valid]),encode(f.loc[valid,list(expected)]))
    assert len(f)==r['rows'] and int(valid.sum())==r['valid']
    assert int((old.formula_input_valid&~valid).sum())==r['newly_invalid']
    assert int((valid&f.beta_history_reference_breaks.gt(0)).sum())==r['valid_with_historical_reference_break']
    assert f.date.between('2024-01-01','2025-12-30').all()
    assert f.loc[valid_source,'beta_history_end'].lt(f.loc[valid_source,'date']).all()
    sample=f.loc[valid].copy()
    sample['key_hash']=[hashlib.sha256(('beta-native/'+d+'/'+k).encode()).hexdigest() for d,k in zip(sample.date,sample.code)]
    sample=sample.sort_values('key_hash').groupby('half',sort=True).head(8)
    cases=[]
    for row in sample.sort_values(['date','code']).itertuples():
        d=pd.read_parquet(sources[row.code],columns=['date','close','tradestatus'],
            filters=[('date','>=',row.beta_history_reference_start),('date','<',row.date)])
        d=d.loc[d.tradestatus.eq(1)].sort_values('date',ascending=False)
        assert len(d)==21 and d.date.iloc[0]==row.beta_history_end
        ic=d.date.map(indices['sh.000001' if row.code.startswith('sh.') else 'sz.399001']).to_numpy()
        cl=d.close.to_numpy(); sx=cl[:-1]/cl[1:]-1; ix=ic[:-1]/ic[1:]-1
        # Literal native arithmetic on DCP1..21 and ICP1..21, newest first.
        bta=(sum(sx*ix)/20-sum(sx)/20*sum(ix)/20)/(sum(ix*ix)/20-(sum(ix)/20)**2)
        np.testing.assert_allclose(bta,row.KB01,rtol=0,atol=2e-10)
        cases.append(dict(date=row.date,code=row.code,first_source_date=d.date.iloc[-1],
            last_source_date=d.date.iloc[0],history_closes=len(d),native_beta=float(bta),stored_beta=row.KB01))
    proof=dict(passed=True,feature_report_sha256=sha(root/'feature_report.json'),rows=len(f),valid=int(valid.sum()),
        all_previous_48_values_unchanged=True,all_covariances_independently_rebuilt=True,
        all_new_integer_encodings_rebuilt=True,input_source_dates_strictly_prior=True,
        native_history_arithmetic_cases=len(cases),native_history_closes=21*len(cases),native_cases=cases,
        native_client_values_verified=False,outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'feature_verification.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k!='native_cases'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
