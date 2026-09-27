"""Scale visible price movements by strictly prior stock-specific true range."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_relative as relative
from . import tail_formula_complete as complete
from .corporate_cash import save_json,sha

ROOT=Path('data/research/tail_formula_volatility')
PROTOCOL=Path('config/tail_formula_volatility_protocol.json')
NORMALIZED=['A01','A02','A03','A05','A06','A07','A13','A14','A18','A19','D01','D02','D03','D04','C02','C03','C04','C07','C08']
HEADER=complete.HEADER+'\n'.join(f'{n}:={complete.EXPRESSIONS[n]};' for n in NORMALIZED)+'\n'
HEADER+='TR0:=MAX(HHV(H,B0)-LLV(L,B0),MAX(ABS(HHV(H,B0)-REF(C,B0)),ABS(LLV(L,B0)-REF(C,B0))));\n'
HEADER+='PAT:=('+ '+'.join(f'REF(TR0,B{i})' for i in range(20))+')/20;\nVP20:=100*PAT/DCP1;\n'
EXPRESSIONS={('N'+n if n in NORMALIZED else n):(n+'/VP20' if n in NORMALIZED else expr)
             for n,expr in complete.EXPRESSIONS.items()}
EXPRESSIONS['V01']='VP20'


def source_files():
    r=json.loads((base.SOURCE/'feature_report.json').read_text())
    for name,digest in r['source_sha256'].items():
        assert sha(Path(name))==digest
    return list(r['source_sha256'])


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace frozen volatility-scaled inputs')
    ROOT.mkdir(parents=True,exist_ok=True)
    p=json.loads((complete.ROOT/'feature_verification.json').read_text())
    r=json.loads((complete.ROOT/'feature_report.json').read_text())
    assert p['passed'] and p['feature_report_sha256']==sha(complete.ROOT/'feature_report.json')
    assert r['features_sha256']==sha(complete.ROOT/'features.parquet')
    files=source_files();c=base.conn();c.read_parquet(files).create_view('raw_daily')
    atr=c.sql('''WITH d AS(SELECT date,code,high::DOUBLE AS h,low::DOUBLE AS l,close::DOUBLE AS cl,
        lag(close::DOUBLE) OVER(PARTITION BY code ORDER BY date) AS pc FROM raw_daily
        WHERE date>='2023-06-01' AND date<='2025-12-30' AND tradestatus=1),
        t AS(SELECT *,greatest(h-l,abs(h-pc),abs(l-pc)) AS tr FROM d),
        a AS(SELECT date,code,avg(tr) OVER w AS atr20,count(tr) OVER w AS n FROM t
        WINDOW w AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING))
        SELECT date,code,atr20 FROM a WHERE date>='2024-01-01' AND n=20 ORDER BY date,code''').df()
    atr.to_parquet(ROOT/'prior_atr.parquet',index=False,compression='zstd')
    f=pd.read_parquet(complete.ROOT/'features.parquet').merge(atr,on=['date','code'],how='left',validate='one_to_one')
    f['V01']=100*f.atr20/f.preclose
    for n in NORMALIZED:
        f['N'+n]=f[n]/f.V01
    f['formula_input_valid'] &= f.atr20.gt(0)&np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    f=f.sort_values(['date','code']).reset_index(drop=True)
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    report=dict(protocol_sha256=sha(PROTOCOL),complete_report_sha256=sha(complete.ROOT/'feature_report.json'),
        daily_report_sha256=sha(base.SOURCE/'feature_report.json'),prior_atr_sha256=sha(ROOT/'prior_atr.parquet'),
        features_sha256=sha(ROOT/'features.parquet'),rows=len(f),valid=int(f.formula_input_valid.sum()),expressions=EXPRESSIONS,
        native_header=HEADER,native_history_alignment_verified=False,software_compilation_verified=False,
        outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',report)
    return {k:v for k,v in report.items() if k not in ['expressions','native_header']}


def verify_features():
    r=json.loads((ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL)
    assert r['complete_report_sha256']==sha(complete.ROOT/'feature_report.json')
    assert r['daily_report_sha256']==sha(base.SOURCE/'feature_report.json')
    assert r['prior_atr_sha256']==sha(ROOT/'prior_atr.parquet') and r['features_sha256']==sha(ROOT/'features.parquet')
    # Independent pandas rolling construction from every issuer's original daily file.
    rebuilt=[]
    for path in source_files():
        d=pd.read_parquet(path,columns=['date','code','high','low','close','tradestatus'],
            filters=[('date','>=','2023-06-01'),('date','<=','2025-12-30')])
        d=d.loc[d.tradestatus.eq(1)].sort_values('date').copy()
        prev=d.close.astype(float).shift()
        tr=pd.concat([d.high-d.low,(d.high-prev).abs(),(d.low-prev).abs()],axis=1).max(axis=1)
        d['atr20']=tr.rolling(20,min_periods=20).mean().shift()
        rebuilt.append(d.loc[d.date.ge('2024-01-01')&d.atr20.notna(),['date','code','atr20']])
    atr=pd.concat(rebuilt).sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(pd.read_parquet(ROOT/'prior_atr.parquet'),atr,check_dtype=False,rtol=0,atol=2e-12)
    old=pd.read_parquet(complete.ROOT/'features.parquet');f=pd.read_parquet(ROOT/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    expected=old[['date','code','preclose',*NORMALIZED]].merge(atr,on=['date','code'],how='left',validate='one_to_one')
    np.testing.assert_allclose(f.V01,100*expected.atr20/expected.preclose,rtol=0,atol=2e-12,equal_nan=True)
    for n in NORMALIZED:
        np.testing.assert_allclose(f['N'+n],expected[n]/(100*expected.atr20/expected.preclose),rtol=0,atol=2e-10,equal_nan=True)
    valid=old.formula_input_valid&expected.atr20.gt(0)&np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    assert valid.equals(f.formula_input_valid) and int(valid.sum())==r['valid']
    result=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=len(f),
        all_prior_atr_independently_rebuilt=True,normalized_fields=19,previous_values_unchanged=True,
        outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True,native_history_alignment_verified=False)
    save_json(ROOT/'feature_verification.json',result)
    return result


def setup():
    base.ROOT=ROOT;base.FEATURES=ROOT;base.PROTOCOL=PROTOCOL
    base.EXPRESSIONS=EXPRESSIONS;base.HEADER=HEADER;relative.PROTOCOL=PROTOCOL


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['features','verify_features','model','verify_model','scores','calibrate','analyze','diagnose'])
    a=p.parse_args()
    if a.stage in ['features','verify_features']:
        result=globals()[a.stage]()
    else:
        setup()
        if a.stage in ['model','verify_model']:
            result=getattr(relative,a.stage)('relative')
        elif a.stage=='calibrate':
            result=base.calibrate(max_p95=None)
        else:
            result=getattr(base,a.stage)()
    print(json.dumps(result,ensure_ascii=False,indent=2))
