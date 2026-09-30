"""Completed morning extrema as visible native inputs for a late-day formula."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_dense_path as price
from .corporate_cash import MINUTES,save_json,sha

STEM='tail_formula_morning_range'
ROOT=Path('data/research')/STEM
INPUTS=ROOT/'inputs'
PROTOCOL=Path('config')/(STEM+'_input_protocol.json')
INTENT=Path('config')/(STEM+'_intent.json')
MINUTE_MANIFEST=Path('data/research/economic_winner/input_manifest.json')
META=price.META
RAW=['date','code','timestamp','open','high','low','close','volume']
EXTRA_HEADER='''AMDT:=VALUEWHEN(TIME=1130,DATE);
AMBC:=VALUEWHEN(TIME=1130,B0);
AMCLK:=VALUEWHEN(TIME=1130,COUNT(TIME>=930 AND TIME<=1130,121));
AMGD:=VALUEWHEN(TIME=1130,COUNT(O>0 AND H>0 AND L>0 AND C>0 AND V>=0 AND H+0.0001>=MAX(MAX(O,C),L) AND L-0.0001<=MIN(O,C) AND ABS(O*100-ROUND(O*100))<=0.01 AND ABS(H*100-ROUND(H*100))<=0.01 AND ABS(L*100-ROUND(L*100))<=0.01 AND ABS(C*100-ROUND(C*100))<=0.01,121));
AMHC:=VALUEWHEN(TIME=1130,HHV(IF(V>0,ROUND(H*100),0),121));
AMLC:=VALUEWHEN(TIME=1130,LLV(IF(V>0,ROUND(L*100),999999999),121));
AMVV:=VALUEWHEN(TIME=1130,SUM(V,121));
AMQC:=ROUND(Q*100);
AMREADY:=AMDT=DATE AND AMBC=121 AND AMCLK=121 AND AMGD=121 AND AMVV>0 AND AMHC>0 AND AMLC>0 AND AMLC<999999999 AND AMQC>0 AND VP20>0;
'''
HEADER=price.HEADER+EXTRA_HEADER
NEW_EXPRESSIONS={n:f'IF(AMREADY,100*(AMQC/{d}-1)/VP20,DRAWNULL)' for n,d in [('AMHD','AMHC'),('AMLD','AMLC')]}
ARMS={'control':price.ARMS['control'],'morning':{**price.ARMS['control'],**NEW_EXPRESSIONS}}
EXPRESSIONS=ARMS['morning']


def checked():
    p=json.loads(PROTOCOL.read_text());price.checked()
    assert p['arms']==ARMS and p['native_header']==HEADER and p['expected_keys']==1258085
    assert p['intent_sha256']==sha(INTENT) and p['expected_minute_labels']==121
    assert not p['new_2026_prices_allowed'] and p['no_new_2023_raw']
    for f,h in p['source_hashes'].items():assert sha(Path(f))==h,f
    gate=json.loads(Path(p['conditional_gate']).read_text());assert gate['passed'] and not gate['supports_2024_extension']
    complete=json.loads(Path(p['conditional_completion']).read_text());assert complete['passed']
    return p


def original():
    f=pd.read_parquet(price.INPUTS/'features.parquet',columns=[*META,*ARMS['control']])
    return f.loc[f.date.ge('2024-01-01') & f.date.lt('2026-01-01')].reset_index(drop=True)


def cache():
    p=checked();assert not (INPUTS/'input_manifest.json').exists();INPUTS.mkdir(parents=True,exist_ok=True)
    f=original();assert len(f)==1258085
    keys=f[['date','code']].copy();remaining=keys.copy();plans=[]
    coverage=json.loads((ROOT/'cache_timestamp_coverage_v2.json').read_text());assert coverage['passed']
    primary=json.loads(MINUTE_MANIFEST.read_text())['source_sha256']
    for spec in p['cache_sources']:
        name=spec['name'];report=Path(spec['report']);r=json.loads(report.read_text())
        if name=='touch':
            parent=report.parent;manifest=json.loads((parent/'manifest.json').read_text())
            assert r['manifest_sha256']==sha(parent/'manifest.json') and manifest['economic_manifest_sha256']==sha(MINUTE_MANIFEST)
            files=r['parts_sha256'];amount='amount';physical='timestamp'
        elif name=='diagnostic':
            parent=report.parent;file=parent/'raw_days.parquet';files={str(file):r['sha256'][file.name]}
            for source in json.loads((parent/'source_index.json').read_text()):
                assert source['download_checksum_matched'] and source['minute_sha256']==primary[source['minute_path']]
            amount='turnover';physical='timestamp'
        else:
            assert name=='external';parent=report.parent;file=parent/'raw_prefix.parquet';files={str(file):r['outputs_sha256']['raw_prefix']}
            for file,h in r['minute_files_sha256'].items():assert primary[file]==h
            a=json.loads((ROOT/'alternate_clock_coverage.json').read_text());assert a['passed'] and a['input_report_sha256']==sha(report)
            amount='turnover';physical="strptime(date||' '||label,'%Y-%m-%d %H%M')"
        for file,h in files.items():assert sha(Path(file))==h
        if name!='external':
            family=spec['family'];records=[r for r in coverage['records'] if r['source_family']==family];assert len(records)==1
            # Coverage part order follows sorted family records, including empty families.
            index=next(i for i,r in enumerate(coverage['records']) if r['source_family']==family)
            part=ROOT/'coverage_parts_v2'/f'part_{index:03d}.parquet';assert sha(part)==records[0]['output_sha256']
            cf=pd.read_parquet(part)
            known=cf.loc[cf.mask0.eq(2**64-1)&cf.mask1.eq(2**57-1),['date','code']]
        else:known=pd.read_parquet(ROOT/'alternate_full_keys.parquet')
        matched=remaining.merge(known,on=['date','code'],validate='one_to_one')
        keyfile=INPUTS/(name+'_cache_keys.parquet');matched.to_parquet(keyfile,index=False,compression='zstd')
        plans.append(dict(name=name,files=files,keys=str(keyfile),keys_sha256=sha(keyfile),rows=len(matched),physical_timestamp=physical,amount_column=amount,report_sha256=sha(report)))
        remaining=remaining.merge(matched.assign(cached=True),on=['date','code'],how='left',validate='one_to_one')
        remaining=remaining.loc[~remaining.cached.eq(True),['date','code']].reset_index(drop=True)
    keys.to_parquet(INPUTS/'keys.parquet',index=False,compression='zstd');remaining.to_parquet(INPUTS/'missing_full_window_keys.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),keys_sha256=sha(INPUTS/'keys.parquet'),missing_keys_sha256=sha(INPUTS/'missing_full_window_keys.parquet'),
        cache_plans=plans,reused_full_window_keys=sum(s['rows'] for s in plans),new_full_window_keys=len(remaining),
        only_complete_source_proven_window_caches_reused=True,partial_four_or_thirty_bars_not_full_morning_extrema=True,
        no_new_price_values_read=True,no_new_2026_prices_read=True,no_exit_rules=True)
    assert r['reused_full_window_keys']+r['new_full_window_keys']==1258085
    save_json(INPUTS/'input_manifest.json',r);return {k:r[k] for k in ['reused_full_window_keys','new_full_window_keys']}


def raw():
    checked();m=json.loads((INPUTS/'input_manifest.json').read_text());assert m['protocol_sha256']==sha(PROTOCOL)
    assert not (INPUTS/'raw_report.json').exists()
    assert m['missing_keys_sha256']==sha(INPUTS/'missing_full_window_keys.parquet')
    parts={};cached_rows=0
    for spec in m['cache_plans']:
        assert spec['keys_sha256']==sha(Path(spec['keys']))
        for file,h in spec['files'].items():assert sha(Path(file))==h
        out=INPUTS/('reused_'+spec['name']+'.parquet')
        meta=out.with_suffix('.json');identity=dict(input_manifest_sha256=sha(INPUTS/'input_manifest.json'),cache_spec=spec)
        if meta.exists():
            receipt=json.loads(meta.read_text());assert all(receipt[k]==v for k,v in identity.items()) and receipt['sha256']==sha(out)
        else:
            assert not out.exists(),'Unreceipted raw fragment must be investigated, not overwritten'
            c=base.conn();c.read_parquet(list(spec['files'])).create_view('cached');c.read_parquet(spec['keys']).create_view('keys')
            # For label-only storage, full timestamp identity was proven before prices.
            ts=spec['physical_timestamp']
            d=c.sql(f'''WITH r AS(SELECT date,code,{ts} AS timestamp,open::DOUBLE AS open,high::DOUBLE AS high,
                low::DOUBLE AS low,close::DOUBLE AS close,volume::DOUBLE AS volume FROM cached)
                SELECT r.* FROM r JOIN keys USING(date,code) WHERE strftime(timestamp,'%H%M') BETWEEN '0930' AND '1130'
                ORDER BY date,code,timestamp''').df();c.close()
            assert len(d)==spec['rows']*121 and not d.duplicated(['date','code','timestamp']).any()
            d.to_parquet(out,index=False,compression='zstd')
            receipt=dict(**identity,sha256=sha(out),rows=len(d));save_json(meta,receipt)
        cached_rows+=receipt['rows'];parts[str(out)]=receipt['sha256']
    missing=pd.read_parquet(INPUTS/'missing_full_window_keys.parquet');codes=sorted(missing.code.unique())
    hashes=json.loads(MINUTE_MANIFEST.read_text())['source_sha256'];folder=INPUTS/'raw_parts';folder.mkdir(exist_ok=True)
    sources={}
    for start in range(0,len(codes),64):
        subset=codes[start:start+64];out=folder/f'part_{start//64:03d}.parquet';meta=out.with_suffix('.json')
        files=[MINUTES/code[:2].upper()/(code[3:]+'.parquet') for code in subset]
        identity=dict(codes=subset,input_manifest_sha256=sha(INPUTS/'input_manifest.json'),extractor_sha256=sha(Path(__file__)))
        if meta.exists():
            receipt=json.loads(meta.read_text());assert all(receipt[k]==v for k,v in identity.items()) and receipt['sha256']==sha(out)
        else:
            for file in files:assert sha(file)==hashes[str(file)]
            c=base.conn();c.read_parquet([str(f) for f in files]).create_view('original');c.register('keys',missing.loc[missing.code.isin(subset)])
            d=c.sql('''WITH r AS(SELECT lower(exchange)||'.'||symbol AS code,strftime(timestamp,'%Y-%m-%d') AS date,
                timestamp,open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,close::DOUBLE AS close,volume::DOUBLE AS volume
                FROM original WHERE timestamp>=TIMESTAMP '2024-01-01' AND timestamp<TIMESTAMP '2026-01-01'
                AND strftime(timestamp,'%H%M') BETWEEN '0930' AND '1130')
                SELECT r.date,r.code,r.timestamp,r.open,r.high,r.low,r.close,r.volume FROM r JOIN keys USING(date,code)
                ORDER BY date,code,timestamp''').df();c.close()
            d.to_parquet(out,index=False,compression='zstd');receipt=dict(**identity,sha256=sha(out),rows=len(d));save_json(meta,receipt)
        sources.update({str(file):hashes[str(file)] for file in files});parts[str(out)]=receipt['sha256']
        print(json.dumps(dict(codes=start+len(subset),total=len(codes))),flush=True)
    r=dict(protocol_sha256=sha(PROTOCOL),input_manifest_sha256=sha(INPUTS/'input_manifest.json'),parts_sha256=parts,
        cached_rows=cached_rows,reused_full_window_keys=m['reused_full_window_keys'],new_full_window_keys=m['new_full_window_keys'],
        source_sha256=sources,no_partial_morning_as_full_extreme=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'raw_report.json',r);return {k:r[k] for k in ['cached_rows','reused_full_window_keys','new_full_window_keys']}


def aggregates():
    r=json.loads((INPUTS/'raw_report.json').read_text());assert r['protocol_sha256']==sha(PROTOCOL)
    for f,h in r['parts_sha256'].items():assert sha(Path(f))==h
    c=base.conn();c.read_parquet(list(r['parts_sha256'])).create_view('raw')
    d=c.sql('''WITH b AS(SELECT *,coalesce(timestamp=date_trunc('minute',timestamp)
        AND isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close) AND isfinite(volume)
        AND least(open,high,low,close)>0 AND volume>=0 AND high+.0001>=greatest(open,close,low)
        AND low-.0001<=least(open,close) AND abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
        AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001,false) AS good FROM raw)
        SELECT date,code,count(*) AS bars,count(DISTINCT strftime(timestamp,'%H%M')) AS clocks,
        count(*) FILTER(WHERE good) AS good_bars,count(*) FILTER(WHERE volume>0) AS active,
        max(floor(high*100+.5)) FILTER(WHERE volume>0) AS high_cents,
        min(floor(low*100+.5)) FILTER(WHERE volume>0) AS low_cents
        FROM b GROUP BY date,code ORDER BY date,code''').df();c.close();return d


def features():
    checked();assert not (INPUTS/'feature_report.json').exists()
    old=original();a=aggregates();a.to_parquet(INPUTS/'morning_aggregates.parquet',index=False,compression='zstd')
    f=old.merge(a,on=['date','code'],how='left',validate='one_to_one')
    valid=f.bars.eq(121)&f.clocks.eq(121)&f.good_bars.eq(121)&f.active.gt(0)&f.high_cents.gt(0)&f.low_cents.gt(0)&f.V01.gt(0)&np.isfinite(f.V01)&f.A04.gt(0)
    q=np.floor(f.A04*100+.5)
    with np.errstate(all='ignore'):
        for n,d in [('AMHD','high_cents'),('AMLD','low_cents')]:f[n]=(100*(q/f[d]-1)/f.V01).where(valid)
    valid &= np.isfinite(f[list(NEW_EXPRESSIONS)]).all(axis=1)
    f['prior_formula_input_valid']=f.formula_input_valid;f['morning_input_valid']=valid;f['formula_input_valid'] &= valid
    f.to_parquet(INPUTS/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(INPUTS/'features.parquet'),raw_report_sha256=sha(INPUTS/'raw_report.json'),
        morning_aggregates_sha256=sha(INPUTS/'morning_aggregates.parquet'),rows=len(f),valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((f.prior_formula_input_valid&~f.formula_input_valid).sum()),expressions=EXPRESSIONS,native_header=HEADER,
        price_only_morning_extrema_quality_not_amount_model=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True,
        software_compilation_verified=False,native_source_parity_verified=False)
    save_json(INPUTS/'feature_report.json',r)
    for file in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:(INPUTS/file).symlink_to((price.INPUTS/file).resolve())
    return {k:r[k] for k in ['rows','valid','newly_invalid']}


def verify():
    checked();r=json.loads((INPUTS/'feature_report.json').read_text());assert r['protocol_sha256']==sha(PROTOCOL)
    assert r['features_sha256']==sha(INPUTS/'features.parquet') and r['raw_report_sha256']==sha(INPUTS/'raw_report.json')
    f=pd.read_parquet(INPUTS/'features.parquet');old=original()
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    np.testing.assert_array_equal(f.prior_formula_input_valid,old.formula_input_valid)
    raw=json.loads((INPUTS/'raw_report.json').read_text());c=base.conn();c.read_parquet(list(raw['parts_sha256'])).create_view('raw');c.register('original',old)
    expected=c.sql('''WITH v AS(SELECT *,timestamp=date_trunc('minute',timestamp) AND
        isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close) AND isfinite(volume)
        AND open>0 AND high>0 AND low>0 AND close>0 AND volume>=0 AND high+.0001>=open AND high+.0001>=close
        AND high+.0001>=low AND low-.0001<=open AND low-.0001<=close
        AND abs(open*100-round(open*100))<=.01000000001 AND abs(high*100-round(high*100))<=.01000000001
        AND abs(low*100-round(low*100))<=.01000000001 AND abs(close*100-round(close*100))<=.01000000001 AS ok FROM raw),
        a AS(SELECT date,code,count(*) AS n,count(DISTINCT timestamp) AS labels,
            bool_and(coalesce(ok,false)) AS all_ok,sum(volume) AS vol,
            max(round(high*100)) FILTER(WHERE volume>0) AS hi,min(round(low*100)) FILTER(WHERE volume>0) AS lo
            FROM v GROUP BY date,code),b AS(SELECT *,coalesce(n=121 AND labels=121 AND all_ok AND vol>0 AND hi>0 AND lo>0
            AND isfinite(V01) AND V01>0 AND A04>0,false) AS valid FROM original LEFT JOIN a USING(date,code))
        SELECT date,code,valid,CASE WHEN valid THEN 100*(round(A04*100)/hi-1)/V01 END AS AMHD,
            CASE WHEN valid THEN 100*(round(A04*100)/lo-1)/V01 END AS AMLD FROM b ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(f[['date','code']],expected[['date','code']],check_exact=True)
    np.testing.assert_array_equal(f.morning_input_valid,expected.valid)
    encode=lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999))
    diff=0.
    for n in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[n],expected[n],rtol=0,atol=2e-11,equal_nan=True)
        ok=expected.valid;np.testing.assert_array_equal(encode(f.loc[ok,n]),encode(expected.loc[ok,n]))
        diff=max(diff,float(np.max(np.abs(f.loc[ok,n]-expected.loc[ok,n]))))
    final=old.formula_input_valid & expected.valid;np.testing.assert_array_equal(f.formula_input_valid,final)
    names=re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS);assert len(names)==len({n.casefold() for n in names})
    out=dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),rows=len(f),valid=int(final.sum()),max_difference=diff,
        all_48_values_and_metadata_unchanged=True,all_raw_window_quality_extrema_ratios_and_encodings_independently_rebuilt=True,
        effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        same_quality_controls_required_before_fitting=not final.equals(old.formula_input_valid),
        no_unknown_zero_imputation=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'feature_verification.json',out);return out


def native_values(bars,quote,atr,outside=1000000.):
    """Replay the literal extra header on a bounded 1-minute array."""
    b=bars.sort_values('timestamp');times=b.timestamp.dt.hour*100+b.timestamp.dt.minute
    size=len(b);env=dict(Q=float(quote),VP20=float(atr),DRAWNULL=np.nan,
        DATE=np.r_[0,np.ones(size+2)],TIME=np.r_[1459,times.to_numpy(),1131,1449],
        B0=np.r_[241,np.arange(1,size+3)],ROUND=lambda a:np.floor(a+.5),ABS=np.abs,
        MAX=np.maximum,MIN=np.minimum,IF=np.where)
    for col,name in [('open','O'),('high','H'),('low','L'),('close','C'),('volume','V')]:
        env[name]=np.r_[outside,b[col].to_numpy(float),outside,outside]
    env['COUNT']=lambda a,n:pd.Series(np.asarray(a,float)).rolling(int(n),min_periods=int(n)).sum().to_numpy()
    for name,method in [('SUM','sum'),('HHV','max'),('LLV','min')]:
        env[name]=lambda a,n,m=method:getattr(pd.Series(np.asarray(a,float)).rolling(int(n),min_periods=int(n)),m)().to_numpy()
    def valuewhen(mask,values):
        values=np.broadcast_to(values,len(mask));result=np.full(len(mask),np.nan)
        result[np.asarray(mask,bool)]=values[np.asarray(mask,bool)]
        return pd.Series(result).ffill().to_numpy()
    env['VALUEWHEN']=valuewhen
    for line in EXTRA_HEADER.splitlines():
        name,expr=line.rstrip(';').split(':=');expr=re.sub(r'(?<![<>=!])=(?!=)','==',expr)
        if name=='AMCLK':expr="VALUEWHEN(TIME==1130,COUNT((TIME>=930)&(TIME<=1130),121))"
        elif name=='AMGD':
            cond=expr.split('COUNT(',1)[1].rsplit(',121)',1)[0]
            expr='VALUEWHEN(TIME==1130,COUNT('+' & '.join('('+s+')' for s in cond.split(' AND '))+',121))'
        elif name=='AMREADY':expr=' & '.join('('+s+')' for s in expr.split(' AND '))
        env[name]=eval(expr,{'__builtins__':{}},env)
    with np.errstate(all='ignore'):
        return np.asarray([np.asarray(eval(e,{'__builtins__':{}},env))[-1] for e in NEW_EXPRESSIONS.values()])


def native():
    p=checked();fv=json.loads((INPUTS/'feature_verification.json').read_text())
    assert fv['passed'] and fv['feature_report_sha256']==sha(INPUTS/'feature_report.json')
    f=pd.read_parquet(INPUTS/'features.parquet');encode=lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999))
    for start in range(0,len(f),50000):
        d=f.iloc[start:start+50000];env=dict(AMREADY=d.morning_input_valid.to_numpy(),AMQC=np.floor(d.A04.to_numpy()*100+.5),
            AMHC=d.high_cents.to_numpy(),AMLC=d.low_cents.to_numpy(),VP20=d.V01.to_numpy(),IF=np.where,DRAWNULL=np.nan)
        for name,e in NEW_EXPRESSIONS.items():
            with np.errstate(all='ignore'):actual=eval(e,{'__builtins__':{}},env)
            np.testing.assert_allclose(actual,d[name],rtol=0,atol=2e-11,equal_nan=True)
            good=d.morning_input_valid.to_numpy();np.testing.assert_array_equal(encode(actual[good]),encode(d.loc[good,name]))
    sample=f[['date','code']].copy();sample['half']=sample.date.str[:4]+np.where(sample.date.str[5:7].le('06'),'H1','H2')
    sample['key_hash']=[hashlib.sha256((date+code).encode()).hexdigest() for date,code in zip(sample.date,sample.code)]
    sample=sample.sort_values(['half','key_hash']).groupby('half').head(24).sort_values(['date','code'])
    assert len(sample)==96;rr=json.loads((INPUTS/'raw_report.json').read_text());c=base.conn();c.read_parquet(list(rr['parts_sha256'])).create_view('raw');c.register('keys',sample[['date','code']])
    fields=','.join('r.'+n for n in RAW)
    observed=c.sql(f'SELECT {fields} FROM raw r JOIN keys USING(date,code) ORDER BY date,code,timestamp').df();c.close()
    hashes=json.loads(MINUTE_MANIFEST.read_text())['source_sha256'];probes=[];index=f.set_index(['date','code'])
    for code,group in sample.groupby('code',sort=True):
        source=MINUTES/code[:2].upper()/(code[3:]+'.parquet');assert sha(source)==hashes[str(source)]
        c=base.conn();c.register('days',group[['date']]);actual=c.sql(f'''WITH r AS(SELECT strftime(timestamp,'%Y-%m-%d') AS date,
            lower(exchange)||'.'||symbol AS code,timestamp,open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,
            close::DOUBLE AS close,volume::DOUBLE AS volume FROM read_parquet('{source}')
            WHERE timestamp>=TIMESTAMP '2024-01-01' AND timestamp<TIMESTAMP '2026-01-01'
            AND strftime(timestamp,'%H%M') BETWEEN '0930' AND '1130')
            SELECT r.* FROM r JOIN days USING(date) ORDER BY date,code,timestamp''').df();c.close()
        cached=observed.loc[observed.code.eq(code)].reset_index(drop=True)
        pd.testing.assert_frame_equal(cached,actual,check_exact=True,check_dtype=False)
        for date in sorted(group.date.unique()):
            bars=actual.loc[actual.date.eq(date)]
            row=index.loc[(date,code)];expected=row[list(NEW_EXPRESSIONS)].to_numpy(float)
            with np.errstate(all='ignore'):got=native_values(bars,row.A04,row.V01)
            np.testing.assert_allclose(got,expected,rtol=0,atol=2e-11,equal_nan=True)
            np.testing.assert_allclose(native_values(bars,row.A04,row.V01,outside=2000000.),got,rtol=0,atol=2e-11,equal_nan=True)
            probes.append(dict(date=date,code=code,bars=len(bars),input_valid=bool(row.formula_input_valid),source_sha256=hashes[str(source)]))
    assert len(probes)==96
    # Constant price, exact integer scale and nonpositive-volume extrema.
    synthetic=pd.DataFrame(dict(timestamp=pd.date_range('2025-01-02 09:30',periods=121,freq='min'),
        open=10.,high=10.,low=10.,close=10.,volume=1000.))
    np.testing.assert_allclose(native_values(synthetic,10.,2.),[0.,0.],rtol=0,atol=0)
    varied=synthetic.copy();varied.loc[0,['open','high','low','close']]=[10.,11.,9.,10.];varied.loc[0,'volume']=0
    np.testing.assert_allclose(native_values(varied,10.,2.),[0.,0.],rtol=0,atol=0)
    varied.loc[0,'volume']=1000
    scaled=varied.copy();scaled[['open','high','low','close']]*=5
    np.testing.assert_allclose(native_values(scaled,50.,2.),native_values(varied,10.,2.),rtol=0,atol=2e-11)
    prior=price.prior.INPUTS/'native_input_verification.json';assert json.loads(prior.read_text())['passed']
    out=dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),all_new_literal_values_and_encodings_checked=2*len(f),
        source_samples=probes,source_sample_days=len(probes),source_sample_bars=len(observed),
        reused_48_native_input_verification_sha256=sha(prior),all_48_full_values_unchanged_before_proof_reuse=True,
        literal_1130_header_with_date_guard_cent_rounding_and_positive_volume_extrema_verified=True,
        flat_scale_zero_volume_and_outside_window_probes_passed=True,
        helper_sha256=sha(Path(__file__)),software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'native_input_verification.json',out);return {k:out[k] for k in ['source_sample_days','source_sample_bars','all_new_literal_values_and_encodings_checked']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['cache','raw','features','verify','native'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
