"""Full 13:01--14:20 prices, without changing the original 50-field domain."""
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_morning_range as prior
from . import tail_formula_volume_memory as original_source
from .corporate_cash import save_json, sha

STEM = 'tail_formula_afternoon_prefix'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
INTENT = Path('config') / (STEM + '_intent.json')
META, CONTROL = prior.META, prior.EXPRESSIONS
COLUMNS = [f'af_c{i:02d}' for i in range(80)]
WINDOW_CLOCKS = [1300+i if i<=59 else 1400+i-60 for i in range(1,81)]
SEQUENCE_GUARD = '+'.join(f'IF(REF(TIME,{79-i})={clock},IF(REF(DATE,{79-i})=DATE,1,0),0)' for i,clock in enumerate(WINDOW_CLOCKS))
EXTRA_HEADER = '''AFDT:=VALUEWHEN(TIME=1420,DATE);
''' + f'AFCLK:=VALUEWHEN(TIME=1420,{SEQUENCE_GUARD});\n' + '''AFGD:=VALUEWHEN(TIME=1420,COUNT(C>0 AND ABS(C*100-ROUND(C*100))<=0.01,80));
AFCC:=VALUEWHEN(TIME=1420,ROUND(C*100));
AFREADY:=AFDT=DATE AND AFCLK=80 AND AFGD=80 AND AFCC>0 AND VP20>0;
''' + '\n'.join(f'AFP{i:02d}:=VALUEWHEN(TIME=1420,ROUND(REF(C,{79-i})*100));' for i in range(79)) + '\n'
NEW_EXPRESSIONS = {f'AF{i:02d}': f'IF(AFREADY,100*(AFP{i:02d}/AFCC-1)/VP20,DRAWNULL)' for i in range(79)}
EXPRESSIONS = {**CONTROL, **NEW_EXPRESSIONS}
HEADER = prior.HEADER + EXTRA_HEADER


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['intent_sha256'] == sha(INTENT) and p['arms'] == {'control':CONTROL,'memory':EXPRESSIONS}
    assert p['native_header'] == HEADER and p['expected_keys'] == 1815129
    assert p['window_labels'] == 80 and p['maximum_new_fits'] == 4
    assert not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items(): assert sha(Path(file)) == digest, file
    gate = json.loads(Path(p['conditional_gate']).read_text())
    assert gate['passed'] and not gate['supports_further_validation']
    text = subprocess.run(['git','show','HEAD:docs/selection-formula.md'],text=True,capture_output=True,check=True).stdout
    assert sha(PROTOCOL) in text and sha(INTENT) in text
    return p


def quote_frame(d, keys):
    """Independent pivot keeps duplicate/bad/missing labels visible."""
    d = d.copy(); d['slot'] = d.timestamp.dt.hour * 60 + d.timestamp.dt.minute - 781
    assert d.slot.between(0,79).all()
    d['clock'] = d.timestamp.dt.strftime('%H%M')
    good = (d.timestamp.eq(d.timestamp.dt.floor('min')) & d.timestamp.dt.strftime('%Y-%m-%d').eq(d.date)
            & np.isfinite(d.close) & d.close.gt(0) & (100*d.close-np.floor(100*d.close+.5)).abs().le(.01))
    d['good'] = good
    grouped = d.groupby(['date','code'],sort=True)
    summary = grouped.agg(af_bars=('close','size'),af_clocks=('clock','nunique'),af_good=('good','sum'))
    values = d.groupby(['date','code','slot']).close.max().unstack('slot').reindex(columns=range(80))
    values.columns = COLUMNS
    out = keys.merge(summary.join(values).reset_index(),on=['date','code'],how='left',validate='one_to_one')
    for name in ['af_bars','af_clocks','af_good']: out[name] = out[name].fillna(0).astype('int64')
    return out.sort_values(['date','code']).reset_index(drop=True)


def raw():
    p = checked(); audit=json.loads(Path(p['source_metadata_audit']).read_text())
    path=INPUTS / 'quote_report.json'; assert not path.exists()
    assert audit['keys_sha256']==sha(INPUTS / 'keys.parquet') and audit['missing_keys_sha256']==sha(INPUTS / 'missing_cache_keys.parquet')
    missing=pd.read_parquet(INPUTS / 'missing_cache_keys.parquet'); sources={i['code']:i for i in audit['files']}
    jobs=[]
    for item in audit['cache_plans']:
        assert sha(Path(item['keys']))==item['keys_sha256']
        jobs.append(dict(name='reused_'+item['name'],keys_file=item['keys'],keys_sha256=item['keys_sha256'],
            files=item['files'],timestamp=item['physical_timestamp'],cached=True))
    codes=sorted(missing.code.unique())
    for start in range(0,len(codes),64):
        subset=codes[start:start+64]; file=INPUTS / 'quote_parts' / f'keys_{start//64:03d}.parquet'
        file.parent.mkdir(exist_ok=True)
        selected_keys=missing.loc[missing.code.isin(subset)].reset_index(drop=True)
        if file.exists():
            pd.testing.assert_frame_equal(pd.read_parquet(file),selected_keys,check_exact=True)
        else:
            selected_keys.to_parquet(file,index=False,compression='zstd')
        jobs.append(dict(name=f'fresh_{start//64:03d}',keys_file=str(file),keys_sha256=sha(file),
            files={sources[code]['file']:sources[code]['expected_sha256'] for code in subset},cached=False))
    records=[]
    for job in jobs:
        keys=pd.read_parquet(job['keys_file']); assert sha(Path(job['keys_file']))==job['keys_sha256']
        file=INPUTS / 'quote_parts' / (job['name']+'.parquet'); file.parent.mkdir(exist_ok=True)
        meta=file.with_suffix('.json'); identity=dict(protocol_sha256=sha(PROTOCOL),extractor_sha256=sha(Path(__file__)),job=job)
        if meta.exists():
            receipt=json.loads(meta.read_text());assert all(receipt[k]==v for k,v in identity.items()) and receipt['sha256']==sha(file)
        else:
            assert not file.exists(), 'Investigate an unreceipted fragment instead of replacing it'
            for source,digest in job['files'].items(): assert sha(Path(source))==digest,source
            c=base.conn(); c.read_parquet(list(job['files'])).create_view('source'); c.register('keys',keys)
            if job['cached']:
                body=f"SELECT date,code,{job['timestamp']} AS timestamp,close::DOUBLE AS close FROM source"
            else:
                body="SELECT strftime(timestamp,'%Y-%m-%d') AS date,lower(exchange)||'.'||symbol AS code,timestamp,close::DOUBLE AS close FROM source WHERE timestamp>=TIMESTAMP '2023-01-01' AND timestamp<TIMESTAMP '2026-01-01'"
            d=c.sql(f'''WITH r AS({body}) SELECT r.* FROM r JOIN keys USING(date,code)
                WHERE timestamp>=TIMESTAMP '2023-01-01' AND timestamp<TIMESTAMP '2026-01-01'
                AND strftime(timestamp,'%H%M') BETWEEN '1301' AND '1420' ORDER BY date,code,timestamp''').df()
            c.register('d',d)
            fields=','.join(f'max(close) FILTER(WHERE date_diff(\'minute\',date_trunc(\'day\',timestamp),timestamp)-781={i}) AS {name}' for i,name in enumerate(COLUMNS))
            result=c.sql(f'''WITH a AS(SELECT *,coalesce(timestamp=date_trunc('minute',timestamp)
                AND strftime(timestamp,'%Y-%m-%d')=date AND isfinite(close) AND close>0
                AND abs(close*100-round(close*100))<=.01,false) AS good FROM d),
                b AS(SELECT date,code,count(*) AS af_bars,count(DISTINCT strftime(timestamp,'%H%M')) AS af_clocks,
                sum(good::INT) AS af_good,{fields} FROM a GROUP BY date,code)
                SELECT k.*,coalesce(af_bars,0)::BIGINT AS af_bars,coalesce(af_clocks,0)::BIGINT AS af_clocks,
                coalesce(af_good,0)::BIGINT AS af_good,{','.join(COLUMNS)} FROM keys k LEFT JOIN b USING(date,code) ORDER BY date,code''').df()
            expected=quote_frame(d,keys)
            pd.testing.assert_frame_equal(result,expected,check_exact=True,check_dtype=False)
            result.to_parquet(file,index=False,compression='zstd');c.close()
            receipt=dict(**identity,sha256=sha(file),rows=len(result),raw_bars=len(d),all_price_slots_and_clock_quality_independent_pandas_equal=True)
            save_json(meta,receipt)
        records.append(dict(file=str(file),sha256=receipt['sha256'],rows=receipt['rows'],raw_bars=receipt['raw_bars'],cached=job['cached'],receipt_sha256=sha(meta)))
        print(json.dumps(dict(quoted=job['name'],rows=receipt['rows'],completed=len(records),total=len(jobs))),flush=True)
    assert sum(i['rows'] for i in records)==1815129
    save_json(path,dict(passed=True,protocol_sha256=sha(PROTOCOL),source_metadata_audit_sha256=sha(Path(p['source_metadata_audit'])),
        parts=records,rows=1815129,raw_bars=sum(i['raw_bars'] for i in records),
        reused_keys=sum(i['rows'] for i in records if i['cached']),only_uncached_key_windows_extracted=True,
        bad_missing_duplicate_or_zero_volume_positions_not_skipped=True,new_2026_prices_read=False,no_exit_rules=True))
    return dict(quote_report_sha256=sha(path),parts=len(records),raw_bars=sum(i['raw_bars'] for i in records))


def measure(prices,atr):
    prices=np.asarray(prices,float);atr=np.asarray(atr,float);assert prices.shape==(len(atr),80)
    cents=np.floor(prices*100+.5)
    with np.errstate(all='ignore'): return 100*(cents[:,:79]/cents[:,79:]-1)/atr[:,None]


def literal(prices,atr,good):
    env=dict(AFCC=np.floor(prices[:,79]*100+.5),VP20=atr,AFREADY=good,IF=np.where,DRAWNULL=np.nan)
    env.update({f'AFP{i:02d}':np.floor(prices[:,i]*100+.5) for i in range(79)})
    with np.errstate(all='ignore'):
        return np.column_stack([eval(e,{'__builtins__':{}},env) for e in NEW_EXPRESSIONS.values()])


def native_value(prices,atr,outside=1000000.,window_clocks=None):
    clocks=[1300]+(WINDOW_CLOCKS if window_clocks is None else list(window_clocks))+[1449,1450]
    env=dict(C=np.r_[outside,prices,outside,outside],VP20=float(atr),DATE=np.full(len(clocks),20240101),
        TIME=np.array(clocks),IF=np.where,ABS=np.abs,ROUND=lambda x:np.floor(x+.5),DRAWNULL=np.nan,
        REF=lambda x,n:pd.Series(x).shift(int(n)).to_numpy(),
        COUNT=lambda x,n:pd.Series(np.asarray(x,float)).rolling(int(n),min_periods=int(n)).sum().to_numpy(),
        VALUEWHEN=lambda mask,value:np.asarray(value)[np.flatnonzero(mask)[-1]] if np.ndim(value) else value)
    for line in EXTRA_HEADER.splitlines():
        name,expr=line.rstrip(';').split(':=');expr=re.sub(r'(?<![<>=!])=(?!=)','==',expr)
        if name=='AFGD':expr='VALUEWHEN(TIME==1420,COUNT((C>0) & (ABS(C*100-ROUND(C*100))<=0.01),80))'
        elif name=='AFREADY':expr=expr.replace('AFDT==DATE','AFDT==DATE[-2]').replace(' AND ',' and ')
        env[name]=eval(expr,{'__builtins__':{}},env)
    return np.asarray([float(eval(e,{'__builtins__':{}},env)) for e in NEW_EXPRESSIONS.values()])


def prepare():
    p=checked();path=INPUTS / 'feature_report.json';assert not path.exists()
    qr=json.loads((INPUTS / 'quote_report.json').read_text());assert qr['passed'] and qr['protocol_sha256']==sha(PROTOCOL)
    old=original_source.original();index=old.set_index(['date','code'],drop=False);parts=[];invalid=0;valid_count=0;checks=0
    for i,item in enumerate(qr['parts']):
        file=Path(item['file']);assert sha(file)==item['sha256'];q=pd.read_parquet(file)
        f=index.loc[pd.MultiIndex.from_frame(q[['date','code']])].reset_index(drop=True)
        pd.testing.assert_frame_equal(f[['date','code']],q[['date','code']],check_exact=True)
        prices=q[COLUMNS].to_numpy(float);good=q.af_bars.eq(80)&q.af_clocks.eq(80)&q.af_good.eq(80)&f.V01.gt(0)&np.isfinite(f.V01)&np.isfinite(prices).all(axis=1)
        values=measure(prices,f.V01.to_numpy(float));values[~good]=np.nan
        fresh=pd.DataFrame(values,columns=list(NEW_EXPRESSIONS));out=pd.concat([f,fresh],axis=1)
        out['prior_formula_input_valid']=f.formula_input_valid;out['afternoon_prefix_valid']=good
        out['formula_input_valid']=f.formula_input_valid & good
        c=base.conn();c.register('q',q);c.register('old',f[['date','code','V01','formula_input_valid']])
        expr=','.join(f'CASE WHEN valid THEN 100*(round({name}*100)/round(af_c79*100)-1)/V01 END AS AF{j:02d}' for j,name in enumerate(COLUMNS[:79]))
        expected=c.sql(f'''WITH a AS(SELECT *,coalesce(af_bars=80 AND af_clocks=80 AND af_good=80 AND isfinite(V01) AND V01>0,false) AS valid FROM q JOIN old USING(date,code)) SELECT date,code,valid,{expr} FROM a ORDER BY date,code''').df();c.close()
        np.testing.assert_array_equal(good,expected.valid)
        native=literal(prices,f.V01.to_numpy(float),good.to_numpy())
        for j,name in enumerate(NEW_EXPRESSIONS):
            for other in [expected[name].to_numpy(float),native[:,j]]:
                np.testing.assert_allclose(values[:,j],other,rtol=0,atol=2e-11,equal_nan=True)
                finite=np.isfinite(other);encode=lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999))
                np.testing.assert_array_equal(encode(values[finite,j]),encode(other[finite]))
            checks+=len(f)
        target=INPUTS / 'feature_parts' / f'part_{i:03d}.parquet';target.parent.mkdir(exist_ok=True)
        receipt_file=target.with_suffix('.json')
        identity=dict(protocol_sha256=sha(PROTOCOL),extractor_sha256=sha(Path(__file__)),quote_part_sha256=item['sha256'])
        if receipt_file.exists():
            receipt=json.loads(receipt_file.read_text())
            assert all(receipt[k]==v for k,v in identity.items()) and receipt['sha256']==sha(target)
            pd.testing.assert_frame_equal(pd.read_parquet(target),out,check_exact=True)
        else:
            assert not target.exists(), 'Investigate an unreceipted feature fragment'
            out.to_parquet(target,index=False,compression='zstd')
            save_json(receipt_file,dict(**identity,sha256=sha(target),rows=len(out)))
        parts.append(str(target))
        invalid+=int((f.formula_input_valid & ~good).sum());valid_count+=int(out.formula_input_valid.sum())
        print(json.dumps(dict(projected=i+1,total=len(qr['parts']),newly_invalid=invalid)),flush=True)
    c=base.conn();c.read_parquet(parts).create_view('parts');c.register('old',old)
    columns=[*META,*CONTROL];rebuilt=c.sql('SELECT '+','.join(columns)+' FROM parts ORDER BY date,code').df()
    pd.testing.assert_frame_equal(rebuilt[[*META[:-1],*CONTROL]],old[[*META[:-1],*CONTROL]],check_exact=True)
    # Domain reductions must be separately audited before fitting controls.
    save_json(INPUTS / 'input_domain_audit.json',dict(passed=invalid==0,original_valid=1602413,
        new_valid=valid_count,newly_invalid=invalid,all_original_50_values_and_metadata_exact=True,
        stop_new_fits_if_domain_changes=True,new_2026_prices_read=False))
    assert invalid==0 and valid_count==1602413
    pd.testing.assert_series_equal(rebuilt.formula_input_valid,old.formula_input_valid,check_names=True)
    c.execute("COPY (SELECT * FROM parts ORDER BY date,code) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",[str(INPUTS / 'features.parquet')]);c.close();del rebuilt
    for name in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        (INPUTS/name).symlink_to((prior.INPUTS/name).resolve())
    helpers=INPUTS / 'prefix_inputs.tdx';helpers.write_text(EXTRA_HEADER+'\n'.join(f'{k}:={v};' for k,v in NEW_EXPRESSIONS.items())+'\n')
    report=dict(protocol_sha256=sha(PROTOCOL),quote_report_sha256=sha(INPUTS / 'quote_report.json'),
        features_sha256=sha(INPUTS / 'features.parquet'),feature_part_hashes={s:sha(Path(s)) for s in parts},
        rows=len(old),valid=valid_count,newly_invalid=invalid,expressions=EXPRESSIONS,native_header=HEADER,
        scalar_path_and_encoding_checks=checks,no_current_or_future_whole_day_values_used=True,
        software_compilation_verified=False,native_source_parity_verified=False,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(path,report)
    proof=dict(passed=True,feature_report_sha256=sha(path),all_original_50_values_and_metadata_exact=True,
        all_79_SQL_native_arithmetic_and_integer_encodings_equal=True,effective_input_intersection_unchanged=True,
        all_80_clocks_prices_and_bad_positions_independent_pivot_verified=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS / 'feature_verification.json',proof)
    probes=[];lookup=json.loads(Path(p['source_metadata_audit']).read_text());sources={s['code']:s for s in lookup['files']}
    for item in p['native_probe_keys']:
        date,code=item['date'],item['code'];row=index.loc[(date,code)];source=Path(sources[code]['file'])
        assert sha(source)==sources[code]['expected_sha256']
        bars=pd.read_parquet(source,columns=['timestamp','close'],filters=[('timestamp','>=',pd.Timestamp(date+' 13:01')),('timestamp','<=',pd.Timestamp(date+' 14:20'))]).sort_values('timestamp')
        assert len(bars)==80 and not bars.timestamp.duplicated().any()
        actual=native_value(bars.close.to_numpy(),row.V01);expected=measure(bars.close.to_numpy()[None,:],np.array([row.V01]))[0]
        np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-11)
        saved=pd.read_parquet(INPUTS / 'features.parquet',columns=list(NEW_EXPRESSIONS),filters=[('date','=',date),('code','=',code)])
        assert len(saved)==1
        np.testing.assert_allclose(actual,saved.iloc[0].to_numpy(float),rtol=0,atol=2e-11)
        np.testing.assert_array_equal(actual,native_value(bars.close.to_numpy(),row.V01,.01))
        probes.append(dict(**item,raw_bars=80,source_sha256=sources[code]['expected_sha256']))
    save_json(INPUTS / 'native_input_verification.json',dict(passed=True,feature_report_sha256=sha(path),
        feature_verification_sha256=sha(INPUTS / 'feature_verification.json'),helper_sha256=sha(helpers),
        samples=probes,scalar_checks=checks,all_actual_extra_native_79_expressions_and_encodings_replayed=True,
        raw_80_clock_header_rounding_guard_and_future_exclusion_verified=True,
        software_compilation_verified=False,native_source_parity_verified=False,new_2026_prices_read=False,no_exit_rules=True))
    return dict(feature_report_sha256=sha(path),rows=len(old),valid=valid_count,newly_invalid=invalid)
