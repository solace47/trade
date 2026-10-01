"""The entire pre-tail afternoon volume distribution, including real zero activity."""
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_afternoon_prefix as clocks
from . import tail_formula_morning_range as prior
from . import tail_formula_volume_memory as original_source
from .corporate_cash import save_json,sha

STEM='tail_formula_afternoon_volume_profile'
ROOT=Path('data/research')/STEM
INPUTS=ROOT/'inputs'
PROTOCOL=Path('config')/(STEM+'_input_protocol.json')
INTENT=Path('config')/(STEM+'_intent.json')
META,CONTROL=prior.META,prior.EXPRESSIONS
COLUMNS=[f'af_v{i:02d}' for i in range(80)]
EXTRA_HEADER='AVDT:=VALUEWHEN(TIME=1420,DATE);\n'+f'AVCLK:=VALUEWHEN(TIME=1420,{clocks.SEQUENCE_GUARD});\n'+'''AVGOOD:=VALUEWHEN(TIME=1420,COUNT(V>=0 AND V-V=0,80));
AVTOTAL:=VALUEWHEN(TIME=1420,SUM(V,80));
AVREADY:=AVDT=DATE AND AVCLK=80 AND AVGOOD=80 AND AVTOTAL>=0;
'''+ '\n'.join(f'AVP{i:02d}:=VALUEWHEN(TIME=1420,REF(V,{79-i}));' for i in range(80))+'\n'
NEW_EXPRESSIONS={f'AV{i:02d}':f'IF(AVREADY,IF(AVTOTAL>0,100*AVP{i:02d}/AVTOTAL,0),DRAWNULL)' for i in range(80)}
EXPRESSIONS={**CONTROL,**NEW_EXPRESSIONS}
HEADER=prior.HEADER+EXTRA_HEADER


def checked():
    p=json.loads(PROTOCOL.read_text())
    assert p['intent_sha256']==sha(INTENT) and p['arms']=={'control':CONTROL,'memory':EXPRESSIONS}
    assert p['native_header']==HEADER and p['expected_keys']==1815129
    assert p['window_labels']==80 and p['maximum_new_fits']==4 and not p['new_2026_prices_allowed']
    for file,digest in p['source_hashes'].items():assert sha(Path(file))==digest,file
    g=json.loads(Path(p['conditional_gate']).read_text());assert g['passed'] and not g['supports_further_validation']
    text=subprocess.run(['git','show','HEAD:docs/selection-formula.md'],text=True,capture_output=True,check=True).stdout
    assert sha(PROTOCOL) in text and sha(INTENT) in text
    return p


def raw():
    p=checked();audit=json.loads(Path(p['source_metadata_audit']).read_text())
    report=INPUTS/'volume_report.json';assert not report.exists();records=[]
    lookup=json.loads(Path(p['original_version_audit']).read_text());sources={s['code']:s for s in lookup['files']}
    for item in audit['source_jobs']:
        parent=Path(item['receipt']);assert sha(parent)==item['receipt_sha256']
        old=json.loads(parent.read_text());job=old['job'];keys=pd.read_parquet(job['keys_file']).sort_values(['date','code']).reset_index(drop=True)
        assert sha(Path(job['keys_file']))==job['keys_sha256']==item['keys_sha256']
        file=INPUTS/'volume_parts'/(job['name']+'.parquet');file.parent.mkdir(exist_ok=True)
        reference_files={sources[code]['file']:sources[code]['expected_sha256'] for code in keys.code.unique()} if job['cached'] else {}
        meta=file.with_suffix('.json');identity=dict(protocol_sha256=sha(PROTOCOL),extractor_sha256=sha(Path(__file__)),parent_receipt_sha256=sha(parent),job=job,cached_reference_files=reference_files)
        if meta.exists():
            receipt=json.loads(meta.read_text());assert all(receipt[k]==v for k,v in identity.items()) and receipt['sha256']==sha(file)
        else:
            assert not file.exists(),'Investigate an unreceipted volume fragment'
            for source,digest in job['files'].items():assert sha(Path(source))==digest,source
            c=base.conn();c.read_parquet(list(job['files'])).create_view('source');c.register('keys',keys)
            body=f"SELECT date,code,{job['timestamp']} AS timestamp,volume::DOUBLE AS volume FROM source" if job['cached'] else "SELECT strftime(timestamp,'%Y-%m-%d') AS date,lower(exchange)||'.'||symbol AS code,timestamp,volume::DOUBLE AS volume FROM source WHERE timestamp>=TIMESTAMP '2023-01-01' AND timestamp<TIMESTAMP '2026-01-01'"
            d=c.sql(f'''WITH r AS({body}) SELECT r.* FROM r JOIN keys USING(date,code)
                WHERE timestamp>=TIMESTAMP '2023-01-01' AND timestamp<TIMESTAMP '2026-01-01'
                AND strftime(timestamp,'%H%M') BETWEEN '1301' AND '1420' ORDER BY date,code,timestamp''').df()
            # All clocks were proved on the same bytes by the preceding study.
            # A changed/missing/duplicate clock stops extraction rather than
            # inventing rows; bad volume values are retained in the wide frame.
            assert len(d)==len(keys)*80
            pd.testing.assert_frame_equal(d[['date','code']].iloc[::80].reset_index(drop=True),keys,check_exact=True)
            np.testing.assert_array_equal(d.timestamp.dt.hour*100+d.timestamp.dt.minute,np.tile(clocks.WINDOW_CLOCKS,len(keys)))
            assert d.timestamp.eq(d.timestamp.dt.floor('min')).all() and d.timestamp.dt.strftime('%Y-%m-%d').eq(d.date).all()
            if job['cached']:
                for source,digest in reference_files.items():assert sha(Path(source))==digest,source
                c.read_parquet(list(reference_files)).create_view('reference_source')
                reference=c.sql('''WITH r AS(SELECT strftime(timestamp,'%Y-%m-%d') AS date,
                    lower(exchange)||'.'||symbol AS code,timestamp,volume::DOUBLE AS volume FROM reference_source
                    WHERE timestamp>=TIMESTAMP '2023-01-01' AND timestamp<TIMESTAMP '2026-01-01')
                    SELECT r.* FROM r JOIN keys USING(date,code) WHERE strftime(timestamp,'%H%M') BETWEEN '1301' AND '1420'
                    ORDER BY date,code,timestamp''').df()
                pd.testing.assert_frame_equal(reference,d,check_exact=True)
            expected=keys.copy();v=d.volume.to_numpy(float).reshape(len(keys),80)
            expected['volume_bars']=80;expected['volume_clocks']=80;expected['volume_good']=(np.isfinite(v)&(v>=0)).sum(axis=1)
            expected=pd.concat([expected,pd.DataFrame(v,columns=COLUMNS)],axis=1)
            c.register('d',d)
            fields=','.join(f'max(volume) FILTER(WHERE date_diff(\'minute\',date_trunc(\'day\',timestamp),timestamp)-781={i}) AS {name}' for i,name in enumerate(COLUMNS))
            result=c.sql(f'''SELECT date,code,count(*)::BIGINT AS volume_bars,
                count(DISTINCT strftime(timestamp,'%H%M'))::BIGINT AS volume_clocks,
                sum(coalesce(isfinite(volume) AND volume>=0,false)::INT)::BIGINT AS volume_good,{fields}
                FROM d GROUP BY date,code ORDER BY date,code''').df();c.close()
            pd.testing.assert_frame_equal(result,expected,check_exact=True,check_dtype=False)
            result.to_parquet(file,index=False,compression='zstd')
            receipt=dict(**identity,sha256=sha(file),rows=len(result),raw_bars=len(d),all_80_volume_values_and_clocks_independent_reshape_equal=True,
                cached_volumes_all_equal_raw_reference=job['cached'])
            save_json(meta,receipt)
        records.append(dict(file=str(file),sha256=receipt['sha256'],rows=receipt['rows'],raw_bars=receipt['raw_bars'],cached=job['cached'],receipt_sha256=sha(meta)))
        print(json.dumps(dict(volumes=job['name'],completed=len(records),total=len(audit['source_jobs']))),flush=True)
    assert sum(i['rows'] for i in records)==1815129
    save_json(report,dict(passed=True,protocol_sha256=sha(PROTOCOL),source_metadata_audit_sha256=sha(Path(p['source_metadata_audit'])),
        parts=records,rows=1815129,raw_bars=sum(i['raw_bars'] for i in records),reused_keys=sum(i['rows'] for i in records if i['cached']),
        all_source_versions_rechecked=True,only_uncached_volume_windows_extracted=True,zero_bad_volume_positions_not_skipped=True,
        new_2026_prices_or_volume_read=False,no_exit_rules=True))
    return dict(volume_report_sha256=sha(report),rows=1815129,parts=len(records))


def measure(volumes):
    volumes=np.asarray(volumes,float);assert volumes.ndim==2 and volumes.shape[1]==80
    total=volumes.sum(axis=1)
    with np.errstate(all='ignore'):return np.where(total[:,None]>0,100*volumes/total[:,None],0.)


def literal(volumes,good):
    env=dict(AVTOTAL=volumes.sum(axis=1),AVREADY=good,IF=np.where,DRAWNULL=np.nan)
    env.update({f'AVP{i:02d}':volumes[:,i] for i in range(80)})
    with np.errstate(all='ignore'):return np.column_stack([eval(e,{'__builtins__':{}},env) for e in NEW_EXPRESSIONS.values()])


def native_value(volumes,outside=1000000.,window_clocks=None):
    times=[1300]+(clocks.WINDOW_CLOCKS if window_clocks is None else list(window_clocks))+[1449,1450]
    env=dict(V=np.r_[outside,volumes,outside,outside],TIME=np.array(times),DATE=np.full(len(times),20240101),
        IF=np.where,DRAWNULL=np.nan,REF=lambda x,n:pd.Series(x).shift(int(n)).to_numpy(),
        SUM=lambda x,n:pd.Series(np.asarray(x,float)).rolling(int(n),min_periods=int(n)).sum().to_numpy(),
        COUNT=lambda x,n:pd.Series(np.asarray(x,float)).rolling(int(n),min_periods=int(n)).sum().to_numpy(),
        VALUEWHEN=lambda mask,value:np.asarray(value)[np.flatnonzero(mask)[-1]] if np.ndim(value) else value)
    with np.errstate(all='ignore'):
        for line in EXTRA_HEADER.splitlines():
            name,e=line.rstrip(';').split(':=');e=re.sub(r'(?<![<>=!])=(?!=)','==',e)
            if name=='AVGOOD':e='VALUEWHEN(TIME==1420,COUNT((V>=0) & ((V-V)==0),80))'
            elif name=='AVREADY':e=e.replace('AVDT==DATE','AVDT==DATE[-2]').replace(' AND ',' and ')
            env[name]=eval(e,{'__builtins__':{}},env)
        return np.asarray([float(eval(e,{'__builtins__':{}},env)) for e in NEW_EXPRESSIONS.values()])


def prepare():
    p=checked();path=INPUTS/'feature_report.json';assert not path.exists()
    report=json.loads((INPUTS/'volume_report.json').read_text());assert report['passed'] and report['protocol_sha256']==sha(PROTOCOL)
    old=original_source.original();index=old.set_index(['date','code'],drop=False)
    parts=[];invalid=0;valid_count=0;checks=0;zero_windows=0
    for i,item in enumerate(report['parts']):
        source=Path(item['file']);assert sha(source)==item['sha256'];q=pd.read_parquet(source)
        f=index.loc[pd.MultiIndex.from_frame(q[['date','code']])].reset_index(drop=True)
        pd.testing.assert_frame_equal(f[['date','code']],q[['date','code']],check_exact=True)
        v=q[COLUMNS].to_numpy(float);good=q.volume_bars.eq(80)&q.volume_clocks.eq(80)&q.volume_good.eq(80)
        values=measure(v);values[~good]=np.nan
        out=pd.concat([f,pd.DataFrame(values,columns=list(NEW_EXPRESSIONS))],axis=1)
        out['prior_formula_input_valid']=f.formula_input_valid;out['volume_profile_valid']=good
        out['formula_input_valid']=f.formula_input_valid & good
        c=base.conn();c.register('q',q)
        expr=','.join(f'CASE WHEN valid AND total>0 THEN 100*{name}/total WHEN valid THEN 0. END AS AV{j:02d}' for j,name in enumerate(COLUMNS))
        expected=c.sql(f'''WITH a AS(SELECT *,volume_bars=80 AND volume_clocks=80 AND volume_good=80 AS valid,
            {'+'.join(COLUMNS)} AS total FROM q) SELECT date,code,valid,{expr} FROM a ORDER BY date,code''').df();c.close()
        np.testing.assert_array_equal(good,expected.valid);native=literal(v,good.to_numpy())
        for j,name in enumerate(NEW_EXPRESSIONS):
            for other in [expected[name].to_numpy(float),native[:,j]]:
                np.testing.assert_allclose(values[:,j],other,rtol=0,atol=2e-11,equal_nan=True)
                finite=np.isfinite(other);encode=lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999))
                np.testing.assert_array_equal(encode(values[finite,j]),encode(other[finite]))
            checks+=len(f)
        zero_windows+=int((good & (v.sum(axis=1)==0)).sum())
        target=INPUTS/'feature_parts'/f'part_{i:03d}.parquet';target.parent.mkdir(exist_ok=True)
        receipt_file=target.with_suffix('.json');identity=dict(protocol_sha256=sha(PROTOCOL),extractor_sha256=sha(Path(__file__)),volume_part_sha256=item['sha256'])
        if receipt_file.exists():
            receipt=json.loads(receipt_file.read_text());assert all(receipt[k]==v for k,v in identity.items()) and receipt['sha256']==sha(target)
            pd.testing.assert_frame_equal(pd.read_parquet(target),out,check_exact=True)
        else:
            assert not target.exists(),'Investigate an unreceipted profile fragment'
            out.to_parquet(target,index=False,compression='zstd');save_json(receipt_file,dict(**identity,sha256=sha(target),rows=len(out)))
        parts.append(str(target));invalid+=int((f.formula_input_valid & ~good).sum());valid_count+=int(out.formula_input_valid.sum())
        print(json.dumps(dict(projected=i+1,total=len(report['parts']),newly_invalid=invalid)),flush=True)
    c=base.conn();c.read_parquet(parts).create_view('parts')
    rebuilt=c.sql('SELECT '+','.join([*META,*CONTROL])+' FROM parts ORDER BY date,code').df()
    pd.testing.assert_frame_equal(rebuilt[[*META[:-1],*CONTROL]],old[[*META[:-1],*CONTROL]],check_exact=True)
    save_json(INPUTS/'input_domain_audit.json',dict(passed=invalid==0,original_valid=1602413,new_valid=valid_count,
        newly_invalid=invalid,all_original_50_values_and_metadata_exact=True,stop_new_fits_if_domain_changes=True,new_2026_prices_read=False))
    assert invalid==0 and valid_count==1602413
    pd.testing.assert_series_equal(rebuilt.formula_input_valid,old.formula_input_valid,check_names=True)
    c.execute("COPY (SELECT * FROM parts ORDER BY date,code) TO ? (FORMAT PARQUET, COMPRESSION ZSTD)",[str(INPUTS/'features.parquet')]);c.close();del rebuilt
    for name in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:(INPUTS/name).symlink_to((prior.INPUTS/name).resolve())
    helper=INPUTS/'volume_inputs.tdx';helper.write_text(EXTRA_HEADER+'\n'.join(f'{k}:={v};' for k,v in NEW_EXPRESSIONS.items())+'\n')
    save_json(path,dict(protocol_sha256=sha(PROTOCOL),volume_report_sha256=sha(INPUTS/'volume_report.json'),
        features_sha256=sha(INPUTS/'features.parquet'),feature_part_hashes={f:sha(Path(f)) for f in parts},
        rows=len(old),valid=valid_count,newly_invalid=invalid,known_zero_volume_windows=zero_windows,
        expressions=EXPRESSIONS,native_header=HEADER,scalar_volume_profile_and_encoding_checks=checks,
        no_failed_price_path_appended=True,software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True))
    save_json(INPUTS/'feature_verification.json',dict(passed=True,feature_report_sha256=sha(path),
        all_original_50_values_and_metadata_exact=True,all_80_SQL_native_arithmetic_and_integer_encodings_equal=True,
        effective_input_intersection_unchanged=True,all_real_zero_activity_not_missing_distinguished=True,
        new_2026_prices_read=False,no_exit_rules=True))
    lookup=json.loads(Path(p['original_version_audit']).read_text());sources={s['code']:s for s in lookup['files']};probes=[]
    for item in p['native_probe_keys']:
        date,code=item['date'],item['code'];source=Path(sources[code]['file']);assert sha(source)==sources[code]['expected_sha256']
        bars=pd.read_parquet(source,columns=['timestamp','volume'],filters=[('timestamp','>=',pd.Timestamp(date+' 13:01')),('timestamp','<=',pd.Timestamp(date+' 14:20'))]).sort_values('timestamp')
        assert len(bars)==80 and not bars.timestamp.duplicated().any()
        actual=native_value(bars.volume.to_numpy());expected=measure(bars.volume.to_numpy()[None,:])[0]
        np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-11)
        saved=pd.read_parquet(INPUTS/'features.parquet',columns=list(NEW_EXPRESSIONS),filters=[('date','=',date),('code','=',code)])
        assert len(saved)==1;np.testing.assert_allclose(actual,saved.iloc[0].to_numpy(float),rtol=0,atol=2e-11)
        np.testing.assert_array_equal(actual,native_value(bars.volume.to_numpy(),.01))
        scaled=native_value(5*bars.volume.to_numpy());np.testing.assert_allclose(actual,scaled,rtol=0,atol=2e-11)
        np.testing.assert_array_equal(np.floor(100*actual+10000+.000001),np.floor(100*scaled+10000+.000001))
        probes.append(dict(**item,raw_bars=80,source_sha256=sources[code]['expected_sha256']))
    save_json(INPUTS/'native_input_verification.json',dict(passed=True,feature_report_sha256=sha(path),
        feature_verification_sha256=sha(INPUTS/'feature_verification.json'),helper_sha256=sha(helper),samples=probes,
        scalar_checks=checks,all_actual_80_native_expressions_and_encodings_replayed=True,
        raw_80_clock_header_zero_activity_unit_scaling_and_future_exclusion_verified=True,
        software_compilation_verified=False,native_source_parity_verified=False,new_2026_prices_read=False,no_exit_rules=True))
    return dict(feature_report_sha256=sha(path),rows=len(old),valid=valid_count,newly_invalid=invalid,known_zero_volume_windows=zero_windows)
