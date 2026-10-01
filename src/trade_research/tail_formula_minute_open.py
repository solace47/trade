"""Separate first-price changes from within-minute changes on active pairs."""
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from . import tail_formula_baseline as prior
from . import tail_formula_additive as base
from .research_io import save_json,sha,check_runtime,check_sources

STEM='tail_formula_minute_open'
ROOT=Path('data/research')/STEM
INPUTS=ROOT/'inputs'
PROTOCOL=Path('config')/(STEM+'_input_protocol.json')
INTENT=Path('config')/(STEM+'_intent.json')
META,CONTROL=prior.META,prior.CONTROL
FIELDS={k:[f'mo_{k}{i:02d}' for i in range(20,50)] for k in 'ohlcv'}
RAW_FIELDS=dict(o='open',h='high',l='low',c='close',v='volume')
NEW_EXPRESSIONS={
    'MOGAP':'IF(MOREADY,IF(MOPV>0,(MOPN*MOGV/MOPV-MOGS)/VP20,0),DRAWNULL)',
    'MOBODY':'IF(MOREADY,IF(MOPV>0,(MOPN*MOBV/MOPV-MOBS)/VP20,0),DRAWNULL)'}
EXPRESSIONS={**CONTROL,**NEW_EXPRESSIONS}
EXTRA_HEADER='''MOPA:=V>0 AND REF(V,1)>0;
MOG:=IF(MOPA,100*LN(O/REF(C,1)),0);
MOB:=IF(MOPA,100*LN(C/O),0);
MOPV:=VALUEWHEN(TIME=1449,SUM(IF(MOPA,V,0),29));
MOPN:=VALUEWHEN(TIME=1449,COUNT(MOPA,29));
MOGV:=VALUEWHEN(TIME=1449,SUM(MOG*V,29));
MOBV:=VALUEWHEN(TIME=1449,SUM(MOB*V,29));
MOGS:=VALUEWHEN(TIME=1449,SUM(MOG,29));
MOBS:=VALUEWHEN(TIME=1449,SUM(MOB,29));
MOOGOOD:=VALUEWHEN(TIME=1449,COUNT(NOT(MOPA) OR (O>0 AND O-O=0 AND H>=O AND O>=L AND ABS(O*100-ROUND(O*100))<=0.01),29));
MOPGOOD:=VALUEWHEN(TIME=1449,COUNT(H>=C AND C>=L AND L>0 AND H-H=0 AND L-L=0 AND C-C=0 AND V>=0 AND V-V=0 AND ABS(H*100-ROUND(H*100))<=0.01 AND ABS(L*100-ROUND(L*100))<=0.01 AND ABS(C*100-ROUND(C*100))<=0.01,30));
MOREADY:=RTCLK AND VALUEWHEN(TIME=1449,DATE)=DATE AND VALUEWHEN(TIME=1449,REF(DATE,29))=DATE AND MOOGOOD=29 AND MOPGOOD=30 AND VP20>0;
'''
HEADER=prior.HEADER+EXTRA_HEADER
NATIVE_GATE='MOREADY'


def checked():
    check_runtime();p=json.loads(PROTOCOL.read_text())
    assert subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}'])==PROTOCOL.read_bytes()
    assert p['intent_sha256']==sha(INTENT) and p['arms']=={'control':CONTROL,'memory':EXPRESSIONS}
    assert p['native_header']==HEADER and p['expected_keys']==1815129 and p['expected_original_valid']==1602413
    assert p['maximum_new_fits']==4 and p['newly_invalid_allowed']==0 and not p['new_2026_prices_allowed']
    check_sources(p['source_hashes'])
    inventory=json.loads(Path(p['source_inventory']).read_text())
    assert inventory['new_values_read'] is False
    p.update(inventory)
    g=json.loads(Path(p['conditional_gate']).read_text())
    assert g['passed'] and not g['supports_further_validation']
    return p


def measure(opens,closes,volumes,scale):
    o,c,v=[np.asarray(x,float) for x in [opens,closes,volumes]]
    assert o.shape==c.shape==v.shape and o.shape[1]==30
    o,c=[np.floor(x*100+.5)/100 for x in [o,c]]
    pair=(v[:,1:]>0)&(v[:,:-1]>0)
    with np.errstate(all='ignore'):
        gap=np.where(pair,100*np.log(o[:,1:]/c[:,:-1]),0)
        body=np.where(pair,100*np.log(c[:,1:]/o[:,1:]),0)
    weight=np.where(pair,v[:,1:],0);total=weight.sum(axis=1);n=pair.sum(axis=1)
    values=[]
    for x in [gap,body]:
        weighted=np.divide((weight*x).sum(axis=1),total,out=np.zeros(len(o)),where=total>0)
        values.append((n*weighted-x.sum(axis=1))/np.asarray(scale))
    return np.column_stack(values),pair


def transform(raw,old):
    d=old[['date','code','formula_input_valid','A04','V01','NA05']].merge(raw,on=['date','code'],how='left',validate='one_to_one')
    o,h,l,c,v=[d[FIELDS[k]].to_numpy(float) for k in 'ohlcv']
    good=d.mo_bars.eq(30)&d.mo_clocks.eq(30)&d.mo_good_bars.eq(30)
    for x in [h,l,c]:
        good&=np.isfinite(x).all(axis=1)&(x>0).all(axis=1)&(abs(x*100-np.floor(x*100+.5))<=.01).all(axis=1)
    good&=(h>=c).all(axis=1)&(c>=l).all(axis=1)&np.isfinite(v).all(axis=1)&(v>=0).all(axis=1)&(v==np.floor(v)).all(axis=1)
    values,pair=measure(o,c,v,d.V01.to_numpy())
    open_good=np.isfinite(o[:,1:])&(o[:,1:]>0)&(h[:,1:]>=o[:,1:])&(o[:,1:]>=l[:,1:])&(abs(o[:,1:]*100-np.floor(o[:,1:]*100+.5))<=.01)
    good&=(~pair|open_good).all(axis=1)&np.isfinite(values).all(axis=1)&np.isfinite(d.V01)&d.V01.gt(0)
    ok=old.formula_input_valid.to_numpy()
    np.testing.assert_array_equal(np.floor(c[ok,-1]*100+.5),np.floor(old.loc[ok,'A04']*100+.5))
    reconstructed=np.floor(100*old.A04/(1+old.NA05*old.V01/100)+.5)/100
    np.testing.assert_array_equal(np.floor(c[ok,0]*100+.5),np.floor(reconstructed[ok]*100+.5))
    # Independently build every active-pair log term in SQL.
    sql=base.conn();sql.register('d',d)
    expressions={name:[] for name in ['pair','volume','gap','body','gapv','bodyv']}
    for i in range(21,50):
        active=f'(mo_v{i}>0 AND mo_v{i-1}>0)'
        safe=f'isfinite(mo_o{i}) AND mo_o{i}>0 AND isfinite(mo_c{i-1}) AND mo_c{i-1}>0 AND isfinite(mo_c{i}) AND mo_c{i}>0'
        gap=f'CASE WHEN {active} AND {safe} THEN 100*ln(round(mo_o{i}*100)/round(mo_c{i-1}*100)) WHEN {active} THEN NULL ELSE 0 END'
        body=f'CASE WHEN {active} AND {safe} THEN 100*ln(round(mo_c{i}*100)/round(mo_o{i}*100)) WHEN {active} THEN NULL ELSE 0 END'
        expressions['pair'].append(f'({active})::INT');expressions['volume'].append(f'CASE WHEN {active} THEN mo_v{i} ELSE 0 END')
        for name,x in [('gap',gap),('body',body)]:
            expressions[name].append(x);expressions[name+'v'].append(f'({x})*mo_v{i}')
    aggregates=','.join('+'.join(parts)+' AS '+name for name,parts in expressions.items())
    expected=sql.sql('WITH a AS(SELECT date,code,V01,'+aggregates+' FROM d) SELECT date,code,'+
        'CASE WHEN volume>0 THEN (pair*gapv/volume-gap)/V01 ELSE 0 END AS MOGAP,'+
        'CASE WHEN volume>0 THEN (pair*bodyv/volume-body)/V01 ELSE 0 END AS MOBODY FROM a ORDER BY date,code').df();sql.close()
    np.testing.assert_allclose(values[good],expected.loc[good,list(NEW_EXPRESSIONS)],rtol=0,atol=2e-10)
    encode=lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999))
    np.testing.assert_array_equal(encode(values[good]),encode(expected.loc[good,list(NEW_EXPRESSIONS)].to_numpy()))
    values[~good]=np.nan
    out=d[['date','code']].copy();out['minute_open_valid']=good
    for i,name in enumerate(NEW_EXPRESSIONS):out[name]=values[:,i]
    return out


def native_values(o,c,v,scale):
    """Literal twenty-nine native terms with shares converted to lots."""
    o,c=[np.floor(x*100+.5)/100 for x in [o,c]]
    n=np.zeros(len(o));pv=n.copy();gs=n.copy();bs=n.copy();gv=n.copy();bv=n.copy()
    lots=v/100
    for i in range(1,30):
        pair=(lots[:,i]>0)&(lots[:,i-1]>0)
        g=np.zeros(len(o));b=g.copy()
        with np.errstate(all='ignore'):
            g[pair]=100*np.log(o[pair,i]/c[pair,i-1])
            b[pair]=100*np.log(c[pair,i]/o[pair,i])
        n+=pair;pv+=np.where(pair,lots[:,i],0)
        gs+=g;bs+=b;gv+=g*lots[:,i];bv+=b*lots[:,i]
    out=[]
    for sums,weighted in [(gs,gv),(bs,bv)]:
        avg=np.divide(weighted,pv,out=np.zeros(len(o)),where=pv>0)
        out.append((n*avg-sums)/scale)
    return np.column_stack(out)


def extract(p,old):
    folder=INPUTS/'parts';folder.mkdir(parents=True,exist_ok=True)
    codes=sorted(old.code.unique());pieces=[];receipts={}
    for offset in range(0,len(codes),64):
        subset=codes[offset:offset+64];keys=old.loc[old.code.isin(subset)]
        paths=[Path(p['minute_template'].format(exchange=x[:2].upper(),symbol=x[3:])) for x in subset]
        for path in paths:assert sha(path)==p['raw_sources'][str(path)],path
        file=folder/f'part_{offset//64:03d}.parquet';meta_file=file.with_suffix('.json')
        if meta_file.exists():
            meta=json.loads(meta_file.read_text())
            assert meta['protocol_sha256']==sha(PROTOCOL) and meta['implementation_sha256']==sha(Path(__file__))
            assert meta['codes']==subset and meta['sha256']==sha(file)
        else:
            sql=base.conn();sql.read_parquet([str(x) for x in paths]).create_view('raw')
            sql.register('keys',keys[['date','code']])
            pivots=','.join(f"max({name}) FILTER(WHERE clock='14{i:02d}') AS mo_{k}{i:02d}"
                for k,name in RAW_FIELDS.items() for i in range(20,50))
            quality=' AND '.join(f'isfinite({n}) AND {n}>0 AND abs({n}-round({n},2))<=.0001' for n in ['high','low','close'])
            data=sql.sql(f'''WITH s AS(SELECT lower(exchange)||'.'||symbol AS code,
                strftime(timestamp,'%Y-%m-%d') AS date,strftime(timestamp,'%H%M') AS clock,timestamp,
                open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,close::DOUBLE AS close,volume::DOUBLE AS volume
                FROM raw WHERE timestamp>=TIMESTAMP '2023-01-01' AND timestamp<TIMESTAMP '2026-01-01'
                AND strftime(timestamp,'%H%M') BETWEEN '1420' AND '1449'),
                t AS(SELECT s.*,coalesce(timestamp=date_trunc('minute',timestamp) AND {quality}
                    AND high+.0001>=greatest(close,low) AND low-.0001<=least(close,high)
                    AND isfinite(volume) AND volume>=0 AND volume=floor(volume),false) AS good
                    FROM s JOIN keys USING(date,code))
                SELECT date,code,count(*) AS mo_bars,count(DISTINCT clock) AS mo_clocks,
                    count(*) FILTER(WHERE good) AS mo_good_bars,{pivots}
                FROM t GROUP BY date,code ORDER BY date,code''').df();sql.close()
            data.to_parquet(file,index=False,compression='zstd')
            save_json(meta_file,dict(protocol_sha256=sha(PROTOCOL),implementation_sha256=sha(Path(__file__)),
                codes=subset,sha256=sha(file),rows=len(data),source_hashes={str(x):p['raw_sources'][str(x)] for x in paths},
                price_interval=['2023-01-01','2026-01-01'],minute_clock=['1420','1449'],no_2026_rows=True))
        raw=pd.read_parquet(file);receipts[str(file)]=sha(file)
        # Match previously frozen HLCV values for all original valid keys.
        cached=[]
        for path,spec in p['prior_window_parts'].items():
            selected=sorted(set(subset)&set(spec['codes']))
            if not selected:continue
            assert sha(Path(path))==spec['sha256']
            columns=['date','code',*[f'mp_{k}{i}' for k in 'hlcv' for i in range(21,50)]]
            cached.append(pd.read_parquet(path,columns=columns,filters=[('code','in',selected)]))
        previous=pd.concat(cached,ignore_index=True)
        wanted=keys.loc[keys.formula_input_valid,['date','code']]
        a=wanted.merge(raw,on=['date','code'],how='left',validate='one_to_one')
        b=wanted.merge(previous,on=['date','code'],how='left',validate='one_to_one')
        for k in 'hlcv':
            current=a[[f'mo_{k}{i}' for i in range(21,50)]].to_numpy(float)
            earlier=b[[f'mp_{k}{i}' for i in range(21,50)]].to_numpy(float)
            if k!='v':current,earlier=[np.floor(x*100+.5) for x in [current,earlier]]
            np.testing.assert_array_equal(current,earlier)
        piece=transform(raw,keys)
        joined=keys[['date','code','V01']].merge(raw,on=['date','code'],how='left',validate='one_to_one')
        o,c,v=[joined[FIELDS[k]].to_numpy(float) for k in 'ocv']
        known=piece.minute_open_valid.to_numpy()
        native=native_values(o[known],c[known],v[known],joined.loc[known,'V01'].to_numpy())
        expected=piece.loc[known,list(NEW_EXPRESSIONS)].to_numpy(float)
        np.testing.assert_allclose(native,expected,rtol=0,atol=2e-10)
        encode=lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999))
        np.testing.assert_array_equal(encode(native),encode(expected))
        pieces.append(piece)
        print(json.dumps(dict(source_codes=offset+len(subset),total_codes=len(codes))),flush=True)
    return pieces,receipts


def prepare():
    p=checked();assert not (INPUTS/'feature_report.json').exists();INPUTS.mkdir(parents=True,exist_ok=True)
    old=prior.original();parts,receipts=extract(p,old)
    values=pd.concat(parts).sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(values[['date','code']],old[['date','code']],check_exact=True)
    out=old.merge(values,on=['date','code'],validate='one_to_one')
    invalid=int((old.formula_input_valid&~out.minute_open_valid).sum())
    audit=dict(passed=invalid==0,rows=len(out),original_valid=int(old.formula_input_valid.sum()),
        new_valid=int((old.formula_input_valid&out.minute_open_valid).sum()),newly_invalid=invalid,
        stop_fit_if_domain_changes=True,no_new_group_economics=True,new_2026_prices_read=False)
    save_json(INPUTS/'input_domain_audit.json',audit)
    assert invalid==0,'Fix a matched-domain protocol before any fit'
    out['prior_formula_input_valid']=old.formula_input_valid;out['formula_input_valid']&=out.minute_open_valid
    pd.testing.assert_frame_equal(out[[*META,*CONTROL]],old,check_exact=True)
    out.to_parquet(INPUTS/'features.parquet',index=False,compression='zstd')
    for name in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        (INPUTS/name).symlink_to(Path('../../tail_formula_intraday_scale/inputs')/name)
    adapter=INPUTS/'minute_open_inputs.tdx';adapter.write_text(EXTRA_HEADER+''.join(f'{n}:={e};\n' for n,e in NEW_EXPRESSIONS.items()))
    report=dict(protocol_sha256=sha(PROTOCOL),implementation_sha256=sha(Path(__file__)),
        features_sha256=sha(INPUTS/'features.parquet'),rows=len(out),valid=int(out.formula_input_valid.sum()),newly_invalid=0,
        expressions=EXPRESSIONS,native_header=HEADER,source_hashes=p['source_hashes'],parts_sha256=receipts,
        adapter_sha256=sha(adapter),all_old_HLCV_cached_windows_cent_values_equal=True,
        all_raw_windows_direct_SQL_and_literal_native_encodings_equal=True,
        all_original_fifty_metadata_and_validity_unchanged=True,new_group_outcomes_read=False,new_2026_prices_read=False,
        no_exit_rules=True,software_compilation_verified=False,native_source_parity_verified=False)
    save_json(INPUTS/'feature_report.json',report)
    proof=dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),
        all_original_fifty_values_keys_domain_equal=True,all_new_values_SQL_native_encoding_equal=True,
        mathematical_replay_only=True,software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'feature_verification.json',proof);save_json(INPUTS/'native_input_verification.json',proof)
    return dict(feature_report_sha256=sha(INPUTS/'feature_report.json'),**audit)
