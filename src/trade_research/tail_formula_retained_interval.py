"""Maximum remaining separation of active three-bar endpoint ranges."""
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_baseline as prior
from .research_io import save_json, sha, check_sources, check_runtime

STEM = 'tail_formula_retained_interval'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
INTENT = Path('config') / (STEM + '_intent.json')
META, CONTROL = prior.META, prior.EXPRESSIONS
FIELDS = {k: [f'mp_{k}{i:02d}' for i in range(21, 50)] for k in 'hlcv'}
RAW_FIELDS = {'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'}
WINDOW_COLUMNS = ['date','code','mp_bars','mp_clocks','mp_good_bars',
                  *[n for names in FIELDS.values() for n in names]]

EXTRA_HEADER = '''RGHC:=INTPART(H*100+0.5);
RGLC:=INTPART(L*100+0.5);
RGGOOD:=VALUEWHEN(TIME=1449,COUNT(H>=C AND C>=L AND L>0 AND H-H=0 AND L-L=0 AND C-C=0 AND V>=0 AND V-V=0 AND ABS(H*100-RGHC)<=0.01 AND ABS(L*100-RGLC)<=0.01 AND ABS(C*100-INTPART(C*100+0.5))<=0.01,29));
RGDT:=VALUEWHEN(TIME=1449,DATE);
RGFIRST:=VALUEWHEN(TIME=1449,REF(DATE,28));
RGREADY:=(RGDT=DATE) AND (RGFIRST=DATE) AND RTCLK AND (RGGOOD=29) AND (RMC>0) AND (VP20>0);
'''
for k in range(27):
    refs = [f'REF(V,{j})' if j else 'V' for j in [k,k+1,k+2]]
    active = ' AND '.join(f'({v}>0)' for v in refs)
    EXTRA_HEADER += f'RGU{k:02d}:=IF({active},MAX(LLV(IF(V>0,RGLC,999999999),{k+1})-REF(RGHC,{k+2}),0),0);\n'
    EXTRA_HEADER += f'RGD{k:02d}:=IF({active},MAX(REF(RGLC,{k+2})-HHV(IF(V>0,RGHC,0),{k+1}),0),0);\n'
def nested_max(names):
    out = names[0]
    for name in names[1:]: out = f'MAX({out},{name})'
    return out

def extract_history(p, old):
    hist = old.loc[old.date.lt('2024-01-01')]
    codes = sorted(hist.code.unique()); parts = []; receipts = {}
    folder = INPUTS/'historical_parts'; folder.mkdir(parents=True,exist_ok=True)
    for offset in range(0,len(codes),64):
        subset = codes[offset:offset+64]
        paths = [Path('data/hf/pilot/data/stock_1m')/code[:2].upper()/(code[3:]+'.parquet') for code in subset]
        for source in paths:
            digest = p['historical_raw_sources'][str(source)]
            assert sha(source)==digest,source
            receipts[str(source)] = digest
        file = folder/f'part_{offset//64:03d}.parquet'; meta_file = file.with_suffix('.json')
        if meta_file.exists():
            meta = json.loads(meta_file.read_text())
            assert meta['protocol_sha256']==sha(PROTOCOL) and meta['implementation_sha256']==sha(Path(__file__))
            assert meta['codes']==subset and meta['sha256']==sha(file)
        else:
            sql = base.conn(); sql.read_parquet([str(f) for f in paths]).create_view('raw')
            sql.register('keys',hist.loc[hist.code.isin(subset),['date','code']])
            pivots = ','.join(f"max({name if k=='v' else f'round({name},2)'}) FILTER(WHERE clock='14{i:02d}') AS mp_{k}{i:02d}" for k,name in RAW_FIELDS.items() for i in range(21,50))
            good = ' AND '.join(f'isfinite({n}) AND {n}>0 AND abs({n}-round({n},2))<=.0001' for n in ['high','low','close'])
            raw = sql.sql(f'''WITH s AS(SELECT lower(exchange)||'.'||symbol AS code,
                strftime(timestamp,'%Y-%m-%d') AS date,strftime(timestamp,'%H%M') AS clock,
                timestamp,high::DOUBLE AS high,low::DOUBLE AS low,close::DOUBLE AS close,volume::DOUBLE AS volume
                FROM raw WHERE timestamp>=TIMESTAMP '2023-01-01' AND timestamp<TIMESTAMP '2024-01-01'
                AND strftime(timestamp,'%H%M') BETWEEN '1421' AND '1449'),
                t AS(SELECT s.*,coalesce(timestamp=date_trunc('minute',timestamp) AND {good}
                    AND round(high,2)>=round(close,2) AND round(close,2)>=round(low,2)
                    AND isfinite(volume) AND volume>=0 AND volume=floor(volume),false) AS good
                    FROM s JOIN keys USING(date,code))
                SELECT date,code,count(*) AS mp_bars,count(DISTINCT clock) AS mp_clocks,
                    count(*) FILTER(WHERE good) AS mp_good_bars,{pivots}
                FROM t GROUP BY date,code ORDER BY date,code''').df(); sql.close()
            raw.to_parquet(file,index=False,compression='zstd')
            save_json(meta_file,dict(protocol_sha256=sha(PROTOCOL),implementation_sha256=sha(Path(__file__)),
                codes=subset,sha256=sha(file),rows=len(raw),source_hashes={str(s):receipts[str(s)] for s in paths},
                only_2023_1421_1449_HLCV_read=True,new_2026_prices_read=False))
        raw = pd.read_parquet(file,columns=WINDOW_COLUMNS)
        part = transform(raw,hist.loc[hist.code.isin(subset)])
        parts.append(part)
        print(json.dumps(dict(historical_codes=offset+len(subset),total_codes=len(codes)),ensure_ascii=False),flush=True)
    return parts,receipts

def prepare():
    p = checked(); assert not (INPUTS/'feature_report.json').exists()
    INPUTS.mkdir(parents=True,exist_ok=True)
    old = prior.original(); pieces = []; receipts = {}; reused_rows = 0
    for file,digest in p['cached_window_parts'].items():
        file = Path(file); assert sha(file)==digest
        raw = pd.read_parquet(file,columns=WINDOW_COLUMNS)
        assert raw.date.ge('2024-01-01').all() and raw.date.lt('2026-01-01').all()
        codes = p['cached_window_codes'][str(file)]
        piece = transform(raw,old.loc[old.date.ge('2024-01-01') & old.code.isin(codes)])
        pieces.append(piece); receipts[str(file)] = digest; reused_rows += len(piece)
    assert reused_rows == 1258085
    print(json.dumps(dict(cached_current_rows=reused_rows,no_current_raw_reextraction=True)),flush=True)
    history,sources = extract_history(p,old); pieces += history
    values = pd.concat(pieces,ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(values[['date','code']],old[['date','code']],check_exact=True)
    out = old.merge(values,on=['date','code'],validate='one_to_one')
    invalid = int((old.formula_input_valid & ~out.interval_source_valid).sum())
    audit = dict(passed=invalid==0,rows=len(out),original_valid=int(old.formula_input_valid.sum()),
        new_valid=int((old.formula_input_valid & out.interval_source_valid).sum()),newly_invalid=invalid,
        stop_fit_if_domain_changes=True,new_group_outcomes_read=False,new_2026_prices_read=False)
    save_json(INPUTS/'input_domain_audit.json',audit)
    assert invalid==0, 'Freeze a matched-domain protocol before any fits'
    out['prior_formula_input_valid'] = old.formula_input_valid
    out['formula_input_valid'] &= out.interval_source_valid
    pd.testing.assert_frame_equal(out[[*META,*CONTROL]],old,check_exact=True)
    out.to_parquet(INPUTS/'features.parquet',index=False,compression='zstd')
    for name in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        (INPUTS/name).symlink_to((prior.INPUTS/name).resolve())
    adapter = INPUTS/'interval_inputs.tdx'
    adapter.write_text(EXTRA_HEADER+''.join(f'{n}:={e};\n' for n,e in NEW_EXPRESSIONS.items()))
    report = dict(protocol_sha256=sha(PROTOCOL),implementation_sha256=sha(Path(__file__)),
        features_sha256=sha(INPUTS/'features.parquet'),rows=len(out),valid=int(out.formula_input_valid.sum()),
        newly_invalid=invalid,expressions=EXPRESSIONS,native_header=HEADER,
        cached_window_parts=receipts,historical_raw_sources=sources,adapter_sha256=sha(adapter),
        all_window_math_native_terms_and_independent_SQL_equal=True,all_encodings_equal=True,
        no_current_raw_reextraction=True,current_source_fields_previously_exposed=True,
        new_group_outcomes_read=False,software_compilation_verified=False,native_source_parity_verified=False,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'feature_report.json',report)
    proof = dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),
        all_original_50_metadata_keys_and_domain_exact=True,
        all_scalar_widths_native_SQL_and_integer_encodings_rebuilt=True,
        unknown_and_zero_volume_semantics_verified=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'feature_verification.json',proof)
    save_json(INPUTS/'native_input_verification.json',dict(**proof,
        adapter_sha256=sha(adapter),full_native_terms_and_all_known_windows_equal=True,
        software_compilation_verified=False,native_source_parity_verified=False))
    return dict(feature_report_sha256=sha(INPUTS/'feature_report.json'),**audit)
EXTRA_HEADER += 'RGUPMAX:=VALUEWHEN(TIME=1449,' + nested_max([f'RGU{k:02d}' for k in range(27)]) + ');\n'
EXTRA_HEADER += 'RGDNMAX:=VALUEWHEN(TIME=1449,' + nested_max([f'RGD{k:02d}' for k in range(27)]) + ');\n'
NEW_EXPRESSIONS = {'RGUP':'IF(RGREADY,100*RGUPMAX/(RMC*VP20),DRAWNULL)',
                   'RGDOWN':'IF(RGREADY,100*RGDNMAX/(RMC*VP20),DRAWNULL)'}
EXPRESSIONS = {**CONTROL, **NEW_EXPRESSIONS}
HEADER = prior.HEADER + EXTRA_HEADER
NATIVE_GATE = 'RGREADY'

def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['intent_sha256'] == sha(INTENT)
    assert p['arms'] == {'control':CONTROL,'memory':EXPRESSIONS}
    assert p['native_header'] == HEADER and p['window_size'] == 29
    assert p['maximum_new_fits'] == 4 and not p['new_2026_prices_allowed']
    check_runtime(); check_sources(p['source_hashes'])
    gate = json.loads(Path(p['conditional_gate']).read_text())
    assert gate['passed'] and not gate['supports_further_validation']
    text = subprocess.run(['git','show','HEAD:docs/selection-formula.md'],text=True,capture_output=True,check=True).stdout
    assert sha(PROTOCOL) in text and sha(INTENT) in text
    p['historical_raw_sources'] = json.loads(Path(p['historical_raw_report']).read_text())['source_sha256']
    return p

def measure(high, low, volume):
    h,l,v = [np.asarray(x,float) for x in [high,low,volume]]
    assert h.shape == l.shape == v.shape and h.shape[1] == 29
    h,l = [np.floor(x*100+.5) for x in [h,l]]
    active = v > 0
    future_low = np.minimum.accumulate(np.where(active,l,999999999)[:,::-1],axis=1)[:,::-1]
    future_high = np.maximum.accumulate(np.where(active,h,0)[:,::-1],axis=1)[:,::-1]
    formation = active[:,:-2] & active[:,1:-1] & active[:,2:]
    up = np.where(formation,np.maximum(future_low[:,2:]-h[:,:-2],0),0)
    down = np.where(formation,np.maximum(l[:,:-2]-future_high[:,2:],0),0)
    return np.column_stack([up.max(axis=1),down.max(axis=1)])

def native_values(high, low, volume):
    """Execute the literal 27 native terms, including zero-volume masks."""
    h,l,v = [np.asarray(x,float) for x in [high,low,volume]]
    assert h.shape == l.shape == v.shape and h.shape[1] == 29
    env = dict(RGHC=np.floor(h*100+.5),RGLC=np.floor(l*100+.5),V=v,
        IF=np.where,MAX=np.maximum,
        REF=lambda x,n:x[:,28-int(n)],
        LLV=lambda x,n:np.min(x[:,-int(n):],axis=1),
        HHV=lambda x,n:np.max(x[:,-int(n):],axis=1))
    values = {'U':[],'D':[]}
    for line in EXTRA_HEADER.splitlines():
        if not re.match(r'RG[UD]\d{2}:=',line): continue
        name,expression = line.rstrip(';').split(':=',1)
        expression = expression.replace('IF((V>0) AND','IF((REF(V,0)>0) AND').replace(' AND ',' & ')
        # Unshifted V in formation denotes the current anchor, whereas IF(V>0,...)
        # inside LLV/HHV denotes its full loaded minute series.
        values[name[2]].append(eval(expression,{'__builtins__':{}},env))
    return np.column_stack([np.max(values[k],axis=0) for k in ['U','D']])

def transform(raw, old):
    d = old[['date','code','formula_input_valid','A04','V01']].merge(raw,on=['date','code'],how='left',validate='one_to_one')
    h,l,c,v = [d[FIELDS[k]].to_numpy(float) for k in 'hlcv']
    good = d.mp_bars.eq(29) & d.mp_clocks.eq(29) & d.mp_good_bars.eq(29)
    for x in [h,l,c]:
        good &= np.isfinite(x).all(axis=1) & (x>0).all(axis=1) & (abs(x*100-np.floor(x*100+.5))<=.01).all(axis=1)
    good &= (h>=c).all(axis=1) & (c>=l).all(axis=1)
    good &= np.isfinite(v).all(axis=1) & (v>=0).all(axis=1) & (v==np.floor(v)).all(axis=1)
    good &= np.isfinite(d.A04) & d.A04.gt(0) & np.isfinite(d.V01) & d.V01.gt(0)
    with np.errstate(all='ignore'):
        widths = measure(h,l,v)
        values = 100*widths/(np.floor(d.A04.to_numpy()*100+.5)*d.V01.to_numpy())[:,None]
    good &= np.isfinite(values).all(axis=1)
    values[~good] = np.nan

    # Independent wide SQL performs every individual endpoint comparison.
    sql = base.conn(); sql.register('d',d)
    terms = {'up':[],'down':[]}
    for j in range(23,50):
        active = ' AND '.join(f'mp_v{i:02d}>0' for i in [j-2,j-1,j])
        lo = 'least('+','.join(f'CASE WHEN mp_v{i:02d}>0 THEN round(mp_l{i:02d}*100) ELSE 999999999 END' for i in range(j,50))+')' if j<49 else 'round(mp_l49*100)'
        hi = 'greatest('+','.join(f'CASE WHEN mp_v{i:02d}>0 THEN round(mp_h{i:02d}*100) ELSE 0 END' for i in range(j,50))+')' if j<49 else 'round(mp_h49*100)'
        terms['up'].append(f'CASE WHEN {active} THEN greatest({lo}-round(mp_h{j-2:02d}*100),0) ELSE 0 END')
        terms['down'].append(f'CASE WHEN {active} THEN greatest(round(mp_l{j-2:02d}*100)-{hi},0) ELSE 0 END')
    quality = ' AND '.join(f'isfinite(mp_{k}{i:02d}) AND mp_{k}{i:02d}>0 AND abs(mp_{k}{i:02d}*100-round(mp_{k}{i:02d}*100))<=.01' for k in 'hlc' for i in range(21,50))
    quality += ' AND '+' AND '.join(f'mp_h{i:02d}>=mp_c{i:02d} AND mp_c{i:02d}>=mp_l{i:02d} AND isfinite(mp_v{i:02d}) AND mp_v{i:02d}>=0 AND mp_v{i:02d}=floor(mp_v{i:02d})' for i in range(21,50))
    queries = ','.join(f'greatest({",".join(terms[k])}) AS {k}' for k in ['up','down'])
    expected = sql.sql(f'''SELECT date,code,coalesce(mp_bars=29 AND mp_clocks=29 AND mp_good_bars=29
        AND {quality} AND isfinite(A04) AND A04>0 AND isfinite(V01) AND V01>0,false) AS good,
        {queries} FROM d ORDER BY date,code''').df(); sql.close()
    np.testing.assert_array_equal(good,expected.good)
    np.testing.assert_array_equal(widths[good],expected.loc[good,['up','down']])
    np.testing.assert_array_equal(widths[good],native_values(h[good],l[good],v[good]))
    reference = 100*expected[['up','down']].to_numpy()/(np.floor(d.A04.to_numpy()*100+.5)*d.V01.to_numpy())[:,None]
    np.testing.assert_allclose(values[good],reference[good],rtol=0,atol=2e-12)
    encode = lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999))
    np.testing.assert_array_equal(encode(values[good]),encode(reference[good]))
    out = d[['date','code']].copy(); out['interval_source_valid'] = good
    for i,name in enumerate(NEW_EXPRESSIONS): out[name] = values[:,i]
    return out
