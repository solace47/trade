"""Fixed regressions of the visible late-minute clock, not prior daily prices.

Reuse the verified 2024–2025 30-quote cache; extract only the missing 2023
training windows. Raw integer cents and centered moments are also used by
the native minute expression. The old full-window polynomial study remains
a separate, mandatory 2025 control.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_feature_subsample as source
from . import tail_formula_path_variance as cached
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_tail_regression'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_protocol.json')
MANIFEST = Path('data/research/tail_formula_long48/inputs/source_manifest.json')
WINDOWS = [5, 10, 20, 30]
META = source.META
PRICE_COLUMNS = cached.PRICE_COLUMNS
QUOTE_META = ['date', 'code', 'pv_bars', 'pv_clocks', 'pv_good_bars']


def declarations():
    lines = ['RMC:=ROUND(Q*100);', 'RTC00:=0;']
    lines += [f'RTC{i:02d}:=VALUEWHEN(TIME=1449,ROUND(REF(C,{i})*100))-RMC;' for i in range(1, 30)]
    clocks = ['VALUEWHEN(TIME=1449,TIME)=1449']
    clocks += [f'VALUEWHEN(TIME=1449,REF(TIME,{i}))={1449-i}' for i in range(1, 30)]
    lines += ['RTCLK:=' + ' AND '.join(clocks) + ';',
        'RTGOOD:=VALUEWHEN(TIME=1449,COUNT(C>0 AND ABS(C*100-ROUND(C*100))<=0.01,30));',
        'RTREADY:=RTCLK AND RTGOOD=30 AND RMC>0 AND VP20>0;']
    for n in WINDOWS:
        s = '+'.join(f'RTC{i:02d}' for i in range(n))
        w = '+'.join(f'({n-1-2*i}*RTC{i:02d})' for i in range(n))
        squares = '+'.join(f'(RTC{i:02d}*RTC{i:02d})' for i in range(n))
        d = n*(n*n-1)
        lines += [f'RTS{n:02d}:={s};', f'RTW{n:02d}:={w};',
            f'RTV{n:02d}:={n}*({squares})-RTS{n:02d}*RTS{n:02d};',
            f'RTX{n:02d}:=-{d}*RTS{n:02d}-{3*n*(n-1)}*RTW{n:02d};']
    return '\n'.join(lines) + '\n'


EXTRA_HEADER = declarations()
NEW_EXPRESSIONS = {}
for n in WINDOWS:
    d = n*(n*n-1)
    NEW_EXPRESSIONS.update({
        f'TRB{n:02d}': f'IF(RTREADY,600*{n-1}*RTW{n:02d}/({d}*RMC*VP20),DRAWNULL)',
        f'TRQ{n:02d}': f'IF(RTREADY,IF(RTV{n:02d}>0,300*RTW{n:02d}*RTW{n:02d}/({n*n-1}*RTV{n:02d}),0),DRAWNULL)',
        f'TRE{n:02d}': f'IF(RTREADY,100*RTX{n:02d}/({d*n}*RMC*VP20),DRAWNULL)'})
ARMS = {'control': source.ARMS['norm'], 'regression': {**source.ARMS['norm'], **NEW_EXPRESSIONS}}
EXPRESSIONS = ARMS['regression']
HEADER = source.HEADER + EXTRA_HEADER


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['arms'] == ARMS and p['native_header'] == HEADER and p['windows'] == WINDOWS
    assert p['expected_keys'] == 1815129 and p['threshold'] == .995
    assert not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for folder in [source.INPUTS, cached.ROOT]:
        r = json.loads((folder/'feature_report.json').read_text())
        v = json.loads((folder/'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(folder/'feature_report.json')
        assert r['features_sha256'] == sha(folder/'features.parquet')
    r = json.loads((cached.ROOT/'feature_report.json').read_text())
    assert r['window_report_sha256'] == sha(cached.ROOT/'window_report.json')
    r = json.loads((source.INPUTS/'full_label_report.json').read_text())
    v = json.loads((source.INPUTS/'full_label_verification.json').read_text())
    assert v['passed'] and v['label_report_sha256'] == sha(source.INPUTS/'full_label_report.json')
    assert r['labels_sha256'] == sha(source.INPUTS/'full_labels.parquet')
    return p


def windows():
    checked(); assert not (INPUTS/'window_report.json').exists()
    folder = INPUTS/'parts_2023'; folder.mkdir(parents=True, exist_ok=True)
    keys = pd.read_parquet(source.INPUTS/'features.parquet', columns=['date','code'],
        filters=[('date','<','2024-01-01')])
    assert len(keys) == 557044 and keys.date.ge('2023-01-01').all()
    sources = json.loads(MANIFEST.read_text())['minute_sha256']
    names = sorted(keys.code.unique()); receipts = {}; parts = {}
    for offset in range(0, len(names), 64):
        codes = names[offset:offset+64]
        path = folder/f'part_{offset//64:03d}.parquet'; meta_path = path.with_suffix('.json')
        paths = [MINUTES/code[:2].upper()/(code[3:]+'.parquet') for code in codes]
        for file in paths:
            assert sha(file) == sources[str(file)]
            receipts[str(file)] = sources[str(file)]
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            assert meta['codes'] == codes and meta['protocol_sha256'] == sha(PROTOCOL)
            assert meta['extractor_sha256'] == sha(Path(__file__)) and meta['sha256'] == sha(path)
        else:
            c = base.conn(); c.read_parquet([str(file) for file in paths]).create_view('raw')
            c.register('keys', keys.loc[keys.code.isin(codes)])
            pivots = ','.join(f"max(round(close,2)) FILTER(WHERE clock='14{i:02d}') AS pv_c{i:02d}" for i in range(20,50))
            out = c.sql(f'''WITH s AS(SELECT lower(exchange)||'.'||symbol AS code,
                strftime(timestamp,'%Y-%m-%d') AS date,strftime(timestamp,'%H%M') AS clock,
                timestamp,close::DOUBLE AS close FROM raw WHERE timestamp>=TIMESTAMP '2023-01-01'
                AND timestamp<TIMESTAMP '2024-01-01' AND strftime(timestamp,'%H%M') BETWEEN '1420' AND '1449'),
                selected AS(SELECT s.*,coalesce(timestamp=date_trunc('minute',timestamp) AND isfinite(close)
                AND close>0 AND abs(close-round(close,2))<=.0001,false) AS good FROM s JOIN keys USING(date,code))
                SELECT date,code,count(*) AS pv_bars,count(DISTINCT clock) AS pv_clocks,
                count(*) FILTER(WHERE good) AS pv_good_bars,{pivots}
                FROM selected GROUP BY date,code ORDER BY date,code''').df(); c.close()
            out.to_parquet(path, index=False, compression='zstd')
            meta = dict(codes=codes,protocol_sha256=sha(PROTOCOL),extractor_sha256=sha(Path(__file__)),
                sha256=sha(path),rows=len(out),price_first='2023-01-01',price_last='2023-12-29',
                raw_price_columns_read=['timestamp','exchange','symbol','close'])
            save_json(meta_path, meta)
        parts[str(path)] = meta['sha256']
        print(json.dumps(dict(codes=offset+len(codes),total_codes=len(names))), flush=True)
    old = json.loads((cached.ROOT/'window_report.json').read_text())
    for file, digest in old['parts_sha256'].items(): assert sha(Path(file)) == digest
    report = dict(protocol_sha256=sha(PROTOCOL),extractor_sha256=sha(Path(__file__)),
        raw_2023_parts_sha256=parts,raw_2023_source_sha256=receipts,
        old_2024_2025_window_report_sha256=sha(cached.ROOT/'window_report.json'),
        reused_2024_2025_parts_sha256=old['parts_sha256'],
        new_raw_price_extraction_only_2023=True, no_new_2024_2025_raw_window_extraction=True,
        window_first='1420',window_last='1449',no_evaluation_labels_read=True,new_2026_prices_read=False)
    save_json(INPUTS/'window_report.json',report)
    return dict(new_2023_symbols=len(names),new_2023_parts=len(parts),
        old_2024_2025_parts_reused=len(old['parts_sha256']),window_report_sha256=sha(INPUTS/'window_report.json'))


def quotes(keys):
    r = json.loads((INPUTS/'window_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['extractor_sha256'] == sha(Path(__file__))
    assert r['old_2024_2025_window_report_sha256'] == sha(cached.ROOT/'window_report.json')
    files = {**r['raw_2023_parts_sha256'], **r['reused_2024_2025_parts_sha256']}
    for file, digest in files.items(): assert sha(Path(file)) == digest
    wide = pd.concat([pd.read_parquet(file, columns=[*QUOTE_META,*PRICE_COLUMNS]) for file in files],ignore_index=True)
    return keys.merge(wide,on=['date','code'],how='left',validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)


def regression(cents_newest_first, atr, n):
    x = np.asarray(cents_newest_first,float)[:,:n]
    delta = x-x[:,:1]
    s = delta.sum(axis=1); w = (delta*(n-1-2*np.arange(n))).sum(axis=1)
    v = n*np.square(delta).sum(axis=1)-np.square(s); d = n*(n*n-1)
    e = -d*s-3*n*(n-1)*w
    with np.errstate(all='ignore'):
        b = 600*(n-1)*w/(d*x[:,0]*atr)
        q = np.divide(300*w*w,(n*n-1)*v,out=np.zeros(len(x)),where=v>0)
        residual = 100*e/(d*n*x[:,0]*atr)
    return np.column_stack([b,q,residual])


def quote_valid(wide, atr):
    price = wide[PRICE_COLUMNS].to_numpy(float)
    return (wide[['pv_bars','pv_clocks','pv_good_bars']].eq(30).all(axis=1).to_numpy()
        & np.isfinite(price).all(axis=1) & (price>0).all(axis=1)
        & np.isfinite(atr) & (atr>0))


def prepare():
    p = checked(); assert not (INPUTS/'feature_report.json').exists()
    old = pd.read_parquet(source.INPUTS/'features.parquet',columns=[*META,*ARMS['control']])
    h = quotes(old[['date','code']]); atr = old.V01.to_numpy(float)
    pd.testing.assert_frame_equal(old[['date','code']],h[['date','code']],check_exact=True)
    good = quote_valid(h,atr); price = h[PRICE_COLUMNS].to_numpy(float)
    cents = np.floor(price[:,::-1]*100+.5)
    f = old.copy(); f['prior_formula_input_valid'] = old.formula_input_valid
    for n in WINDOWS:
        values = regression(cents,atr,n)
        for i, kind in enumerate(['TRB','TRQ','TRE']):
            f[f'{kind}{n:02d}'] = pd.Series(values[:,i]).where(good)
    f['tail_regression_valid'] = good
    f['formula_input_valid'] &= good & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    np.testing.assert_allclose(h.loc[f.formula_input_valid,'pv_c49'],old.loc[f.formula_input_valid,'A04'],rtol=0,atol=.0001)
    h.to_parquet(INPUTS/'quotes.parquet',index=False,compression='zstd')
    f.to_parquet(INPUTS/'features.parquet',index=False,compression='zstd')
    flat = {str(n):int((cents[good,:n].max(axis=1)==cents[good,:n].min(axis=1)).sum()) for n in WINDOWS}
    r = dict(protocol_sha256=sha(PROTOCOL),implementation_sha256=sha(Path(__file__)),
        window_report_sha256=sha(INPUTS/'window_report.json'),quotes_sha256=sha(INPUTS/'quotes.parquet'),
        features_sha256=sha(INPUTS/'features.parquet'),rows=len(f),prior_valid=int(old.formula_input_valid.sum()),
        valid=int(f.formula_input_valid.sum()),newly_invalid=int((old.formula_input_valid&~f.formula_input_valid).sum()),
        by_year=f.assign(year=f.date.str[:4]).groupby('year').agg(rows=('code','size'),
            prior_valid=('prior_formula_input_valid','sum'),valid=('formula_input_valid','sum')).to_dict('index'),
        actual_constant_price_windows=flat,expressions=EXPRESSIONS,native_header=HEADER,
        raw_integer_cents_and_prior_atr_used=True,matched_control_uses_same_input_validity=True,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True,
        software_compilation_verified=False,native_source_parity_verified=False)
    save_json(INPUTS/'feature_report.json',r)
    for file in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        (INPUTS/file).symlink_to((source.INPUTS/file).resolve())
    return {k:r[k] for k in ['rows','prior_valid','valid','newly_invalid','by_year','actual_constant_price_windows']}


def verify():
    checked(); r = json.loads((INPUTS/'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(INPUTS/'features.parquet')
    assert r['quotes_sha256'] == sha(INPUTS/'quotes.parquet') and r['window_report_sha256'] == sha(INPUTS/'window_report.json')
    f = pd.read_parquet(INPUTS/'features.parquet')
    old = pd.read_parquet(source.INPUTS/'features.parquet',columns=[*META,*ARMS['control']])
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    np.testing.assert_array_equal(f.prior_formula_input_valid,old.formula_input_valid)
    h = quotes(old[['date','code']]); pd.testing.assert_frame_equal(h,pd.read_parquet(INPUTS/'quotes.parquet'),check_exact=True)
    assert len(f) == r['rows'] == 1815129 and not f.duplicated(['date','code']).any()
    c = base.conn(); c.register('quotes',h); c.register('old',old[['date','code','V01']])
    xs = ','.join(f'round(pv_c{49-i}*100)::DOUBLE AS x{i:02d}' for i in range(30))
    valid = ' AND '.join(f'isfinite(pv_c{i}) AND pv_c{i}>0' for i in range(20,50))
    moments = []; fields = []
    for n in WINDOWS:
        s = '+'.join(f'x{i:02d}' for i in range(n)); w = '+'.join(f'({n-1-2*i}*x{i:02d})' for i in range(n))
        squares = '+'.join(f'(x{i:02d}*x{i:02d})' for i in range(n)); d = n*(n*n-1)
        moments += [f'({s}) AS s{n}',f'({w}) AS w{n}',f'({n}*({squares})-({s})*({s})) AS v{n}']
        fields += [f'CASE WHEN valid THEN 600*{n-1}*w{n}/({d}*x00*V01) END AS TRB{n:02d}',
            f'CASE WHEN valid THEN CASE WHEN v{n}>0 THEN 300*w{n}*w{n}/({n*n-1}*v{n}) ELSE 0. END END AS TRQ{n:02d}',
            f'CASE WHEN valid THEN 100*({d}*({n}*x00-s{n})-{3*n*(n-1)}*w{n})/({d*n}*x00*V01) END AS TRE{n:02d}']
    expected = c.sql(f'''WITH x AS(SELECT date,code,V01,{xs},coalesce(pv_bars=30 AND pv_clocks=30
        AND pv_good_bars=30 AND isfinite(V01) AND V01>0 AND {valid},false) AS valid
        FROM quotes JOIN old USING(date,code)),m AS(SELECT *,{','.join(moments)} FROM x)
        SELECT date,code,valid,{','.join(fields)} FROM m ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(f[['date','code']],expected[['date','code']],check_exact=True)
    np.testing.assert_array_equal(f.tail_regression_valid,expected.valid)
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name],expected[name],rtol=0,atol=2e-11,equal_nan=True)
        finite = np.isfinite(expected[name]); encode = lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999))
        np.testing.assert_array_equal(encode(f.loc[finite,name]),encode(expected.loc[finite,name]))
    final = old.formula_input_valid & expected.valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    np.testing.assert_array_equal(f.formula_input_valid,final)
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names})
    proof = dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),rows=len(f),valid=int(final.sum()),
        all_48_values_keys_and_original_validity_preserved=True,
        all_twelve_statistics_independent_uncentered_sql_moments=True,all_integer_encodings_equal=True,
        exact_matched_control_validity=True,effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        no_new_training_or_evaluation_labels=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'feature_verification.json',proof); return proof


def replay_native(prices, atr, outside=1000000.):
    price = np.asarray(prices,float); assert price.shape == (30,)
    env = dict(C=np.r_[outside,price,outside,outside],VP20=float(atr),Q=price[-1],DRAWNULL=np.nan,
        TIME=np.r_[1419,np.arange(1420,1450),1450,1451],IF=np.where,ABS=np.abs,ROUND=lambda x:np.floor(x+.5),
        REF=lambda x,n:pd.Series(x).shift(int(n)).to_numpy(),
        COUNT=lambda x,n:pd.Series(np.asarray(x,float)).rolling(int(n),min_periods=int(n)).sum().to_numpy(),
        VALUEWHEN=lambda mask,values:np.asarray(values)[np.flatnonzero(mask)[-1]])
    with np.errstate(all='ignore'):
        for line in EXTRA_HEADER.splitlines():
            name,expr = line.rstrip(';').split(':=')
            expr = re.sub(r'(?<![<>=!])=(?!=)','==',expr)
            if name == 'RTCLK': expr = expr.replace(' AND ',' and ')
            elif name == 'RTGOOD': expr = 'VALUEWHEN(TIME==1449,COUNT((C>0) & (ABS(C*100-ROUND(C*100))<=0.01),30))'
            elif name == 'RTREADY': expr = expr.replace(' AND ',' and ')
            env[name] = eval(expr,{'__builtins__':{}},env)
        return np.asarray([float(eval(expr,{'__builtins__':{}},env)) for expr in NEW_EXPRESSIONS.values()])


def native():
    checked();v = json.loads((INPUTS/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(INPUTS/'feature_report.json')
    f = pd.read_parquet(INPUTS/'features.parquet'); h = pd.read_parquet(INPUTS/'quotes.parquet')
    pd.testing.assert_frame_equal(f[['date','code']],h[['date','code']],check_exact=True)
    checks = 0
    for start in range(0,len(f),50000):
        ff = f.iloc[start:start+50000]; hh = h.iloc[start:start+50000]
        cents = np.floor(hh[PRICE_COLUMNS].to_numpy(float)[:,::-1]*100+.5)
        env = dict(RMC=cents[:,0],VP20=ff.V01.to_numpy(),RTC00=np.zeros(len(ff)),
            RTREADY=ff.tail_regression_valid.to_numpy(),IF=np.where,DRAWNULL=np.nan)
        env.update({f'RTC{i:02d}':cents[:,i]-cents[:,0] for i in range(1,30)})
        with np.errstate(all='ignore'):
            for line in EXTRA_HEADER.splitlines()[34:]:
                name,expr = line.rstrip(';').split(':='); env[name] = eval(expr,{'__builtins__':{}},env)
            for name,expr in NEW_EXPRESSIONS.items():
                got = eval(expr,{'__builtins__':{}},env); expected = ff[name].to_numpy(float)
                np.testing.assert_allclose(got,expected,rtol=0,atol=2e-11,equal_nan=True)
                good = np.isfinite(expected); encode = lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999))
                np.testing.assert_array_equal(encode(got[good]),encode(expected[good])); checks += len(ff)
    sample = f.loc[f.formula_input_valid].copy()
    sample['identity'] = [hashlib.sha256((d+'|'+s+'|tail-regression-v1').encode()).hexdigest() for d,s in zip(sample.date,sample.code)]
    sample['period'] = sample.date.str[:4]+np.where(sample.date.str[5:7].lt('07'),'H1','H2')
    sample = sample.sort_values('identity').groupby('period',sort=True).head(8).sort_values(['date','code'])
    sources = json.loads((INPUTS/'window_report.json').read_text())['raw_2023_source_sha256']
    sources = {**json.loads((cached.ROOT/'window_report.json').read_text())['source_sha256'],**sources}
    receipts = []
    for row in sample.itertuples():
        file = MINUTES/row.code[:2].upper()/(row.code[3:]+'.parquet'); assert sha(file) == sources[str(file)]
        raw = pd.read_parquet(file,columns=['timestamp','close'],filters=[('timestamp','>=',pd.Timestamp(row.date+' 14:20')),
            ('timestamp','<=',pd.Timestamp(row.date+' 14:49'))]).sort_values('timestamp')
        assert raw.timestamp.dt.strftime('%H%M').tolist() == [f'14{i:02d}' for i in range(20,50)]
        assert (np.abs(raw.close-raw.close.round(2))<=.0001).all()
        got = replay_native(raw.close.to_numpy(),row.V01); expected = np.asarray([getattr(row,n) for n in NEW_EXPRESSIONS])
        np.testing.assert_allclose(got,expected,rtol=0,atol=2e-11)
        np.testing.assert_array_equal(got,replay_native(raw.close.to_numpy(),row.V01,.01))
        encode = lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999))
        np.testing.assert_array_equal(encode(got),encode(expected))
        receipts.append(dict(date=row.date,code=row.code,source_sha256=sources[str(file)],minutes=30))
    assert len(receipts) == 48
    helper = INPUTS/'tail_regression_inputs.tdx';helper.write_text(EXTRA_HEADER+'\n'.join(f'{k}:={e};' for k,e in NEW_EXPRESSIONS.items())+'\n')
    out = dict(passed=True,protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(INPUTS/'feature_report.json'),
        feature_verification_sha256=sha(INPUTS/'feature_verification.json'),helper_sha256=sha(helper),samples=receipts,
        scalar_checks=checks,all_twelve_literal_moments_and_encodings_replayed=True,
        six_halves_48_actual_raw_minute_samples=True,future_and_outside_window_price_pollution_unchanged=True,
        single_argument_round_and_centered_integer_moments=True,
        software_compilation_verified=False,native_source_parity_verified=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'native_input_verification.json',out);return {k:v for k,v in out.items() if k!='samples'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['windows','prepare','verify','native'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
