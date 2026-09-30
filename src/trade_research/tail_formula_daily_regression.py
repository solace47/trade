"""Completed daily price regressions, with a matched 48-field control.

Slope/R-square/residual are standard regression statistics also present in
the frozen Qlib library. This study uses raw *prior* daily closes, integer
cents, fixed five horizons, and the existing before-10:00 execution labels.
It does not reuse Qlib's historical pool, adjustment scheme, or targets.
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
from .corporate_cash import DAILY, save_json, sha

STEM = 'tail_formula_daily_regression'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_protocol.json')
WINDOWS = [5, 10, 20, 30, 60]
META = source.META
PRICES = [f'rc{i:02d}' for i in range(1, 61)]
HISTORY_META = ['date', 'code', 'history_rows', 'history_good', 'first_history_date', 'last_history_date']
NEW_EXPRESSIONS = {f'{kind}{n:02d}': f'YJREG01.{kind}{n:02d}#DAY'
                   for n in WINDOWS for kind in ['RB', 'RQ', 'RE']}
ARMS = {'control': source.ARMS['norm'], 'regression': {**source.ARMS['norm'], **NEW_EXPRESSIONS}}
EXPRESSIONS = ARMS['regression']
HEADER = source.HEADER


def helper_text():
    lines = ['READY:=BARSCOUNT(C)>=61 AND REF(COUNT(C>0 AND ABS(C*100-ROUND(C*100,0))<=0.01,60),1)=60;']
    lines += [f'RPC{i:02d}:=ROUND(REF(C,{i})*100,0);' for i in range(1, 61)]
    for n in WINDOWS:
        total = '+'.join(f'RPC{i:02d}' for i in range(1, n+1))
        moment = '+'.join(f'({n+1-2*i}*RPC{i:02d})' for i in range(1, n+1))
        squares = '+'.join(f'(RPC{i:02d}*RPC{i:02d})' for i in range(1, n+1))
        d = n*(n*n-1)
        lines += [f'RS{n:02d}:={total};', f'RW{n:02d}:={moment};',
            f'RV{n:02d}:={n}*({squares})-RS{n:02d}*RS{n:02d};',
            f'RX{n:02d}:={d}*({n}*RPC01-RS{n:02d})-{3*n*(n-1)}*RW{n:02d};',
            f'RB{n:02d}:IF(READY,600*RW{n:02d}/({d}*RPC01),DRAWNULL);',
            f'RQ{n:02d}:IF(READY,IF(RV{n:02d}>0,300*RW{n:02d}*RW{n:02d}/({n*n-1}*RV{n:02d}),0),DRAWNULL);',
            f'RE{n:02d}:IF(READY,100*RX{n:02d}/({d*n}*RPC01),DRAWNULL);']
    return '\n'.join(lines)+'\n'


HELPER = helper_text()


def regression(cents, n):
    """Newest first; flat history has slope/residual/R-square all zero."""
    x = np.asarray(cents, dtype=float)[:, :n]
    s = x.sum(axis=1)
    w = (x*(n+1-2*np.arange(1, n+1))).sum(axis=1)
    v = n*np.square(x).sum(axis=1)-np.square(s)
    d = n*(n*n-1)
    e = d*(n*x[:, 0]-s)-3*n*(n-1)*w
    with np.errstate(all='ignore'):
        b = 600*w/(d*x[:, 0])
        q = np.divide(300*w*w, (n*n-1)*v, out=np.zeros(len(x)), where=v>0)
        residual = 100*e/(d*n*x[:, 0])
    return np.column_stack([b, q, residual])


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['arms'] == ARMS and p['windows'] == WINDOWS
    assert p['helper_source'] == HELPER and p['native_header'] == HEADER
    assert p['expected_keys'] == 1815129 and p['threshold'] == .995
    assert not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    r = json.loads((source.INPUTS/'feature_report.json').read_text())
    v = json.loads((source.INPUTS/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(source.INPUTS/'feature_report.json')
    assert r['features_sha256'] == sha(source.INPUTS/'features.parquet')
    lr = json.loads((source.INPUTS/'full_label_report.json').read_text())
    lv = json.loads((source.INPUTS/'full_label_verification.json').read_text())
    assert lv['passed'] and lv['label_report_sha256'] == sha(source.INPUTS/'full_label_report.json')
    assert lr['labels_sha256'] == sha(source.INPUTS/'full_labels.parquet')
    return p


def daily_files():
    names = pd.read_parquet(source.INPUTS/'features.parquet', columns=['code']).code.unique()
    return [DAILY/(code.replace('.', '_')+'.parquet') for code in sorted(names)]


def history(files, first):
    c = base.conn(); c.read_parquet([str(p) for p in files]).create_view('daily')
    lags = ','.join(f'lag(cents,{i}) OVER w AS rc{i:02d}' for i in range(1, 61))
    f = c.sql(f'''WITH a AS(SELECT date,code,close::DOUBLE AS raw_close,
        round(100*close::DOUBLE)::BIGINT AS cents,adjustflag::DOUBLE AS adj FROM daily
        WHERE tradestatus=1 AND date BETWEEN '{first}' AND '2025-12-30'),
        b AS(SELECT *,coalesce(isfinite(raw_close) AND raw_close>0 AND
             abs(raw_close-cents/100.)<=.0001 AND adj=3,false) AS good FROM a),
        h AS(SELECT date,code,{lags},count(*) OVER z AS history_rows,
          sum(good::INT) OVER z AS history_good,min(date) OVER z AS first_history_date,
          max(date) OVER z AS last_history_date FROM b
          WINDOW w AS(PARTITION BY code ORDER BY date),
          z AS(PARTITION BY code ORDER BY date ROWS BETWEEN 60 PRECEDING AND 1 PRECEDING))
        SELECT * FROM h WHERE date>='2023-01-01' ORDER BY date,code''').df()
    c.close(); return f


def prepare():
    p = checked(); assert not (INPUTS/'feature_report.json').exists()
    INPUTS.mkdir(parents=True, exist_ok=True)
    files = daily_files()
    receipts = {str(file): sha(file) for file in files}
    save_json(INPUTS/'daily_source_manifest.json', dict(source_sha256=receipts,
        first_requested=p['history_first'], last_requested='2025-12-30',
        only_active_raw_daily_close_and_quality_fields_read=True, new_2026_prices_read=False))
    old = pd.read_parquet(source.INPUTS/'features.parquet', columns=[*META, *ARMS['control']])
    h = history(files, p['history_first'])
    h = old[['date','code']].merge(h, on=['date','code'], how='left', validate='one_to_one')
    pd.testing.assert_frame_equal(old[['date','code']], h[['date','code']], check_exact=True)
    cents = h[PRICES].to_numpy(float)
    good = h.history_rows.eq(60) & h.history_good.eq(60) & h.last_history_date.lt(h.date)
    good &= np.isfinite(cents).all(axis=1) & (cents>0).all(axis=1)
    f = old.copy(); f['prior_formula_input_valid'] = old.formula_input_valid
    for n in WINDOWS:
        values = regression(cents, n)
        for i,kind in enumerate(['RB','RQ','RE']):
            f[f'{kind}{n:02d}'] = pd.Series(values[:,i]).where(good)
    f['history_input_valid'] = good
    f['formula_input_valid'] &= good & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    h.to_parquet(INPUTS/'history.parquet', index=False, compression='zstd')
    f.to_parquet(INPUTS/'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), implementation_sha256=sha(Path(__file__)),
        source_hashes=p['source_hashes'], daily_source_manifest_sha256=sha(INPUTS/'daily_source_manifest.json'),
        history_sha256=sha(INPUTS/'history.parquet'), features_sha256=sha(INPUTS/'features.parquet'),
        rows=len(f), prior_valid=int(old.formula_input_valid.sum()), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        history_valid=int(good.sum()), expressions=EXPRESSIONS, native_header=HEADER, helper_source=HELPER,
        by_year=f.assign(year=f.date.str[:4]).groupby('year').agg(rows=('code','size'),
            prior_valid=('prior_formula_input_valid','sum'),valid=('formula_input_valid','sum')).to_dict('index'),
        flat_windows={str(n):int((f[f'RQ{n:02d}'].eq(0)&good).sum()) for n in WINDOWS},
        raw_completed_daily_prices_not_qfq=True, matched_control_uses_same_input_validity=True,
        current_daily_prices_not_used=True, target_not_joined=True, new_2026_prices_read=False,
        software_compilation_verified=False, native_source_parity_verified=False, no_exit_rules=True)
    save_json(INPUTS/'feature_report.json', r)
    for file in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        (INPUTS/file).symlink_to((source.INPUTS/file).resolve())
    return {k:r[k] for k in ['rows','prior_valid','valid','newly_invalid','by_year','flat_windows']}


def verify():
    p = checked(); r = json.loads((INPUTS/'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(INPUTS/'features.parquet')
    assert r['history_sha256'] == sha(INPUTS/'history.parquet')
    manifest = json.loads((INPUTS/'daily_source_manifest.json').read_text())
    assert r['daily_source_manifest_sha256'] == sha(INPUTS/'daily_source_manifest.json')
    for file,digest in manifest['source_sha256'].items(): assert sha(Path(file)) == digest
    f = pd.read_parquet(INPUTS/'features.parquet')
    old = pd.read_parquet(source.INPUTS/'features.parquet', columns=[*META, *ARMS['control']])
    pd.testing.assert_frame_equal(f[[n for n in old.columns if n!='formula_input_valid']],
        old.drop(columns='formula_input_valid'),check_exact=True)
    np.testing.assert_array_equal(f.prior_formula_input_valid,old.formula_input_valid)
    assert len(f)==p['expected_keys'] and not f.duplicated(['date','code']).any()
    # Independent direct rolling moments on source daily rows, rather than
    # recomputing the saved 60-column lag vectors or calling regression().
    c = base.conn(); c.read_parquet(list(manifest['source_sha256'])).create_view('daily')
    c.register('keys',old[['date','code']])
    parts=[]; windows=[]
    for n in WINDOWS:
        parts += [f'sum(cents) OVER z{n} AS s{n}',f'sum(cents*cents) OVER z{n} AS q{n}',
                  f'sum(row_id*cents) OVER z{n} AS t{n}']
        windows.append(f'z{n} AS(PARTITION BY code ORDER BY date ROWS BETWEEN {n} PRECEDING AND 1 PRECEDING)')
    query=f'''WITH a AS(SELECT date,code,close::DOUBLE AS raw_close,
       round(100*close::DOUBLE)::BIGINT AS cents,adjustflag::DOUBLE AS adj,
       row_number() OVER(PARTITION BY code ORDER BY date)::BIGINT AS row_id
       FROM daily WHERE tradestatus=1 AND date BETWEEN '{p['history_first']}' AND '2025-12-30'),
       b AS(SELECT *,coalesce(isfinite(raw_close) AND raw_close>0 AND abs(raw_close-cents/100.)<=.0001
          AND adj=3,false) AS good FROM a),
       h AS(SELECT date,code,row_id,lag(cents) OVER(PARTITION BY code ORDER BY date) AS last_cents,
          count(*) OVER z60 AS h_rows,sum(good::INT) OVER z60 AS h_good,
          min(date) OVER z60 AS first_date,max(date) OVER z60 AS last_date,{','.join(parts)}
          FROM b WINDOW {','.join(windows)}),
       m AS(SELECT *,{','.join(f'(2*t{n}-(2*row_id-{n}-1)*s{n})::DOUBLE AS w{n}' for n in WINDOWS)},
          {','.join(f'({n}*q{n}-s{n}*s{n})::DOUBLE AS v{n}' for n in WINDOWS)} FROM h)
       SELECT k.date,k.code,coalesce(h_rows=60 AND h_good=60 AND last_cents>0 AND last_date<k.date,false) AS valid,
         first_date,last_date,{','.join(expr for n in WINDOWS for expr in [
          f'600*w{n}/({n*(n*n-1)}*last_cents) AS RB{n:02d}',
          f'CASE WHEN v{n}>0 THEN 300*w{n}*w{n}/({n*n-1}*v{n}) ELSE 0. END AS RQ{n:02d}',
          f'100*({n*(n*n-1)}*({n}*last_cents-s{n})-{3*n*(n-1)}*w{n})/({n*n*(n*n-1)}*last_cents) AS RE{n:02d}'])}
       FROM keys k LEFT JOIN m ON k.date=m.date AND k.code=m.code ORDER BY k.date,k.code'''
    rebuilt = c.sql(query).df(); c.close()
    pd.testing.assert_frame_equal(f[['date','code']],rebuilt[['date','code']],check_exact=True)
    np.testing.assert_array_equal(f.history_input_valid,rebuilt.valid)
    for name in NEW_EXPRESSIONS:
        expected=rebuilt[name].where(rebuilt.valid).to_numpy(float)
        np.testing.assert_allclose(f[name],expected,rtol=0,atol=2e-11,equal_nan=True)
        finite=np.isfinite(expected)
        enc=lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999))
        np.testing.assert_array_equal(enc(f.loc[finite,name].to_numpy()),enc(expected[finite]))
    good=old.formula_input_valid & rebuilt.valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    np.testing.assert_array_equal(f.formula_input_valid,good)
    h=pd.read_parquet(INPUTS/'history.parquet',columns=HISTORY_META)
    pd.testing.assert_series_equal(h.first_history_date,rebuilt.first_date,check_names=False)
    pd.testing.assert_series_equal(h.last_history_date,rebuilt.last_date,check_names=False)
    assert int(good.sum())==r['valid'] and int((old.formula_input_valid&~good).sum())==r['newly_invalid']
    proof=dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),rows=len(f),
        valid=int(good.sum()),all_48_values_and_all_original_keys_unchanged=True,
        all_fifteen_features_independent_direct_sql_rolling_moments=True,all_integer_encodings_equal=True,
        exact_matched_control_validity=True,current_and_future_daily_prices_not_used=True,
        no_new_training_or_evaluation_labels=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'feature_verification.json',proof)
    return proof


def replay_helper(closes):
    closes=np.asarray(closes,float)
    env=dict(C=closes,DRAWNULL=np.nan,IF=np.where,ABS=np.abs,ROUND=np.round,
        REF=lambda a,n:pd.Series(a).shift(int(n)).to_numpy(),
        COUNT=lambda a,n:pd.Series(np.asarray(a,float)).rolling(int(n),min_periods=int(n)).sum().to_numpy(),
        BARSCOUNT=lambda a:np.arange(1,len(a)+1))
    with np.errstate(all='ignore'):
        for line in HELPER.splitlines():
            name,expr=line.rstrip(';').split(':=') if ':=' in line else line.rstrip(';').split(':')
            if name=='READY':
                expr='(BARSCOUNT(C)>=61) & (REF(COUNT((C>0) & (ABS(C*100-ROUND(C*100,0))<=0.01),60),1)==60)'
            env[name]=eval(expr,{'__builtins__':{}},env)
    return np.column_stack([env[n] for n in NEW_EXPRESSIONS])


def native():
    import pyarrow.parquet as pq
    checked();r=json.loads((INPUTS/'feature_report.json').read_text())
    proof=json.loads((INPUTS/'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256']==sha(INPUTS/'feature_report.json')
    f=pd.read_parquet(INPUTS/'features.parquet');f=f.loc[f.formula_input_valid].copy()
    f['identity']=[hashlib.sha256((d+'|'+s+'|daily-regression-native-v1').encode()).hexdigest()
        for d,s in zip(f.date,f.code)]
    f['period']=f.date.str[:4]+np.where(f.date.str[5:7].lt('07'),'H1','H2')
    samples=f.sort_values('identity').groupby('period',sort=True).head(8).sort_values(['date','code'])
    receipts=[]
    sources=json.loads((INPUTS/'daily_source_manifest.json').read_text())['source_sha256']
    for row in samples.itertuples():
        path=DAILY/(row.code.replace('.','_')+'.parquet');assert sha(path)==sources[str(path)]
        d=pq.read_table(path,columns=['date','close','tradestatus','adjustflag'],
            filters=[('date','>=','2022-06-01'),('date','<',row.date)]).to_pandas()
        d=d.loc[d.tradestatus.eq(1)].sort_values('date').tail(60)
        assert len(d)==60 and d.date.iloc[-1]<row.date and d.adjustflag.eq(3).all()
        prior=d.close.to_numpy(float)
        expected=np.asarray([getattr(row,n) for n in NEW_EXPRESSIONS])
        a=replay_helper(np.r_[prior,17.])[-1]
        np.testing.assert_allclose(a,expected,rtol=0,atol=2e-11)
        np.testing.assert_array_equal(replay_helper(np.r_[prior,.01])[-1],a)
        np.testing.assert_array_equal(replay_helper(np.r_[prior,1000000.])[-1],a)
        np.testing.assert_array_equal(replay_helper(np.r_[prior,17.,.01,1000000.])[60],a)
        enc=lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999))
        np.testing.assert_array_equal(enc(a),enc(expected))
        receipts.append(dict(date=row.date,code=row.code,first=d.date.iloc[0],last=d.date.iloc[-1],
            rows=60,daily_source_sha256=sources[str(path)]))
    assert len(receipts)==48
    helper=INPUTS/'YJREG01.tdx';helper.write_text(HELPER)
    out=dict(passed=True,protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(INPUTS/'feature_report.json'),
        feature_verification_sha256=sha(INPUTS/'feature_verification.json'),helper_sha256=sha(helper),samples=receipts,
        six_halves_48_actual_daily_samples_replayed=True,current_and_future_daily_price_pollution_unchanged=True,
        all_fifteen_native_arithmetic_and_integer_encodings_replayed=True,
        helper_uses_only_completed_prior_daily_closes=True,requires_unadjusted_aligned_active_daily_history=True,
        software_compilation_verified=False,native_source_parity_verified=False,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'native_input_verification.json',out)
    return {k:v for k,v in out.items() if k!='samples'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','verify','native'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
