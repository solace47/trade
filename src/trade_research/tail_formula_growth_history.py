"""One completed-day growth-style input with explicit stock-session alignment."""
import argparse
import hashlib
import json
import socket
import subprocess
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd

from . import tail_formula_additive as numeric
from . import tail_formula_baseline as baseline
from .research_io import check_runtime, check_sources, save_json, sha

ROOT = Path('data/research/tail_formula_growth_history')
PROTOCOL = Path('config/tail_formula_growth_history_inputs.json')
INDEX_CODES = ['sh.000001', 'sz.399001', 'sz.399006']
EXPRESSION = 'IF(GHOK,100*(GHP1/GHP6-ZHP1/ZHP6),DRAWNULL)'
HEADER = baseline.HEADER + '''
GHP1:=REF("SZ399006$CLOSE",B0);
GHP6:=REF("SZ399006$CLOSE",B5);
ZHP1:=REF("SZ399001$CLOSE",B0);
ZHP6:=REF("SZ399001$CLOSE",B5);
GHOK:=GHP1>0 AND GHP6>0 AND ZHP1>0 AND ZHP6>0 AND REF(DATE,B0)<DATE AND REF(DATE,B5)<REF(DATE,B0);
'''


def checked():
    check_runtime()
    assert subprocess.check_output(['git', 'show', 'HEAD:'+str(PROTOCOL)]) == PROTOCOL.read_bytes()
    p = json.loads(PROTOCOL.read_text())
    check_sources(p['source_hashes'])
    return p


def collect(p):
    assert not (ROOT/'index_source_report.json').exists()
    ROOT.mkdir(exist_ok=True)
    folder = ROOT/'raw'; folder.mkdir(exist_ok=True)
    socket.setdefaulttimeout(20)
    login = bs.login()
    assert login.error_code == '0', (login.error_code, login.error_msg)
    records, receipts = [], dict(p['source_hashes'])
    fields = 'date,code,open,high,low,close'
    try:
        for code in INDEX_CODES:
            path = folder/(code.replace('.', '_')+'.json')
            assert not path.exists()
            result = bs.query_history_k_data_plus(code, fields,
                start_date=p['index_first'], end_date=p['index_last'], frequency='d', adjustflag='3')
            assert result.error_code == '0', (result.error_code, result.error_msg)
            rows = []
            while result.next(): rows.append(result.get_row_data())
            assert result.error_code == '0'
            save_json(path, dict(code=code, first=p['index_first'], last=p['index_last'],
                fields=fields.split(','), frequency='d', adjustflag='3', records=rows))
            receipts[str(path)] = sha(path)
            f = pd.DataFrame(rows, columns=fields.split(','))
            assert len(f) and f.code.eq(code).all() and f.date.between(p['index_first'], p['index_last']).all()
            for name in ['open', 'high', 'low', 'close']: f[name] = pd.to_numeric(f[name], errors='raise')
            assert not f.duplicated(['date', 'code']).any()
            assert np.isfinite(f[['open','high','low','close']]).all().all()
            assert f[['open','high','low','close']].gt(0).all().all()
            assert f.high.add(.0001).ge(f[['open','low','close']].max(axis=1)).all()
            assert f.low.sub(.0001).le(f[['open','high','close']].min(axis=1)).all()
            records.append(f)
            print(json.dumps(dict(collected=code, rows=len(f), first=f.date.min(), last=f.date.max())), flush=True)
    finally:
        bs.logout()
    f = pd.concat(records).sort_values(['code', 'date']).reset_index(drop=True)
    calendar = pd.read_parquet(p['calendar'], columns=['calendar_date','is_trading_day'])
    days = sorted(calendar.loc[calendar.is_trading_day.eq('1') & calendar.calendar_date.between(p['index_first'],p['index_last']), 'calendar_date'])
    for code in INDEX_CODES: assert f.loc[f.code.eq(code), 'date'].tolist() == days
    f.to_parquet(ROOT/'indices.parquet', index=False, compression='zstd')
    receipts[str(ROOT/'indices.parquet')] = sha(ROOT/'indices.parquet')
    save_json(ROOT/'index_source_report.json', dict(passed=True, protocol_sha256=sha(PROTOCOL),
        source_hashes=receipts, rows=len(f), days_per_index=len(days),
        no_new_fits=True, no_new_economic_groups=True, new_2026_prices_read=False))


def align(stock_days, indices):
    """Use the stock's completed sessions, including suspension gaps."""
    assert not stock_days.duplicated(['date','code']).any()
    assert not indices.duplicated(['date','code']).any()
    s = stock_days.sort_values(['code','date']).copy()
    grouped = s.groupby('code', sort=False).date
    s['gh_date1'], s['gh_date6'] = grouped.shift(1), grouped.shift(6)
    for prefix, code in [('G','sz.399006'), ('Z','sz.399001'), ('S','sh.000001')]:
        prices = indices.loc[indices.code.eq(code)].set_index('date').close
        s[prefix+'1'] = s.gh_date1.map(prices)
        s[prefix+'6'] = s.gh_date6.map(prices)
    s['GH5'] = 100*(s.G1/s.G6-s.Z1/s.Z6)
    s['GH_valid'] = np.isfinite(s[['G1','G6','Z1','Z6']]).all(axis=1) & s[['G1','G6','Z1','Z6']].gt(0).all(axis=1)
    return s.sort_values(['date','code']).reset_index(drop=True)


def features(p):
    source = json.loads((ROOT/'index_source_report.json').read_text())
    assert source['passed'] and source['protocol_sha256'] == sha(PROTOCOL)
    check_sources(source['source_hashes'])
    folder = ROOT/'inputs'; folder.mkdir(exist_ok=False)
    old = pd.read_parquet(p['original_features'])
    assert old.date.ge('2023-01-01').all() and old.date.lt('2026-01-01').all()
    files = [Path(p['stock_daily_root'])/(code.replace('.', '_')+'.parquet') for code in sorted(old.code.unique())]
    assert all(file.exists() for file in files)
    receipts = {str(file):sha(file) for file in files}
    save_json(ROOT/'stock_session_manifest.json', dict(source_sha256=receipts,
        allowed_columns=['date','code','tradestatus'], first=p['index_first'], last=p['index_last']))
    c = numeric.conn()
    stock = c.execute('''SELECT date,code FROM read_parquet(?) WHERE tradestatus=1
        AND date>=? AND date<=? ORDER BY code,date''',
        [list(receipts),p['index_first'],p['index_last']]).df()
    c.close()
    a = align(stock, pd.read_parquet(ROOT/'indices.parquet'))
    a = old[['date','code']].merge(a,on=['date','code'],how='left',validate='one_to_one')
    assert len(a)==len(old) and a[['date','code']].equals(old[['date','code']])
    f = old.copy(); f['GH5'] = a.GH5.where(a.GH_valid.fillna(False))
    f['formula_input_valid'] &= a.GH_valid.fillna(False)
    a.to_parquet(folder/'aligned_indices.parquet',index=False,compression='zstd')
    f.to_parquet(folder/'features.parquet',index=False,compression='zstd')
    definitions = {**baseline.EXPRESSIONS,'GH5':EXPRESSION}
    report = dict(protocol_sha256=sha(PROTOCOL), source_hashes={**source['source_hashes'],**receipts},
        features_sha256=sha(folder/'features.parquet'),aligned_indices_sha256=sha(folder/'aligned_indices.parquet'),
        stock_session_manifest_sha256=sha(ROOT/'stock_session_manifest.json'),
        rows=len(f),valid=int(f.formula_input_valid.sum()),original_valid=int(old.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        expressions=definitions,native_header=HEADER, no_new_fits=True,
        no_new_economic_groups=True,new_2026_prices_read=False,software_compilation_verified=False,
        native_source_parity_verified=False)
    save_json(folder/'feature_report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k in ['rows','valid','original_valid','newly_invalid']},ensure_ascii=False),flush=True)


def verify(p):
    folder = ROOT/'inputs'; report = json.loads((folder/'feature_report.json').read_text())
    assert not (folder/'feature_verification.json').exists()
    check_sources(report['source_hashes'])
    assert report['features_sha256']==sha(folder/'features.parquet')
    assert report['aligned_indices_sha256']==sha(folder/'aligned_indices.parquet')
    c = numeric.conn(); manifest=json.loads((ROOT/'stock_session_manifest.json').read_text())
    c.read_parquet(list(manifest['source_sha256'])).create_view('stock')
    c.read_parquet(str(ROOT/'indices.parquet')).create_view('indices')
    c.read_parquet(p['original_features']).create_view('old')
    rebuilt = c.execute('''WITH h AS(SELECT date,code,lag(date,1) OVER w AS gh_date1,
        lag(date,6) OVER w AS gh_date6 FROM stock WHERE tradestatus=1 AND date>=? AND date<=?
        WINDOW w AS(PARTITION BY code ORDER BY date))
        SELECT o.date,o.code,h.gh_date1,h.gh_date6,g1.close AS G1,g6.close AS G6,
        z1.close AS Z1,z6.close AS Z6,s1.close AS S1,s6.close AS S6
        FROM old o LEFT JOIN h USING(date,code)
        LEFT JOIN indices g1 ON g1.code='sz.399006' AND g1.date=h.gh_date1
        LEFT JOIN indices g6 ON g6.code='sz.399006' AND g6.date=h.gh_date6
        LEFT JOIN indices z1 ON z1.code='sz.399001' AND z1.date=h.gh_date1
        LEFT JOIN indices z6 ON z6.code='sz.399001' AND z6.date=h.gh_date6
        LEFT JOIN indices s1 ON s1.code='sh.000001' AND s1.date=h.gh_date1
        LEFT JOIN indices s6 ON s6.code='sh.000001' AND s6.date=h.gh_date6
        ORDER BY o.date,o.code''',[p['index_first'],p['index_last']]).df()
    a = pd.read_parquet(folder/'aligned_indices.parquet')
    pd.testing.assert_frame_equal(a[rebuilt.columns],rebuilt,check_dtype=False,check_exact=True)
    old=pd.read_parquet(p['original_features']);f=pd.read_parquet(folder/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns],old,check_exact=True)
    calculated=100*((rebuilt.G1-rebuilt.G6)/rebuilt.G6-(rebuilt.Z1-rebuilt.Z6)/rebuilt.Z6)
    np.testing.assert_allclose(f.GH5,calculated.where(a.GH_valid.fillna(False)),rtol=0,atol=2e-12,equal_nan=True)
    valid=old.formula_input_valid
    own=np.where(old.code.str.startswith('sh.'),100*(a.S1/a.S6-1),100*(a.Z1/a.Z6-1))
    np.testing.assert_allclose(old.loc[valid,'I02'],own[valid],rtol=0,atol=2e-12)
    assert a.loc[valid,'gh_date6'].lt(a.loc[valid,'gh_date1']).all()
    assert a.loc[valid,'gh_date1'].lt(a.loc[valid,'date']).all()
    values=f.loc[valid,'GH5'].to_numpy()
    expected=np.floor(np.clip(100*values+10000+.000001,0,999999)).astype('int32')
    encoded=c.sql('''SELECT floor(least(greatest(100*GH5+10000+.000001,0),999999))::INT AS x
        FROM read_parquet('''+"'"+str(folder/'features.parquet')+"'"+''') WHERE formula_input_valid ORDER BY date,code''').df().x.to_numpy()
    np.testing.assert_array_equal(expected,encoded);c.close()
    receipts=dict(report['source_hashes'])
    for file in [folder/'feature_report.json',folder/'features.parquet',folder/'aligned_indices.parquet',ROOT/'stock_session_manifest.json']:
        receipts[str(file)]=sha(file)
    save_json(folder/'feature_verification.json',dict(passed=True,source_hashes=receipts,
        feature_report_sha256=sha(folder/'feature_report.json'), all_stock_session_dates_and_index_prices_SQL_rebuilt=True,
        all_original_I02_references_on_valid_keys_match=True, all_original_fifty_values_and_flags_exact=True,
        all_new_integer_inputs_SQL_verified=True,effective_input_intersection_unchanged=report['newly_invalid']==0,
        no_new_fits=True,no_new_economic_groups=True,new_2026_prices_read=False))
    print('完整股日、六个指数引用、原I02及新整数编码核准',flush=True)


def native(p):
    folder=ROOT/'inputs';proof=json.loads((folder/'feature_verification.json').read_text())
    assert proof['passed'] and proof['effective_input_intersection_unchanged']
    assert proof['feature_report_sha256']==sha(folder/'feature_report.json')
    check_sources(proof['source_hashes'])
    destination=folder/'complete_receipt.json';assert not destination.exists()
    f=pd.read_parquet(folder/'features.parquet')
    frames=[]
    for half, q in f.loc[f.formula_input_valid].groupby('half'):
        q=q.assign(fixed_hash=[hashlib.sha256((d+'|'+c).encode()).hexdigest() for d,c in zip(q.date,q.code)])
        frames.append(q.sort_values('fixed_hash').head(8))
    samples=pd.concat(frames).sort_values(['date','code']).reset_index(drop=True)
    assert len(samples)==48
    prices={}
    for code in INDEX_CODES:
        raw=json.loads((ROOT/'raw'/(code.replace('.','_')+'.json')).read_text())
        prices[code]={row[0]:float(row[-1]) for row in raw['records']}
    records=[]
    for row in samples.itertuples():
        path=Path(p['stock_daily_root'])/(row.code.replace('.','_')+'.parquet')
        stock=pd.read_parquet(path,columns=['date','tradestatus'],
            filters=[('date','>=',p['index_first']),('date','<',row.date)])
        dates=stock.loc[stock.tradestatus.eq(1),'date'].sort_values().tail(6).tolist()
        assert len(dates)==6 and dates[-1]<row.date
        arrays={code:np.concatenate([np.repeat(prices[code][date],241) for date in dates]+
                    [np.repeat(987654.321,230)]) for code in INDEX_CODES}
        def replay(arrays):
            g=arrays['sz.399006'];z=arrays['sz.399001'];b0=230;b5=230+5*241
            return 100*(g[-1-b0]/g[-1-b5]-z[-1-b0]/z[-1-b5])
        value=replay(arrays)
        np.testing.assert_allclose(value,row.GH5,rtol=0,atol=2e-12)
        changed={code:array.copy() for code,array in arrays.items()}
        for array in changed.values():array[-230:]=-12345.678
        assert replay(changed)==value
        scaled={code:array*scale for (code,array),scale in zip(arrays.items(),[10.,.1,100.])}
        np.testing.assert_allclose(replay(scaled),value,rtol=0,atol=2e-12)
        records.append(dict(date=row.date,code=row.code,date1=dates[-1],date6=dates[0],
            raw_source_native_bar_offset_value=value,encoded=int(np.floor(100*value+10000+.000001))))
    save_json(folder/'native_replay.json',dict(passed=True,samples=records,
        counts_are_the_frozen_mathematical_241_and_230_grid_not_verified_client_grid=True,
        all_raw_index_prices_and_stock_dates_rebuilt=True,current_day_values_do_not_affect_input=True,
        independent_index_unit_rescaling_preserves_input=True,software_compilation_verified=False,
        native_source_parity_verified=False))
    receipts=dict(proof['source_hashes'])
    for file in [PROTOCOL,folder/'feature_verification.json',folder/'native_replay.json']:
        receipts[str(file)]=sha(file)
    save_json(destination,dict(passed=True,source_hashes=receipts,rows=len(f),valid=int(f.formula_input_valid.sum()),
        original_fifty_and_flags_unchanged=True,effective_input_intersection_unchanged=True,
        all_new_values_and_encoding_SQL_verified=True,original_I02_reference_control_rebuilt=True,
        samples=48,software_compilation_verified=False,native_source_parity_verified=False,
        no_new_fits=True,no_new_economic_groups=True,new_2026_prices_read=False))
    print(json.dumps(dict(input_complete_sha256=sha(destination))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['collect','features','verify','native'])
    args=parser.parse_args();globals()[args.stage](checked())
