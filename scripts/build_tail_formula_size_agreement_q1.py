"""Build and independently verify the unchanged two size inputs on cached Q1 rows."""
import argparse
import json
import struct

import numpy as np
import pandas as pd

from verify_index_minute_probe import replay
from trade_research import tail_formula_size_agreement_q1 as study
from trade_research.corporate_cash import save_json, sha
from trade_research.turnover_reference import CALENDAR


def sources():
    p = study.policy(); study.checked_model(); root = study.ROOT / 'indices'
    reports = [json.loads((root / (name + '_report.json')).read_text()) for name in ['daily', 'minute']]
    for r in reports:
        assert r['protocol_sha256'] == sha(study.PROTOCOL)
        assert r['model_freeze_report_sha256'] == sha(study.ROOT / 'model_freeze_report.json')
    oldr = json.loads((study.OLD / 'inputs/feature_report.json').read_text())
    assert oldr['features_sha256'] == sha(study.OLD / 'inputs/features.parquet')
    old = pd.read_parquet(study.OLD / 'inputs/features.parquet')
    assert old.date.between(p['signal_first'], p['signal_last']).all()
    assert len(old) == 161781 and int(old.formula_input_valid.sum()) == 139353
    return p, root, reports, old


def build():
    p, root, (dr, mr), old = sources(); out = study.OUT
    assert not (out / 'feature_report.json').exists(); out.mkdir(exist_ok=True)
    assert dr['daily_sha256'] == sha(root / 'daily.parquet')
    daily = pd.read_parquet(root / 'daily.parquet'); indexed = daily.set_index(['code', 'date'])
    points = []; audits = []
    assert {(s['symbol'], s['date']) for s in mr['sessions']} == {(code, day) for code in p['symbols'] for day in old.date.unique()}
    for s in mr['sessions']:
        rows = []
        if 'path' in s:
            from pathlib import Path
            assert sha(Path(s['path'])) == s['sha256']; rows = json.loads(Path(s['path']).read_text())
        points.append(dict(date=s['date'], index_code=s['symbol'], **study.extended.prefix_points(rows)))
        d = indexed.loc[s['symbol'], s['date']]; prices = np.array([x['price_raw'] / 100 for x in rows])
        full = bool(len(prices) == 240 and (prices > 0).all() and (prices >= d.low - .0051).all()
            and (prices <= d.high + .0051).all() and abs(prices[-1] - d.close) <= .0051)
        audits.append(dict(date=s['date'], index_code=s['symbol'], rows=len(rows), full_day_source_valid=full))
    points = pd.DataFrame(points).sort_values(['date', 'index_code']).reset_index(drop=True)
    f = old.copy(); f['sc_prior_date'] = f.float_source_date
    for name, code in [('small', 'sh.000852'), ('large', 'sh.000300')]:
        prior = daily.loc[daily.code.eq(code), ['date', 'close']].rename(columns={'date':'sc_prior_date', 'close':f'sc_{name}_prior'})
        f = f.merge(prior, on='sc_prior_date', how='left', validate='many_to_one')
        q = points.loc[points.index_code.eq(code)].drop(columns='index_code').rename(columns={k:f'sc_{name}_{k}' for k in ['p48','p20','prefix_valid']})
        f = f.merge(q, on='date', how='left', validate='many_to_one')
    values = f[['sc_small_prior','sc_large_prior','sc_small_p48','sc_small_p20','sc_large_p48','sc_large_p20']]
    valid = (f.sc_prior_date.lt(f.date) & np.isfinite(values).all(axis=1) & values.gt(0).all(axis=1)
        & f.sc_small_prefix_valid.fillna(False) & f.sc_large_prefix_valid.fillna(False))
    f['SZ01'] = (100 * (f.sc_small_p48/f.sc_small_prior - f.sc_large_p48/f.sc_large_prior)).where(valid)
    f['SZ02'] = (100 * (f.sc_small_p48/f.sc_small_p20 - f.sc_large_p48/f.sc_large_p20)).where(valid)
    f['size_context_valid'] = valid; f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(study.extended.EXPRESSIONS)]).all(axis=1)
    f = f.sort_values(['date','code']).reset_index(drop=True)
    audit = pd.DataFrame(audits).sort_values(['date','index_code']).reset_index(drop=True)
    for name, table in [('features',f),('index_points',points),('source_audit',audit)]:
        table.to_parquet(out/(name+'.parquet'), index=False, compression='zstd')
    r = dict(protocol_sha256=sha(study.PROTOCOL), model_freeze_report_sha256=sha(study.ROOT/'model_freeze_report.json'),
        original_feature_report_sha256=sha(study.OLD/'inputs/feature_report.json'), daily_report_sha256=sha(root/'daily_report.json'),
        minute_report_sha256=sha(root/'minute_report.json'), features_sha256=sha(out/'features.parquet'),
        index_points_sha256=sha(out/'index_points.parquet'), source_audit_sha256=sha(out/'source_audit.parquet'),
        rows=len(f), valid=int(f.formula_input_valid.sum()), newly_invalid=int((f.prior_formula_input_valid & ~f.formula_input_valid).sum()),
        index_sessions=len(points), invalid_prefix_sessions=int((~points.prefix_valid).sum()), full_day_source_conflicts=int((~audit.full_day_source_valid).sum()),
        expressions=study.extended.EXPRESSIONS, native_header=study.extended.HEADER, original_48_inputs_unchanged=True,
        full_day_source_audit_used_for_selection=False, prior_q1_original_outcomes_exposed=True, new_2026_index_prices_read=True,
        new_2026_stock_prices_read=False, new_intersection_outcomes_read=False, strict_blind=False, no_exit_rules=True)
    save_json(out/'feature_report.json',r); return {k:v for k,v in r.items() if k not in ['expressions','native_header']}


def verify():
    from pathlib import Path
    p, root, (dr, mr), old = sources(); out = study.OUT
    r = json.loads((out/'feature_report.json').read_text()); actual = pd.read_parquet(out/'features.parquet')
    for key,path in [('protocol_sha256',study.PROTOCOL),('features_sha256',out/'features.parquet'),
                     ('index_points_sha256',out/'index_points.parquet'),('source_audit_sha256',out/'source_audit.parquet'),
                     ('daily_report_sha256',root/'daily_report.json'),('minute_report_sha256',root/'minute_report.json')]:
        assert r[key] == sha(path)
    frames = []
    cal = pd.read_parquet(CALENDAR)
    days = cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between(p['warmup_first'],p['signal_last']),'calendar_date'].tolist()
    for name,digest in dr['raw_files_sha256'].items():
        assert sha(Path(name)) == digest; d = json.loads(Path(name).read_text())
        assert (d['first'],d['last']) == (p['warmup_first'],p['signal_last']) and d['code'] in p['symbols']
        f = pd.DataFrame(d['records']); assert f.date.tolist() == days
        for field in ['open','high','low','close']:
            f[field] = f[field].astype(float)
        assert f.code.eq(d['code']).all() and np.isfinite(f[['open','high','low','close']]).all().all()
        assert f.low.gt(0).all() and f.low.le(f[['open','close']].min(axis=1)).all() and f.high.ge(f[['open','close']].max(axis=1)).all()
        frames.append(f)
    daily = pd.concat(frames).sort_values(['code','date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(daily,pd.read_parquet(root/'daily.parquet'),check_exact=True)
    points = []; audits = []; perturb = 0; wire_count = 0
    for s in mr['sessions']:
        rows = []
        if 'path' in s:
            assert sha(Path(s['path'])) == s['sha256']
            stored = json.loads(Path(s['path']).read_text())
            assert s['wire']
            for wire in s['wire']:
                assert sha(Path(wire['response'])) == wire['response_sha256'] and sha(Path(wire['request'])) == wire['request_sha256']
                packet = Path(wire['request']).read_bytes()
                assert packet[:12] == bytes.fromhex('0c01300001010d000d00b40f')
                assert struct.unpack('<IB6s',packet[12:]) == (int(s['date'].replace('-','')),1,s['symbol'][3:].encode())
                rows = replay(Path(wire['response']).read_bytes()); assert rows == stored; wire_count += 1
        valid = len(rows) >= 228 and all(x['sequence']==i and x['price_raw']>0 for i,x in enumerate(rows[:228]))
        points.append(dict(date=s['date'],index_code=s['symbol'],p48=rows[227]['price_raw']/100 if valid else np.nan,
            p20=rows[199]['price_raw']/100 if valid else np.nan,prefix_valid=valid))
        d = daily.set_index(['code','date']).loc[s['symbol'],s['date']]; prices = [x['price_raw']/100 for x in rows]
        full = len(prices)==240 and all(d.low-.0051<=v<=d.high+.0051 and v>0 for v in prices) and abs(prices[-1]-d.close)<=.0051
        audits.append(dict(date=s['date'],index_code=s['symbol'],rows=len(rows),full_day_source_valid=full))
        if prices:
            assert full, 'Investigate source contradiction; never drop dates based on outcomes'
        if valid:
            changed = [dict(x) for x in rows]
            for x in changed[228:]:
                x['price_raw'] = -999999; x['sequence'] = -1
            assert study.extended.prefix_points(rows) == study.extended.prefix_points(changed); perturb += 1
    points = pd.DataFrame(points).sort_values(['date','index_code']).reset_index(drop=True)
    audit = pd.DataFrame(audits).sort_values(['date','index_code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(points,pd.read_parquet(out/'index_points.parquet'),check_exact=True)
    pd.testing.assert_frame_equal(audit,pd.read_parquet(out/'source_audit.parquet'),check_exact=True)
    pd.testing.assert_frame_equal(actual[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    c = study.base.conn(); c.register('old',old); c.register('d',daily); c.register('pt',points)
    expected = c.sql('''SELECT o.date,o.code,o.float_source_date AS sc_prior_date,
        s.close AS sc_small_prior,l.close AS sc_large_prior,spt.p48 AS sc_small_p48,spt.p20 AS sc_small_p20,
        lpt.p48 AS sc_large_p48,lpt.p20 AS sc_large_p20,spt.prefix_valid AS sc_small_prefix_valid,lpt.prefix_valid AS sc_large_prefix_valid,
        100*(spt.p48*l.close-lpt.p48*s.close)/(s.close*l.close) AS SZ01,
        100*(spt.p48*lpt.p20-lpt.p48*spt.p20)/(spt.p20*lpt.p20) AS SZ02
        FROM old o LEFT JOIN d s ON s.code='sh.000852' AND s.date=o.float_source_date
        LEFT JOIN d l ON l.code='sh.000300' AND l.date=o.float_source_date
        LEFT JOIN pt spt ON spt.index_code='sh.000852' AND spt.date=o.date
        LEFT JOIN pt lpt ON lpt.index_code='sh.000300' AND lpt.date=o.date ORDER BY o.date,o.code''').df(); c.close()
    values = expected[['sc_small_prior','sc_large_prior','sc_small_p48','sc_small_p20','sc_large_p48','sc_large_p20']]
    valid = np.isfinite(values).all(axis=1) & values.gt(0).all(axis=1) & expected.sc_prior_date.lt(expected.date)
    valid &= expected.sc_small_prefix_valid.fillna(False) & expected.sc_large_prefix_valid.fillna(False)
    expected[['SZ01','SZ02']] = expected[['SZ01','SZ02']].where(valid, np.nan)
    pd.testing.assert_frame_equal(actual[expected.columns],expected,check_dtype=False,rtol=0,atol=2e-12)
    final = old.formula_input_valid & valid
    assert actual.prior_formula_input_valid.equals(old.formula_input_valid) and actual.formula_input_valid.equals(final)
    for name,expression in study.extended.NEW_EXPRESSIONS.items():
        variables = dict(SM48=expected.sc_small_p48,SMP=expected.sc_small_prior,SM20=expected.sc_small_p20,
            LG48=expected.sc_large_p48,LGP=expected.sc_large_prior,LG20=expected.sc_large_p20)
        np.testing.assert_allclose(eval(expression,{'__builtins__':{}},variables).where(valid),expected[name],rtol=0,atol=2e-12,equal_nan=True)
    for name in ['SZ01','SZ02']:
        np.testing.assert_array_equal(np.floor(np.clip(100*actual.loc[final,name]+10000+.000001,0,999999)),
            np.floor(np.clip(100*expected.loc[final,name]+10000+.000001,0,999999)))
    assert r['valid']==int(final.sum()) and r['newly_invalid']==int((old.formula_input_valid & ~final).sum())
    assert r['full_day_source_conflicts']==int((~audit.full_day_source_valid).sum()) and r['index_sessions']==len(points)
    result = dict(passed=True,feature_report_sha256=sha(out/'feature_report.json'),rows=len(actual),valid=int(final.sum()),
        wire_replays=wire_count,after_cutoff_perturbation_checks=perturb,original_48_inputs_unchanged=True,
        stock_prior_dates_reused_from_verified_original_inputs=True,all_new_values_validity_encoding_and_native_arithmetic_rebuilt=True,
        new_2026_index_prices_read=True,new_2026_stock_prices_read=False,new_intersection_outcomes_read=False,strict_blind=False,no_exit_rules=True)
    save_json(out/'feature_verification.json',result); return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('stage',choices=['build','verify'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
