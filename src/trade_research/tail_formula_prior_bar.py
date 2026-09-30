"""Last completed active stock-day return and closing position; fixed input audit."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_morning_range as prior
from .corporate_cash import save_json, sha
from .turnover_reference import CALENDAR

STEM = 'tail_formula_prior_bar'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
INTENT = Path('config') / (STEM + '_intent.json')
CACHE = Path('data/research/limitup_premium_environment/history.parquet')
DAILY_REPORT = Path('data/research/tail_formula_1000/feature_report.json')
DAY_PROOF = Path('data/research/tail_formula_daily_pressure/native_input_verification.json')
META = prior.META
EXTRA_HEADER = ('PYHC:=ROUND(REF(HHV(H,B0),B0)*100);\n'
                'PYLC:=ROUND(REF(LLV(L,B0),B0)*100);\n'
                'PYCC:=ROUND(DCP1*100);\nPYPC:=ROUND(DCP2*100);\n'
                'PYREADY:=PYHC>=PYCC AND PYCC>=PYLC AND PYLC>0 AND PYPC>0 AND VP20>0 '
                'AND ABS(REF(HHV(H,B0),B0)*100-PYHC)<=0.01 '
                'AND ABS(REF(LLV(L,B0),B0)*100-PYLC)<=0.01 '
                'AND ABS(DCP1*100-PYCC)<=0.01 AND ABS(DCP2*100-PYPC)<=0.01;\n')
HEADER = prior.HEADER + EXTRA_HEADER
NEW_EXPRESSIONS = {
    'PY01': 'IF(PYREADY,100*(PYCC/MAX(PYPC,1)-1)/VP20,DRAWNULL)',
    'PY02': 'IF(PYREADY,100*(2*PYCC-PYHC-PYLC)/MAX(PYHC-PYLC,1),DRAWNULL)'}
ARMS = {'control': prior.EXPRESSIONS, 'prior': {**prior.EXPRESSIONS, **NEW_EXPRESSIONS}}
EXPRESSIONS = ARMS['prior']
PRIMITIVES = ['py_prev_date', 'py_comp_date', 'py_high', 'py_low', 'py_close', 'py_comp_close',
              'py_prev_adjustflag', 'py_comp_adjustflag']


def checked():
    p = json.loads(PROTOCOL.read_text()); prior.checked()
    assert p['arms'] == ARMS and p['native_header'] == HEADER and p['intent_sha256'] == sha(INTENT)
    assert not p['new_2026_prices_allowed'] and p['no_new_downloads']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    gate = json.loads(Path(p['conditional_gate']).read_text())
    complete = json.loads(Path(p['conditional_completion']).read_text())
    assert gate['passed'] and not gate['supports_2024_extension'] and complete['passed']
    for file, digest in complete['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    report = json.loads((prior.INPUTS / 'feature_report.json').read_text())
    for kind in ['feature', 'native_input']:
        v = json.loads((prior.INPUTS / (kind + '_verification.json')).read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(prior.INPUTS / 'feature_report.json')
    assert report['features_sha256'] == sha(prior.INPUTS / 'features.parquet')
    coverage = json.loads((ROOT / 'daily_cache_date_coverage_verification.json').read_text())
    assert coverage['passed'] and coverage['cache_sha256'] == sha(CACHE)
    daily = json.loads(DAILY_REPORT.read_text()); cache_report = json.loads(CACHE.with_name('input_report.json').read_text())
    for file, digest in daily['source_sha256'].items():
        assert cache_report['sha256'][file] == digest and sha(Path(file)) == digest, file
    return p


def original():
    f = pd.read_parquet(prior.INPUTS / 'features.parquet')
    assert len(f) == 1258085 and f.date.ge('2024-01-01').all() and f.date.lt('2026-01-01').all()
    return f


def market_gaps(keys):
    cal = pd.read_parquet(CALENDAR)
    dates = sorted(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between('2023-01-01', '2025-12-31'), 'calendar_date'])
    ranks = {day: i for i, day in enumerate(dates)}
    result = keys[['date', 'code']].copy()
    result['py_prev_market_gap'] = keys.date.map(ranks) - keys.py_prev_date.map(ranks)
    result['py_comp_market_gap'] = keys.py_prev_date.map(ranks) - keys.py_comp_date.map(ranks)
    assert result[['py_prev_market_gap', 'py_comp_market_gap']].ge(1).all().all()
    return result


def production_primitives(keys):
    d = pd.read_parquet(CACHE, columns=['date', 'code', 'high', 'low', 'close', 'tradestatus', 'adjustflag'],
                        filters=[('date', '>=', '2023-01-01'), ('date', '<', '2025-12-30')])
    d = d.loc[d.tradestatus.eq(1)].sort_values(['code', 'date'])
    groups = {code: part.reset_index(drop=True) for code, part in d.groupby('code', sort=False)}
    parts = []
    for code, k in keys.groupby('code', sort=False):
        g = groups[code]; idx = np.searchsorted(g.date.to_numpy(), k.date.to_numpy(), side='left') - 1
        assert (idx >= 1).all()
        a, b = g.iloc[idx].reset_index(drop=True), g.iloc[idx - 1].reset_index(drop=True)
        result = k.reset_index(drop=True).copy()
        for name, values in [('py_prev_date', a.date), ('py_comp_date', b.date), ('py_high', a.high),
                             ('py_low', a.low), ('py_close', a.close), ('py_comp_close', b.close),
                             ('py_prev_adjustflag', a.adjustflag), ('py_comp_adjustflag', b.adjustflag)]:
            result[name] = values.to_numpy()
        parts.append(result)
    out = pd.concat(parts, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(out[['date', 'code']], keys, check_exact=True)
    assert out.py_comp_date.lt(out.py_prev_date).all() and out.py_prev_date.lt(out.date).all()
    return out


def prepare():
    checked(); assert not (INPUTS / 'feature_report.json').exists(); INPUTS.mkdir(parents=True, exist_ok=True)
    f = original(); a = production_primitives(f[['date', 'code']]); prices = ['py_high', 'py_low', 'py_close', 'py_comp_close']
    good = np.isfinite(a[prices].to_numpy()).all(axis=1) & a[prices].gt(0).all(axis=1).to_numpy()
    cents = np.floor(a[prices].to_numpy() * 100 + .5)
    good &= (np.abs(a[prices].to_numpy() * 100 - cents) <= .01).all(axis=1)
    good &= (cents[:, 0] >= cents[:, 2]) & (cents[:, 2] >= cents[:, 1])
    good &= a.py_prev_adjustflag.eq(3).to_numpy() & a.py_comp_adjustflag.eq(3).to_numpy()
    good &= np.isfinite(f.V01.to_numpy()) & f.V01.gt(0).to_numpy()
    f['prior_bar_prior_formula_input_valid'] = f.formula_input_valid
    f['prior_bar_input_valid'] = good; f['formula_input_valid'] &= good
    for name in PRIMITIVES:
        f[name] = a[name]
    for i, name in enumerate(['PYHC', 'PYLC', 'PYCC', 'PYPC']):
        f[name] = cents[:, i]
    with np.errstate(all='ignore'):
        f['PY01'] = 100 * (f.PYCC / f.PYPC - 1) / f.V01
        f['PY02'] = 100 * (2 * f.PYCC - f.PYHC - f.PYLC) / np.maximum(f.PYHC - f.PYLC, 1)
    f.loc[~good, ['PYHC', 'PYLC', 'PYCC', 'PYPC', *NEW_EXPRESSIONS]] = np.nan
    gaps = market_gaps(a)
    for name in ['py_prev_market_gap', 'py_comp_market_gap']:
        f[name] = gaps[name]
    a.to_parquet(INPUTS / 'prior_bar_primitives.parquet', index=False, compression='zstd')
    f.to_parquet(INPUTS / 'features.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), features_sha256=sha(INPUTS / 'features.parquet'),
        primitives_sha256=sha(INPUTS / 'prior_bar_primitives.parquet'), daily_cache_sha256=sha(CACHE),
        rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((f.prior_bar_prior_formula_input_valid & ~f.formula_input_valid).sum()),
        prior_market_gap_above_one=int(f.py_prev_market_gap.gt(1).sum()),
        comparison_market_gap_above_one=int(f.py_comp_market_gap.gt(1).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, all_complete_daily_cache_reused=True,
        no_new_downloads=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True,
        software_compilation_verified=False, native_source_parity_verified=False)
    save_json(INPUTS / 'feature_report.json', report)
    for file in ['full_labels.parquet', 'full_label_report.json', 'full_label_verification.json']:
        (INPUTS / file).symlink_to((prior.INPUTS / file).resolve())
    return {key: report[key] for key in ['rows', 'valid', 'newly_invalid', 'prior_market_gap_above_one', 'comparison_market_gap_above_one']}


def verify():
    checked(); report = json.loads((INPUTS / 'feature_report.json').read_text())
    assert report['protocol_sha256'] == sha(PROTOCOL) and report['features_sha256'] == sha(INPUTS / 'features.parquet')
    assert report['primitives_sha256'] == sha(INPUTS / 'prior_bar_primitives.parquet')
    f = pd.read_parquet(INPUTS / 'features.parquet'); old = original()
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    np.testing.assert_array_equal(f.prior_bar_prior_formula_input_valid, old.formula_input_valid)
    daily = json.loads(DAILY_REPORT.read_text()); c = base.conn(); c.execute('SET threads=1'); c.execute("SET memory_limit='4GB'")
    c.read_parquet(list(daily['source_sha256'])).create_view('daily'); c.register('keys', old[['date', 'code', 'V01']])
    expected = c.sql("""WITH active AS(
        SELECT code,date AS py_prev_date,high AS py_high,low AS py_low,close AS py_close,adjustflag AS py_prev_adjustflag,
            lag(date) OVER(PARTITION BY code ORDER BY date) AS py_comp_date,
            lag(close) OVER(PARTITION BY code ORDER BY date) AS py_comp_close,
            lag(adjustflag) OVER(PARTITION BY code ORDER BY date) AS py_comp_adjustflag
        FROM daily WHERE tradestatus=1 AND date>='2023-01-01' AND date<'2025-12-30'),
        selected AS(SELECT k.*,a.* EXCLUDE(code) FROM keys k ASOF LEFT JOIN active a
            ON k.code=a.code AND k.date>a.py_prev_date),
        rounded AS(SELECT *,round(py_high*100) AS hc,round(py_low*100) AS lc,
            round(py_close*100) AS cc,round(py_comp_close*100) AS pc FROM selected),
        quality AS(SELECT *,coalesce(isfinite(py_high) AND py_high>0 AND isfinite(py_low) AND py_low>0
            AND isfinite(py_close) AND py_close>0 AND isfinite(py_comp_close) AND py_comp_close>0
            AND abs(py_high*100-hc)<=.01 AND abs(py_low*100-lc)<=.01 AND abs(py_close*100-cc)<=.01
            AND abs(py_comp_close*100-pc)<=.01 AND hc>=cc AND cc>=lc
            AND py_prev_adjustflag=3 AND py_comp_adjustflag=3 AND isfinite(V01) AND V01>0,false) AS valid FROM rounded)
        SELECT *,CASE WHEN valid THEN 100.*(cc/pc-1)/V01 END AS PY01,
            CASE WHEN valid THEN 100.*(2*cc-hc-lc)/greatest(hc-lc,1) END AS PY02
        FROM quality ORDER BY date,code""").df(); c.close()
    pd.testing.assert_frame_equal(f[['date', 'code']], expected[['date', 'code']], check_exact=True)
    pd.testing.assert_frame_equal(f[PRIMITIVES], expected[PRIMITIVES], check_exact=True)
    np.testing.assert_array_equal(f.prior_bar_input_valid, expected.valid)
    for name, short in zip(['PYHC', 'PYLC', 'PYCC', 'PYPC'], ['hc', 'lc', 'cc', 'pc']):
        np.testing.assert_array_equal(f[name], expected[short].where(expected.valid))
    encode = lambda x: np.floor(np.clip(100*x + 10000 + .000001, 0, 999999)); diff = 0.
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name], expected[name], rtol=0, atol=2e-10, equal_nan=True)
        valid = expected.valid
        np.testing.assert_array_equal(encode(f.loc[valid, name]), encode(expected.loc[valid, name]))
        diff = max(diff, float(np.max(np.abs(f.loc[valid, name] - expected.loc[valid, name]))))
    gaps = market_gaps(expected)
    for name in ['py_prev_market_gap', 'py_comp_market_gap']:
        np.testing.assert_array_equal(f[name], gaps[name])
    final = old.formula_input_valid & expected.valid; np.testing.assert_array_equal(f.formula_input_valid, final)
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len({name.casefold() for name in names})
    out = dict(passed=True, feature_report_sha256=sha(INPUTS / 'feature_report.json'), rows=len(f), valid=int(final.sum()),
        all_50_values_and_all_metadata_unchanged=True, max_difference=diff,
        all_dates_raw_prices_flags_cents_features_and_encodings_independently_verified=True,
        current_day_end_of_day_rows_never_used_for_current_signal=True, invalid_prior_rows_not_skipped=True,
        effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        same_quality_controls_required_before_fitting=not final.equals(old.formula_input_valid),
        no_new_downloads=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'feature_verification.json', out); return out


def native_values(previous, current_bars=230, outside=1000000.):
    """Replay literal dynamic REF and HHV/LLV over two completed stock days."""
    arrays = {name: [] for name in ['H', 'L', 'C', 'B0']}
    for day in previous:
        n = day.get('bars', 241)
        for name, column in [('H', 'high'), ('L', 'low'), ('C', 'close')]:
            arrays[name].extend([day[column]] * n)
        arrays['B0'].extend(range(1, n + 1))
    for name in ['H', 'L', 'C']:
        arrays[name].extend([outside] * current_bars)
    arrays['B0'].extend(range(1, current_bars + 1))
    env = {name: np.asarray(values, float) for name, values in arrays.items()}
    n = len(env['C'])
    def ref(values, offsets):
        values = np.broadcast_to(values, n); offsets = np.broadcast_to(offsets, n)
        result = np.full(n, np.nan)
        for i, count in enumerate(offsets):
            if np.isfinite(count) and i >= int(count):
                result[i] = values[i - int(count)]
        return result
    def rolling(values, lengths, method):
        lengths = np.broadcast_to(lengths, n); result = np.full(n, np.nan)
        for i, count in enumerate(lengths):
            if np.isfinite(count) and count >= 1 and i + 1 >= int(count):
                result[i] = method(values[i + 1 - int(count):i + 1])
        return result
    env.update(REF=ref, HHV=lambda a,b:rolling(a,b,np.max), LLV=lambda a,b:rolling(a,b,np.min),
        ROUND=lambda a:np.floor(a + .5), MAX=np.maximum, ABS=np.abs, IF=np.where, DRAWNULL=np.nan,
        VP20=2., B1=env['B0'] + ref(env['B0'], env['B0']))
    env['DCP1'] = ref(env['C'], env['B0']); env['DCP2'] = ref(env['C'], env['B1'])
    for line in EXTRA_HEADER.splitlines():
        name, expression = line.rstrip(';').split(':=')
        if name == 'PYREADY':
            expression = ' & '.join('(' + part + ')' for part in expression.split(' AND '))
        env[name] = eval(expression, {'__builtins__': {}}, env)
    with np.errstate(all='ignore'):
        return np.asarray([np.asarray(eval(expression, {'__builtins__': {}}, env))[-1] for expression in NEW_EXPRESSIONS.values()])


def native():
    checked(); fv = json.loads((INPUTS / 'feature_verification.json').read_text())
    assert fv['passed'] and fv['feature_report_sha256'] == sha(INPUTS / 'feature_report.json')
    f = pd.read_parquet(INPUTS / 'features.parquet'); encode = lambda x:np.floor(np.clip(100*x + 10000 + .000001, 0, 999999))
    for start in range(0, len(f), 50000):
        d = f.iloc[start:start + 50000]; env = {name:d[name].to_numpy() for name in ['PYHC', 'PYLC', 'PYCC', 'PYPC']}
        env.update(PYREADY=d.prior_bar_input_valid.to_numpy(), VP20=d.V01.to_numpy(), MAX=np.maximum, IF=np.where, DRAWNULL=np.nan)
        for name, expression in NEW_EXPRESSIONS.items():
            with np.errstate(all='ignore'):
                values = eval(expression, {'__builtins__': {}}, env)
            np.testing.assert_allclose(values, d[name], rtol=0, atol=2e-10, equal_nan=True)
            valid = d.prior_bar_input_valid.to_numpy()
            np.testing.assert_array_equal(encode(values[valid]), encode(d.loc[valid, name]))
    proof = json.loads(DAY_PROOF.read_text()); assert proof['passed'] and proof['samples'] == 32
    assert proof['complete_prior_day_high_low_close_and_new_native_expression_verified']
    daily = json.loads(DAILY_REPORT.read_text()); index = f.set_index(['date', 'code']); cases = []
    for case in proof['cases']:
        row = index.loc[(case['date'], case['code'])]
        assert row.prior_bar_input_valid and row.py_prev_date == case['last'] and row.py_comp_date >= case['first']
        path = Path('data/baostock/market_2020_2026/daily') / (case['code'].replace('.', '_') + '.parquet')
        assert sha(path) == daily['source_sha256'][str(path)]
        days = pd.read_parquet(path, columns=['date', 'high', 'low', 'close', 'tradestatus'],
            filters=[('date', '>=', case['first']), ('date', '<', case['date'])])
        days = days.loc[days.tradestatus.eq(1)].sort_values('date').tail(2)
        assert days.date.tolist() == [row.py_comp_date, row.py_prev_date]
        result = native_values(days.to_dict('records'))
        result[0] *= 2 / row.V01
        np.testing.assert_allclose(result, row[list(NEW_EXPRESSIONS)].to_numpy(float), rtol=0, atol=2e-10)
        np.testing.assert_array_equal(encode(result), encode(row[list(NEW_EXPRESSIONS)].to_numpy(float)))
        cases.append(dict(date=case['date'], code=case['code'], prior_date=row.py_prev_date,
            comparison_date=row.py_comp_date, old_raw_high_low_close_proof_reused=True, encoded_equal=True))
    flat = [dict(high=10., low=10., close=10.)] * 2
    np.testing.assert_array_equal(native_values(flat), [0., 0.])
    up = [flat[0], dict(high=10.2, low=9.8, close=10.1)]
    np.testing.assert_allclose(native_values(up), [.5, 50.], rtol=0, atol=2e-10)
    down = [flat[0], dict(high=10.2, low=9.8, close=9.9, bars=120)]
    np.testing.assert_allclose(native_values(down), [-.5, -50.], rtol=0, atol=2e-10)
    scaled = [{key:value*5 if key != 'bars' else value for key, value in day.items()} for day in up]
    np.testing.assert_allclose(native_values(scaled), native_values(up), rtol=0, atol=2e-10)
    for count in [1, 121, 230]:
        np.testing.assert_allclose(native_values(up, current_bars=count, outside=0.), [.5, 50.], rtol=0, atol=2e-10)
        np.testing.assert_allclose(native_values(up, current_bars=count, outside=1e9), [.5, 50.], rtol=0, atol=2e-10)
    assert np.isnan(native_values([flat[0], dict(high=9., low=10., close=10.)])).all()
    assert np.isnan(native_values([dict(high=0., low=0., close=0.), flat[0]])).all()
    assert np.isnan(native_values([flat[0], dict(high=10.2, low=9.8, close=10.101)])).all()
    source = prior.INPUTS / 'native_input_verification.json'
    out = dict(passed=True, feature_report_sha256=sha(INPUTS / 'feature_report.json'),
        feature_verification_sha256=sha(INPUTS / 'feature_verification.json'), all_new_literal_values_and_encodings_checked=2*len(f),
        reused_50_native_proof_sha256=sha(source), reused_complete_daily_high_low_close_proof_sha256=sha(DAY_PROOF),
        all_50_values_and_daily_versions_unchanged_before_proof_reuse=True,
        literal_dynamic_REF_and_previous_day_HHV_LLV_integer_cents_and_stock_day_boundaries_verified=True,
        source_cases=cases, source_sample_days=len(cases), flat_up_down_short_day_scale_invalid_and_outside_checks_passed=True,
        no_new_raw_minute_extraction=True, software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'native_input_verification.json', out); return {key:value for key,value in out.items() if key != 'source_cases'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage', choices=['prepare', 'verify', 'native'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
