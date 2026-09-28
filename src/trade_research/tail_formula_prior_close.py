"""Strictly previous closing-window volume and price, with independent replay."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_float as previous
from . import tail_formula_prior_close_raw as raw
from .corporate_cash import MINUTES, save_json, sha
from .turnover_reference import CALENDAR

STEM, ROOT, PROTOCOL = raw.STEM, raw.ROOT, raw.PROTOCOL
CONTROL_INPUTS = Path('data/research') / (STEM + '_control_inputs')
PROBES = Path('data/research/tail_formula_daily_efficiency/native_input_verification.json')
NEW_EXPRESSIONS = {
    'CA01': 'REF((SUM(V,4)/4)/MAX((SUM(V,29)-SUM(V,4))/25,1),B0)',
    'CA02': '100*(INTPART(DCP1*100+0.5)/MAX(INTPART(REF(C,B0+4)*100+0.5),1)-1)/V01'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER + 'CAP:=REF(SUM(V,29)-SUM(V,4),B0);\n'
GATE = 'CAP>=25 AND REF(TIME,B0)=1500 AND REF(TIME,B0+4)=1456 AND REF(TIME,B0+28)=1432'
ORIGINAL_NATIVE_CORE = base.native_core
CLOCKS = [f'14{i:02d}' for i in range(32, 60)] + ['1500']


def native_core(*args, **kwargs):
    text = ORIGINAL_NATIVE_CORE(*args, **kwargs)
    assert text.count('CORE:SC>') == 1
    return text.replace('CORE:SC>', 'CORE:' + GATE + ' AND SC>')


def checked_raw():
    p, m = raw.checked()
    r = json.loads((ROOT / 'raw_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['producer_sha256'] == sha(Path(raw.__file__))
    assert r['schedule_report_sha256'] == sha(ROOT / 'schedule_report.json')
    for file, digest in r['parts_sha256'].items():
        assert sha(Path(file)) == digest
    return p, m, r


def aggregate(f):
    f = f.copy()
    prices = f[['open', 'high', 'low', 'close']].to_numpy(float)
    cents = np.rint(prices * 100)
    good = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1) & (np.abs(prices-cents/100) <= .0001).all(axis=1)
    good &= (cents[:, 1] >= cents.max(axis=1)) & (cents[:, 2] <= cents.min(axis=1))
    good &= np.isfinite(f.volume) & f.volume.ge(0) & f.volume.eq(np.floor(f.volume))
    good &= f.timestamp.eq(f.timestamp.dt.floor('min')) & f.clock.isin(CLOCKS)
    f['good'] = good
    f['volume25'] = f.volume.where(f.clock.le('1456'), 0)
    f['volume4'] = f.volume.where(f.clock.ge('1457'), 0)
    f['p56c'] = np.where(f.clock.eq('1456'), cents[:, 3], np.nan)
    f['p00c'] = np.where(f.clock.eq('1500'), cents[:, 3], np.nan)
    out = f.groupby(['code', 'date'], sort=True).agg(bars=('clock', 'size'), clocks=('clock', 'nunique'), good_bars=('good', 'sum'),
        first=('clock', 'min'), last=('clock', 'max'), volume25=('volume25', 'sum'), volume4=('volume4', 'sum'),
        p56c=('p56c', 'max'), p00c=('p00c', 'max')).reset_index()
    out['window_complete'] = out.bars.eq(29) & out.clocks.eq(29) & out.good_bars.eq(29) & out['first'].eq('1432') & out['last'].eq('1500')
    out['window_valid'] = out.window_complete & out.volume25.ge(2500)
    return out


def windows():
    p, m, r = checked_raw()
    assert not (ROOT / 'window_report.json').exists()
    out = pd.concat([aggregate(pd.read_parquet(file)) for file in r['parts_sha256']], ignore_index=True).sort_values(['code', 'date']).reset_index(drop=True)
    assert not out.duplicated(['date', 'code']).any() and int(out.bars.sum()) == r['raw_minutes']
    out.to_parquet(ROOT / 'windows.parquet', index=False, compression='zstd')
    report = dict(raw_report_sha256=sha(ROOT / 'raw_report.json'), windows_sha256=sha(ROOT / 'windows.parquet'), rows=len(out),
        complete=int(out.window_complete.sum()), valid=int(out.window_valid.sum()),
        below_native_volume_floor=int((out.window_complete & ~out.window_valid).sum()), raw_minutes=r['raw_minutes'])
    save_json(ROOT / 'window_report.json', report)
    return report


def features():
    p, m, raw_report = checked_raw()
    assert not (ROOT / 'feature_report.json').exists()
    wr = json.loads((ROOT / 'window_report.json').read_text())
    assert wr['raw_report_sha256'] == sha(ROOT / 'raw_report.json') and wr['windows_sha256'] == sha(ROOT / 'windows.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    mapping = pd.read_parquet(ROOT / 'prior_dates.parquet')
    windows = pd.read_parquet(ROOT / 'windows.parquet')
    aliases = {name: 'pc_' + name for name in windows if name not in ['date', 'code']}
    assert not set(aliases.values()).intersection(old.columns)
    windows = windows.rename(columns={**aliases, 'date': 'source_date'})
    f = old.merge(mapping, on=['date', 'code'], validate='one_to_one').merge(windows, on=['code', 'source_date'], how='left', validate='one_to_one')
    f = f.sort_values(['date', 'code']).reset_index(drop=True)
    good = f.pc_window_valid.fillna(False).astype(bool) & f.source_date.lt(f.date) & np.isfinite(f.V01) & f.V01.gt(0)
    f['prior_close_valid'] = good
    f['CA01'] = ((f.pc_volume4/4) / (f.pc_volume25/25)).where(good)
    f['CA02'] = (100*(f.pc_p00c/f.pc_p56c-1)/f.V01).where(good)
    f['prior_formula_input_valid'] = old.formula_input_valid
    f['formula_input_valid'] &= good
    cal = pd.read_parquet(CALENDAR)
    dates = sorted(cal.loc[cal.is_trading_day.eq('1'), 'calendar_date'])
    ranks = {d: i for i, d in enumerate(dates)}
    f['source_market_gap'] = f.date.map(ranks)-f.source_date.map(ranks)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    keys = pd.read_parquet(labels.ROOT / 'full_labels.parquet', columns=['date', 'code', 'next_date', 'known15'])
    train = f.loc[f.formula_input_valid, ['date', 'code']].merge(keys.loc[keys.known15], on=['date', 'code'], validate='one_to_one')
    counts = {}
    for fold, spec in p['folds'].items():
        z = train.loc[train.date.ge(spec['training_start']) & train.next_date.lt(spec['training_end'])]
        counts[fold] = dict(rows=len(z), days=z.date.nunique(), last_observation=z.next_date.max())
    report = dict(protocol_sha256=sha(PROTOCOL), window_report_sha256=sha(ROOT / 'window_report.json'),
        schedule_report_sha256=sha(ROOT / 'schedule_report.json'), features_sha256=sha(ROOT / 'features.parquet'),
        rows=len(f), valid=int(f.formula_input_valid.sum()), newly_invalid=int((old.formula_input_valid & ~good).sum()),
        unreliable_window_from_previously_valid=int((old.formula_input_valid & ~f.pc_window_complete.fillna(False)).sum()),
        volume_floor_from_previously_valid=int((old.formula_input_valid & f.pc_window_complete.fillna(False) & f.pc_volume25.lt(2500)).sum()),
        valid_source_gaps=int((f.formula_input_valid & f.source_market_gap.gt(1)).sum()), training_counts=counts,
        calendar_sha256=sha(CALENDAR), expressions=EXPRESSIONS, native_header=HEADER, native_core_gate=GATE,
        native_source_parity_verified=False, software_compilation_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', report)
    return {k: v for k, v in report.items() if k not in ['expressions', 'native_header']}


def verify_features():
    p, m, raw_report = checked_raw()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT / 'features.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_names=False, check_exact=True)
    c = base.conn()
    checks = ' AND '.join(f'isfinite({x}) AND {x}>0 AND abs({x}-round({x}*100)/100)<=.0001' for x in ['open', 'high', 'low', 'close'])
    sql = f'''WITH g AS(SELECT *,coalesce({checks} AND round(high*100)>=greatest(round(open*100),round(low*100),round(close*100))
        AND round(low*100)<=least(round(open*100),round(high*100),round(close*100))
        AND isfinite(volume) AND volume>=0 AND volume=floor(volume) AND timestamp=date_trunc('minute',timestamp)
        AND strftime(timestamp,'%H%M')=clock AND clock BETWEEN '1432' AND '1500',false) AS good FROM raw),
        a AS(SELECT code,date,count(*) AS bars,count(DISTINCT clock) AS clocks,sum(good::INT) AS good_bars,
        min(clock) AS first,max(clock) AS last,sum(volume) FILTER(WHERE clock<='1456') AS volume25,
        sum(volume) FILTER(WHERE clock>='1457') AS volume4,max(round(close*100)) FILTER(WHERE clock='1456') AS p56c,
        max(round(close*100)) FILTER(WHERE clock='1500') AS p00c FROM g GROUP BY code,date)
        SELECT *,bars=29 AND clocks=29 AND good_bars=29 AND first='1432' AND last='1500' AS window_complete,
        window_complete AND volume25>=2500 AS window_valid FROM a ORDER BY code,date'''
    parts = []
    for file in raw_report['parts_sha256']:
        c.read_parquet(file).create_view('raw', replace=True)
        parts.append(c.sql(sql).df())
    w = pd.concat(parts, ignore_index=True).sort_values(['code', 'date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(pd.read_parquet(ROOT / 'windows.parquet'), w, check_dtype=False, rtol=0, atol=0)
    c.register('windows', w)
    expected = c.sql(f'''WITH d AS(SELECT date,code,lag(date) OVER(PARTITION BY code ORDER BY date) AS source_date
        FROM read_parquet('{raw.SCHEDULE}/stock_days.parquet'))
        SELECT f.date,f.code,d.source_date,w.* EXCLUDE(code,date),coalesce(w.window_valid AND d.source_date<f.date AND isfinite(f.V01) AND f.V01>0,false) AS good,
        CASE WHEN good THEN 25*volume4/(4*volume25) END AS ca01,
        CASE WHEN good THEN 100*(p00c-p56c)/p56c/f.V01 END AS ca02,
        f.formula_input_valid AND good AS final_valid
        FROM read_parquet('{previous.ROOT}/features.parquet') f JOIN d USING(date,code)
        LEFT JOIN windows w ON w.code=f.code AND w.date=d.source_date ORDER BY f.date,f.code''').df()
    pd.testing.assert_series_equal(f.source_date, expected.source_date, check_exact=True)
    for name in w.columns.drop(['date', 'code']):
        pd.testing.assert_series_equal(f['pc_' + name], expected[name], check_names=False, check_dtype=False, check_exact=True)
    np.testing.assert_array_equal(f.prior_close_valid, expected.good)
    np.testing.assert_array_equal(f.formula_input_valid, expected.final_valid)
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name], expected[name.lower()], rtol=0, atol=2e-12, equal_nan=True)
    enc = lambda a: np.floor(np.clip(100*np.asarray(a)+10000+.000001, 0, 999999)).astype('int32')
    for name in NEW_EXPRESSIONS:
        np.testing.assert_array_equal(enc(f.loc[f.formula_input_valid, name]), enc(expected.loc[expected.final_valid, name.lower()]))
    # The software uses lots. Check the actual denominator floor over all valid inputs.
    g = f.loc[f.formula_input_valid]
    native_ratio = (g.pc_volume4/100/4) / np.maximum(g.pc_volume25/100/25, 1)
    np.testing.assert_allclose(native_ratio, g.CA01, rtol=0, atol=2e-10)
    np.testing.assert_array_equal(enc(native_ratio), enc(g.CA01))
    c.register('flags', expected[['date', 'code', 'final_valid']])
    assert r['calendar_sha256'] == sha(CALENDAR)
    gaps = c.sql(f'''WITH dates AS(SELECT calendar_date,row_number() OVER(ORDER BY calendar_date) AS n
        FROM read_parquet('{CALENDAR}') WHERE is_trading_day='1')
        SELECT f.date,f.code,a.n-b.n AS gap FROM read_parquet('{ROOT}/features.parquet') f
        JOIN dates a ON a.calendar_date=f.date JOIN dates b ON b.calendar_date=f.source_date ORDER BY f.date,f.code''').df()
    pd.testing.assert_frame_equal(f[['date', 'code']], gaps[['date', 'code']], check_exact=True)
    np.testing.assert_array_equal(f.source_market_gap, gaps.gap)
    for fold, spec in p['folds'].items():
        counts = c.sql(f'''SELECT count(*) AS rows,count(DISTINCT l.date) AS days,max(l.next_date) AS last_observation
            FROM read_parquet('{labels.ROOT}/full_labels.parquet') l JOIN flags f USING(date,code)
            WHERE l.known15 AND f.final_valid AND l.date>='{spec['training_start']}' AND l.next_date<'{spec['training_end']}' ''').df().iloc[0].to_dict()
        assert counts == r['training_counts'][fold]
    c.close()
    assert r['valid'] == int(f.formula_input_valid.sum()) and r['newly_invalid'] == int((old.formula_input_valid & ~expected.final_valid).sum())
    assert r['unreliable_window_from_previously_valid'] == int((old.formula_input_valid & ~expected.window_complete.fillna(False)).sum())
    assert r['volume_floor_from_previously_valid'] == int((old.formula_input_valid & expected.window_complete.fillna(False) & expected.volume25.lt(2500)).sum())
    assert r['valid_source_gaps'] == int((expected.final_valid & gaps.gap.gt(1)).sum())
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER and r['native_core_gate'] == GATE
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names})
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f), valid=r['valid'],
        raw_window_checks_and_all_lag_mappings_independently_rebuilt=True, original_48_values_and_quality_preserved=True,
        all_new_features_native_lot_units_and_encodings_rebuilt=True, all_training_counts_rebuilt=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof)
    return proof


def native_values(close, lots, clocks, current_bars, atr):
    def ref(a, n):
        a = np.asarray(a)
        return np.r_[np.full(int(n), np.nan), a[:-int(n)]]
    env = dict(C=np.asarray(close), V=np.asarray(lots), TIME=np.asarray(clocks), B0=current_bars, V01=atr,
        REF=ref, SUM=lambda x, n: pd.Series(x).rolling(int(n)).sum().to_numpy(), MAX=np.maximum, INTPART=np.trunc)
    env['DCP1'] = ref(env['C'], current_bars)
    env['CAP'] = ref(env['SUM'](env['V'], 29)-env['SUM'](env['V'], 4), current_bars)
    values = [float(eval(expr, {'__builtins__': {}}, env)[-1]) for expr in NEW_EXPRESSIONS.values()]
    gate = all(bool(eval(part.replace('=1500', '==1500').replace('=1456', '==1456').replace('=1432', '==1432'), {'__builtins__': {}}, env)[-1]) for part in GATE.split(' AND '))
    return values, gate


def native():
    p, m, r = checked_raw()
    v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    samples = json.loads(PROBES.read_text())['samples']
    assert len(samples) == 32
    f = pd.read_parquet(ROOT / 'features.parquet').set_index(['date', 'code'])
    cases, total = [], 0
    for sample in samples:
        day, code = sample['date'], sample['code']
        row = f.loc[(day, code)]
        path = MINUTES / code[:2].upper() / (code[3:] + '.parquet')
        assert sha(path) == sample['source_sha256']
        q = pd.read_parquet(path, columns=['timestamp', 'close', 'volume'], filters=[
            ('timestamp', '>=', pd.Timestamp(row.source_date)), ('timestamp', '<=', pd.Timestamp(day + ' 14:49'))])
        q['date'] = q.timestamp.dt.strftime('%Y-%m-%d')
        q = q.loc[q.date.isin([row.source_date, day])].sort_values('timestamp')
        assert q.timestamp.is_unique and q.timestamp.max() == pd.Timestamp(day + ' 14:49')
        prior = q.loc[q.date.eq(row.source_date)]
        assert len(prior) == 241 and prior.timestamp.iloc[-1] == pd.Timestamp(row.source_date + ' 15:00')
        current_bars = int(q.date.eq(day).sum())
        close, lots = q.close.to_numpy(float), q.volume.to_numpy(float)/100
        clocks = q.timestamp.dt.strftime('%H%M').astype(int).to_numpy()
        values, gate = native_values(close, lots, clocks, current_bars, row.V01)
        mutated_close, mutated_lots = close.copy(), lots.copy()
        mutated_close[-current_bars:] = 98765
        mutated_lots[-current_bars:] = 76543
        again, same_gate = native_values(mutated_close, mutated_lots, clocks, current_bars, row.V01)
        np.testing.assert_array_equal(values, again)
        assert same_gate == gate
        if row.formula_input_valid:
            assert gate
            np.testing.assert_allclose(values, [row.CA01, row.CA02], rtol=0, atol=2e-10)
            encode = lambda a: np.floor(np.clip(100*np.asarray(a)+10000+.000001, 0, 999999))
            np.testing.assert_array_equal(encode(values), encode([row.CA01, row.CA02]))
        cases.append(dict(date=day, code=code, source_date=row.source_date, source_sha256=sample['source_sha256'],
            minutes=len(q), valid=bool(row.formula_input_valid), native_gate=bool(gate)))
        total += len(q)
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), reused_probe_keys_sha256=sha(PROBES), samples=cases,
        raw_minutes=total, actual_native_ref_sum_gate_lot_units_and_encodings_verified=True,
        arbitrary_current_day_price_and_volume_changes_cannot_change_prior_inputs=True,
        native_source_parity_verified=False, software_compilation_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', proof)
    return {k: v for k, v in proof.items() if k != 'samples'}


def control_inputs():
    import os
    r = json.loads((ROOT / 'feature_report.json').read_text())
    v = json.loads((ROOT / 'feature_verification.json').read_text())
    n = json.loads((ROOT / 'native_input_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    assert n['passed'] and n['feature_verification_sha256'] == sha(ROOT / 'feature_verification.json')
    assert r['newly_invalid'] > 0, 'Unchanged intersections must reuse the original 48 models'
    assert not (CONTROL_INPUTS / 'feature_report.json').exists()
    CONTROL_INPUTS.mkdir(parents=True, exist_ok=True)
    os.link(ROOT / 'features.parquet', CONTROL_INPUTS / 'features.parquet')
    assert sha(CONTROL_INPUTS / 'features.parquet') == r['features_sha256']
    report = dict(protocol_sha256=sha(PROTOCOL), source_feature_report_sha256=sha(ROOT / 'feature_report.json'),
        features_sha256=r['features_sha256'], rows=r['rows'], valid=r['valid'], expressions=previous.EXPRESSIONS,
        native_header=HEADER, native_core_gate=GATE, same_exact_table_and_validity=True,
        all_extra_columns_are_unused_metadata=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(CONTROL_INPUTS / 'feature_report.json', report)
    proof = dict(passed=True, feature_report_sha256=sha(CONTROL_INPUTS / 'feature_report.json'),
        source_feature_verification_sha256=sha(ROOT / 'feature_verification.json'), rows=r['rows'], same_exact_table_bytes=True,
        original_48_expressions_unchanged=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(CONTROL_INPUTS / 'feature_verification.json', proof)
    return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['windows', 'features', 'verify_features', 'native', 'control_inputs'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
