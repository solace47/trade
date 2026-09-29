"""Brokerage versus bank context, preserving the original 48 stock inputs."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from .corporate_cash import MINUTES, save_json, sha
from .turnover_reference import CALENDAR

STEM = 'tail_formula_financial_context'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
NEW_EXPRESSIONS = {'FI01': '100*(FB48/FBP-FK48/FKP)', 'FI02': '100*(FB48/FB20-FK48/FK20)'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
native_core = base.native_core
HEADER = previous.HEADER
for name, symbol in [('FB', 'SZ399975'), ('FK', 'SZ399986')]:
    for minute in [48, 20]:
        HEADER += f'{name}{minute}:=VALUEWHEN(TIME=14{minute},"{symbol}$CLOSE");\n'
    HEADER += f'{name}P:=REF("{symbol}$CLOSE",B0);\n'


def checked_sources():
    p = json.loads(PROTOCOL.read_text()); r = json.loads((ROOT / 'source_report.json').read_text())
    for name, digest in p['source_hashes'].items():
        assert sha(Path(name)) == digest
    proof_path = Path('data/research/tail_formula_financial_daily_probe/verification.json')
    assert sha(proof_path) == p['daily_probe_verification_sha256']
    assert json.loads(proof_path.read_text())['source_usable']
    assert r['protocol_sha256'] == sha(PROTOCOL)
    assert p['previous_feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert p['daily_feature_report_sha256'] == sha(base.SOURCE / 'feature_report.json')
    assert p['calendar_sha256'] == sha(CALENDAR)
    old = json.loads((previous.ROOT / 'feature_report.json').read_text())
    proof = json.loads((previous.ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert old['features_sha256'] == r['source_feature_sha256'] == sha(previous.ROOT / 'features.parquet')
    assert r['indices_sha256'] == sha(ROOT / 'indices.parquet')
    return previous.context.market.stock.source_files()


def prefix_points(rows):
    prefix = rows[:228]
    valid = len(prefix) == 228 and all(r['sequence'] == i and np.isfinite(r['price_raw']) and r['price_raw'] > 0 and r['price_raw'] == int(r['price_raw']) for i, r in enumerate(prefix))
    return dict(p48=rows[227]['price_raw'] / 100 if valid else np.nan,
        p20=rows[199]['price_raw'] / 100 if valid else np.nan, prefix_valid=valid)


def features():
    assert not (ROOT / 'feature_report.json').exists(), 'Do not replace frozen financial context'
    files = checked_sources(); source = json.loads((ROOT / 'source_report.json').read_text())
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    assert {(s['symbol'], s['date']) for s in source['sessions']} == {
        (code, day) for code in ['sz.399975', 'sz.399986'] for day in old.date.unique()}
    indices = pd.read_parquet(ROOT / 'indices.parquet').set_index(['code', 'date'])
    points = []; audits = []
    for s in source['sessions']:
        if 'path' in s:
            path = Path(s['path']); assert sha(path) == s['sha256']; rows = json.loads(path.read_text())
        else:
            rows = []
        points.append(dict(date=s['date'], index_code=s['symbol'], **prefix_points(rows)))
        day = indices.loc[(s['symbol'], s['date'])]; prices = np.array([x['price_raw'] for x in rows]) / 100
        full = bool(len(prices) == 240 and (prices > 0).all() and
            (prices >= day.low - .0051).all() and (prices <= day.high + .0051).all() and abs(prices[-1] - day.close) <= .0051)
        audits.append(dict(date=s['date'], index_code=s['symbol'], rows=len(rows), full_day_source_valid=full,
            last_minus_daily_close=float(prices[-1] - day.close) if len(prices) else None))
    points = pd.DataFrame(points).sort_values(['date', 'index_code']).reset_index(drop=True)
    audit = pd.DataFrame(audits).sort_values(['date', 'index_code']).reset_index(drop=True)
    c = base.conn(); c.read_parquet(files).create_view('daily'); c.register('indices', indices.reset_index())
    prior = c.sql('''WITH traded AS (
        SELECT date,code,lag(date) OVER(PARTITION BY code ORDER BY date) AS fi_prior_date
        FROM daily WHERE tradestatus=1 AND date BETWEEN '2023-06-01' AND '2025-12-30')
        SELECT t.date,t.code,t.fi_prior_date,s.close AS fi_broker_prior,l.close AS fi_bank_prior
        FROM traded t LEFT JOIN indices s ON s.code='sz.399975' AND s.date=t.fi_prior_date
        LEFT JOIN indices l ON l.code='sz.399986' AND l.date=t.fi_prior_date
        WHERE t.date>='2024-01-01' ORDER BY t.date,t.code''').df(); c.close()
    f = old.merge(prior, on=['date', 'code'], how='left', validate='one_to_one')
    for name, code in [('broker', 'sz.399975'), ('bank', 'sz.399986')]:
        q = points.loc[points.index_code.eq(code)].drop(columns='index_code').rename(
            columns={k: f'fi_{name}_{k}' for k in ['p48', 'p20', 'prefix_valid']})
        f = f.merge(q, on='date', how='left', validate='many_to_one')
    values = f[['fi_broker_prior', 'fi_bank_prior', 'fi_broker_p48', 'fi_broker_p20', 'fi_bank_p48', 'fi_bank_p20']]
    valid = (f.fi_prior_date.lt(f.date) & values.gt(0).all(axis=1) & np.isfinite(values).all(axis=1)
        & f.fi_broker_prefix_valid.fillna(False) & f.fi_bank_prefix_valid.fillna(False))
    f['FI01'] = (100 * (f.fi_broker_p48 / f.fi_broker_prior - f.fi_bank_p48 / f.fi_bank_prior)).where(valid)
    f['FI02'] = (100 * (f.fi_broker_p48 / f.fi_broker_p20 - f.fi_bank_p48 / f.fi_bank_p20)).where(valid)
    f['financial_context_valid'] = valid; f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    cal = pd.read_parquet(CALENDAR)
    days = sorted(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between('2023-06-01', '2025-12-30'), 'calendar_date'])
    ranks = {day: index for index, day in enumerate(days)}
    f['fi_prior_gap'] = f.date.map(ranks) - f.fi_prior_date.map(ranks)
    f = f.sort_values(['date', 'code']).reset_index(drop=True)
    points.to_parquet(ROOT / 'index_points.parquet', index=False, compression='zstd')
    audit.to_parquet(ROOT / 'source_audit.parquet', index=False, compression='zstd')
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), previous_feature_report_sha256=sha(previous.ROOT / 'feature_report.json'),
        daily_feature_report_sha256=sha(base.SOURCE / 'feature_report.json'), source_report_sha256=sha(ROOT / 'source_report.json'),
        calendar_sha256=sha(CALENDAR), index_points_sha256=sha(ROOT / 'index_points.parquet'),
        source_audit_sha256=sha(ROOT / 'source_audit.parquet'), features_sha256=sha(ROOT / 'features.parquet'),
        rows=len(f), valid=int(f.formula_input_valid.sum()), previous_valid=int(f.prior_formula_input_valid.sum()),
        newly_invalid=int((f.prior_formula_input_valid & ~f.formula_input_valid).sum()),
        valid_with_prior_stock_day_gaps=int((f.formula_input_valid & f.fi_prior_gap.gt(1)).sum()),
        index_sessions=len(points), invalid_prefix_sessions=int((~points.prefix_valid).sum()),
        full_day_source_conflicts=int((~audit.full_day_source_valid).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, full_day_source_audit_used_for_selection=False,
        native_source_price_parity_verified=False, software_compilation_verified=False,
        all_previous_48_inputs_retained=True, new_selection_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['expressions', 'native_header']}


def native_values(broker, bank, clocks, current_count):
    env = dict(BROKER=np.asarray(broker, float), BANK=np.asarray(bank, float), TIME=np.asarray(clocks),
        B0=int(current_count), VALUEWHEN=lambda mask, x: x[np.flatnonzero(mask)[-1]],
        REF=lambda x, n: x[-1-int(n)])
    for line in HEADER[len(previous.HEADER):].splitlines():
        name, expression = line.rstrip(';').split(':=')
        expression = expression.replace('"SZ399975$CLOSE"', 'BROKER').replace('"SZ399986$CLOSE"', 'BANK')
        expression = re.sub(r'(?<![<>=!])=(?!=)', '==', expression)
        env[name] = eval(expression, {'__builtins__': {}}, env)
    return {name: float(eval(expression, {'__builtins__': {}}, env)) for name, expression in NEW_EXPRESSIONS.items()}


def native():
    checked_sources()
    proof = json.loads((ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    probes_path = Path('data/research/tail_formula_volume_response/native_input_verification.json')
    assert sha(probes_path) == '5c305f6188190bcf2543c6e6befec526d87000ea0aba3b6ab682201e7177441d'
    probes = json.loads(probes_path.read_text()); assert probes['passed']
    report = json.loads((ROOT / 'source_report.json').read_text())
    sessions = {(v['symbol'], v['date']): v for v in report['sessions']}
    actual = pd.read_parquet(ROOT / 'features.parquet').set_index(['date', 'code'])
    details = []
    for sample in probes['samples']:
        day, code = sample['date'], sample['code']; row = actual.loc[day, code]
        assert row.formula_input_valid and '2024-01-01' <= day <= '2025-12-30'
        path = MINUTES / code[:2].upper() / (code[3:]+'.parquet'); assert sha(path) == sample['source_sha256']
        raw = pd.read_parquet(path, columns=['timestamp'], filters=[('timestamp', '>=', pd.Timestamp(day+' 09:30')),
            ('timestamp', '<=', pd.Timestamp(day+' 14:49'))]).sort_values('timestamp')
        clocks = (100*raw.timestamp.dt.hour+raw.timestamp.dt.minute).to_numpy()
        expected = [930] + [100*(m//60)+m%60 for m in list(range(571, 691))+list(range(781, 890))]
        assert clocks.tolist() == expected and len(clocks) == 230
        arrays = []
        for symbol, prior in [('sz.399975', row.fi_broker_prior), ('sz.399986', row.fi_bank_prior)]:
            item = sessions[symbol, day]; file = Path(item['path']); assert sha(file) == item['sha256']
            points = json.loads(file.read_text()); assert len(points) == 240
            # A prior stock trading day's final aligned index close precedes today's bars.
            # The unavailable 09:30 representative point is deliberately NaN; neither field reads it.
            arrays.append(np.r_[prior, np.nan, [v['price_raw']/100 for v in points[:229]]])
        values = native_values(*arrays, np.r_[1500, clocks], len(clocks))
        changed = [v.copy() for v in arrays]
        for v in changed: v[-1] = -999999.  # 14:49 is beyond the fixed index cutoff.
        future = native_values(*[np.r_[v, .01, 1e6] for v in changed], np.r_[1500, clocks, 1450, 1451], len(clocks)+2)
        for name in NEW_EXPRESSIONS:
            np.testing.assert_allclose(values[name], row[name], rtol=0, atol=2e-12)
            assert values[name] == future[name]
            assert np.floor(100*values[name]+10000+1e-6).clip(0, 999999) == np.floor(100*row[name]+10000+1e-6).clip(0, 999999)
        details.append(dict(date=day, code=code, stock_prior_date=row.fi_prior_date, source_sha256=sample['source_sha256'], **values))
    assert len(details) == 32
    r = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), samples=details,
        native_cross_symbol_anchors_and_previous_stock_bar_replayed=True, actual_stock_prefix_rows=7360,
        cutoff_and_future_perturbations=32, native_source_price_parity_verified=False, software_compilation_verified=False,
        new_stock_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', r)
    return {k: v for k, v in r.items() if k != 'samples'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['features', 'native'], default='features', nargs='?')
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
