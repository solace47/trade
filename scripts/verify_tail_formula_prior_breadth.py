"""Independent wire, past-date, arithmetic and encoding audit for prior breadth."""
import json
from pathlib import Path
import re
import struct

import numpy as np
import pandas as pd

from collect_tail_formula_prior_breadth import replay
from trade_research import tail_formula_prior_breadth as study
from trade_research.corporate_cash import save_json, sha

ROOT = study.ROOT


def main():
    files = study.checked_sources(); cfg = json.loads(study.PROTOCOL.read_text())
    r = json.loads((ROOT / 'feature_report.json').read_text()); s = json.loads((ROOT / 'source_report.json').read_text())
    for key, path in [('protocol_sha256', study.PROTOCOL), ('source_report_sha256', ROOT / 'source_report.json'),
        ('features_sha256', ROOT / 'features.parquet'), ('counts_sha256', ROOT / 'counts.parquet'),
        ('calendar_sha256', study.CALENDAR)]:
        assert r[key] == sha(path)
    records = []; wire_count = 0
    for session in s['sessions']:
        for attempt in session['attempts']:
            for name, digest in attempt['wire_sha256'].items():
                assert sha(Path(name)) == digest
            if attempt['status'] != 'passed':
                continue
            packet = Path(attempt['request_path']).read_bytes()
            parts = struct.unpack('<HIHHHH6sHHHHIIH', packet)
            assert parts == (0x10c, 0x01016408, 0x1c, 0x1c, 0x052d, int(session['symbol'].startswith('sh.')),
                session['symbol'][3:].encode(), 9, 1, cfg['start'], cfg['count'], 0, 0, 0)
            rows = replay(Path(attempt['response_path']), cfg); assert rows == attempt['records']
            records.extend(dict(code=session['symbol'], **x) for x in rows); wire_count += 1
    source = pd.DataFrame(records).sort_values(['code', 'date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(source, pd.read_parquet(ROOT / 'counts.parquet'), check_exact=True)
    calendar = pd.read_parquet(study.CALENDAR)
    days = sorted(calendar.loc[calendar.is_trading_day.eq('1') & calendar.calendar_date.between('2023-06-01', '2025-12-30'), 'calendar_date'])
    for symbol in cfg['symbols']:
        assert source.loc[source.code.eq(symbol), 'date'].tolist() == days
    assert source[['up_count', 'down_count']].ge(0).all().all()
    assert source.up_count.add(source.down_count).gt(0).all()
    assert len(s['overlap_checks']) == 14 and all(x['matched'] for x in s['overlap_checks'])
    indexed_counts = source.set_index(['code', 'date'])
    checks = 0
    for path in ['data/research/tail_formula_prior_breadth_probe/report.json', 'data/research/tail_formula_intraday_breadth_probe/report.json']:
        raw = json.loads(Path(path).read_text())
        for session in raw['sessions']:
            rows = next((x['records'] for x in session['attempts'] if x['status'] == 'nonempty'), [])
            for row in rows:
                if 'minute' in row and row['minute'] != 900:
                    continue
                actual = indexed_counts.loc[(session['symbol'], row['date'])]
                assert (actual.up_count, actual.down_count) == (row['up_count'], row['down_count']); checks += 1
    assert checks == 14
    old = pd.read_parquet(study.previous.ROOT / 'features.parquet'); actual = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(actual[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    assert actual.prior_formula_input_valid.equals(old.formula_input_valid)
    np.testing.assert_array_equal(old.index_code, np.where(old.code.str.startswith('sh.'), 'sh.000001', 'sz.399001'))
    c = study.base.conn(); c.read_parquet(files).create_view('daily')
    traded = c.sql("SELECT date,code FROM daily WHERE tradestatus=1 AND date BETWEEN '2023-06-01' AND '2025-12-30' ORDER BY code,date").df(); c.close()
    assert not traded.duplicated(['date', 'code']).any()
    source['value'] = 100 * (1 - source.down_count / (source.up_count + source.down_count))
    mapped = source.set_index(['code', 'date']).value.to_dict()
    symbols = np.where(traded.code.str.startswith('sh.'), 'sh.000001', 'sz.399001')
    traded['value'] = [mapped.get(key, np.nan) for key in zip(symbols, traded.date)]
    groups = traded.groupby('code', sort=False)
    for i in range(1, 6):
        traded[f'br_date_{i}'] = groups.date.shift(i)
        traded[f'br_value_{i}'] = groups.value.shift(i)
    cols = ['date', 'code'] + [f'br_{kind}_{i}' for i in range(1, 6) for kind in ['date', 'value']]
    f = old[['date', 'code', 'index_code']].merge(traded[cols], on=['date', 'code'], how='left', validate='one_to_one')
    pd.testing.assert_frame_equal(actual[cols], f[cols], check_dtype=False, rtol=0, atol=2e-12)
    values = f[[f'br_value_{i}' for i in range(1, 6)]]
    valid = np.isfinite(values).all(axis=1) & values.ge(0).all(axis=1) & values.le(100).all(axis=1)
    for i in range(1, 6):
        valid &= f[f'br_date_{i}'].lt(f.date)
    expected = pd.DataFrame(dict(BR01=f.br_value_1.where(valid),
        BR02=((4 * f.br_value_1 - f.br_value_2 - f.br_value_3 - f.br_value_4 - f.br_value_5) / 5).where(valid)))
    np.testing.assert_allclose(actual[list(expected)], expected, rtol=0, atol=2e-12, equal_nan=True)
    final = old.formula_input_valid & valid & np.isfinite(expected).all(axis=1)
    assert actual.breadth_valid.equals(valid) and actual.formula_input_valid.equals(final)
    np.testing.assert_array_equal(np.floor(np.clip(actual.loc[final, list(expected)] * 100 + 10000 + .000001, 0, 999999)),
        np.floor(np.clip(expected.loc[final] * 100 + 10000 + .000001, 0, 999999)))
    ranks = {day: i for i, day in enumerate(days)}
    gaps = f.date.map(ranks) - f.br_date_1.map(ranks)
    spans = f.br_date_1.map(ranks) - f.br_date_5.map(ranks) + 1
    np.testing.assert_allclose(actual.br_prior_gap, gaps, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_allclose(actual.br_history_span, spans, rtol=0, atol=0, equal_nan=True)
    assert r['valid_with_prior_stock_day_gaps'] == int((final & gaps.gt(1)).sum())
    assert r['valid_with_five_day_gaps'] == int((final & spans.gt(5)).sum())
    assert r['rows'] == len(actual) and r['valid'] == int(final.sum())
    assert r['newly_invalid'] == int((old.formula_input_valid & ~final).sum())
    perturbations = 0
    for day, group in f.groupby('date', sort=False):
        altered = {key: value if key[1] < day else -99999 for key, value in mapped.items()}
        for i in range(1, 6):
            rebuilt = np.array([altered.get(key, np.nan) for key in zip(group.index_code, group[f'br_date_{i}'])])
            np.testing.assert_allclose(rebuilt, group[f'br_value_{i}'], rtol=0, atol=0, equal_nan=True)
        perturbations += 1
    assert r['expressions'] == study.EXPRESSIONS and r['native_header'] == study.HEADER
    names = re.findall(r'(?m)^([A-Za-z][A-Za-z0-9]*):=', study.HEADER) + list(study.EXPRESSIONS)
    assert len(names) == len(set(names))
    assert '100*INDEXADV/(INDEXADV+INDEXDEC)' in study.HEADER
    variables = {f'BRD{i}': f[f'br_value_{i}'] for i in range(1, 6)}
    for i in range(1, 6):
        assert f'BRD{i}:=REF(BRR,B{i-1});' in study.HEADER
    for name, expression in study.NEW_EXPRESSIONS.items():
        value = eval(expression, {'__builtins__': {}}, variables).where(valid)
        np.testing.assert_allclose(value, expected[name], rtol=0, atol=2e-12, equal_nan=True)
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(actual), valid=int(final.sum()),
        original_wire_replays=wire_count, daily_and_five_minute_close_overlap_checks=checks,
        all_five_stock_prior_dates_and_breadth_values_independently_rebuilt=True, original_48_inputs_unchanged=True,
        all_integer_encodings_and_validity_verified=True, after_cutoff_perturbation_checks=perturbations,
        native_expression_arithmetic_verified=True, native_source_parity_verified=False, software_compilation_verified=False,
        new_selection_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
