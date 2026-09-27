"""Previous twenty stock-day volume rebased to the latest known float proxy."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_history_direction as history
from . import tail_formula_prior_day as adapter
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_history_turnover'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')
EXPRESSIONS = {**previous.EXPRESSIONS, 'HT01': '10000*REF(SUM(V,B19),B0)/PSH'}
HEADER = previous.HEADER
HISTORY_COLUMNS = ['date', 'code', 'hd_volume20', 'hd_rows20', 'hd_good20',
                   'hd_first_date', 'hd_last_date', 'hd_reference_date',
                   'hd_market_span', 'hd_last_gap', 'history_direction_valid']


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    reports = {}
    for prefix, root in [('previous', previous.ROOT), ('history', history.ROOT)]:
        for name in ['feature_report', 'feature_verification']:
            path = root / (name + '.json')
            assert p[prefix + '_' + name + '_sha256'] == sha(path)
            reports[prefix + '_' + name] = json.loads(path.read_text())
        r, v = reports[prefix + '_feature_report'], reports[prefix + '_feature_verification']
        assert v['passed'] and v['feature_report_sha256'] == sha(root / 'feature_report.json')
        assert r['features_sha256'] == sha(root / 'features.parquet')
    path = history.ROOT / 'native_input_verification.json'
    assert p['history_native_input_verification_sha256'] == sha(path)
    n = json.loads(path.read_text())
    assert n['passed'] and n['feature_report_sha256'] == sha(history.ROOT / 'feature_report.json')
    assert n['feature_verification_sha256'] == sha(history.ROOT / 'feature_verification.json')
    assert reports['history_feature_report']['history_sha256'] == sha(history.ROOT / 'history.parquet')
    assert p['history_stock_days'] == 20
    assert p['float_denominator'] == 'latest_prior_stock_day_float_shares_proxy'
    return reports, n


def features():
    checked_sources()
    assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    h = pd.read_parquet(history.ROOT / 'features.parquet', columns=HISTORY_COLUMNS)
    f = old.merge(h, on=['date', 'code'], how='left', validate='one_to_one')
    f = f.sort_values(['date', 'code']).reset_index(drop=True)
    valid = (f.history_direction_valid & f.hd_rows20.eq(20) & f.hd_good20.eq(20)
             & f.hd_volume20.gt(0) & np.isfinite(f.hd_volume20)
             & f.hd_last_date.lt(f.date) & f.hd_last_date.eq(f.float_source_date)
             & f.float_source_valid & f.float_shares_proxy.gt(0))
    f['history_turnover_valid'] = valid
    f['HT01'] = (100 * f.hd_volume20 / f.float_shares_proxy).where(valid)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL),
        previous_feature_report_sha256=sha(previous.ROOT / 'feature_report.json'),
        history_feature_report_sha256=sha(history.ROOT / 'feature_report.json'),
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f),
        valid=int(f.formula_input_valid.sum()), previous_valid=int(old.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        valid_with_market_gaps=int((f.formula_input_valid & (f.hd_market_span.gt(20) | f.hd_last_gap.gt(1))).sum()),
        first_history_date=f.hd_first_date.min(), last_history_date=f.hd_last_date.max(),
        expressions=EXPRESSIONS, native_header=HEADER,
        same_prior_float_rebases_all_twenty_days=True, sum_of_actual_historical_daily_turnover=False,
        all_previous_48_inputs_retained=True, new_selection_outcomes_read=False,
        native_FINANCE7_parity_verified=False, software_compilation_verified=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['expressions', 'native_header']}


def verify_features():
    checked_sources()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL)
    assert r['features_sha256'] == sha(ROOT / 'features.parquet')
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],
                                 old.drop(columns='formula_input_valid'), check_exact=True)
    h = pd.read_parquet(history.ROOT / 'features.parquet', columns=HISTORY_COLUMNS)
    pd.testing.assert_frame_equal(f[HISTORY_COLUMNS], h, check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_names=False)
    c = base.conn()
    expected = c.sql(f'''SELECT o.date,o.code,
        h.history_direction_valid AND h.hd_rows20=20 AND h.hd_good20=20
        AND h.hd_volume20>0 AND isfinite(h.hd_volume20) AND h.hd_last_date<o.date
        AND h.hd_last_date=o.float_source_date AND o.float_source_valid AND o.float_shares_proxy>0 AS valid,
        h.hd_volume20*o.float_prior_turn/o.float_prior_volume AS ht
        FROM read_parquet('{previous.ROOT}/features.parquet') o
        JOIN read_parquet('{history.ROOT}/features.parquet') h USING(date,code)
        ORDER BY date,code''').df()
    c.close()
    pd.testing.assert_frame_equal(f[['date', 'code']], expected[['date', 'code']], check_exact=True)
    assert f.history_turnover_valid.equals(expected.valid)
    expected_ht = expected.ht.where(expected.valid)
    np.testing.assert_allclose(f.HT01, expected_ht, rtol=2e-15, atol=2e-12, equal_nan=True)
    valid = old.formula_input_valid & expected.valid & np.isfinite(expected_ht)
    assert valid.equals(f.formula_input_valid)
    assert r['valid'] == int(valid.sum())
    assert r['newly_invalid'] == int((old.formula_input_valid & ~valid).sum())
    # Different operation order and the native lot/share conversion must retain every integer input.
    native = 10000 * (h.hd_volume20 / 100) / old.float_shares_proxy
    encode = lambda x: np.floor(np.clip(x * 100 + 10000 + .000001, 0, 999999))
    for values in [expected_ht, native]:
        np.testing.assert_array_equal(encode(f.loc[valid, 'HT01']), encode(values[valid]))
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f),
        valid=int(valid.sum()), all_keys_old_inputs_history_windows_and_validity_verified=True,
        independent_denominator_algebra_and_native_units_verified=True,
        every_new_integer_encoding_verified=True, original_48_inputs_unchanged=True,
        history_source_proof_reused=True, new_selection_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof)
    return proof


def native():
    _, prior_native = checked_sources()
    proof = json.loads((ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    f = pd.read_parquet(ROOT / 'features.parquet', columns=['date', 'code', 'HT01',
        'float_shares_proxy', 'hd_first_date', 'hd_last_date', 'hd_reference_date',
        'hd_volume20', 'history_turnover_valid']).set_index(['date', 'code'])
    active = pd.read_parquet(history.ROOT / 'history.parquet', columns=['date', 'code'])
    active_dates = {code: set(g.date) for code, g in active.groupby('code')}
    cases = []; checked = {}; total = 0; maximum = 0.; excluded_paused_bars = 0
    for case in prior_native['cases']:
        row = f.loc[(case['date'], case['code'])]
        path = MINUTES / case['code'][:2].upper() / (case['code'][3:] + '.parquet')
        if str(path) not in checked:
            assert sha(path) == prior_native['source_sha256'][str(path)]
            checked[str(path)] = prior_native['source_sha256'][str(path)]
        q = pd.read_parquet(path, columns=['timestamp', 'volume'], filters=[
            ('timestamp', '>=', pd.Timestamp(row.hd_reference_date)),
            ('timestamp', '<=', pd.Timestamp(case['date'] + ' 14:49:00'))])
        eligible = q.timestamp.dt.strftime('%Y-%m-%d').isin(active_dates[case['code']])
        # The minute vendor includes all-zero placeholder days during suspension.
        # The frozen window is twenty active stock days, as in the reused source proof.
        assert q.loc[~eligible, 'volume'].eq(0).all()
        excluded_paused_bars += int((~eligible).sum())
        q = q.loc[eligible].sort_values('timestamp').reset_index(drop=True)
        dates = q.timestamp.dt.strftime('%Y-%m-%d')
        assert dates.nunique() == 22 and not q.timestamp.duplicated().any()
        b0 = q.groupby(dates).cumcount().to_numpy() + 1
        position = np.arange(len(q)); offset = b0.astype(float)
        # Reproduce B1..B19 at every bar, so REF evaluates the historical B19, not today's length.
        for _ in range(19):
            usable = np.isfinite(offset) & (position >= offset)
            new = np.full(len(q), np.nan)
            new[usable] = offset[usable] + b0[(position[usable] - offset[usable]).astype(int)]
            offset = new
        prior_end = len(q) - 1 - b0[-1]
        count = int(offset[prior_end]); begin = prior_end - count + 1
        assert dates.iloc[begin] == row.hd_first_date and dates.iloc[prior_end] == row.hd_last_date
        volumes = q.volume.iloc[begin:prior_end + 1].to_numpy(dtype=float)
        assert dates.iloc[begin:prior_end + 1].nunique() == 20
        if row.history_turnover_valid:
            np.testing.assert_allclose(volumes.sum(), row.hd_volume20, rtol=0, atol=1e-7)
            value = 10000 * (volumes / 100).sum() / row.float_shares_proxy
            maximum = max(maximum, float(abs(value - row.HT01)))
            assert np.floor(np.clip(value * 100 + 10000 + .000001, 0, 999999)) == np.floor(np.clip(row.HT01 * 100 + 10000 + .000001, 0, 999999))
        else:
            assert pd.isna(row.HT01)
        total += len(q)
        cases.append(dict(date=case['date'], code=case['code'], raw_minutes=len(q),
            valid=bool(row.history_turnover_valid), previous_stock_days=20))
    assert len(cases) == 32
    r = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'),
        history_native_proof_sha256=sha(history.ROOT / 'native_input_verification.json'),
        samples=len(cases), raw_minutes=total, cases=cases, source_sha256=checked,
        maximum_native_scalar_difference=maximum,
        excluded_zero_volume_suspension_placeholder_bars=excluded_paused_bars,
        client_minute_history_must_omit_suspension_placeholders=True,
        sampled_recursive_minute_offsets_and_volume_sum_verified=True,
        float_denominator_is_daily_source_proxy_not_client_verified=True,
        software_compilation_verified=False, native_client_data_parity_verified=False,
        new_selection_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', r)
    return {k: v for k, v in r.items() if k not in ['cases', 'source_sha256']}


def configure():
    checked_sources()
    v = json.loads((ROOT / 'native_input_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    assert v['feature_verification_sha256'] == sha(ROOT / 'feature_verification.json')
    adapter.STEM = STEM; adapter.ROOT = ROOT; adapter.PROTOCOL = PROTOCOL
    adapter.COMBINED_PROTOCOL = COMBINED_PROTOCOL
    adapter.CONTROL = Path('data/research') / (STEM + '_control')
    adapter.EXPRESSIONS = EXPRESSIONS; adapter.HEADER = HEADER
    for fold in ['2024', 'recent', 'combined']:
        assert json.loads((Path('config') / (STEM + '_' + fold + '_protocol.json')).read_text())['inputs_protocol_sha256'] == sha(PROTOCOL)


if __name__ == '__main__':
    from . import tail_formula_context_2024 as linkage
    from . import tail_formula_recent as study
    from . import tail_formula_relative as relative
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'native', 'model', 'verify_model',
        'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--fold', choices=['2024', 'recent', 'combined', 'control'], default='2024')
    a = p.parse_args()
    if a.stage in ['features', 'verify_features', 'native']:
        result = globals()[a.stage]()
    else:
        configure()
        if a.stage == 'analyze':
            assert all((Path('data/research') / (STEM + '_' + fold) / 'selection_verification.json').exists()
                       for fold in ['2024', 'recent', '2025', 'control'])
        if a.fold == 'control':
            assert a.stage in ['freeze', 'verify', 'analyze']
            result = (linkage.common_analysis(adapter.CONTROL, COMBINED_PROTOCOL) if a.stage == 'analyze'
                      else adapter.control(a.stage + '_control'))
        else:
            adapter.setup(a.fold)
            if a.fold == 'combined':
                assert a.stage in ['freeze', 'verify', 'analyze']
                result = (linkage.common_analysis(linkage.COMBINED, linkage.PROTOCOL) if a.stage == 'analyze'
                          else getattr(linkage, 'combine' if a.stage == 'freeze' else 'verify_combined')())
            elif a.stage in ['model', 'verify_model']:
                result = getattr(relative, a.stage)('relative')
            elif a.stage == 'verify_scores':
                result = verify_scores()
            elif a.stage in ['freeze', 'verify']:
                result = getattr(study, a.stage)()
            else:
                result = getattr(base, a.stage)()
    print(json.dumps(result, ensure_ascii=False, indent=2))
