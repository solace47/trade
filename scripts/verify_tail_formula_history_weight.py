"""Rebuild prior-day volume-weighted prices from original daily files."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_history_weight as study
from trade_research.corporate_cash import MINUTES, save_json, sha


def lag_matrix(a):
    out = np.full((len(a), 20), np.nan)
    for n in range(1, 21):
        out[n:, n-1] = a[:-n]
    return out


def main():
    root = study.ROOT
    r = json.loads((root/'feature_report.json').read_text())
    for key, path in [('protocol_sha256', study.PROTOCOL),
                      ('previous_feature_report_sha256', study.previous.ROOT/'feature_report.json'),
                      ('daily_feature_report_sha256', study.base.SOURCE/'feature_report.json'),
                      ('features_sha256', root/'features.parquet'), ('calendar_sha256', study.CALENDAR)]:
        assert r[key] == sha(path)
    paths = study.source_files()
    f = pd.read_parquet(root/'features.parquet')
    old = pd.read_parquet(study.previous.ROOT/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],
                                  old.drop(columns='formula_input_valid'), check_exact=True)
    assert f.prior_formula_input_valid.equals(old.formula_input_valid)
    groups = {code: q for code, q in f.groupby('code', sort=False)}
    cal = pd.read_parquet(study.CALENDAR)
    days = sorted(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between('2023-06-01', '2025-12-30'), 'calendar_date'])
    ranks = {d: i for i, d in enumerate(days)}
    checked = 0
    cache = {}
    for path in paths:
        code = Path(path).stem.replace('_', '.')
        if code not in groups:
            continue
        q = groups[code]
        d = pd.read_parquet(path, columns=['date', 'code', 'close', 'volume', 'preclose', 'adjustflag', 'tradestatus'],
                            filters=[('date', '>=', '2023-06-01'), ('date', '<=', '2025-12-30')])
        d = d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True)
        assert d.code.eq(code).all() and not d.date.duplicated().any()
        lookup = {day: i for i, day in enumerate(d.date)}
        indices = np.array([lookup[day] for day in q.date])
        cp = lag_matrix(d.close.to_numpy(dtype=float))
        vp = lag_matrix(d.volume.to_numpy(dtype=float))
        good = (np.isfinite(d.close) & d.close.gt(0) & (d.close-d.close.round(2)).abs().le(.0001)
                & np.isfinite(d.volume) & d.volume.gt(0) & d.volume.eq(np.floor(d.volume)) & d.adjustflag.eq(3))
        count = np.minimum(np.arange(len(d)), 20)
        count_good = good.astype(int).rolling(20, min_periods=1).sum().shift().to_numpy()
        breaks = (d.preclose-d.close.shift()).abs().gt(.005).astype(int).rolling(20, min_periods=1).sum().shift().to_numpy()
        first = pd.Series([d.date.iloc[max(0, i-20)] if i else None for i in range(len(d))])
        last = d.date.shift()
        for name, expected in [('hc_rows', count), ('hc_good', count_good), ('hc_reference_breaks', breaks)]:
            np.testing.assert_allclose(q[name], np.asarray(expected)[indices], atol=0, rtol=0, equal_nan=True)
        assert q.hc_first_date.tolist() == first.iloc[indices].tolist()
        assert q.hc_last_date.tolist() == last.iloc[indices].tolist()
        span = last.map(ranks)-first.map(ranks)+1
        np.testing.assert_allclose(q.hc_market_span, span.iloc[indices], atol=0, rtol=0, equal_nan=True)
        np.testing.assert_allclose(q[study.PRICE_COLUMNS], cp[indices], atol=0, rtol=0, equal_nan=True)
        np.testing.assert_allclose(q[study.VOLUME_COLUMNS], vp[indices], atol=0, rtol=0, equal_nan=True)
        # Independent window-level weighted averages, without a difference of large second moments.
        means, variances, sums = [], [], []
        for j in indices:
            weights = vp[j]
            center = np.average(cp[j], weights=weights)
            means.append(center)
            variances.append(np.average(np.square(cp[j]-center), weights=weights))
            sums.append(sum(weights))
        means, variances, sums = map(np.asarray, [means, variances, sums])
        for name, value in [('hc_weighted_close', means), ('hc_weighted_variance', variances), ('hc_total_volume', sums)]:
            np.testing.assert_allclose(q[name], value, atol=2e-12, rtol=1e-12, equal_nan=True)
        valid = ((count[indices] == 20) & (count_good[indices] == 20)
                 & np.isfinite(cp[indices]).all(axis=1) & np.isfinite(vp[indices]).all(axis=1)
                 & (cp[indices] > 0).all(axis=1) & (vp[indices] > 0).all(axis=1) & q.V01.gt(0).to_numpy()
                 & (last.iloc[indices].to_numpy() < q.date.to_numpy()))
        np.testing.assert_array_equal(q.history_weight_valid, valid)
        values = np.column_stack([100*(q.price_1449.to_numpy()/means-1)/q.V01.to_numpy(),
                                  100*np.sqrt(variances)/means/q.V01.to_numpy()])
        values[~valid] = np.nan
        np.testing.assert_allclose(q[['K01', 'K02']], values, atol=2e-10, rtol=0, equal_nan=True)
        final = valid & q.prior_formula_input_valid.to_numpy() & np.isfinite(q[list(study.EXPRESSIONS)]).all(axis=1).to_numpy()
        np.testing.assert_array_equal(q.formula_input_valid, final)
        encode = lambda a: np.floor(np.clip(100*a+10000+.000001, 0, 999999))
        np.testing.assert_array_equal(encode(q.loc[final, ['K01', 'K02']].to_numpy()), encode(values[final]))
        checked += len(q)
        cache[code] = d
    assert checked == len(f) == r['rows']
    valid = f.formula_input_valid
    assert int(valid.sum()) == r['valid']
    assert int((f.prior_formula_input_valid & ~valid).sum()) == r['newly_invalid']
    assert int((valid & f.hc_reference_breaks.gt(0)).sum()) == r['valid_with_reference_break']
    assert int((valid & f.hc_market_span.gt(20)).sum()) == r['valid_with_historical_gaps']
    assert float(f.loc[valid, 'hc_market_span'].max()) == r['maximum_valid_history_market_span']
    assert f.loc[valid, 'K02'].ge(0).all()
    assert r['expressions'] == study.EXPRESSIONS and r['native_header'] == study.HEADER
    symbols = re.findall(r'(?m)^([A-Za-z][A-Za-z0-9]*):=', study.HEADER) + list(study.EXPRESSIONS)
    symbols += [f'T{i:02d}' for i in range(1,65)] + [f'X{i:02d}' for i in range(1,51)]
    assert len(symbols) == len(set(symbols))
    for i in range(1,21):
        assert f'HDV{i:02d}:=REF(SUM(V,B0),B{i-1});' in study.HEADER
    sample = f.loc[valid].copy()
    sample['hash'] = [hashlib.sha256(f'history-weight-v1|{d}|{c}'.encode()).hexdigest() for d, c in zip(sample.date, sample.code)]
    sample = sample.sort_values('hash').groupby('half', sort=True).head(8)
    cases = []
    manifest = Path('data/research/economic_winner/input_manifest.json')
    hashes = json.loads(manifest.read_text())['source_sha256']
    for _, row in sample.iterrows():
        raw = cache[row.code]
        past = raw.loc[raw.date.lt(row.date)].tail(20)
        assert len(past) == 20 and past.date.max() < row.date
        corrupted = raw.copy()
        corrupted.loc[corrupted.date.ge(row.date), ['close', 'volume']] = [999999., 1e15]
        pd.testing.assert_frame_equal(corrupted.loc[corrupted.date.lt(row.date)].tail(20), past, check_exact=True)
        p = past.close.to_numpy(dtype=float)
        v = past.volume.to_numpy(dtype=float)
        w = sum(x*y for x, y in zip(p, v))/sum(v)
        dispersion = sum(y*(x-w)**2 for x, y in zip(p, v))/sum(v)
        np.testing.assert_allclose([100*(row.price_1449/w-1)/row.V01, 100*dispersion**.5/w/row.V01],
                                  row[['K01', 'K02']].to_numpy(dtype=float), atol=2e-10, rtol=0)
        source = MINUTES/row.code[:2].upper()/(row.code[3:]+'.parquet')
        assert sha(source) == hashes[str(source)]
        minute = pd.read_parquet(source, columns=['timestamp', 'close', 'volume'],
            filters=[('timestamp','>=',pd.Timestamp(past.date.min())), ('timestamp','<',pd.Timestamp(row.date))])
        minute['date'] = minute.timestamp.dt.strftime('%Y-%m-%d')
        agg = minute.sort_values('timestamp').groupby('date').agg(cl=('close', 'last'), v=('volume','sum')).tail(20)
        aligned = agg.index.tolist() == past.date.tolist()
        details = dict(date=row.date, code=row.code, prior_dates_aligned=aligned, prior_days=len(agg),
                       reference_breaks=int(row.hc_reference_breaks), field_differences={})
        if aligned:
            for field, x, y in [('close', agg.cl.round(2).to_numpy(), p), ('volume', agg.v.to_numpy(), v)]:
                if not np.array_equal(x, y):
                    details['field_differences'][field] = float(np.max(np.abs(x-y)))
        cases.append(details)
    assert len(cases) == 32
    proof = dict(passed=True, feature_report_sha256=sha(root/'feature_report.json'), rows=len(f), valid=int(valid.sum()),
        all_prior_price_volume_windows_and_metadata_rebuilt=True, all_inputs_and_integer_encodings_rebuilt=True,
        original_48_values_and_keys_unchanged=True, fixed_native_cases=cases, native_case_count=32, prior_daily_points=640,
        native_probe_date_mismatches=sum(not x['prior_dates_aligned'] for x in cases),
        native_probe_field_mismatches=sum(bool(x['field_differences']) for x in cases),
        current_and_future_contamination_does_not_change_prior_inputs=True, all_native_variable_names_unique=True,
        native_source_parity_verified=False, software_compilation_verified=False,
        outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root/'feature_verification.json', proof)
    return {k:v for k,v in proof.items() if k != 'fixed_native_cases'}


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
