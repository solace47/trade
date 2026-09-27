"""Independently reconstruct minute closing locations and inspect raw windows."""
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_minute_pressure as study
from trade_research.corporate_cash import save_json, sha


def raw_window(raw, date):
    clock = raw.timestamp.dt.strftime('%H%M')
    x = raw.loc[raw.timestamp.dt.strftime('%Y-%m-%d').eq(date) & clock.between('1421', '1449')].copy()
    clock = x.timestamp.dt.strftime('%H%M')
    good = x.timestamp.eq(x.timestamp.dt.floor('min'))
    rounded = {}
    for key, field in study.FIELDS.items():
        a = x[field].astype(float)
        rounded[key] = a if key == 'v' else np.floor(a * 100 + .5) / 100
        good &= np.isfinite(a)
        if key == 'v':
            good &= a.ge(0) & a.eq(np.floor(a))
        else:
            good &= a.gt(0) & (a - rounded[key]).abs().le(.0001)
    good &= rounded['h'].ge(rounded['c']) & rounded['c'].ge(rounded['l'])
    arrays = {key: np.array([a.loc[clock.eq(f'14{n:02d}')].max() for n in range(21, 50)])
              for key, a in rounded.items()}
    return dict(mp_bars=len(x), mp_clocks=clock.nunique(), mp_good_bars=int(good.sum())), arrays


def main():
    _, old, source_hashes = study.checked_source()
    root = study.ROOT
    r = json.loads((root / 'feature_report.json').read_text())
    for key, path in [('protocol_sha256', study.PROTOCOL),
                      ('previous_feature_report_sha256', study.previous.ROOT / 'feature_report.json'),
                      ('window_report_sha256', root / 'window_report.json'),
                      ('features_sha256', root / 'features.parquet')]:
        assert r[key] == sha(path)
    windows = json.loads((root / 'window_report.json').read_text())
    assert windows['extractor_sha256'] == sha(Path(study.__file__))
    assert windows['protocol_sha256'] == sha(study.PROTOCOL)
    assert windows['minute_manifest_sha256'] == sha(study.MANIFEST)
    for path, digest in windows['parts_sha256'].items():
        assert sha(Path(path)) == digest
    for path, digest in windows['source_sha256'].items():
        assert digest == source_hashes[path] and sha(Path(path)) == digest
    expected_paths = {str(study.MINUTES / code[:2].upper() / (code[3:] + '.parquet')) for code in old.code.unique()}
    assert expected_paths == set(windows['source_sha256'])
    f = pd.read_parquet(root / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],
                                  old.drop(columns='formula_input_valid'), check_exact=True)
    assert f.prior_formula_input_valid.equals(old.formula_input_valid)
    c = study.base.conn()
    c.register('original', old)
    c.read_parquet(list(windows['parts_sha256'])).create_view('wide')
    linked = c.sql('SELECT o.date,o.code,w.* EXCLUDE(date,code) FROM original o LEFT JOIN wide w USING(date,code) ORDER BY date,code').df()
    pd.testing.assert_frame_equal(f[linked.columns], linked, check_exact=True)
    # Subtract the two distances to the extremes, separately from the producer's 2*C-H-L.
    position = lambda n: f'((mp_c{n}-mp_l{n})-(mp_h{n}-mp_c{n}))/greatest(mp_h{n}-mp_l{n},.01)'
    weighted = lambda n: f'({position(n)})*mp_v{n}'
    n29 = '+'.join(weighted(n) for n in range(21, 50))
    n4 = '+'.join(weighted(n) for n in range(46, 50))
    v29 = '+'.join(f'mp_v{n}' for n in range(21, 50))
    v4 = '+'.join(f'mp_v{n}' for n in range(46, 50))
    c.execute(f'CREATE VIEW aggregates AS SELECT *,{n29} AS mp_num29,{n4} AS mp_num4,{v29} AS mp_vol29,{v4} AS mp_vol4 FROM wide')
    valid_terms = ' AND '.join(f'isfinite(mp_{k}{n})' for k in study.FIELDS for n in range(21, 50))
    valid_terms += ' AND ' + ' AND '.join(f'mp_l{n}>0 AND mp_h{n}>=mp_c{n} AND mp_c{n}>=mp_l{n} AND mp_v{n}>=0'
                                         for n in range(21, 50))
    expected = c.sql(f'''WITH j AS(SELECT o.*,a.* EXCLUDE(date,code),
        coalesce(mp_bars=29 AND mp_clocks=29 AND mp_good_bars=29 AND {valid_terms}
        AND mp_vol29>0 AND mp_vol4>0 AND mp_vol29>mp_vol4,false) AS minute_pressure_valid
        FROM original o LEFT JOIN aggregates a USING(date,code))
        SELECT date,code,mp_num29,mp_num4,mp_vol29,mp_vol4,minute_pressure_valid,
        CASE WHEN minute_pressure_valid THEN 100*mp_num29/mp_vol29 END AS W01,
        CASE WHEN minute_pressure_valid THEN 100*(mp_num4/mp_vol4-(mp_num29-mp_num4)/(mp_vol29-mp_vol4)) END AS W02,
        formula_input_valid AND minute_pressure_valid AND isfinite(W01) AND isfinite(W02) AS formula_input_valid
        FROM j ORDER BY date,code''').df()
    c.close()
    pd.testing.assert_frame_equal(f[expected.columns], expected, check_dtype=False, rtol=1e-10, atol=2e-9)
    valid = f.formula_input_valid
    for name in study.NEW_EXPRESSIONS:
        a, b = f.loc[valid, name].to_numpy(), expected.loc[valid, name].to_numpy()
        np.testing.assert_allclose(a, b, atol=2e-9, rtol=0)
        np.testing.assert_array_equal(np.floor(np.clip(100*a+10000+.000001, 0, 999999)),
                                      np.floor(np.clip(100*b+10000+.000001, 0, 999999)))
    assert f.loc[valid, 'W01'].between(-100-1e-8, 100+1e-8).all()
    assert f.loc[valid, 'W02'].between(-200-1e-8, 200+1e-8).all()
    assert int(valid.sum()) == r['valid']
    assert int((old.formula_input_valid & ~valid).sum()) == r['newly_invalid']
    flat = (f[study.COLUMNS['h']].to_numpy() == f[study.COLUMNS['l']].to_numpy()).all(axis=1)
    assert int((valid & flat).sum()) == r['flat_valid_windows']
    np.testing.assert_allclose(f.loc[valid & flat, ['W01', 'W02']], 0, atol=1e-10, rtol=0)
    for key, old_key in [('mp_c49', 'price_1449'), ('mp_vol29', 'v29'), ('mp_vol4', 'v4')]:
        pd.testing.assert_series_equal(f.loc[valid, key], old.loc[valid, old_key], check_names=False, check_dtype=False, check_exact=True)
    assert r['expressions'] == study.EXPRESSIONS and r['native_header'] == study.HEADER
    names = re.findall(r'(?m)^([A-Za-z][A-Za-z0-9]*):=', study.HEADER) + list(study.EXPRESSIONS)
    assert len(names) == len(set(names))
    samples = []
    for half in ['2024H1', '2024H2', '2025H1', '2025H2']:
        for stratum, mask, count in [('valid', valid, 6), ('invalid', ~valid, 2)]:
            pool = f.loc[f.half.eq(half) & mask].copy()
            pool['sample_key'] = [hashlib.sha256(f'minute-pressure-v1|{d}|{code}'.encode()).hexdigest()
                                  for d, code in zip(pool.date, pool.code)]
            for _, row in pool.sort_values('sample_key').head(count).iterrows():
                path = study.MINUTES / row.code[:2].upper() / (row.code[3:] + '.parquet')
                raw = pd.read_parquet(path, columns=['timestamp', 'high', 'low', 'close', 'volume'],
                    filters=[('timestamp', '>=', pd.Timestamp(row.date+' 14:21:00')),
                             ('timestamp', '<', pd.Timestamp(row.date+' 14:50:00'))])
                counts, a = raw_window(raw, row.date)
                for key, value in counts.items():
                    assert row[key] == value
                for key, values in a.items():
                    np.testing.assert_allclose(values, row[study.COLUMNS[key]].to_numpy(dtype=float), rtol=0, atol=1e-12, equal_nan=True)
                contaminant = pd.DataFrame(dict(timestamp=[pd.Timestamp(row.date+' 14:50:00'), pd.Timestamp(row.date+' 15:00:00')],
                    high=[999999., .01], low=[.01, 999999.], close=[999999., .01], volume=[1e15, -1.]))
                counts2, a2 = raw_window(pd.concat([raw, contaminant], ignore_index=True), row.date)
                assert counts == counts2
                for key in a:
                    np.testing.assert_array_equal(a[key], a2[key])
                if row.formula_input_valid:
                    # Integer-cent reference eliminates cancellation in the fractional price representation.
                    h, l, q = [[int(math.floor(x*100+.5)) for x in a[k]] for k in ['h', 'l', 'c']]
                    terms = [((z-y)-(x-z))/max(x-y, 1)*v for x, y, z, v in zip(h, l, q, a['v'])]
                    w1 = 100*sum(terms)/sum(a['v'])
                    w2 = 100*(sum(terms[-4:])/sum(a['v'][-4:])-sum(terms[:-4])/sum(a['v'][:-4]))
                    np.testing.assert_allclose([w1, w2], row[['W01','W02']].to_numpy(dtype=float), rtol=0, atol=2e-9)
                samples.append(dict(date=row.date, code=row.code, stratum=stratum, raw_minutes=len(raw)))
    assert len(samples) == 32
    proof = dict(passed=True, feature_report_sha256=sha(root/'feature_report.json'), rows=len(f), valid=int(valid.sum()),
        all_previous_keys_and_48_values_unchanged=True, all_window_links_aggregates_inputs_and_encodings_rebuilt=True,
        previous_volume_and_endpoint_fields_reconciled=True, sampled_raw_windows=samples,
        sampled_raw_minutes=sum(x['raw_minutes'] for x in samples), integer_cent_reference_checked=True,
        future_minute_contamination_does_not_change_inputs=True, all_native_variable_names_unique=True,
        outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root/'feature_verification.json', proof)
    return {k:v for k,v in proof.items() if k != 'sampled_raw_windows'}


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
