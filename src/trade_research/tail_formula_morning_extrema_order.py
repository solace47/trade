"""Latest positive-volume morning extrema order, without new raw extraction."""
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_morning_range as prior
from . import tail_formula_volume_memory as original_source
from .corporate_cash import save_json, sha

STEM = 'tail_formula_morning_extrema_order'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
INTENT = Path('config') / (STEM + '_intent.json')
META, CONTROL = prior.META, prior.EXPRESSIONS


def minimum(terms):
    if len(terms) == 1:
        return terms[0]
    middle = len(terms) // 2
    return 'MIN(' + minimum(terms[:middle]) + ',' + minimum(terms[middle:]) + ')'


def latest_distance(field, extreme):
    terms = [f'IF((REF(V,{i})>0) AND (ROUND(REF({field},{i})*100)={extreme}),{i},121)' for i in range(121)]
    return 'VALUEWHEN(TIME=1130,' + minimum(terms) + ')'


EXTRA_HEADER = ('AMHB:=' + latest_distance('H', 'AMHC') + ';\n'
                'AMLB:=' + latest_distance('L', 'AMLC') + ';\n')
NEW_EXPRESSIONS = {'AMORD': 'IF(AMREADY,100*(AMLB-AMHB)/120,DRAWNULL)',
                   'AMLAGE': 'IF(AMREADY,100*AMLB/120,DRAWNULL)'}
EXPRESSIONS = {**CONTROL, **NEW_EXPRESSIONS}
HEADER = prior.HEADER + EXTRA_HEADER


def measure(high_cents, low_cents, volume):
    """Chronological 121 bars; ties take the latest active label."""
    h, l, v = np.asarray(high_cents, float), np.asarray(low_cents, float), np.asarray(volume, float)
    assert h.shape == l.shape == v.shape and h.ndim == 2 and h.shape[1] == 121
    assert np.isfinite(h).all() and np.isfinite(l).all() and np.isfinite(v).all()
    assert (h >= l).all() and (l > 0).all() and (v >= 0).all() and (v > 0).any(axis=1).all()
    highest = np.where(v > 0, h, -np.inf).max(axis=1)
    lowest = np.where(v > 0, l, np.inf).min(axis=1)
    positions = np.arange(121)
    ht = np.where((v > 0) & (h == highest[:, None]), positions, -1).max(axis=1)
    lt = np.where((v > 0) & (l == lowest[:, None]), positions, -1).max(axis=1)
    return np.column_stack([100 * (ht - lt) / 120, 100 * (120 - lt) / 120])


def literal(high_cents, low_cents, volume):
    """Evaluate the actual finite native header, not an extrema-index proxy."""
    h, l, v = np.asarray(high_cents, float), np.asarray(low_cents, float), np.asarray(volume, float)
    env = dict(H=h / 100, L=l / 100, V=v, REF=lambda a, k: a[:, 120 - k],
               ROUND=lambda x: np.floor(x + .5), IF=np.where, MIN=np.minimum,
               AMHC=np.where(v > 0, h, 0).max(axis=1),
               AMLC=np.where(v > 0, l, 999999999).min(axis=1),
               AMREADY=np.ones(len(h), bool), DRAWNULL=np.nan)
    for line in EXTRA_HEADER.splitlines():
        name, expression = line.rstrip(';').split(':=')
        assert expression.startswith('VALUEWHEN(TIME=1130,') and expression.endswith(')')
        expression = expression[len('VALUEWHEN(TIME=1130,'):-1].replace(' AND ', ' & ')
        expression = re.sub(r'(?<![<>=!])=(?!=)', '==', expression)
        env[name] = eval(expression, {'__builtins__': {}}, env)
    return np.column_stack([eval(e, {'__builtins__': {}}, env) for e in NEW_EXPRESSIONS.values()])


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['intent_sha256'] == sha(INTENT) and p['expressions'] == EXPRESSIONS and p['native_header'] == HEADER
    assert not p['new_2026_prices_allowed'] and p['no_new_raw_extraction']
    for file, digest in p['source_hashes'].items(): assert sha(Path(file)) == digest, file
    gate = json.loads(Path(p['conditional_gate']).read_text())
    assert gate['passed'] and not gate['supports_further_validation']
    committed = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'], text=True, capture_output=True, check=True).stdout
    assert sha(PROTOCOL) in committed and sha(INTENT) in committed
    return p


def raw_sources(p):
    sources = []
    for report_file, first, end in p['raw_report_scopes']:
        report = json.loads(Path(report_file).read_text())
        assert not report['new_2026_prices_read']
        for file, digest in report['parts_sha256'].items():
            assert sha(Path(file)) == digest, file
            sources.append(dict(file=file, sha256=digest, first=first, end=end))
    assert len({s['file'] for s in sources}) == len(sources)
    return sources


def prepare():
    p = checked(); assert not (INPUTS / 'feature_report.json').exists(); INPUTS.mkdir(parents=True, exist_ok=True)
    f = original_source.original(); valid_keys = f.loc[f.formula_input_valid, ['date', 'code']]
    sources = raw_sources(p); records, pieces = [], []
    for index, source in enumerate(sources):
        d = pd.read_parquet(source['file'])
        assert d.date.ge(source['first']).all() and d.date.lt(source['end']).all()
        assert d.timestamp.dt.strftime('%Y-%m-%d').eq(d.date).all()
        d['position'] = d.timestamp.dt.hour * 60 + d.timestamp.dt.minute - 570
        assert d.position.between(0, 120).all() and not d.duplicated(['date', 'code', 'position']).any()
        d['hc'] = np.floor(100 * d.high + .5); d['lc'] = np.floor(100 * d.low + .5)
        d['active_high'] = d.hc.where(d.volume.gt(0)); d['active_low'] = d.lc.where(d.volume.gt(0))
        group = d.groupby(['date', 'code'], sort=False)
        high, low = group.active_high.transform('max'), group.active_low.transform('min')
        top = d.loc[d.volume.gt(0) & d.hc.eq(high)].groupby(['date', 'code']).position.max().rename('high_time')
        bottom = d.loc[d.volume.gt(0) & d.lc.eq(low)].groupby(['date', 'code']).position.max().rename('low_time')
        keys = group.size().rename('raw_rows').to_frame()
        a = keys.join(top).join(bottom).reset_index()
        a['AMORD'] = 100 * (a.high_time - a.low_time) / 120
        a['AMLAGE'] = 100 * (120 - a.low_time) / 120
        # Every originally valid row receives a replay of all 121 bars from
        # the actual native expression, including latest-tie and zero-volume rules.
        v = d.merge(valid_keys, on=['date', 'code'], validate='many_to_one')
        if len(v):
            counts = v.groupby(['date', 'code']).size(); assert counts.eq(121).all()
            arrays = [v.pivot(index=['date', 'code'], columns='position', values=n).sort_index() for n in ['hc', 'lc', 'volume']]
            assert all(g.columns.tolist() == list(range(121)) and g.notna().all().all() for g in arrays)
            values = [g.to_numpy(float) for g in arrays]
            calculated, native = measure(*values), literal(*values)
            np.testing.assert_allclose(calculated, native, rtol=0, atol=2e-12)
            actual = a.set_index(['date', 'code']).loc[arrays[0].index, list(NEW_EXPRESSIONS)].to_numpy(float)
            np.testing.assert_allclose(actual, native, rtol=0, atol=2e-12)
            np.testing.assert_array_equal(np.floor(100 * actual + 10000 + .000001), np.floor(100 * native + 10000 + .000001))
            replayed = len(calculated)
        else:
            replayed = 0
        # Independent SQL uses window extrema followed by conditional MAX,
        # never the pandas transform or its latest-position implementation.
        c = base.conn(); c.register('raw', d[['date', 'code', 'timestamp', 'high', 'low', 'volume']])
        independent = c.sql('''WITH prices AS (SELECT *,round(high*100) AS hc,round(low*100) AS lc,
            extract(hour FROM timestamp)*60+extract(minute FROM timestamp)-570 AS pos FROM raw), ranges AS (
            SELECT *,max(hc) FILTER(WHERE volume>0) OVER(PARTITION BY date,code) AS hi,
                     min(lc) FILTER(WHERE volume>0) OVER(PARTITION BY date,code) AS lo FROM prices)
            SELECT date,code,count(*) AS raw_rows,
              max(pos) FILTER(WHERE volume>0 AND hc=hi) AS high_time,
              max(pos) FILTER(WHERE volume>0 AND lc=lo) AS low_time FROM ranges GROUP BY date,code ORDER BY date,code''').df(); c.close()
        pd.testing.assert_frame_equal(a[['date', 'code', 'raw_rows', 'high_time', 'low_time']].sort_values(['date', 'code']).reset_index(drop=True),
                                     independent, check_exact=True, check_dtype=False)
        pieces.append(a); records.append(dict(index=index, source=source['file'], source_sha256=source['sha256'],
                                             raw_rows=len(d), stock_dates=len(a), native_replayed_valid_keys=replayed))
        print(json.dumps(records[-1]), flush=True)
    a = pd.concat(pieces, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    assert not a.duplicated(['date', 'code']).any()
    assert not len(a[['date', 'code']].merge(f[['date', 'code']], on=['date', 'code'], how='left', indicator=True).query('_merge == "left_only"'))
    combined = f[['date', 'code']].merge(a, on=['date', 'code'], how='left', validate='one_to_one')
    for n in NEW_EXPRESSIONS: f[n] = combined[n].to_numpy()
    assert np.isfinite(f.loc[f.formula_input_valid, list(NEW_EXPRESSIONS)]).all().all()
    assert f.loc[f.formula_input_valid, 'AMORD'].between(-100, 100).all()
    assert f.loc[f.formula_input_valid, 'AMLAGE'].between(0, 100).all()
    assert sum(r['native_replayed_valid_keys'] for r in records) == int(f.formula_input_valid.sum()) == 1602413
    old = original_source.original(); pd.testing.assert_frame_equal(f[old.columns], old, check_exact=True)
    a.to_parquet(INPUTS / 'morning_order_aggregates.parquet', index=False, compression='zstd')
    f.to_parquet(INPUTS / 'features.parquet', index=False, compression='zstd')
    save_json(INPUTS / 'feature_report.json', dict(protocol_sha256=sha(PROTOCOL), features_sha256=sha(INPUTS / 'features.parquet'),
        aggregates_sha256=sha(INPUTS / 'morning_order_aggregates.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=0, raw_records=records, expressions=EXPRESSIONS, native_header=HEADER,
        no_new_raw_extraction=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True))
    save_json(INPUTS / 'feature_verification.json', dict(passed=True, feature_report_sha256=sha(INPUTS / 'feature_report.json'),
        all_original_keys_50_values_and_validity_exact=True, all_extrema_latest_positions_independent_sql_exact=True,
        full_valid_input_domain_unchanged=True, no_future_prices_read=True, new_2026_prices_read=False))
    save_json(INPUTS / 'native_input_verification.json', dict(passed=True, feature_report_sha256=sha(INPUTS / 'feature_report.json'),
        feature_verification_sha256=sha(INPUTS / 'feature_verification.json'), rows=1602413,
        all_121_bars_of_every_valid_stock_date_literal_native_replayed=True,
        all_new_values_and_integer_encodings_equal=True, latest_ties_and_positive_volume_extrema_verified=True,
        software_compilation_verified=False, native_source_parity_verified=False, new_2026_prices_read=False))
    for file in ['full_labels.parquet', 'full_label_report.json', 'full_label_verification.json']:
        (INPUTS / file).symlink_to((prior.INPUTS / file).resolve())
    return dict(rows=len(f), valid=int(f.formula_input_valid.sum()), newly_invalid=0,
                feature_report_sha256=sha(INPUTS / 'feature_report.json'),
                verification_sha256=sha(INPUTS / 'feature_verification.json'),
                native_verification_sha256=sha(INPUTS / 'native_input_verification.json'))
