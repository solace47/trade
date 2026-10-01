"""Audited existing prices for a matched-capacity, two-period prediction test."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from .research_io import check_sources, save_json, sha
from .tail_formula_baseline import CONTROL, HEADER, META
from . import tail_formula_additive as base


PROTOCOL = Path('config/tail_formula_phase_split_input_protocol.json')
ROOT = Path('data/research/tail_formula_phase_split')
TARGETS = ['net', 'tail_with_cost', 'after_close']
FIELDS = ['date', 'code', 'next_date', 'known15', 'opportunity15',
          'known_no_trade', 'adverse_return15']


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert subprocess.check_output(['git', 'show', f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    assert p['arms'] == {'direct': {'net': 64}, 'split': {'tail_with_cost': 32, 'after_close': 32}}
    assert p['maximum_new_fits'] == 12 and not p['target_mean_centered']
    assert not p['new_2026_prices_allowed'] and not p['new_raw_price_reads_allowed']
    assert len(CONTROL) == 50
    check_sources(p['source_hashes'])
    return p


def prepare():
    p = checked()
    assert not ROOT.exists(), 'Do not repeat a completed source gate'
    features = Path(p['features_root'])
    fr = json.loads((features / 'feature_report.json').read_text())
    fv = json.loads((features / 'feature_verification.json').read_text())
    nv = json.loads((features / 'native_input_verification.json').read_text())
    assert fv['passed'] and nv['passed']
    assert fv['feature_report_sha256'] == nv['feature_report_sha256'] == sha(features / 'feature_report.json')
    assert fr['features_sha256'] == sha(features / 'features.parquet')
    f = pd.read_parquet(features / 'features.parquet', columns=[*META, *CONTROL])
    assert len(f) == p['expected_keys'] and int(f.formula_input_valid.sum()) == p['expected_valid_features']
    assert not f.duplicated(['date', 'code']).any()
    base.EXPRESSIONS = CONTROL
    labels = []
    extra = ['day_close', 'price_0959', 'source_valid_0959', 'mark_0959_return15']
    for index, name in enumerate(p['label_roots']):
        source = Path(name)
        r = json.loads((source / 'full_label_report.json').read_text())
        v = json.loads((source / 'full_label_verification.json').read_text())
        assert v['passed'] and v['label_report_sha256'] == sha(source / 'full_label_report.json')
        assert r['labels_sha256'] == sha(source / 'full_labels.parquet')
        if index == 0:
            assert r['window_start'] == '09:31' and r['window_end'] == '09:59'
        else:
            assert r['mark_reference_label'] == '09:59' and v['window_end'] == '09:59'
        d = pd.read_parquet(source / 'full_labels.parquet', columns=FIELDS + extra)
        assert d.date.ge('2023-01-01' if index == 0 else '2024-01-01').all()
        assert d.date.lt('2024-01-01' if index == 0 else '2026-01-01').all()
        labels.append(d)
    l = pd.concat(labels, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    original = pd.read_parquet(features / 'full_labels.parquet', columns=FIELDS)
    original = original.sort_values(['date', 'code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(l[FIELDS], original, check_exact=True)
    assert len(l) == p['expected_keys'] and not l.duplicated(['date', 'code']).any()
    positive = np.isfinite(l[['day_close', 'price_0959']]).all(axis=1) & l[['day_close', 'price_0959']].gt(0).all(axis=1)
    finite_reference = np.isfinite(l.mark_0959_return15) & l.mark_0959_return15.gt(-1)
    usable = l.known15 & finite_reference & positive & l.source_valid_0959
    assert not (l.known15 & finite_reference & ~positive).any(), 'Known reference has invalid bridge prices'
    assert not (l.known15 & finite_reference & ~l.source_valid_0959).any()
    l['target_valid'] = usable
    l['net'] = (100 * np.log1p(l.loc[usable, 'mark_0959_return15'])).reindex(l.index)
    l['after_close'] = (100 * np.log(l.loc[usable, 'price_0959'] / l.loc[usable, 'day_close'])).reindex(l.index)
    l['tail_with_cost'] = l.net - l.after_close
    assert l.loc[~usable, TARGETS].isna().all().all()
    con = base.conn()
    con.register('raw', pd.concat(labels, ignore_index=True))
    sql = con.sql('''SELECT date,code,next_date,
        coalesce(known15 AND isfinite(mark_0959_return15) AND mark_0959_return15>-1
            AND isfinite(day_close) AND day_close>0 AND isfinite(price_0959)
            AND price_0959>0 AND source_valid_0959,false) AS target_valid,
        CASE WHEN target_valid THEN 100*ln(1+mark_0959_return15) END AS net,
        CASE WHEN target_valid THEN 100*ln(price_0959/day_close) END AS after_close,
        net-after_close AS tail_with_cost FROM raw ORDER BY date,code''').df()
    pd.testing.assert_frame_equal(l[['date', 'code', 'next_date', 'target_valid']],
        sql[['date', 'code', 'next_date', 'target_valid']], check_dtype=False)
    np.testing.assert_allclose(l[TARGETS], sql[TARGETS], rtol=0, atol=2e-12, equal_nan=True)
    counts = {}
    con.register('f', f)
    con.register('l', l)
    for fold, spec in p['folds'].items():
        d = l.loc[l.date.ge(spec['training_start']) & l.next_date.lt(spec['training_end']) & usable]
        t = f.loc[f.formula_input_valid].merge(d, on=['date', 'code'], validate='one_to_one')
        assert t.next_date.max() < spec['evaluation_start']
        s = con.sql(f'''SELECT count(*) AS rows, count(DISTINCT l.date) AS days,
            max(next_date) AS last_observation FROM f JOIN l USING(date,code)
            WHERE formula_input_valid AND target_valid AND l.date>='{spec['training_start']}'
            AND next_date<'{spec['training_end']}' ''').df().iloc[0]
        counts[fold] = dict(rows=len(t), days=int(t.date.nunique()), last_observation=t.next_date.max())
        assert counts[fold] == dict(rows=int(s.rows), days=int(s.days), last_observation=s.last_observation)
    con.close()
    ROOT.mkdir()
    output = ROOT / 'targets.parquet'
    l[FIELDS + ['target_valid', *TARGETS]].to_parquet(output, index=False, compression='zstd')
    report = dict(passed=True, protocol_sha256=sha(PROTOCOL), targets_sha256=sha(output),
        keys=len(l), valid_original_inputs=int(f.formula_input_valid.sum()),
        original_training_projection_exact=True, raw_price_targets_sql_rebuilt=True,
        unknown_targets_not_zero_filled=True, folds=counts,
        missing_reference_known=int((l.known15 & ~finite_reference).sum()),
        newly_invalid_price_bridges=0, fits_performed=0, new_selections=0,
        new_raw_prices_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'source_gate.json', report)
    return report
