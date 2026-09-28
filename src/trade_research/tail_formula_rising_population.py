"""Fixed positive-current-day research population, separate from data quality."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as source
from . import tail_formula_prior_day as adapter
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_rising_population'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
COMBINED = Path('config') / (STEM + '_combined_protocol.json')
ORIGINAL = Path('data/research/tail_formula_before1000_model_2025')
VISIBLE = Path('data/research/next_day_winner/visible_base.parquet')
EXPRESSIONS, HEADER = source.EXPRESSIONS, source.HEADER
GATE = 'INTPART(Q*100+0.5)>INTPART(DYNAINFO(3)*100+0.5)'
ORIGINAL_NATIVE_CORE = base.native_core


def native_core(*args, **kwargs):
    text = ORIGINAL_NATIVE_CORE(*args, **kwargs)
    assert text.count('CORE:SC>') == 1
    return text.replace('CORE:SC>', 'CORE:' + GATE + ' AND SC>')


def policy():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    r = json.loads((source.ROOT / 'feature_report.json').read_text())
    assert r['features_sha256'] == sha(source.ROOT / 'features.parquet')
    assert p['expected_features'] == len(EXPRESSIONS) == 48 and not p['new_2026_prices_allowed']
    return p


def features():
    p = policy()
    assert not (ROOT / 'feature_report.json').exists()
    f = pd.read_parquet(source.ROOT / 'features.parquet')
    f['reliable_input_valid'] = f.formula_input_valid
    f['research_population_eligible'] = np.rint(f.price_1449*100) > np.rint(f.preclose*100)
    f['formula_input_valid'] &= f.research_population_eligible
    keys = pd.read_parquet(labels.ROOT / 'full_labels.parquet', columns=['date', 'code', 'next_date', 'known15'])
    candidates = f.loc[f.formula_input_valid, ['date', 'code']].merge(keys.loc[keys.known15], on=['date', 'code'], validate='one_to_one')
    training_counts = {}
    for fold, spec in p['folds'].items():
        z = candidates.loc[candidates.date.ge(spec['training_start']) & candidates.next_date.lt(spec['training_end'])]
        training_counts[fold] = dict(rows=len(z), days=z.date.nunique(), last_observation=z.next_date.max())
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), source_feature_report_sha256=sha(source.ROOT / 'feature_report.json'),
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f), reliable_inputs=int(f.reliable_input_valid.sum()),
        valid=int(f.formula_input_valid.sum()), new_quality_invalid=0,
        excluded_by_research_population=int((f.reliable_input_valid & ~f.research_population_eligible).sum()),
        training_counts=training_counts, quality_and_population_separately_recorded=True, all_48_values_unchanged=True,
        expressions=EXPRESSIONS, native_header=HEADER, native_core_gate=GATE,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['expressions', 'native_header']}


def verify_features():
    p = policy()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT / 'features.parquet')
    old = pd.read_parquet(source.ROOT / 'features.parquet')
    f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.reliable_input_valid, old.formula_input_valid, check_names=False, check_exact=True)
    c = base.conn()
    c.register('old', old[['date', 'code', 'formula_input_valid']])
    expected = c.sql(f'''SELECT o.date,o.code,o.formula_input_valid AS reliable_input_valid,
        coalesce(round(v.price_1449*100)>round(v.preclose*100),false) AS research_population_eligible,
        o.formula_input_valid AND research_population_eligible AS formula_input_valid
        FROM old o JOIN read_parquet('{VISIBLE}') v USING(date,code) ORDER BY date,code''').df()
    pd.testing.assert_frame_equal(f[expected.columns], expected, check_exact=True)
    native_flag = np.trunc(old.price_1449*100+.5) > np.trunc(old.preclose*100+.5)
    np.testing.assert_array_equal(native_flag, f.research_population_eligible)
    np.testing.assert_array_equal(old.A01.gt(0), f.research_population_eligible)
    assert r['rows'] == len(f) and r['valid'] == int(f.formula_input_valid.sum())
    assert r['reliable_inputs'] == int(old.formula_input_valid.sum())
    assert r['excluded_by_research_population'] == int((f.reliable_input_valid & ~f.research_population_eligible).sum())
    assert r['new_quality_invalid'] == 0 and np.isfinite(f.loc[f.formula_input_valid, list(EXPRESSIONS)]).all().all()
    c.register('expected', expected)
    for fold, spec in p['folds'].items():
        counts = c.sql(f'''SELECT count(*) AS rows,count(DISTINCT l.date) AS days,max(l.next_date) AS last_observation
            FROM read_parquet('{labels.ROOT}/full_labels.parquet') l JOIN expected e USING(date,code)
            WHERE l.known15 AND e.formula_input_valid AND l.date>='{spec['training_start']}' AND l.next_date<'{spec['training_end']}' ''').df().iloc[0].to_dict()
        assert counts == r['training_counts'][fold]
    c.close()
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER and r['native_core_gate'] == GATE
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f), valid=r['valid'],
        all_population_flags_sql_rebuilt_from_original_visible_prices=True,
        actual_native_integer_gate_and_original_A01_sign_verified_for_all_rows=True,
        all_48_values_original_quality_and_keys_unchanged=True, all_training_population_counts_sql_rebuilt=True,
        numerical_validity_now_includes_explicit_research_population=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof)
    return proof


def configure(fold):
    p = policy()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    assert r['features_sha256'] == sha(ROOT / 'features.parquet')
    adapter.STEM, adapter.ROOT = STEM, ROOT
    adapter.EXPRESSIONS, adapter.HEADER, adapter.COMBINED_PROTOCOL = EXPRESSIONS, HEADER, COMBINED
    base.native_core = native_core
    for name in ['2024', 'recent']:
        q = json.loads((Path('config') / (STEM + '_' + name + '_protocol.json')).read_text())
        assert q['inputs_protocol_sha256'] == sha(PROTOCOL) and q['parameters'] == p['parameters']
        assert q['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
        assert q['feature_verification_sha256'] == sha(ROOT / 'feature_verification.json')
        assert q['expected_rows'] == r['training_counts'][name]['rows']
        assert q['expected_training_days'] == r['training_counts'][name]['days']
    if fold not in ['control', 'population']:
        adapter.setup(fold)
    base.SOURCE = labels.ROOT


def ancillary(stage, fold):
    assert fold in ['control', 'population']
    root = Path('data/research') / (STEM + '_' + fold)
    f = pd.read_parquet(ROOT / 'features.parquet')
    original = pd.read_parquet(ORIGINAL / 'selection.parquet')
    pd.testing.assert_frame_equal(f[['date', 'code']], original[['date', 'code']], check_exact=True)
    out = original.copy()
    out['selected'] = (original.selected & f.research_population_eligible if fold == 'control'
        else f.date.ge('2025-01-01') & f.date.lt('2026-01-01') & f.formula_input_valid)
    if stage == 'freeze':
        assert not (root / 'selection_report.json').exists()
        assert not any((Path('data/research') / (STEM + '_' + part) / 'analysis_report.json').exists() for part in ['2024', 'recent', '2025', 'control', 'population'])
        root.mkdir(parents=True, exist_ok=True)
        out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
        r = dict(protocol_sha256=sha(COMBINED), inputs_protocol_sha256=sha(PROTOCOL),
            feature_report_sha256=sha(ROOT / 'feature_report.json'), original_selection_report_sha256=sha(ORIGINAL / 'selection_report.json'),
            selection_sha256=sha(root / 'selection.parquet'), selected=int(out.selected.sum()),
            days=out.loc[out.selected, 'date'].nunique(), arm=fold, model_refitted=False,
            new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
        save_json(root / 'selection_report.json', r)
        return r
    assert stage == 'verify'
    r = json.loads((root / 'selection_report.json').read_text())
    assert r['protocol_sha256'] == sha(COMBINED) and r['inputs_protocol_sha256'] == sha(PROTOCOL)
    assert r['selection_sha256'] == sha(root / 'selection.parquet')
    c = base.conn()
    eligible = 'round(v.price_1449*100)>round(v.preclose*100)'
    condition = 's.selected AND ' + eligible if fold == 'control' else "s.date>='2025-01-01' AND s.date<'2026-01-01' AND f.formula_input_valid AND " + eligible
    expected = c.sql(f'''SELECT s.date,s.code,s.half,s.board,s.decision_shares,{condition} AS selected
        FROM read_parquet('{ORIGINAL}/selection.parquet') s JOIN read_parquet('{source.ROOT}/features.parquet') f USING(date,code)
        JOIN read_parquet('{VISIBLE}') v USING(date,code) ORDER BY date,code''').df()
    c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(root / 'selection.parquet'), expected, check_exact=True)
    pd.testing.assert_frame_equal(out, expected, check_exact=True)
    assert r['selected'] == int(expected.selected.sum()) and r['days'] == expected.loc[expected.selected, 'date'].nunique()
    proof = dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'), rows=len(expected),
        all_selection_flags_rebuilt_from_original_visible_prices_quality_and_parent_selection=True,
        all_original_nonselection_fields_unchanged=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'selection_verification.json', proof)
    return proof


def main():
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--fold', choices=['2024', 'recent', 'combined', 'control', 'population'], default='2024')
    a = p.parse_args()
    if a.stage in ['features', 'verify_features']:
        result = globals()[a.stage]()
    else:
        configure(a.fold)
        if a.stage == 'analyze':
            assert a.fold in ['combined', 'control', 'population'], 'Annual reports already include both halves'
            for fold in ['2024', 'recent', '2025', 'control', 'population']:
                root = Path('data/research') / (STEM + '_' + fold)
                v = json.loads((root / 'selection_verification.json').read_text())
                assert v['passed'] and v['selection_report_sha256'] == sha(root / 'selection_report.json')
            root = Path('data/research') / (STEM + '_' + ('2025' if a.fold == 'combined' else a.fold))
            result = evaluation.analyze(root, COMBINED)
        elif a.fold in ['control', 'population']:
            result = ancillary(a.stage, a.fold)
        elif a.fold == 'combined':
            assert a.stage in ['freeze', 'verify']
            result = linkage.combine() if a.stage == 'freeze' else linkage.verify_combined()
        elif a.stage == 'model':
            result = relative.model('relative')
        elif a.stage == 'verify_model':
            q = json.loads(base.PROTOCOL.read_text())
            r = json.loads((base.ROOT / 'model_report.json').read_text())
            assert r['rows'] == q['expected_rows'] and r['days'] == q['expected_training_days']
            assert all(r['parameters'][k] == v for k, v in q['parameters'].items())
            result = relative.verify_model('relative')
        elif a.stage == 'verify_scores':
            result = verify_scores()
        elif a.stage == 'freeze':
            result = study.freeze()
        elif a.stage == 'verify':
            r = json.loads((base.ROOT / 'model_report.json').read_text())
            assert (base.ROOT / 'frozen_numeric_core.tdx').read_text() == native_core(r, r['thresholds'][3]['threshold'], EXPRESSIONS, HEADER)
            result = study.verify()
        else:
            result = base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
