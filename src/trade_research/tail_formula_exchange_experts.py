"""Fit Shanghai and Shenzhen components on the same original 48 inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_exchange_context as inputs
from . import tail_formula_market_experts as mechanics
from . import tail_formula_prior_day as adapter
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_exchange_experts'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
EXPRESSIONS = {**inputs.previous.EXPRESSIONS, 'EX01': inputs.NEW_EXPRESSIONS['EX01']}


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['references'].items():
        assert sha(Path(file)) == digest
    for key, path in [('feature_report_sha256', inputs.ROOT / 'feature_report.json'),
        ('feature_verification_sha256', inputs.ROOT / 'feature_verification.json'),
        ('features_sha256', inputs.ROOT / 'features.parquet'), ('boundary_protocol_sha256', labels.PROTOCOL),
        ('label_report_sha256', labels.ROOT / 'full_label_report.json'),
        ('label_verification_sha256', labels.ROOT / 'full_label_verification.json'),
        ('labels_sha256', labels.ROOT / 'full_labels.parquet')]:
        assert p[key] == sha(path)
    for file in ['feature_verification.json', 'native_input_verification.json']:
        proof = json.loads((inputs.ROOT / file).read_text())
        assert proof['passed'] and proof['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
    assert p['fit_features'] == 48 and p['export_features'] == 49 and p['fit_excludes_gate']
    assert p['gate']['feature'] == 'EX01' and p['gate']['threshold'] == 10050
    assert p['gate']['expression'] == EXPRESSIONS['EX01']
    return p


def setup(fold):
    master = checked_sources()
    adapter.STEM = STEM; adapter.ROOT = inputs.ROOT
    adapter.EXPRESSIONS = EXPRESSIONS; adapter.HEADER = inputs.HEADER
    adapter.COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')
    adapter.setup(fold); base.SOURCE = labels.ROOT
    # These helpers only wrap immutable fitted trees; the original module stays unchanged.
    mechanics.GATE_INDEX = 48; mechanics.GATE = 10050
    for name in ['2024', 'recent']:
        p = json.loads((Path('config') / (STEM + '_' + name + '_protocol.json')).read_text())
        assert p['master_protocol_sha256'] == sha(PROTOCOL) and p['gate'] == master['gate']
        assert p['expected_features'] == len(EXPRESSIONS) == 49 and p['fit_features'] == 48
        assert p['parameters'] == dict(master['parameters_common'], n_estimators=64)


def state_masks(x):
    assert x.shape[1] == 49 and np.isin(x[:, 48], [10000, 10100]).all()
    upper = x[:, 48] > 10050
    return [('lower', ~upper), ('upper', upper)]


def model():
    assert not (base.ROOT / 'model_report.json').exists()
    p = json.loads(base.PROTOCOL.read_text()); t = relative.training('relative')
    assert len(t) == p['expected_training_rows'] and t.date.nunique() == 241
    x = base.encode(t); y = t.target.to_numpy()
    w = 1 / t.groupby('date').code.transform('size').to_numpy()
    components = []; fitted_scores = np.empty(len(t))
    for state, mask in state_masks(x):
        # A fixed routing column must not alter the fitter's feature permutation.
        fitted = GradientBoostingRegressor(**p['parameters']).fit(x[mask, :48], y[mask], sample_weight=w[mask])
        assert fitted.n_features_in_ == 48
        trees = []
        for estimator in fitted.estimators_.ravel():
            tree = estimator.tree_
            trees.append({name: getattr(tree, name).reshape(-1).tolist() for name in [
                'feature', 'threshold', 'children_left', 'children_right', 'n_node_samples',
                'weighted_n_node_samples', 'value', 'impurity']})
        component = dict(state=state, rows=int(mask.sum()), days=t.loc[mask, 'date'].nunique(),
            weight_sum=float(w[mask].sum()), bias=float(fitted.init_.constant_.ravel()[0]),
            learning_rate=.05, trees=trees, fitted_feature_count=int(fitted.n_features_in_))
        fitted_scores[mask] = fitted.predict(x[mask, :48])
        np.testing.assert_allclose(base.predict(x[mask], component), fitted_scores[mask], rtol=0, atol=2e-12)
        components.append(component)
    bias, trees = mechanics.flatten(components, 'split')
    r = dict(protocol_sha256=sha(base.PROTOCOL), master_protocol_sha256=sha(PROTOCOL),
        implementation_sha256=sha(Path(__file__)), feature_report_sha256=sha(inputs.ROOT / 'feature_report.json'),
        label_report_sha256=sha(labels.ROOT / 'full_label_report.json'), variant='relative',
        rows=len(t), days=t.date.nunique(), last_observation=t.next_date.max(),
        training_start=p['training_start'], training_end=p['training_end'],
        feature_names=list(EXPRESSIONS), fit_feature_names=list(inputs.previous.EXPRESSIONS),
        parameters=p['parameters'], gate=p['gate'], learning_rate=.05, bias=bias, trees=trees,
        components=components, new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()),
        new_2025H2_score_groups_read=False, new_2026_prices_read=False, no_exit_rules=True)
    manual = base.predict(x, r)
    np.testing.assert_allclose(manual, fitted_scores, rtol=0, atol=2e-12)
    r['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(manual, q)))
                       for i, q in enumerate(base.QUANTILES)]
    base.ROOT.mkdir(parents=True, exist_ok=True); save_json(base.ROOT / 'model_report.json', r)
    return {k: v for k, v in r.items() if k not in ['trees', 'components']}


def independent_training():
    start, end, where = relative.training_scope(); names = list(EXPRESSIONS)
    f = base.feature_inputs()[['date', 'code', 'formula_input_valid', *names]]
    c = base.conn(); c.register('features', f)
    enc = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d = c.sql(f'''WITH l AS (SELECT date,code,next_date,
        opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target
        FROM read_parquet('{labels.ROOT}/full_labels.parquet') WHERE {where} AND known15)
        SELECT date,code,next_date,target,1./count(*) OVER(PARTITION BY date) AS w,{enc},
        CASE WHEN regexp_full_match(code,'sh[.]60[0-9]{{4}}') THEN 'upper'
             WHEN regexp_full_match(code,'sz[.]00[0-9]{{4}}') THEN 'lower' ELSE NULL END AS state
        FROM features JOIN l USING(date,code) WHERE formula_input_valid ORDER BY date,code''').df(); c.close()
    original = relative.training('relative')
    pd.testing.assert_frame_equal(d[['date', 'code', 'next_date']], original[['date', 'code', 'next_date']], check_exact=True)
    np.testing.assert_allclose(d.target, original.target, rtol=0, atol=2e-12)
    np.testing.assert_array_equal(d[names].to_numpy(dtype='int32'), base.encode(original))
    np.testing.assert_allclose(d.w, 1 / original.groupby('date').code.transform('size'), rtol=0, atol=0)
    assert d.state.notna().all() and d.next_date.lt(end).all() and d.date.ge(start).all()
    for state, mask in state_masks(base.encode(original)):
        np.testing.assert_array_equal(mask, d.state.eq(state))
    return d, original


def verify_model():
    p = json.loads(base.PROTOCOL.read_text()); root = base.ROOT
    r = json.loads((root / 'model_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('master_protocol_sha256', PROTOCOL),
        ('implementation_sha256', Path(__file__)), ('feature_report_sha256', inputs.ROOT / 'feature_report.json'),
        ('label_report_sha256', labels.ROOT / 'full_label_report.json')]:
        assert r[key] == sha(path)
    assert r['parameters'] == p['parameters'] and r['gate'] == p['gate'] and r['learning_rate'] == .05
    assert r['feature_names'] == list(EXPRESSIONS) and r['fit_feature_names'] == list(inputs.previous.EXPRESSIONS)
    assert (r['training_start'], r['training_end']) == (p['training_start'], p['training_end'])
    d, original = independent_training()
    assert len(d) == r['rows'] == p['expected_training_rows'] and d.date.nunique() == r['days'] == 241
    assert r['last_observation'] == original.next_date.max() < p['evaluation_start']
    x = d[list(EXPRESSIONS)].to_numpy(dtype='int32'); y = d.target.to_numpy(); w = d.w.to_numpy()
    assert [v['state'] for v in r['components']] == ['lower', 'upper']
    score = np.full(len(d), np.nan); count = 0; coverage = []
    for component in r['components']:
        mask = d.state.eq(component['state']).to_numpy()
        assert component['rows'] == int(mask.sum()) and component['days'] == d.loc[mask, 'date'].nunique()
        assert component['fitted_feature_count'] == 48
        assert all(max(tree['feature']) < 48 for tree in component['trees'])
        np.testing.assert_allclose(component['weight_sum'], w[mask].sum(), rtol=0, atol=1e-10)
        score[mask], checks = mechanics.verify_component(component, x[mask, :48], y[mask], w[mask], 64)
        count += checks
        coverage.append({k: v for k, v in component.items() if k not in ['bias', 'trees', 'learning_rate']})
    bias, trees = mechanics.flatten(r['components'], 'split')
    assert bias == r['bias'] and trees == r['trees'] and len(trees) == 129
    np.testing.assert_allclose(score, base.predict(x, r), rtol=0, atol=2e-10)
    for q, t in zip(base.QUANTILES, r['thresholds']):
        assert t['training_quantile'] == q
        np.testing.assert_allclose(np.quantile(score, q), t['threshold'], rtol=0, atol=2e-10)
    for control in [Path('data/research') / ('tail_formula_before1000_model_' + root.name.rsplit('_', 1)[1]),
                    Path('data/research') / ('tail_formula_market_experts_pooled_' + root.name.rsplit('_', 1)[1])]:
        old = json.loads((control / 'model_report.json').read_text())
        for key in ['rows', 'days', 'last_observation', 'training_start', 'training_end', 'label_report_sha256']:
            assert old[key] == r[key]
    proof = dict(passed=True, model_report_sha256=sha(root / 'model_report.json'), rows=len(d), days=241,
        node_checks=count, components=coverage, all_node_residual_statistics_and_support_rebuilt=True,
        states_independently_rebuilt_from_exchange_and_six_digit_code=True,
        all_training_keys_targets_encodings_and_global_weights_rebuilt=True,
        component_fit_uses_original_48_features_only=True, original_training_population_unchanged=True,
        routed_scores_equal_native_flattened_IFs=True, original_single_global_quantiles_rebuilt=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'model_verification.json', proof); return proof


def analyze(fold):
    joint = json.loads((ROOT / 'joint_selection_freeze.json').read_text())
    assert joint['passed'] and joint['protocol_sha256'] == sha(PROTOCOL) and len(joint['selections']) == 3
    root = linkage.COMBINED if fold == 'combined' else base.ROOT
    record = next(v for v in joint['selections'] if v['root'] == str(root))
    assert record['selection_report_sha256'] == sha(root / 'selection_report.json')
    return evaluation.analyze(root, linkage.PROTOCOL if fold == 'combined' else base.PROTOCOL)


def main():
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['sources', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    a = p.parse_args(); setup(a.fold)
    if a.stage == 'sources':
        ROOT.mkdir(parents=True, exist_ok=True)
        result = dict(passed=True, protocol_sha256=sha(PROTOCOL), implementation_sha256=sha(Path(__file__)),
            existing_venue_inputs_and_native_mapping_proofs_reused=True, new_group_outcomes_read=False,
            new_raw_price_extraction=False, new_2026_prices_read=False)
        save_json(ROOT / 'source_verification.json', result)
    elif a.stage == 'analyze':
        result = analyze(a.fold)
    elif a.fold == 'combined':
        assert a.stage in ['freeze', 'verify']
        result = getattr(linkage, 'combine' if a.stage == 'freeze' else 'verify_combined')()
    elif a.stage in ['model', 'verify_model']:
        result = globals()[a.stage]()
    elif a.stage == 'verify_scores':
        result = verify_scores(expected_expressions=EXPRESSIONS)
    elif a.stage == 'freeze':
        result = study.freeze()
    elif a.stage == 'verify':
        m = json.loads((base.ROOT / 'model_report.json').read_text())
        assert (base.ROOT / 'frozen_numeric_core.tdx').read_text() == base.native_core(m, m['thresholds'][3]['threshold'], EXPRESSIONS, inputs.HEADER)
        result = study.verify()
    else:
        result = base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
