"""Average four complete boosted models, each omitting one observation quarter."""
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
from . import tail_formula_float as inputs
from . import tail_formula_prior_day as adapter
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .tail_formula_market_experts import verify_component
from .corporate_cash import save_json, sha

STEM = 'tail_formula_quarter_ensemble'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
LR = .05


def setup(fold):
    master = json.loads(PROTOCOL.read_text())
    for key, path in [
        ('feature_report_sha256', inputs.ROOT / 'feature_report.json'),
        ('feature_verification_sha256', inputs.ROOT / 'feature_verification.json'),
        ('features_sha256', inputs.ROOT / 'features.parquet'),
        ('boundary_protocol_sha256', labels.PROTOCOL),
        ('label_report_sha256', labels.ROOT / 'full_label_report.json'),
        ('label_verification_sha256', labels.ROOT / 'full_label_verification.json'),
        ('labels_sha256', labels.ROOT / 'full_labels.parquet'),
        ('component_verifier_sha256', Path(master['component_verifier_file'])),
    ]:
        assert master[key] == sha(path)
    assert master['components'] == 4 and master['flattened_trees'] == 256
    adapter.STEM = STEM; adapter.ROOT = inputs.ROOT
    adapter.EXPRESSIONS = inputs.EXPRESSIONS; adapter.HEADER = inputs.HEADER
    adapter.COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')
    adapter.setup(fold); base.SOURCE = labels.ROOT
    for name in ['2024', 'recent']:
        p = json.loads((Path('config') / (STEM + '_' + name + '_protocol.json')).read_text())
        assert p['master_protocol_sha256'] == sha(PROTOCOL)
        assert p['omitted_quarters'] == master['quarter_sets'][name]
        assert p['parameters'] == master['parameters'] and p['expected_features'] == len(inputs.EXPRESSIONS) == 48


def quarters(next_date):
    return pd.to_datetime(next_date).dt.to_period('Q').astype(str)


def component_masks(next_date, omitted):
    q = quarters(next_date)
    assert sorted(q.unique()) == omitted and len(omitted) == 4
    masks = [(name, q.ne(name).to_numpy()) for name in omitted]
    np.testing.assert_array_equal(np.stack([mask for _, mask in masks]).sum(axis=0), np.full(len(q), 3))
    return masks


def flatten(components):
    assert len(components) == 4
    bias = sum(c['bias'] for c in components) / 4
    trees = [dict(t, value=[v / 4 for v in t['value']]) for c in components for t in c['trees']]
    return bias, trees


def independent_training():
    start, end, where = relative.training_scope(); names = list(inputs.EXPRESSIONS)
    f = base.feature_inputs()[['date', 'code', 'formula_input_valid', *names]]
    c = base.conn(); c.register('features', f)
    enc = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d = c.sql(f'''WITH l AS (SELECT date,code,next_date,
        opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target
        FROM read_parquet('{labels.ROOT}/full_labels.parquet') WHERE {where} AND known15)
        SELECT date,code,next_date,target,1./count(*) OVER(PARTITION BY date) AS w,{enc},
        year(CAST(next_date AS DATE))::VARCHAR||'Q'||quarter(CAST(next_date AS DATE))::VARCHAR AS observation_quarter
        FROM features JOIN l USING(date,code) WHERE formula_input_valid ORDER BY date,code''').df(); c.close()
    original = relative.training('relative')
    pd.testing.assert_frame_equal(d[['date', 'code', 'next_date']], original[['date', 'code', 'next_date']], check_exact=True)
    np.testing.assert_allclose(d.target, original.target, rtol=0, atol=2e-12)
    np.testing.assert_array_equal(d[names].to_numpy(dtype='int32'), base.encode(original))
    np.testing.assert_allclose(d.w, 1 / original.groupby('date').code.transform('size'), rtol=0, atol=0)
    pd.testing.assert_series_equal(d.observation_quarter, quarters(original.next_date), check_names=False)
    assert original.next_date.lt(end).all() and original.date.ge(start).all()
    assert d.groupby('date').next_date.nunique().eq(1).all()
    return d, original


def verify_inputs():
    d, original = independent_training(); p = json.loads(base.PROTOCOL.read_text())
    assert len(d) == p['expected_training_rows'] and d.date.nunique() == p['expected_training_days'] == 241
    coverage = []
    for name, mask in component_masks(original.next_date, p['omitted_quarters']):
        np.testing.assert_array_equal(mask, d.observation_quarter.ne(name))
        coverage.append(dict(omitted_quarter=name, rows=int(mask.sum()), days=int(d.loc[mask, 'date'].nunique()),
                             weight_sum=float(d.loc[mask, 'w'].sum())))
    proof = dict(passed=True, protocol_sha256=sha(base.PROTOCOL), master_protocol_sha256=sha(PROTOCOL),
        rows=len(d), days=d.date.nunique(), last_observation=original.next_date.max(), components=coverage,
        all_keys_targets_integer_inputs_weights_and_quarters_sql_rebuilt=True,
        each_record_used_by_exactly_three_components=True, no_future_labels_or_new_2026_prices_read=True,
        no_exit_rules=True)
    base.ROOT.mkdir(parents=True, exist_ok=True)
    path = base.ROOT / 'training_input_verification.json'
    if path.exists():
        assert json.loads(path.read_text()) == proof
    else:
        save_json(path, proof)
    return proof


def model():
    assert not (base.ROOT / 'model_report.json').exists(), 'Do not refit frozen models'
    p = json.loads(base.PROTOCOL.read_text()); proof = json.loads((base.ROOT / 'training_input_verification.json').read_text())
    assert proof['passed'] and proof['protocol_sha256'] == sha(base.PROTOCOL)
    train = relative.training('relative'); x = base.encode(train); y = train.target.to_numpy()
    w = 1 / train.groupby('date').code.transform('size').to_numpy()
    assert len(train) == p['expected_training_rows'] and train.date.nunique() == 241
    folder = base.ROOT / 'components'; folder.mkdir(exist_ok=True)
    components = []; component_hashes = {}
    for name, mask in component_masks(train.next_date, p['omitted_quarters']):
        path = folder / (name + '.json')
        meta = dict(omitted_quarter=name, protocol_sha256=sha(base.PROTOCOL), implementation_sha256=sha(Path(__file__)),
            training_input_verification_sha256=sha(base.ROOT / 'training_input_verification.json'),
            rows=int(mask.sum()), days=int(train.loc[mask, 'date'].nunique()), weight_sum=float(w[mask].sum()), learning_rate=LR)
        if path.exists():
            component = json.loads(path.read_text())
            assert all(component[k] == v for k, v in meta.items())
        else:
            fitted = GradientBoostingRegressor(**p['parameters']).fit(x[mask], y[mask], sample_weight=w[mask])
            trees = []
            for estimator in fitted.estimators_.ravel():
                t = estimator.tree_
                trees.append({field: getattr(t, field).reshape(-1).tolist() for field in [
                    'feature', 'threshold', 'children_left', 'children_right', 'n_node_samples',
                    'weighted_n_node_samples', 'value', 'impurity']})
            component = dict(meta, bias=float(fitted.init_.constant_.ravel()[0]), trees=trees)
            np.testing.assert_allclose(base.predict(x, component), fitted.predict(x), rtol=0, atol=2e-12)
            save_json(path, component)
        components.append(component); component_hashes[str(path)] = sha(path)
        print(json.dumps(dict(component=name, rows=component['rows'], days=component['days'])), flush=True)
    bias, trees = flatten(components)
    r = dict(protocol_sha256=sha(base.PROTOCOL), master_protocol_sha256=sha(PROTOCOL),
        implementation_sha256=sha(Path(__file__)), feature_report_sha256=sha(inputs.ROOT / 'feature_report.json'),
        label_report_sha256=sha(labels.ROOT / 'full_label_report.json'),
        training_input_verification_sha256=sha(base.ROOT / 'training_input_verification.json'),
        variant='relative', rows=len(train), days=train.date.nunique(), last_observation=train.next_date.max(),
        training_start=p['training_start'], training_end=p['training_end'], feature_names=list(inputs.EXPRESSIONS),
        parameters=GradientBoostingRegressor(**p['parameters']).get_params(), learning_rate=LR, bias=bias, trees=trees,
        components=components, component_hashes=component_hashes, new_2025_score_groups_read=bool(train.date.ge('2025-01-01').any()),
        new_2025H2_score_groups_read=False, new_2026_prices_read=False, no_exit_rules=True)
    manual = base.predict(x, r); separate = np.mean([base.predict(x, c) for c in components], axis=0)
    np.testing.assert_allclose(manual, separate, rtol=0, atol=2e-12)
    r['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(manual, q))) for i, q in enumerate(base.QUANTILES)]
    save_json(base.ROOT / 'model_report.json', r)
    return {k: v for k, v in r.items() if k not in ['trees', 'components', 'component_hashes']}


def verify_model():
    p = json.loads(base.PROTOCOL.read_text()); r = json.loads((base.ROOT / 'model_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('master_protocol_sha256', PROTOCOL),
        ('implementation_sha256', Path(__file__)), ('feature_report_sha256', inputs.ROOT / 'feature_report.json'),
        ('label_report_sha256', labels.ROOT / 'full_label_report.json'),
        ('training_input_verification_sha256', base.ROOT / 'training_input_verification.json')]:
        assert r[key] == sha(path)
    assert r['feature_names'] == list(inputs.EXPRESSIONS) and r['learning_rate'] == LR
    assert all(r['parameters'][k] == v for k, v in p['parameters'].items())
    assert (r['training_start'], r['training_end']) == (p['training_start'], p['training_end'])
    d, original = independent_training()
    assert r['rows'] == len(d) == p['expected_training_rows'] and r['days'] == d.date.nunique() == 241
    assert r['last_observation'] == original.next_date.max() < p['evaluation_start']
    x = d[list(inputs.EXPRESSIONS)].to_numpy(dtype='int32'); y = d.target.to_numpy(); w = d.w.to_numpy()
    assert [c['omitted_quarter'] for c in r['components']] == p['omitted_quarters']
    checks = 0; coverage = []; component_scores = []
    for component in r['components']:
        name = component['omitted_quarter']; mask = d.observation_quarter.ne(name).to_numpy()
        path = base.ROOT / 'components' / (name + '.json')
        assert r['component_hashes'][str(path)] == sha(path) and json.loads(path.read_text()) == component
        assert component['rows'] == int(mask.sum()) and component['days'] == d.loc[mask, 'date'].nunique()
        np.testing.assert_allclose(component['weight_sum'], w[mask].sum(), rtol=0, atol=1e-10)
        rebuilt, count = verify_component(component, x[mask], y[mask], w[mask], 64)
        np.testing.assert_allclose(rebuilt, base.predict(x[mask], component), rtol=0, atol=2e-10)
        component_scores.append(base.predict(x, component)); checks += count
        coverage.append(dict(omitted_quarter=name, rows=int(mask.sum()), days=component['days'], node_checks=count))
    bias, trees = flatten(r['components']); assert bias == r['bias'] and trees == r['trees'] and len(trees) == 256
    score = np.mean(component_scores, axis=0); np.testing.assert_allclose(score, base.predict(x, r), rtol=0, atol=2e-12)
    for q, threshold in zip(base.QUANTILES, r['thresholds']):
        assert q == threshold['training_quantile']
        np.testing.assert_allclose(np.quantile(score, q), threshold['threshold'], rtol=0, atol=2e-12)
    proof = dict(passed=True, model_report_sha256=sha(base.ROOT / 'model_report.json'), rows=len(d), days=241,
        node_checks=checks, components=coverage, every_component_node_residual_statistics_rebuilt=True,
        component_average_equals_flattened_IF_scores=True, full_training_quantiles_rebuilt=True,
        aggregate_training_scores_are_not_out_of_sample=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT / 'model_verification.json', proof); return proof


def selection_gate(write=False):
    records = []
    for fold in ['2024', 'recent', '2025']:
        root = Path('data/research') / (STEM + '_' + fold)
        v = json.loads((root / 'selection_verification.json').read_text()); s = json.loads((root / 'selection_report.json').read_text())
        assert v['passed'] and v['selection_report_sha256'] == sha(root / 'selection_report.json')
        assert s['selection_sha256'] == sha(root / 'selection.parquet')
        if write:
            assert not (root / 'analysis_report.json').exists()
        if fold != '2025':
            m = json.loads((root / 'model_report.json').read_text())
            old = json.loads((Path('data/research') / ('tail_formula_before1000_model_' + fold) / 'model_report.json').read_text())
            for key in ['rows', 'days', 'last_observation', 'training_start', 'training_end', 'feature_report_sha256', 'label_report_sha256', 'parameters', 'feature_names']:
                assert m[key] == old[key]
        records.append(dict(fold=fold, root=str(root), selected=s['selected'],
            selection_report_sha256=sha(root / 'selection_report.json'), selection_verification_sha256=sha(root / 'selection_verification.json')))
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), selections=records, both_folds_and_year_frozen_together=True,
        original_complete_training_population_unchanged=True, no_individual_component_selection=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    path = ROOT / 'joint_selection_freeze.json'
    if write:
        ROOT.mkdir(parents=True, exist_ok=True); assert not path.exists(); save_json(path, proof)
    else:
        assert json.loads(path.read_text()) == proof
    return proof


if __name__ == '__main__':
    from .tail_formula_offset_logit48 import verify_scores
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['verify_inputs', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'gate', 'analyze'])
    parser.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    args = parser.parse_args(); setup(args.fold)
    if args.stage == 'gate':
        result = selection_gate(write=True)
    elif args.stage == 'analyze':
        selection_gate()
        result = evaluation.analyze(linkage.COMBINED if args.fold == 'combined' else base.ROOT,
                                    linkage.PROTOCOL if args.fold == 'combined' else base.PROTOCOL)
    elif args.fold == 'combined':
        assert args.stage in ['freeze', 'verify']
        result = linkage.combine() if args.stage == 'freeze' else linkage.verify_combined()
    elif args.stage in ['verify_inputs', 'model', 'verify_model']:
        result = globals()[args.stage]()
    elif args.stage == 'verify_scores':
        result = verify_scores()
    elif args.stage == 'freeze':
        result = study.freeze()
    elif args.stage == 'verify':
        m = json.loads((base.ROOT / 'model_report.json').read_text())
        assert (base.ROOT / 'frozen_numeric_core.tdx').read_text() == base.native_core(m, m['thresholds'][3]['threshold'], base.EXPRESSIONS, base.HEADER)
        result = study.verify()
    else:
        result = base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))
