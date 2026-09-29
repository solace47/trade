"""Ten chronologically valid components for the frozen 2024 extension."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from . import tail_formula_additive as base
from . import tail_formula_float as original
from . import tail_formula_market_experts as mechanics
from . import tail_formula_quarter_ensemble as quarter
from . import tail_formula_quarter_history2024_inputs as inputs
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = inputs.STEM
ROOT = inputs.ROOT.parent
PROTOCOL = inputs.PROTOCOL
FOLD = None


def setup(fold):
    global FOLD
    FOLD = fold
    master = json.loads(PROTOCOL.read_text())
    base.ROOT = ROOT / fold
    base.PROTOCOL = Path('config') / (STEM + '_' + fold + '_protocol.json')
    relative.PROTOCOL = base.PROTOCOL
    base.FEATURES = inputs.ROOT; base.SOURCE = inputs.ROOT
    base.EXPRESSIONS = original.EXPRESSIONS; base.HEADER = original.HEADER
    p = json.loads(base.PROTOCOL.read_text())
    assert p['master_protocol_sha256'] == sha(PROTOCOL)
    assert all(p[k] == v for k, v in master['folds'][fold].items())
    assert p['parameters'] == master['parameters'] and p['training_quantile'] == master['threshold'] == .995
    for path, digest in p['input_receipts'].items():
        assert sha(Path(path)) == digest
    for file, key in [('feature', 'feature'), ('full_label', 'label')]:
        proof = json.loads((inputs.ROOT / (file + '_verification.json')).read_text())
        assert proof['passed'] and proof[key + '_report_sha256'] == sha(inputs.ROOT / (file + '_report.json'))
    return p


def independent_training():
    p = json.loads(base.PROTOCOL.read_text())
    names = list(original.EXPRESSIONS)
    f = base.feature_inputs()[['date', 'code', 'formula_input_valid', *names]]
    c = base.conn(); c.register('features', f)
    enc = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d = c.sql(f'''WITH l AS(SELECT date,code,next_date,
        opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target
        FROM read_parquet('{inputs.ROOT}/full_labels.parquet') WHERE known15
        AND date>='{p['training_start']}' AND next_date<'{p['training_end']}')
        SELECT date,code,next_date,target,1./count(*) OVER(PARTITION BY date) AS w,{enc},
        year(CAST(next_date AS DATE))::VARCHAR||'Q'||quarter(CAST(next_date AS DATE))::VARCHAR AS observation_quarter
        FROM features JOIN l USING(date,code) WHERE formula_input_valid ORDER BY date,code''').df(); c.close()
    t = relative.training('relative')
    pd.testing.assert_frame_equal(d[['date', 'code', 'next_date']], t[['date', 'code', 'next_date']], check_exact=True)
    np.testing.assert_allclose(d.target, t.target, rtol=0, atol=2e-12)
    np.testing.assert_allclose(d.w, 1 / t.groupby('date').code.transform('size'), rtol=0, atol=0)
    np.testing.assert_array_equal(d[names].to_numpy(dtype='int32'), base.encode(t))
    assert t.date.ge(p['training_start']).all() and t.next_date.lt(p['training_end']).all()
    assert len(t) == p['expected_training_rows'] and t.date.nunique() == p['expected_training_days']
    assert t.next_date.max() == p['expected_last_observation']
    masks = quarter.component_masks(t.next_date, p['omitted_quarters'])
    for name, mask in masks:
        np.testing.assert_array_equal(mask, d.observation_quarter.ne(name))
    return p, d, t, [('full', np.ones(len(t), dtype=bool)), *masks]


def model():
    root = base.ROOT; assert not (root / 'model_report.json').exists()
    p, d, t, masks = independent_training()
    root.mkdir(parents=True, exist_ok=True); (root / 'components').mkdir(exist_ok=True)
    receipt = dict(passed=True, protocol_sha256=sha(base.PROTOCOL),
        feature_report_sha256=sha(inputs.ROOT / 'feature_report.json'),
        label_report_sha256=sha(inputs.ROOT / 'full_label_report.json'),
        rows=len(t), days=t.date.nunique(), first=t.date.min(), last_observation=t.next_date.max(),
        all_training_keys_targets_encodings_global_weights_and_omitted_quarters_sql_verified=True,
        target_day_means_computed_before_input_intersection=True, new_2024_test_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    path = root / 'training_input_verification.json'
    if path.exists():
        assert json.loads(path.read_text()) == receipt
    else:
        save_json(path, receipt)
    x = base.encode(t); y = t.target.to_numpy(); w = 1 / t.groupby('date').code.transform('size').to_numpy()
    components = []; digests = {}
    for name, mask in masks:
        path = root / 'components' / (name + '.json')
        meta = dict(name=name, protocol_sha256=sha(base.PROTOCOL), implementation_sha256=sha(Path(__file__)),
            training_input_verification_sha256=sha(root / 'training_input_verification.json'),
            mask_sha256=hashlib.sha256(np.packbits(mask).tobytes()).hexdigest(), rows=int(mask.sum()),
            days=t.loc[mask, 'date'].nunique(), weight_sum=float(w[mask].sum()), learning_rate=.05,
            parameters=p['parameters'], fitted_parameters=GradientBoostingRegressor(**p['parameters']).get_params())
        if path.exists():
            component = json.loads(path.read_text())
            assert all(component[k] == v for k, v in meta.items())
        else:
            fitted = GradientBoostingRegressor(**p['parameters']).fit(x[mask], y[mask], sample_weight=w[mask])
            trees = []
            for estimator in fitted.estimators_.ravel():
                tree = estimator.tree_
                trees.append({field: getattr(tree, field).reshape(-1).tolist() for field in [
                    'feature', 'threshold', 'children_left', 'children_right', 'n_node_samples',
                    'weighted_n_node_samples', 'value', 'impurity']})
            component = dict(meta, bias=float(fitted.init_.constant_.ravel()[0]), trees=trees,
                             fitted_parameters=fitted.get_params())
            np.testing.assert_allclose(base.predict(x[mask], component), fitted.predict(x[mask]), rtol=0, atol=2e-12)
            save_json(path, component)
        components.append(component); digests[str(path)] = sha(path)
        print(json.dumps(dict(component=name, rows=component['rows'], days=component['days'])), flush=True)
    scores = [base.predict(x, component) for component in components]
    minimum = np.minimum.reduce(scores[1:])
    r = dict(protocol_sha256=sha(base.PROTOCOL), master_protocol_sha256=sha(PROTOCOL),
        implementation_sha256=sha(Path(__file__)), feature_report_sha256=sha(inputs.ROOT / 'feature_report.json'),
        label_report_sha256=sha(inputs.ROOT / 'full_label_report.json'),
        training_input_verification_sha256=sha(root / 'training_input_verification.json'),
        rows=len(t), days=t.date.nunique(), last_observation=t.next_date.max(),
        training_start=p['training_start'], training_end=p['training_end'], feature_names=list(original.EXPRESSIONS),
        parameters=p['parameters'], full=components[0], components=components[1:], component_hashes=digests,
        thresholds={name: dict(training_quantile=.995, threshold=float(np.quantile(values, .995)))
                    for name, values in [('full', scores[0]), ('minimum', minimum)]},
        minimum_training_scores_not_out_of_sample=True, new_2024_test_group_outcomes_read=False,
        rule_selected_after_2025_results=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'model_report.json', r)
    return {k: v for k, v in r.items() if k not in ['full', 'components', 'component_hashes']}


def checked_model():
    p = json.loads(base.PROTOCOL.read_text()); m = json.loads((base.ROOT / 'model_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('master_protocol_sha256', PROTOCOL),
        ('implementation_sha256', Path(__file__)), ('feature_report_sha256', inputs.ROOT / 'feature_report.json'),
        ('label_report_sha256', inputs.ROOT / 'full_label_report.json'),
        ('training_input_verification_sha256', base.ROOT / 'training_input_verification.json')]:
        assert m[key] == sha(path)
    assert m['feature_names'] == list(original.EXPRESSIONS) and m['parameters'] == p['parameters']
    assert m['training_start'] == p['training_start'] and m['training_end'] == p['training_end']
    assert m['last_observation'] < p['evaluation_start']
    assert m['full']['name'] == 'full' and [c['name'] for c in m['components']] == p['omitted_quarters']
    for component in [m['full'], *m['components']]:
        path = base.ROOT / 'components' / (component['name'] + '.json')
        assert m['component_hashes'][str(path)] == sha(path) and json.loads(path.read_text()) == component
        assert len(component['trees']) == 64 and component['learning_rate'] == .05
    return p, m


def verify_model():
    p, m = checked_model(); _, d, t, masks = independent_training()
    assert m['rows'] == len(t) and m['days'] == t.date.nunique() and m['last_observation'] == t.next_date.max()
    x = d[list(original.EXPRESSIONS)].to_numpy(dtype='int32'); y = d.target.to_numpy(); w = d.w.to_numpy()
    checks = 0; scores = []; coverage = []
    for component, (name, mask) in zip([m['full'], *m['components']], masks):
        assert component['name'] == name and component['rows'] == int(mask.sum())
        assert component['days'] == t.loc[mask, 'date'].nunique()
        assert component['mask_sha256'] == hashlib.sha256(np.packbits(mask).tobytes()).hexdigest()
        np.testing.assert_allclose(component['weight_sum'], w[mask].sum(), rtol=0, atol=1e-10)
        rebuilt, count = mechanics.verify_component(component, x[mask], y[mask], w[mask], 64)
        np.testing.assert_allclose(rebuilt, base.predict(x[mask], component), rtol=0, atol=2e-10)
        scores.append(base.predict(x, component)); checks += count
        coverage.append(dict(name=name, rows=component['rows'], days=component['days'], node_checks=count))
    for name, values in [('full', scores[0]), ('minimum', np.minimum.reduce(scores[1:]))]:
        cut = m['thresholds'][name]
        assert cut['training_quantile'] == .995
        np.testing.assert_allclose(np.quantile(values, .995), cut['threshold'], rtol=0, atol=2e-12)
    v = dict(passed=True, model_report_sha256=sha(base.ROOT / 'model_report.json'), rows=len(t), days=t.date.nunique(),
        node_checks=checks, components=coverage, all_five_components_node_statistics_independently_rebuilt=True,
        original_global_date_weights_retained_within_omitted_quarters=True,
        all_training_boundaries_and_labels_next_date_checked=True, full_and_minimum_training_quantiles_rebuilt=True,
        full_native_score_sql_verification_still_required=True, new_2024_test_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT / 'model_verification.json', v); return v


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['model', 'verify_model'])
    p.add_argument('--fold', choices=['h1', 'h2'], required=True)
    a = p.parse_args(); setup(a.fold)
    print(json.dumps(globals()[a.stage](), ensure_ascii=False, indent=2))
