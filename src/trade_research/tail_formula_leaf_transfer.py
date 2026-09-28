"""Keep frozen half-year tree structures and estimate values on the other half."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as inputs
from . import tail_formula_prior_day as adapter
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_leaf_transfer'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
STRUCTURE = ['feature', 'threshold', 'children_left', 'children_right']
LR = .05


def root(variant, fold):
    return Path('data/research') / f'{STEM}_{variant}_{fold}'


def policy():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    assert p['variants'] == ['average', 'refit'] and not p['new_2026_prices_allowed']
    assert p['training_quantile'] == .995 and p['component_trees'] == 64
    return p


def setup(variant, fold):
    p = policy(); assert variant in p['variants']
    adapter.STEM = STEM + '_' + variant; adapter.ROOT = inputs.ROOT
    adapter.EXPRESSIONS = inputs.EXPRESSIONS; adapter.HEADER = inputs.HEADER
    adapter.COMBINED_PROTOCOL = Path('config') / (adapter.STEM + '_combined_protocol.json')
    adapter.setup(fold); base.SOURCE = labels.ROOT
    for name in ['2024', 'recent']:
        q = json.loads((Path('config') / (adapter.STEM + '_' + name + '_protocol.json')).read_text())
        assert q['master_protocol_sha256'] == sha(PROTOCOL) and q['variant'] == variant
        assert all(q[k] == v for k, v in p['folds'][name].items())
        assert q['parameters'] == p['parameters']


def training():
    p = json.loads(base.PROTOCOL.read_text()); t = relative.training('relative')
    t = t.loc[t.date.ge(p['training_split']) | t.next_date.lt(p['training_split'])].reset_index(drop=True)
    t['training_group'] = t.date.ge(p['training_split']).astype(int)
    assert len(t) == p['expected_rows'] and t.date.nunique() == p['expected_days'] == 240
    assert t.groupby('training_group').size().tolist() == p['group_rows']
    assert t.groupby('training_group').date.nunique().tolist() == p['group_days']
    return t


def independent_training():
    p = json.loads(base.PROTOCOL.read_text()); names = list(inputs.EXPRESSIONS)
    c = base.conn()
    encoded = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d = c.sql(f'''WITH known AS(SELECT date,code,next_date,opportunity15
        FROM read_parquet('{labels.ROOT}/full_labels.parquet')
        WHERE known15 AND date>='{p['training_start']}' AND next_date<'{p['training_end']}'
        AND (date>='{p['training_split']}' OR next_date<'{p['training_split']}')),
        targets AS(SELECT *,opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target FROM known),
        joined AS(SELECT date,code,next_date,target,{encoded},
            (date>='{p['training_split']}')::INT AS training_group
            FROM targets JOIN read_parquet('{inputs.ROOT}/features.parquet') USING(date,code)
            WHERE formula_input_valid)
        SELECT *,1./count(*) OVER(PARTITION BY date) AS w FROM joined ORDER BY date,code''').df()
    c.close(); t = training()
    pd.testing.assert_frame_equal(d[['date', 'code', 'next_date', 'training_group']],
        t[['date', 'code', 'next_date', 'training_group']], check_exact=True, check_dtype=False)
    np.testing.assert_array_equal(d[names].to_numpy(dtype='int32'), base.encode(t))
    np.testing.assert_allclose(d.target, t.target, rtol=0, atol=2e-12)
    np.testing.assert_allclose(d.w, 1/t.groupby('date').code.transform('size'), rtol=0, atol=0)
    return d


def source_components():
    p = json.loads(base.PROTOCOL.read_text()); sources = []
    for i, name in enumerate(p['components']):
        path = Path('data/research') / ('tail_formula_half_consensus_model_' + name)
        m = json.loads((path / 'model_report.json').read_text())
        v = json.loads((path / 'model_verification.json').read_text())
        assert v['passed'] and v['model_report_sha256'] == sha(path / 'model_report.json')
        assert m['feature_names'] == list(inputs.EXPRESSIONS) and m['label_report_sha256'] == sha(labels.ROOT / 'full_label_report.json')
        assert m['rows'] == p['group_rows'][i] and m['days'] == p['group_days'][i]
        start = p['training_start'] if i == 0 else p['training_split']
        end = p['training_split'] if i == 0 else p['training_end']
        assert m['training_start'] == start and m['training_end'] == end and m['last_observation'] < end
        assert len(m['trees']) == 64 and m['learning_rate'] == LR
        sources.append((name, m, sha(path / 'model_report.json')))
    return sources


def terminal_nodes(x, tree):
    nodes = np.zeros(len(x), dtype=np.int32)
    feature = np.asarray(tree['feature']); left = np.asarray(tree['children_left']); right = np.asarray(tree['children_right'])
    threshold = np.asarray(tree['threshold'])
    for _ in range(3):
        ix = np.flatnonzero(left[nodes] >= 0)
        n = nodes[ix]
        nodes[ix] = np.where(x[ix, feature[n]] <= threshold[n], left[n], right[n])
    assert (left[nodes] < 0).all()
    return nodes


def refit_component(source, x, y, w):
    """Only x/y/w from the other half are accepted; source values are unused."""
    bias = float(np.average(y, weights=w)); score = np.full(len(y), bias)
    trees, losses = [], []
    for source_tree in source['trees']:
        tree = {k: copy.deepcopy(source_tree[k]) for k in STRUCTURE}
        terminal = terminal_nodes(x, tree); residual = y-score; n = len(tree['feature'])
        total = np.bincount(terminal, weights=w, minlength=n)
        numer = np.bincount(terminal, weights=w*residual, minlength=n)
        counts = np.bincount(terminal, minlength=n)
        for node in range(n-1, -1, -1):
            left, right = tree['children_left'][node], tree['children_right'][node]
            if left >= 0:
                total[node] = total[left]+total[right]
                numer[node] = numer[left]+numer[right]
                counts[node] = counts[left]+counts[right]
        means = np.divide(numer, total, out=np.zeros(n), where=total > 0)
        value = np.where(np.asarray(tree['children_left']) < 0, means, 0.)
        tree['value'] = value.tolist()
        tree['estimation_rows'] = counts.tolist()
        tree['estimation_weights'] = total.tolist()
        tree['estimation_residual_mean'] = means.tolist()
        before = float(np.average(residual**2, weights=w))
        score += LR*value[terminal]
        after = float(np.average((y-score)**2, weights=w)); assert after <= before+2e-12
        trees.append(tree); losses.append(dict(before=before, after=after))
    return dict(bias=bias, learning_rate=LR, trees=trees, estimation_losses=losses)


def flatten(components):
    assert len(components) == 2
    return sum(c['bias'] for c in components)/2, [
        {**{k: copy.deepcopy(t[k]) for k in STRUCTURE}, 'value': [v/2 for v in t['value']]}
        for c in components for t in c['trees']]


def verify_inputs():
    d = independent_training(); sources = source_components(); p = json.loads(base.PROTOCOL.read_text())
    fold = '2024' if p['training_start'] == '2024-01-01' else 'recent'
    control = Path('data/research') / ('tail_formula_half_consistent_ordinary_' + fold)
    sr = json.loads((control / 'selection_report.json').read_text())
    mv = json.loads((control / 'model_verification.json').read_text())
    cm = json.loads((control / 'model_report.json').read_text())
    assert mv['passed'] and mv['model_report_sha256'] == sr['model_report_sha256'] == sha(control / 'model_report.json')
    assert cm['feature_names'] == list(inputs.EXPRESSIONS)
    assert cm['rows'] == len(d) and cm['days'] == d.date.nunique()
    assert all(cm[k] == p[k] for k in ['training_start', 'training_split', 'training_end', 'group_rows', 'group_days'])
    coverage = []
    for i, (name, _, digest) in enumerate(sources):
        own = d.training_group.eq(i); other = ~own
        assert set(d.loc[own, 'date']).isdisjoint(d.loc[other, 'date'])
        coverage.append(dict(structure=name, structure_model_sha256=digest, structure_rows=int(own.sum()),
            estimation_rows=int(other.sum()), structure_days=d.loc[own, 'date'].nunique(),
            estimation_days=d.loc[other, 'date'].nunique()))
    proof = dict(passed=True, protocol_sha256=sha(base.PROTOCOL), master_protocol_sha256=sha(PROTOCOL),
        rows=len(d), days=d.date.nunique(), components=coverage,
        keys_targets_half_groups_integer_inputs_and_weights_sql_rebuilt=True,
        per_component_structure_and_estimation_dates_disjoint=True,
        same_union_as_ordinary64_control=True, ordinary64_model_report_sha256=sha(control / 'model_report.json'),
        new_2026_prices_read=False, no_exit_rules=True)
    base.ROOT.mkdir(parents=True, exist_ok=True)
    path = base.ROOT / 'training_input_verification.json'; assert not path.exists(); save_json(path, proof)
    return proof


def model(variant):
    assert not (base.ROOT / 'model_report.json').exists()
    proof = json.loads((base.ROOT / 'training_input_verification.json').read_text())
    assert proof['passed'] and proof['protocol_sha256'] == sha(base.PROTOCOL)
    p = json.loads(base.PROTOCOL.read_text()); t = training(); x = base.encode(t)
    y = t.target.to_numpy(); w = 1/t.groupby('date').code.transform('size').to_numpy(); components = []
    for i, (name, source, digest) in enumerate(source_components()):
        if variant == 'refit':
            mask = t.training_group.ne(i).to_numpy()
            c = refit_component(source, x[mask], y[mask], w[mask])
            c.update(estimation_rows=int(mask.sum()), estimation_days=t.loc[mask, 'date'].nunique())
        else:
            c = dict(bias=source['bias'], learning_rate=LR, trees=copy.deepcopy(source['trees']))
        c.update(structure=name, structure_model_sha256=digest); components.append(c)
    bias, trees = flatten(components)
    r = dict(protocol_sha256=sha(base.PROTOCOL), master_protocol_sha256=sha(PROTOCOL),
        implementation_sha256=sha(Path(__file__)), feature_report_sha256=sha(inputs.ROOT / 'feature_report.json'),
        label_report_sha256=sha(labels.ROOT / 'full_label_report.json'),
        training_input_verification_sha256=sha(base.ROOT / 'training_input_verification.json'),
        variant='relative', method=variant, parameters=p['parameters'], feature_names=list(inputs.EXPRESSIONS),
        rows=len(t), days=t.date.nunique(), last_observation=t.next_date.max(),
        training_start=p['training_start'], training_split=p['training_split'], training_end=p['training_end'],
        learning_rate=LR, bias=bias, trees=trees, components=components,
        new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()), new_2025H2_score_groups_read=False,
        aggregate_training_scores_are_not_out_of_sample=True, new_2026_prices_read=False, no_exit_rules=True)
    score = base.predict(x, r)
    np.testing.assert_allclose(score, np.mean([base.predict(x, c) for c in components], axis=0), rtol=0, atol=2e-12)
    r['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(score, q))) for i, q in enumerate(base.QUANTILES)]
    save_json(base.ROOT / 'model_report.json', r)
    return {k: v for k, v in r.items() if k not in ['trees', 'components']}


def verify_model(variant):
    p = json.loads(base.PROTOCOL.read_text()); r = json.loads((base.ROOT / 'model_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('master_protocol_sha256', PROTOCOL),
        ('implementation_sha256', Path(__file__)), ('feature_report_sha256', inputs.ROOT / 'feature_report.json'),
        ('label_report_sha256', labels.ROOT / 'full_label_report.json'),
        ('training_input_verification_sha256', base.ROOT / 'training_input_verification.json')]:
        assert r[key] == sha(path)
    assert r['method'] == variant and r['parameters'] == p['parameters'] and r['feature_names'] == list(inputs.EXPRESSIONS)
    assert r['learning_rate'] == LR and all(r[k] == p[k] for k in ['training_start', 'training_split', 'training_end'])
    d = independent_training(); x = d[list(inputs.EXPRESSIONS)].to_numpy(dtype='int32')
    assert len(d) == r['rows'] and d.date.nunique() == r['days'] and d.next_date.max() == r['last_observation']
    checks = zero_leaves = 0; component_scores = []
    for i, ((name, source, digest), c) in enumerate(zip(source_components(), r['components'])):
        assert c['structure'] == name and c['structure_model_sha256'] == digest and len(c['trees']) == 64
        assert c['learning_rate'] == LR
        if variant == 'average':
            assert c['bias'] == source['bias'] and c['trees'] == source['trees']
        else:
            other = d.training_group.ne(i).to_numpy(); z = x[other]
            y = d.loc[other, 'target'].to_numpy(); w = d.loc[other, 'w'].to_numpy()
            assert c['estimation_rows'] == len(y) and c['estimation_days'] == d.loc[other, 'date'].nunique()
            np.testing.assert_allclose(c['bias'], np.dot(w, y)/w.sum(), rtol=0, atol=2e-12)
            score = np.full(len(y), c['bias'])
            for index, (tree, original) in enumerate(zip(c['trees'], source['trees'])):
                assert all(tree[k] == original[k] for k in STRUCTURE)
                residual = y-score; masks = {0: np.ones(len(y), dtype=bool)}; step = np.empty(len(y))
                for node, left in enumerate(tree['children_left']):
                    mask = masks[node]
                    ix = np.flatnonzero(mask); total = float(w[ix].sum())
                    value = float(np.dot(w[ix], residual[ix])/total) if len(ix) else 0.
                    assert tree['estimation_rows'][node] == len(ix)
                    np.testing.assert_allclose(tree['estimation_weights'][node], total, rtol=0, atol=2e-10)
                    np.testing.assert_allclose(tree['estimation_residual_mean'][node], value, rtol=0, atol=2e-10)
                    if left < 0:
                        np.testing.assert_allclose(tree['value'][node], value, rtol=0, atol=2e-10)
                        step[mask] = value; zero_leaves += int(not len(ix))
                    else:
                        assert tree['value'][node] == 0
                        lower = z[:, tree['feature'][node]] <= tree['threshold'][node]
                        masks[left] = mask & lower; masks[tree['children_right'][node]] = mask & ~lower
                    checks += 1
                before = np.dot(w, residual**2)/w.sum(); score += LR*step
                after = np.dot(w, (y-score)**2)/w.sum()
                np.testing.assert_allclose([before, after], [c['estimation_losses'][index]['before'], c['estimation_losses'][index]['after']], rtol=0, atol=2e-12)
                assert after <= before+2e-12
            np.testing.assert_allclose(score, base.predict(z, c), rtol=0, atol=2e-12)
        component_scores.append(base.predict(x, c))
    bias, trees = flatten(r['components']); assert bias == r['bias'] and trees == r['trees'] and len(trees) == 128
    score = np.mean(component_scores, axis=0)
    np.testing.assert_allclose(score, base.predict(x, r), rtol=0, atol=2e-12)
    for q, cut in zip(base.QUANTILES, r['thresholds']):
        assert q == cut['training_quantile']
        np.testing.assert_allclose(np.quantile(score, q), cut['threshold'], rtol=0, atol=2e-12)
    proof = dict(passed=True, model_report_sha256=sha(base.ROOT / 'model_report.json'), rows=len(d), days=r['days'],
        node_checks=checks, zero_support_leaves=zero_leaves, all_source_structures_unchanged=True,
        original_component_node_proofs_reused=True, all_opposite_half_values_independently_rebuilt=variant == 'refit',
        all_opposite_half_losses_nonincreasing=variant == 'refit', full_average_and_training_quantiles_rebuilt=True,
        aggregate_training_scores_are_not_out_of_sample=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT / 'model_verification.json', proof); return proof


def selection_gate():
    for v in ['average', 'refit']:
        for f in ['2024', 'recent', '2025']:
            path = root(v, f); proof = json.loads((path / 'selection_verification.json').read_text())
            assert proof['passed'] and proof['selection_report_sha256'] == sha(path / 'selection_report.json')
    receipt = json.loads((ROOT / 'joint_selection_freeze.json').read_text())
    assert receipt['passed'] and receipt['protocol_sha256'] == sha(PROTOCOL)


if __name__ == '__main__':
    from .tail_formula_offset_logit48 import verify_scores
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['verify_inputs', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    parser.add_argument('--variant', choices=['average', 'refit'], required=True)
    parser.add_argument('--fold', choices=['2024', 'recent', 'combined'], required=True)
    args = parser.parse_args(); setup(args.variant, args.fold)
    if args.stage == 'analyze':
        selection_gate()
        result = evaluation.analyze(linkage.COMBINED if args.fold == 'combined' else base.ROOT,
            linkage.PROTOCOL if args.fold == 'combined' else base.PROTOCOL)
    elif args.fold == 'combined':
        assert args.stage in ['freeze', 'verify']
        result = linkage.combine() if args.stage == 'freeze' else linkage.verify_combined()
    elif args.stage in ['model', 'verify_model']:
        result = globals()[args.stage](args.variant)
    elif args.stage == 'verify_inputs':
        result = verify_inputs()
    elif args.stage == 'verify_scores':
        result = verify_scores()
    elif args.stage in ['freeze', 'verify']:
        result = getattr(study, args.stage)()
    else:
        result = base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))
