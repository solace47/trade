"""Choose boosting length on a later half-year, then refit the whole prior year."""
import argparse
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import GradientBoostingRegressor

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as inputs
from . import tail_formula_prior_day as adapter
from . import tail_formula_recent as study
from .corporate_cash import save_json, sha

STEM = 'tail_formula_chrono_rounds'
PROTOCOL = Path('config')/(STEM+'_protocol.json')
ORIGINAL_NATIVE_CORE = base.native_core


def native_core(model, threshold, expressions=inputs.EXPRESSIONS, header=inputs.HEADER):
    text = ORIGINAL_NATIVE_CORE(model, threshold, expressions, header)
    if not model['trees']:
        suffix = 'SC:='+format(model['bias'], '.17g')+'+;\n'
        assert text.count(suffix) == 1
        text = text.replace(suffix, suffix.replace('+;', ';'))
    return text


def setup(fold):
    p = json.loads(PROTOCOL.read_text())
    assert p['maximum_rounds'] == 256 and p['stages'] == list(range(257))
    assert not p['new_2026_prices_allowed'] and sklearn.__version__ == '1.7.2'
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    combined = Path('config')/(STEM+'_combined_protocol.json')
    q = json.loads(combined.read_text())
    assert q['master_protocol_sha256'] == sha(PROTOCOL)
    control = Path(q['control'])
    for stage in ['selection', 'analysis']:
        v = json.loads((control/(stage+'_verification.json')).read_text())
        assert v['passed'] and v[stage+'_report_sha256'] == q['control_'+stage+'_report_sha256'] == sha(control/(stage+'_report.json'))
    adapter.STEM = STEM
    adapter.ROOT, adapter.EXPRESSIONS, adapter.HEADER = inputs.ROOT, inputs.EXPRESSIONS, inputs.HEADER
    adapter.COMBINED_PROTOCOL = combined
    adapter.setup(fold)
    base.SOURCE = labels.ROOT
    base.native_core = native_core
    for name in ['2024', 'recent']:
        q = json.loads((Path('config')/(STEM+'_'+name+'_protocol.json')).read_text())
        assert q['master_protocol_sha256'] == sha(PROTOCOL) and q['parameters'] == p['parameters']
        assert q['feature_report_sha256'] == sha(inputs.ROOT/'feature_report.json')
        assert q['label_report_sha256'] == sha(labels.ROOT/'full_label_report.json')
        assert q['training_start'] < q['calibration_split'] < q['training_end'] == q['evaluation_start']


def scope(role):
    p = json.loads(base.PROTOCOL.read_text())
    if role == 'inner':
        return p['training_start'], p['calibration_split']
    if role == 'validation':
        return p['calibration_split'], p['training_end']
    assert role == 'full'
    return p['training_start'], p['training_end']


def frame(role, independent=False):
    start, end = scope(role)
    names = list(inputs.EXPRESSIONS)
    f = base.feature_inputs()[['date', 'code', 'formula_input_valid', *names]]
    where = f"date>='{start}' AND next_date<'{end}'"
    if not independent:
        l = base.labels(where)
        l = l.loc[l.known15].copy()
        assert l.opportunity15.isin([0, 1]).all()
        l['target'] = l.opportunity15-l.groupby('date').opportunity15.transform('mean')
        t = f.loc[f.formula_input_valid].merge(l[['date', 'code', 'next_date', 'target']],
                                             on=['date', 'code'], validate='one_to_one')
        t = t.sort_values(['date', 'code']).reset_index(drop=True)
        t['w'] = 1/t.groupby('date').code.transform('size')
        return t
    c = base.conn()
    c.register('features', f)
    c.execute(f'''CREATE VIEW targets AS SELECT date,code,next_date,
        opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target
        FROM read_parquet('{labels.ROOT}/full_labels.parquet') WHERE {where} AND known15''')
    fields = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in names)
    d = c.sql('SELECT date,code,next_date,target,1./count(*) OVER(PARTITION BY date) AS w,'+fields+
              ' FROM features JOIN targets USING(date,code) WHERE formula_input_valid ORDER BY date,code').df()
    c.close()
    return d


def frame_receipt(t):
    keys = '\n'.join(t.date+'|'+t.code+'|'+t.next_date).encode()
    return dict(rows=len(t), days=t.date.nunique(), first_signal=t.date.min(), last_signal=t.date.max(),
        last_observation=t.next_date.max(), keys_sha256=hashlib.sha256(keys).hexdigest(),
        encoded_sha256=hashlib.sha256(base.encode(t).tobytes()).hexdigest(),
        target_sha256=hashlib.sha256(t.target.to_numpy(dtype='float64').tobytes()).hexdigest(),
        weight_sha256=hashlib.sha256(t.w.to_numpy(dtype='float64').tobytes()).hexdigest())


def verify_inputs():
    roles, keys = {}, {}
    for role in ['inner', 'validation', 'full']:
        t, d = frame(role), frame(role, True)
        pd.testing.assert_frame_equal(t[['date', 'code', 'next_date']], d[['date', 'code', 'next_date']], check_exact=True)
        np.testing.assert_allclose(t.target, d.target, rtol=0, atol=2e-12)
        np.testing.assert_allclose(t.w, d.w, rtol=0, atol=0)
        np.testing.assert_array_equal(base.encode(t), d[list(inputs.EXPRESSIONS)].to_numpy(dtype='int32'))
        start, end = scope(role)
        assert t.date.min() >= start and t.next_date.max() < end
        roles[role] = frame_receipt(t)
        keys[role] = pd.MultiIndex.from_frame(t[['date', 'code']])
        if role == 'full':
            full = t[['date', 'code', 'next_date']]
    p = json.loads(base.PROTOCOL.read_text())
    assert not len(keys['inner'].intersection(keys['validation']))
    assert not len(keys['inner'].union(keys['validation']).difference(keys['full']))
    purged = keys['full'].difference(keys['inner'].union(keys['validation']))
    expected = full.loc[full.date.lt(p['calibration_split']) & full.next_date.ge(p['calibration_split'])]
    assert set(purged) == set(pd.MultiIndex.from_frame(expected[['date', 'code']]))
    assert roles['full']['rows'] == p['expected_training_rows'] and roles['full']['days'] == 241
    proof = dict(passed=True, protocol_sha256=sha(base.PROTOCOL), master_protocol_sha256=sha(PROTOCOL),
        feature_report_sha256=sha(inputs.ROOT/'feature_report.json'), label_report_sha256=sha(labels.ROOT/'full_label_report.json'),
        roles=roles, purged_inner_boundary_rows=len(purged), purged_inner_boundary_dates=sorted(expected.date.unique()),
        all_time_roles_keys_targets_weights_and_integer_inputs_independently_rebuilt=True,
        full_year_training_intersection_unchanged=True, new_2026_prices_read=False, no_exit_rules=True)
    base.ROOT.mkdir(parents=True, exist_ok=True)
    save_json(base.ROOT/'training_verification.json', proof)
    return proof


def checked_frame(role):
    v = json.loads((base.ROOT/'training_verification.json').read_text())
    assert v['passed'] and v['protocol_sha256'] == sha(base.PROTOCOL) and v['master_protocol_sha256'] == sha(PROTOCOL)
    t = frame(role)
    assert frame_receipt(t) == v['roles'][role]
    return t


def model_count(calibration):
    if calibration:
        return json.loads(PROTOCOL.read_text())['maximum_rounds']
    v = json.loads((base.ROOT/'calibration_verification.json').read_text())
    assert v['passed'] and v['calibration_report_sha256'] == sha(base.ROOT/'calibration_report.json')
    r = json.loads((base.ROOT/'calibration_report.json').read_text())
    return r['chosen_rounds']


def fit_model(calibration=False):
    role = 'inner' if calibration else 'full'
    stem = 'calibration_model' if calibration else 'model'
    out = base.ROOT/(stem+'_report.json')
    assert not out.exists()
    t = checked_frame(role)
    count = model_count(calibration)
    params = {**json.loads(PROTOCOL.read_text())['parameters'], 'n_estimators': count}
    x, y, w = base.encode(t), t.target.to_numpy(), t.w.to_numpy()
    trees = []
    bias = float(np.average(y, weights=w))
    reuse = None
    if not calibration and count == 64:
        fold = '2024' if scope(role)[0] == '2024-01-01' else 'recent'
        root = Path('data/research')/('tail_formula_before1000_model_'+fold)
        old = json.loads((root/'model_report.json').read_text())
        proof = json.loads((root/'model_verification.json').read_text())
        assert proof['passed'] and proof['model_report_sha256'] == sha(root/'model_report.json')
        assert old['feature_report_sha256'] == sha(inputs.ROOT/'feature_report.json')
        assert old['label_report_sha256'] == sha(labels.ROOT/'full_label_report.json')
        assert (old['training_start'], old['training_end']) == scope(role)
        assert old['rows'] == len(t) and old['feature_names'] == list(inputs.EXPRESSIONS)
        assert all(old['parameters'][k] == v for k, v in params.items())
        trees, bias, params = old['trees'], old['bias'], old['parameters']
        reuse = dict(root=str(root), model_report_sha256=sha(root/'model_report.json'),
                     model_verification_sha256=sha(root/'model_verification.json'))
    elif count:
        estimator = GradientBoostingRegressor(**params)
        estimator.fit(x, y, sample_weight=w)
        for fitted in estimator.estimators_.ravel():
            tree = fitted.tree_
            record = {k: getattr(tree, k).tolist() for k in ['feature', 'threshold', 'children_left',
                'children_right', 'n_node_samples', 'weighted_n_node_samples', 'impurity']}
            record['value'] = tree.value.reshape(-1).tolist()
            trees.append(record)
        params = estimator.get_params()
        bias = float(np.ravel(estimator.init_.constant_)[0])
    start, end = scope(role)
    r = dict(protocol_sha256=sha(base.PROTOCOL), master_protocol_sha256=sha(PROTOCOL),
        training_verification_sha256=sha(base.ROOT/'training_verification.json'),
        feature_report_sha256=sha(inputs.ROOT/'feature_report.json'), label_report_sha256=sha(labels.ROOT/'full_label_report.json'),
        rows=len(t), days=t.date.nunique(), last_observation=t.next_date.max(), parameters=params,
        feature_names=list(inputs.EXPRESSIONS), variant='chronological_round_count', role=role,
        learning_rate=.05, bias=bias, trees=trees, chosen_rounds=count, zero_round_constant_model=count == 0,
        training_start=start, training_end=end, new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()),
        new_2025H2_score_groups_read=False, new_2026_prices_read=False, no_exit_rules=True)
    score = base.predict(x, r)
    if count and reuse is None:
        np.testing.assert_allclose(score, estimator.predict(x), rtol=0, atol=2e-12)
    if reuse is not None:
        r['existing_identical_64_round_model_reused'] = reuse
    if not calibration:
        r['calibration_report_sha256'] = sha(base.ROOT/'calibration_report.json')
        r['calibration_verification_sha256'] = sha(base.ROOT/'calibration_verification.json')
    r['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(score, q)))
                        for i, q in enumerate(base.QUANTILES)]
    save_json(out, r)
    return {k: v for k, v in r.items() if k != 'trees'}


def verify_model(calibration=False):
    stem, role = ('calibration_model', 'inner') if calibration else ('model', 'full')
    r = json.loads((base.ROOT/(stem+'_report.json')).read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('master_protocol_sha256', PROTOCOL),
                      ('training_verification_sha256', base.ROOT/'training_verification.json'),
                      ('feature_report_sha256', inputs.ROOT/'feature_report.json'),
                      ('label_report_sha256', labels.ROOT/'full_label_report.json')]:
        assert r[key] == sha(path)
    count = model_count(calibration)
    assert r['role'] == role and r['chosen_rounds'] == len(r['trees']) == r['parameters']['n_estimators'] == count
    assert r['zero_round_constant_model'] == (count == 0) and r['learning_rate'] == .05
    assert all(r['parameters'][k] == v for k, v in json.loads(PROTOCOL.read_text())['parameters'].items())
    if not calibration:
        assert r['calibration_report_sha256'] == sha(base.ROOT/'calibration_report.json')
        assert r['calibration_verification_sha256'] == sha(base.ROOT/'calibration_verification.json')
    d = frame(role, True)
    x, y, w = d[list(inputs.EXPRESSIONS)].to_numpy(dtype='int32'), d.target.to_numpy(), d.w.to_numpy()
    assert r['feature_names'] == list(inputs.EXPRESSIONS) and len(x) == r['rows']
    assert r['days'] == d.date.nunique() and r['last_observation'] == d.next_date.max() < scope(role)[1]
    assert (r['training_start'], r['training_end']) == scope(role)
    np.testing.assert_allclose(r['bias'], np.average(y, weights=w), rtol=0, atol=2e-12)
    score = np.full(len(d), r['bias'])
    nodes = leaves = 0
    for tree in r['trees']:
        assert len(tree['feature']) <= 15
        residual = y-score
        masks, levels, terminal = {0: np.ones(len(d), dtype=bool)}, {0: 0}, np.empty(len(d))
        for i, left in enumerate(tree['children_left']):
            mask = masks[i]
            weights = w[mask]
            assert levels[i] <= 3 and int(mask.sum()) == tree['n_node_samples'][i]
            mean = np.average(residual[mask], weights=weights)
            variance = np.average((residual[mask]-mean)**2, weights=weights)
            np.testing.assert_allclose(weights.sum(), tree['weighted_n_node_samples'][i], rtol=0, atol=1e-8)
            np.testing.assert_allclose(mean, tree['value'][i], rtol=0, atol=2e-10)
            np.testing.assert_allclose(variance, tree['impurity'][i], rtol=0, atol=2e-10)
            if left < 0:
                assert mask.sum() >= 300
                terminal[mask] = mean
                leaves += 1
            else:
                right = tree['children_right'][i]
                lower = x[:, tree['feature'][i]] <= tree['threshold'][i]
                masks[left], masks[right] = mask & lower, mask & ~lower
                levels[left] = levels[right] = levels[i]+1
            nodes += 1
        score += .05*terminal
    np.testing.assert_allclose(score, base.predict(x, r), rtol=0, atol=2e-10)
    for q, cut in zip(base.QUANTILES, r['thresholds']):
        assert cut['training_quantile'] == q
        np.testing.assert_allclose(np.quantile(score, q), cut['threshold'], rtol=0, atol=2e-10)
    proof = dict(passed=True, **{stem+'_report_sha256': sha(base.ROOT/(stem+'_report.json'))},
        rows=len(d), days=d.date.nunique(), node_checks=nodes, leaf_checks=leaves, chosen_rounds=count,
        all_training_keys_targets_weights_integer_inputs_node_statistics_and_quantiles_rebuilt=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT/(stem+'_verification.json'), proof)
    return proof


def daily_errors(dates, target, score):
    unique, group = np.unique(np.asarray(dates), return_inverse=True)
    values = np.bincount(group, weights=(np.asarray(target)-np.asarray(score))**2)/np.bincount(group)
    return pd.DataFrame({'date': unique, 'mse': values})


def choose_rounds(means):
    assert means and all(np.isfinite(x) for x in means)
    rounded = [Decimal(str(float(value))).quantize(Decimal('.0000000001'), rounding=ROUND_HALF_UP)
               for value in means]
    return min(range(len(means)), key=lambda i: (rounded[i], i))


def checked_calibration_model():
    v = json.loads((base.ROOT/'calibration_model_verification.json').read_text())
    assert v['passed'] and v['calibration_model_report_sha256'] == sha(base.ROOT/'calibration_model_report.json')
    r = json.loads((base.ROOT/'calibration_model_report.json').read_text())
    assert len(r['trees']) == 256 and r['training_end'] == scope('validation')[0]
    return r


def calibrate():
    assert not (base.ROOT/'calibration_report.json').exists()
    r, t = checked_calibration_model(), checked_frame('validation')
    x, score = base.encode(t), np.full(len(t), r['bias'])
    frames, means = [], []
    for stage in range(257):
        if stage:
            tree = r['trees'][stage-1]
            score += .05*np.asarray(tree['value'])[base.leaf_indices(x, tree)]
        d = daily_errors(t.date, t.target, score)
        d['stage'] = stage
        frames.append(d)
        means.append(float(d.mse.mean()))
    days = pd.concat(frames, ignore_index=True)
    days.to_parquet(base.ROOT/'calibration_days.parquet', index=False, compression='zstd')
    chosen = choose_rounds(means)
    report = dict(protocol_sha256=sha(base.PROTOCOL), master_protocol_sha256=sha(PROTOCOL),
        training_verification_sha256=sha(base.ROOT/'training_verification.json'),
        calibration_model_report_sha256=sha(base.ROOT/'calibration_model_report.json'),
        calibration_model_verification_sha256=sha(base.ROOT/'calibration_model_verification.json'),
        daily_errors_sha256=sha(base.ROOT/'calibration_days.parquet'), rows=len(t), days=t.date.nunique(),
        first_signal=t.date.min(), last_observation=t.next_date.max(), mean_errors=means,
        chosen_rounds=chosen, zero_round_mse=means[0], chosen_mse=means[chosen], maximum_stage_is_best=chosen == 256,
        validation_is_later_than_inner_fit=True, full_year_refit_not_yet_done=True,
        final_evaluation_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT/'calibration_report.json', report)
    return {k: v for k, v in report.items() if k != 'mean_errors'}


def tree_sql(tree, node=0):
    left = tree['children_left'][node]
    if left < 0:
        return format(.05*tree['value'][node], '.17e')
    name = list(inputs.EXPRESSIONS)[tree['feature'][node]]
    cut = int(np.floor(tree['threshold'][node]))
    return f'(CASE WHEN {name}<={cut} THEN {tree_sql(tree, left)} ELSE {tree_sql(tree, tree["children_right"][node])} END)'


def verify_calibration():
    m = checked_calibration_model()
    r = json.loads((base.ROOT/'calibration_report.json').read_text())
    for field, path in [('protocol_sha256', base.PROTOCOL), ('master_protocol_sha256', PROTOCOL),
        ('training_verification_sha256', base.ROOT/'training_verification.json'),
        ('calibration_model_report_sha256', base.ROOT/'calibration_model_report.json'),
        ('calibration_model_verification_sha256', base.ROOT/'calibration_model_verification.json'),
        ('daily_errors_sha256', base.ROOT/'calibration_days.parquet')]:
        assert r[field] == sha(path)
    d = frame('validation', True)
    c = base.conn()
    c.register('input', d)
    c.execute('CREATE TEMP TABLE replay AS SELECT *, '+format(m['bias'], '.17e')+'::DOUBLE AS prediction FROM input')
    frames = []
    for stage in range(257):
        if stage:
            c.execute('UPDATE replay SET prediction=prediction+'+tree_sql(m['trees'][stage-1]))
        q = c.sql('SELECT date,avg((target-prediction)*(target-prediction)) AS mse FROM replay GROUP BY date ORDER BY date').df()
        q['stage'] = stage
        frames.append(q)
    expected = pd.concat(frames, ignore_index=True)
    actual = pd.read_parquet(base.ROOT/'calibration_days.parquet')
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=0, atol=2e-12)
    c.register('daily', expected)
    means = c.sql('SELECT stage,avg(mse) AS mean_mse FROM daily GROUP BY stage ORDER BY stage').df()
    c.register('means', means)
    chosen = c.sql('SELECT stage FROM means ORDER BY round(mean_mse,10),stage LIMIT 1').fetchone()[0]
    c.close()
    np.testing.assert_allclose(means.mean_mse, r['mean_errors'], rtol=0, atol=2e-12)
    assert chosen == r['chosen_rounds'] == choose_rounds(r['mean_errors'])
    assert len(d) == r['rows'] and d.date.nunique() == r['days']
    assert d.date.min() == r['first_signal'] and d.next_date.max() == r['last_observation'] < scope('validation')[1]
    assert r['maximum_stage_is_best'] == (chosen == 256)
    proof = dict(passed=True, calibration_report_sha256=sha(base.ROOT/'calibration_report.json'),
        rows=len(d), daily_stage_checks=len(expected), stage_checks=257, chosen_rounds=int(chosen),
        all_cumulative_tree_scores_daily_mse_and_rounded_minimum_choice_independently_rebuilt=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT/'calibration_verification.json', proof)
    return proof


def verify_scores():
    m = json.loads((base.ROOT/'model_report.json').read_text())
    if m['trees']:
        from .tail_formula_offset_logit48 import verify_scores as common
        return common()
    v = json.loads((base.ROOT/'model_verification.json').read_text())
    assert v['passed'] and v['model_report_sha256'] == sha(base.ROOT/'model_report.json')
    r = json.loads((base.ROOT/'score_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('model_report_sha256', base.ROOT/'model_report.json'),
        ('feature_report_sha256', inputs.ROOT/'feature_report.json'), ('scores_sha256', base.ROOT/'scores.parquet')]:
        assert r[key] == sha(path)
    f = base.feature_inputs()[['date', 'code', 'half', 'board', 'decision_shares', 'formula_input_valid']]
    c = base.conn()
    c.register('features', f)
    expected = c.sql('SELECT *,CASE WHEN formula_input_valid THEN '+format(m['bias'], '.17e')+
                     '::DOUBLE ELSE NULL END AS score FROM features ORDER BY date,code').df()
    c.close()
    actual = pd.read_parquet(base.ROOT/'scores.parquet')
    pd.testing.assert_frame_equal(actual, expected, check_exact=True, check_dtype=False)
    for cut in m['thresholds']:
        assert cut['threshold'] == m['bias'] and not actual.score.gt(cut['threshold']).any()
    proof = dict(passed=True, score_report_sha256=sha(base.ROOT/'score_report.json'), rows=len(actual),
        valid=int(actual.formula_input_valid.sum()), max_score_difference=0.0,
        all_constant_scores_nulls_keys_and_strict_threshold_flags_rebuilt=True,
        original_input_encodings_already_bound_by_training_and_feature_proofs=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT/'score_verification.json', proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['verify_inputs', 'calibration_model', 'verify_calibration_model',
        'calibrate', 'verify_calibration', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    parser.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    args = parser.parse_args()
    setup(args.fold)
    if args.stage == 'analyze':
        for fold in ['2024', 'recent', '2025']:
            root = Path('data/research')/(STEM+'_'+fold)
            v = json.loads((root/'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == sha(root/'selection_report.json')
        result = evaluation.analyze(linkage.COMBINED if args.fold == 'combined' else base.ROOT,
                                    linkage.PROTOCOL if args.fold == 'combined' else base.PROTOCOL)
    elif args.fold == 'combined':
        assert args.stage in ['freeze', 'verify']
        result = linkage.combine() if args.stage == 'freeze' else linkage.verify_combined()
    elif args.stage in ['calibration_model', 'model']:
        result = fit_model(args.stage == 'calibration_model')
    elif args.stage in ['verify_calibration_model', 'verify_model']:
        result = verify_model(args.stage == 'verify_calibration_model')
    elif args.stage == 'scores':
        result = base.scores()
    elif args.stage == 'freeze':
        result = study.freeze()
    elif args.stage == 'verify':
        m = json.loads((base.ROOT/'model_report.json').read_text())
        assert (base.ROOT/'frozen_numeric_core.tdx').read_text() == native_core(m, m['thresholds'][3]['threshold'])
        result = study.verify()
    else:
        result = globals()[args.stage]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
