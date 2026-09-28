"""Learn an executable opportunity event, retaining confirmed non-entry zeros."""
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
from .corporate_cash import save_json, sha

STEM = 'tail_formula_executable_opportunity'
PROTOCOL = Path('config')/(STEM+'_protocol.json')


def setup(fold):
    p = json.loads(PROTOCOL.read_text())
    assert p['parameters']['n_estimators'] == 64 and p['expected_features'] == 48
    assert not p['new_2026_prices_allowed']
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
    for name in ['2024', 'recent']:
        q = json.loads((Path('config')/(STEM+'_'+name+'_protocol.json')).read_text())
        assert q['master_protocol_sha256'] == sha(PROTOCOL) and q['parameters'] == p['parameters']
        assert q['feature_report_sha256'] == sha(inputs.ROOT/'feature_report.json')
        assert q['label_report_sha256'] == sha(labels.ROOT/'full_label_report.json')


def utility(known, no_trade, opportunity):
    known, no_trade = np.asarray(known, dtype=bool), np.asarray(no_trade, dtype=bool)
    opportunity = np.asarray(opportunity, dtype=float)
    assert known.shape == no_trade.shape == opportunity.shape and not (known & no_trade).any()
    assert np.isfinite(opportunity[known]).all() and np.isin(opportunity[known], [0, 1]).all()
    return np.where(no_trade, 0., np.where(known, opportunity, np.nan))


def sources():
    p = json.loads(base.PROTOCOL.read_text())
    v = json.loads((labels.ROOT/'full_label_verification.json').read_text())
    r = json.loads((labels.ROOT/'full_label_report.json').read_text())
    assert v['passed'] and v['label_report_sha256'] == sha(labels.ROOT/'full_label_report.json')
    assert r['labels_sha256'] == sha(labels.ROOT/'full_labels.parquet')
    f = base.feature_inputs()[['date', 'code', 'formula_input_valid', *inputs.EXPRESSIONS]]
    c = base.conn()
    l = c.sql(f'''SELECT date,code,next_date,known15,opportunity15,known_no_trade,
        entry_source_unknown,entry_source_valid,period_entry_bad_day,period_bad_symbol,
        entry_recorded,entry_queue_unknown,entry_fill_status
        FROM read_parquet('{labels.ROOT}/full_labels.parquet')
        WHERE date>='{p['training_start']}' AND next_date<'{p['training_end']}' ORDER BY date,code''').df()
    c.close()
    assert l.date.min() >= p['training_start'] and l.next_date.max() < p['training_end'] <= '2025-07-01'
    return f, l


def training(independent=False):
    f, l = sources()
    if not independent:
        l['utility'] = utility(l.known15, l.known_no_trade, l.opportunity15)
        l['target_known'] = l.known15 | l.known_no_trade
        known = l.loc[l.target_known].copy()
        known['target'] = known.utility-known.groupby('date').utility.transform('mean')
        t = f.loc[f.formula_input_valid].merge(known, on=['date', 'code'], validate='one_to_one')
        t = t.sort_values(['date', 'code']).reset_index(drop=True)
        t['w'] = 1/t.groupby('date').code.transform('size')
        return t
    c = base.conn()
    c.register('features', f)
    c.register('original_labels', l)
    fields = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in inputs.EXPRESSIONS)
    d = c.sql('''WITH available AS(SELECT date,code,next_date,known15,known_no_trade,
        CASE WHEN known_no_trade THEN 0.0 ELSE opportunity15 END AS utility FROM original_labels
        WHERE known15 OR known_no_trade), targets AS(SELECT *,utility-avg(utility) OVER(PARTITION BY date) AS target
        FROM available)
        SELECT date,code,next_date,known15,known_no_trade,utility,target,
        1./count(*) OVER(PARTITION BY date) AS w,'''+fields+'''
        FROM features JOIN targets USING(date,code) WHERE formula_input_valid ORDER BY date,code''').df()
    c.close()
    return d


def verify_inputs():
    f, l = sources()
    source_unknown = ~l.entry_source_valid | l.period_entry_bad_day | l.period_bad_symbol
    np.testing.assert_array_equal(l.entry_source_unknown, source_unknown)
    np.testing.assert_array_equal(l.known_no_trade, ~source_unknown & ~l.entry_recorded)
    assert not (l.known15 & l.known_no_trade).any()
    assert not (l.known_no_trade & l.entry_queue_unknown).any()
    u = utility(l.known15, l.known_no_trade, l.opportunity15)
    unknown = ~l.known15 & ~l.known_no_trade
    assert np.isnan(u[unknown]).all() and (u[l.known_no_trade] == 0).all()
    t, d = training(), training(True)
    columns = ['date', 'code', 'next_date', 'known15', 'known_no_trade']
    pd.testing.assert_frame_equal(t[columns], d[columns], check_exact=True)
    np.testing.assert_allclose(t[['utility', 'target', 'w']], d[['utility', 'target', 'w']], rtol=0, atol=2e-12)
    np.testing.assert_array_equal(base.encode(t), d[list(inputs.EXPRESSIONS)].to_numpy(dtype='int32'))
    old = f.loc[f.formula_input_valid, ['date', 'code']].merge(l.loc[l.known15, ['date', 'code']], on=['date', 'code'], validate='one_to_one')
    pd.testing.assert_frame_equal(t.loc[t.known15, ['date', 'code']].reset_index(drop=True),
                                  old.sort_values(['date', 'code']).reset_index(drop=True), check_exact=True)
    assert t.date.nunique() == json.loads(base.PROTOCOL.read_text())['expected_training_days'] == 241
    proof = dict(passed=True, protocol_sha256=sha(base.PROTOCOL), master_protocol_sha256=sha(PROTOCOL),
        feature_report_sha256=sha(inputs.ROOT/'feature_report.json'), label_report_sha256=sha(labels.ROOT/'full_label_report.json'),
        rows=len(t), days=t.date.nunique(), original_known_rows=len(old), added_no_trade_rows=int(t.known_no_trade.sum()),
        base_known_rows=int(l.known15.sum()), base_no_trade_rows=int(l.known_no_trade.sum()),
        base_unknown_rows=int(unknown.sum()), first_signal=t.date.min(), last_observation=t.next_date.max(),
        added_fill_statuses=t.loc[t.known_no_trade].groupby('entry_fill_status').size().to_dict(),
        source_rejection_states_known_events_unknowns_targets_weights_and_integer_inputs_rebuilt=True,
        original_known_training_rows_retained=True, all_original_evaluation_labels_unchanged=True,
        new_2026_prices_read=False, no_exit_rules=True)
    base.ROOT.mkdir(parents=True, exist_ok=True)
    save_json(base.ROOT/'training_verification.json', proof)
    return proof


def model():
    assert not (base.ROOT/'model_report.json').exists()
    v = json.loads((base.ROOT/'training_verification.json').read_text())
    assert v['passed'] and v['protocol_sha256'] == sha(base.PROTOCOL) and v['master_protocol_sha256'] == sha(PROTOCOL)
    t = training()
    assert len(t) == v['rows'] and int(t.known_no_trade.sum()) == v['added_no_trade_rows']
    x, y, w = base.encode(t), t.target.to_numpy(), t.w.to_numpy()
    params = json.loads(PROTOCOL.read_text())['parameters']
    estimator = GradientBoostingRegressor(**params)
    estimator.fit(x, y, sample_weight=w)
    trees = []
    for fitted in estimator.estimators_.ravel():
        tr = fitted.tree_
        item = {k: getattr(tr, k).tolist() for k in ['feature', 'threshold', 'children_left', 'children_right',
                'n_node_samples', 'weighted_n_node_samples', 'impurity']}
        item['value'] = tr.value.reshape(-1).tolist()
        trees.append(item)
    p = json.loads(base.PROTOCOL.read_text())
    r = dict(protocol_sha256=sha(base.PROTOCOL), master_protocol_sha256=sha(PROTOCOL),
        training_verification_sha256=sha(base.ROOT/'training_verification.json'),
        feature_report_sha256=sha(inputs.ROOT/'feature_report.json'), label_report_sha256=sha(labels.ROOT/'full_label_report.json'),
        feature_names=list(inputs.EXPRESSIONS), rows=len(t), days=t.date.nunique(), last_observation=t.next_date.max(),
        variant='relative_executable_opportunity', parameters=estimator.get_params(), learning_rate=.05,
        bias=float(np.ravel(estimator.init_.constant_)[0]), trees=trees, training_includes_known_no_trade=True,
        training_start=p['training_start'], training_end=p['training_end'],
        new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()), new_2025H2_score_groups_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    score = base.predict(x, r)
    np.testing.assert_allclose(score, estimator.predict(x), rtol=0, atol=2e-12)
    r['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(score, q))) for i, q in enumerate(base.QUANTILES)]
    save_json(base.ROOT/'model_report.json', r)
    return {k: v for k, v in r.items() if k != 'trees'}


def verify_model():
    r = json.loads((base.ROOT/'model_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('master_protocol_sha256', PROTOCOL),
        ('training_verification_sha256', base.ROOT/'training_verification.json'),
        ('feature_report_sha256', inputs.ROOT/'feature_report.json'), ('label_report_sha256', labels.ROOT/'full_label_report.json')]:
        assert r[key] == sha(path)
    assert r['feature_names'] == list(inputs.EXPRESSIONS) and len(r['trees']) == 64 and r['learning_rate'] == .05
    assert all(r['parameters'][k] == v for k, v in json.loads(PROTOCOL.read_text())['parameters'].items())
    p = json.loads(base.PROTOCOL.read_text())
    assert r['training_start'] == p['training_start'] and r['training_end'] == p['training_end']
    d = training(True)
    x, y, w = d[list(inputs.EXPRESSIONS)].to_numpy(dtype='int32'), d.target.to_numpy(), d.w.to_numpy()
    assert len(d) == r['rows'] and d.date.nunique() == r['days']
    assert r['last_observation'] == d.next_date.max() < p['training_end']
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
    for q, cut in zip(base.QUANTILES, r['thresholds']):
        assert q == cut['training_quantile']
        np.testing.assert_allclose(np.quantile(score, q), cut['threshold'], rtol=0, atol=2e-10)
    proof = dict(passed=True, model_report_sha256=sha(base.ROOT/'model_report.json'), rows=len(d), days=d.date.nunique(),
        node_checks=nodes, leaf_checks=leaves, all_extended_targets_weights_integer_inputs_nodes_and_quantiles_rebuilt=True,
        original_evaluation_labels_unchanged=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT/'model_verification.json', proof)
    return proof


def tree_sql(tree, node=0):
    left = tree['children_left'][node]
    if left < 0:
        return format(.05*tree['value'][node], '.17e')
    name = list(inputs.EXPRESSIONS)[tree['feature'][node]]
    cut = int(np.floor(tree['threshold'][node]))
    return f'(CASE WHEN {name}<={cut} THEN {tree_sql(tree, left)} ELSE {tree_sql(tree, tree["children_right"][node])} END)'


def verify_scores():
    m = json.loads((base.ROOT/'model_report.json').read_text())
    v = json.loads((base.ROOT/'model_verification.json').read_text())
    assert v['passed'] and v['model_report_sha256'] == sha(base.ROOT/'model_report.json')
    r = json.loads((base.ROOT/'score_report.json').read_text())
    for key, path in [('protocol_sha256', base.PROTOCOL), ('model_report_sha256', base.ROOT/'model_report.json'),
        ('feature_report_sha256', inputs.ROOT/'feature_report.json'), ('scores_sha256', base.ROOT/'scores.parquet')]:
        assert r[key] == sha(path)
    f = base.feature_inputs()[['date', 'code', 'half', 'board', 'decision_shares', 'formula_input_valid', *inputs.EXPRESSIONS]]
    c = base.conn()
    c.register('features', f)
    fields = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in inputs.EXPRESSIONS)
    c.sql('SELECT date,code,'+fields+' FROM features WHERE formula_input_valid').create_view('encoded')
    expression = format(m['bias'], '.17e')+'+'+'+'.join(tree_sql(t) for t in m['trees'])
    c.sql('SELECT date,code,'+expression+' AS score FROM encoded').create_view('rebuilt')
    expected = c.sql('''SELECT date,code,half,board,decision_shares,formula_input_valid,score
        FROM features LEFT JOIN rebuilt USING(date,code) ORDER BY date,code''').df()
    actual = pd.read_parquet(base.ROOT/'scores.parquet')
    pd.testing.assert_frame_equal(actual.drop(columns='score'), expected.drop(columns='score'), check_exact=True, check_dtype=False)
    np.testing.assert_allclose(actual.score, expected.score, rtol=0, atol=2e-11, equal_nan=True)
    assert actual.score.notna().equals(actual.formula_input_valid)
    p = json.loads(base.PROTOCOL.read_text())
    c.sql(f'''SELECT r.score FROM rebuilt r JOIN read_parquet('{labels.ROOT}/full_labels.parquet') l USING(date,code)
        WHERE l.date>='{p['training_start']}' AND l.next_date<'{p['training_end']}'
        AND (l.known15 OR l.known_no_trade)''').create_view('training_scores')
    scores = c.sql('SELECT score FROM training_scores').df().score
    assert len(scores) == m['rows']
    for cut in m['thresholds']:
        np.testing.assert_allclose(np.quantile(scores, cut['training_quantile']), cut['threshold'], rtol=0, atol=2e-11)
        np.testing.assert_array_equal(actual.score.gt(cut['threshold']), expected.score.gt(cut['threshold']))
    c.close()
    proof = dict(passed=True, score_report_sha256=sha(base.ROOT/'score_report.json'), rows=len(actual),
        valid=int(actual.formula_input_valid.sum()), max_score_difference=float((actual.score-expected.score).abs().max()),
        extended_training_rows=len(scores), all_integer_inputs_tree_scores_extended_training_quantiles_and_flags_rebuilt=True,
        inference_pool_unchanged=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(base.ROOT/'score_verification.json', proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['verify_inputs', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
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
    elif args.stage == 'scores':
        result = base.scores()
    elif args.stage == 'freeze':
        result = study.freeze()
    elif args.stage == 'verify':
        m = json.loads((base.ROOT/'model_report.json').read_text())
        assert (base.ROOT/'frozen_numeric_core.tdx').read_text() == base.native_core(m, m['thresholds'][3]['threshold'], inputs.EXPRESSIONS, inputs.HEADER)
        result = study.verify()
    else:
        result = globals()[args.stage]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
