"""Add a date-level market forecast to an unchanged stock-relative model."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from . import tail_formula_additive as base
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as original
from . import tail_formula_prior_day as adapter
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .tail_formula_offset_logit48 import verify_scores
from .corporate_cash import save_json, sha

STEM = 'tail_formula_market_sum'
ROOT = Path('data/research') / STEM
MASTER = Path('config') / (STEM + '_protocol.json')
NAMES = ['J02', 'J03', 'J04']
SOURCE = Path('data/research/tail_formula_before1000')


def checked_sources():
    p = json.loads(MASTER.read_text())
    assert p['market_names'] == NAMES and not p['new_2026_prices_allowed']
    for path, digest in p['references'].items():
        assert sha(Path(path)) == digest
    r = json.loads((original.ROOT / 'feature_report.json').read_text())
    v = json.loads((original.ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(original.ROOT / 'feature_report.json')
    assert r['features_sha256'] == sha(original.ROOT / 'features.parquet')
    for fold in ['2024', 'recent']:
        stock_component(fold)
    return p


def stock_component(fold):
    root = Path('data/research') / ('tail_formula_before1000_model_' + fold)
    m = json.loads((root / 'model_report.json').read_text())
    v = json.loads((root / 'model_verification.json').read_text())
    p = json.loads((Path('config') / (STEM + '_' + fold + '_protocol.json')).read_text())
    assert v['passed'] and v['model_report_sha256'] == sha(root / 'model_report.json')
    assert m['feature_report_sha256'] == p['feature_report_sha256'] == sha(original.ROOT / 'feature_report.json')
    assert m['label_report_sha256'] == p['label_report_sha256'] == sha(SOURCE / 'full_label_report.json')
    assert m['feature_names'] == list(original.EXPRESSIONS) and m['variant'] == 'relative'
    assert len(m['trees']) == 64 and m['learning_rate'] == .05
    assert (m['training_start'], m['training_end']) == (p['training_start'], p['training_end'])
    assert (m['rows'], m['days']) == (p['expected_rows'], 241)
    assert m['last_observation'] < p['training_end'] == p['evaluation_start']
    return root, m


def encode_market(frame):
    return np.floor(np.clip(100 * frame[NAMES].to_numpy() + 10000 + .000001, 0, 999999)).astype('int32')


def inputs():
    checked_sources()
    out = ROOT / 'input_reuse_verification.json'
    assert not out.exists()
    f = pd.read_parquet(original.ROOT / 'features.parquet', columns=['date', 'code', 'formula_input_valid', *NAMES])
    assert len(f) == 1258085 and int(f.formula_input_valid.sum()) == 1117397
    visible = f.loc[f.formula_input_valid].copy()
    visible['exchange'] = visible.code.str[:2]
    grouped = visible.groupby(['date', 'exchange'], sort=True)
    assert grouped[NAMES].nunique().eq(1).all().all()
    markets = grouped[NAMES].first().reset_index()
    assert len(markets) == 968 and markets.groupby('date').size().eq(2).all()
    assert set(markets.exchange) == {'sh', 'sz'} and markets.date.lt('2026-01-01').all()
    c = base.conn(); c.register('original_input', f)
    checks = c.sql('SELECT date,substr(code,1,2) AS exchange,' +
        ','.join(f'min({name}) AS {name},count(DISTINCT {name}) AS count_{name}' for name in NAMES) +
        ' FROM original_input WHERE formula_input_valid GROUP BY date,exchange ORDER BY date,exchange').df()
    assert checks[['count_' + n for n in NAMES]].eq(1).all().all()
    pd.testing.assert_frame_equal(markets, checks[['date', 'exchange', *NAMES]], check_exact=True)
    c.register('markets', markets)
    encoded = c.sql('SELECT ' + ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in NAMES) +
                    ' FROM markets ORDER BY date,exchange').df()
    np.testing.assert_array_equal(encoded.to_numpy(), encode_market(markets)); c.close()
    ROOT.mkdir(parents=True, exist_ok=True)
    markets.to_parquet(ROOT / 'market_inputs.parquet', index=False, compression='zstd')
    save_json(out, dict(passed=True, protocol_sha256=sha(MASTER), feature_report_sha256=sha(original.ROOT/'feature_report.json'),
        original_feature_verification_sha256=sha(original.ROOT/'feature_verification.json'),
        market_inputs_sha256=sha(ROOT/'market_inputs.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        market_rows=len(markets), market_days=markets.date.nunique(),
        all_date_exchange_uniqueness_values_and_encodings_independently_rebuilt=True,
        original_48_inputs_and_quality_unchanged=True, native_input_proof_reused=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True))
    return dict(passed=True, input_reuse_sha256=sha(out), market_rows=len(markets), market_days=markets.date.nunique())


def setup(fold):
    checked_sources()
    v = json.loads((ROOT / 'input_reuse_verification.json').read_text())
    assert v['passed'] and v['protocol_sha256'] == sha(MASTER)
    assert v['market_inputs_sha256'] == sha(ROOT / 'market_inputs.parquet')
    adapter.STEM = STEM; adapter.ROOT = original.ROOT
    adapter.EXPRESSIONS = original.EXPRESSIONS; adapter.HEADER = original.HEADER
    adapter.COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')
    adapter.setup(fold); base.SOURCE = SOURCE


def market_training():
    start, end, where = relative.training_scope()
    labels = base.labels(where)
    known = labels.loc[labels.known15]
    assert known.opportunity15.isin([0, 1]).all()
    daily = known.groupby('date', as_index=False).agg(target=('opportunity15', 'mean'),
        known=('code', 'size'), next_date=('next_date', 'max'))
    t = pd.read_parquet(ROOT / 'market_inputs.parquet').merge(daily, on='date', validate='many_to_one')
    t = t.sort_values(['date', 'exchange']).reset_index(drop=True)
    assert len(t) == 482 and t.date.nunique() == 241 and t.groupby('date').size().eq(2).all()
    assert t.date.ge(start).all() and t.next_date.lt(end).all()
    t['weight'] = .5
    return t


def market_model():
    path = base.ROOT; assert not (path / 'market_model_report.json').exists()
    cfg = json.loads(base.PROTOCOL.read_text()); t = market_training()
    x, y, w = encode_market(t), t.target.to_numpy(), t.weight.to_numpy()
    estimator = GradientBoostingRegressor(**cfg['market_parameters']).fit(x, y, sample_weight=w)
    trees = []
    for fitted in estimator.estimators_.ravel():
        tree = fitted.tree_
        row = {key: getattr(tree, key).tolist() for key in ['feature', 'threshold', 'children_left', 'children_right',
            'n_node_samples', 'weighted_n_node_samples', 'impurity']}
        row['value'] = tree.value.reshape(-1).tolist(); trees.append(row)
    path.mkdir(parents=True, exist_ok=True)
    t.to_parquet(path / 'market_training.parquet', index=False, compression='zstd')
    m = dict(protocol_sha256=sha(base.PROTOCOL), inputs_protocol_sha256=sha(MASTER),
        input_reuse_verification_sha256=sha(ROOT/'input_reuse_verification.json'),
        label_report_sha256=sha(SOURCE/'full_label_report.json'), training_sha256=sha(path/'market_training.parquet'),
        parameters=estimator.get_params(), rows=len(t), days=t.date.nunique(),
        training_start=cfg['training_start'], training_end=cfg['training_end'], last_observation=t.next_date.max(),
        feature_names=NAMES, learning_rate=.05, bias=float(estimator.init_.constant_.ravel()[0]), trees=trees,
        target='same_date_all_known_base_opportunity15', stock_rows_not_treated_as_independent_market_observations=True,
        new_2025H2_score_groups_read=False, new_2026_prices_read=False, no_exit_rules=True)
    np.testing.assert_allclose(base.predict(x, m), estimator.predict(x), rtol=0, atol=2e-12)
    all_inputs = pd.read_parquet(ROOT/'market_inputs.parquet')
    all_inputs['market_score'] = base.predict(encode_market(all_inputs), m)
    all_inputs.to_parquet(path/'market_scores.parquet', index=False, compression='zstd')
    m['market_scores_sha256'] = sha(path/'market_scores.parquet')
    save_json(path/'market_model_report.json', m)
    return {k:v for k,v in m.items() if k != 'trees'}


def market_tree_sql(tree, node=0):
    if tree['children_left'][node] < 0:
        return format(.05 * tree['value'][node], '.17e')
    return (f"(CASE WHEN X{tree['feature'][node]+1:02d}<={math.floor(tree['threshold'][node])} THEN " +
        market_tree_sql(tree, tree['children_left'][node]) + ' ELSE ' +
        market_tree_sql(tree, tree['children_right'][node]) + ' END)')


def verify_market():
    path = base.ROOT; cfg = json.loads(base.PROTOCOL.read_text())
    m = json.loads((path/'market_model_report.json').read_text())
    for key, source in [('protocol', base.PROTOCOL), ('inputs_protocol', MASTER),
        ('input_reuse_verification', ROOT/'input_reuse_verification.json'), ('label_report', SOURCE/'full_label_report.json'),
        ('training', path/'market_training.parquet'), ('market_scores', path/'market_scores.parquet')]:
        assert m[key+'_sha256'] == sha(source)
    assert m['feature_names'] == NAMES and len(m['trees']) == 32 and m['learning_rate'] == .05
    assert all(m['parameters'][k] == v for k,v in cfg['market_parameters'].items())
    start, end, where = relative.training_scope()
    c = base.conn()
    c.execute(f"CREATE VIEW labels AS SELECT * FROM read_parquet('{SOURCE}/full_labels.parquet') WHERE {where} AND known15")
    c.execute("CREATE VIEW daily AS SELECT date,sum(opportunity15)::DOUBLE/count(*) AS target,count(*) AS known,max(next_date) AS next_date FROM labels GROUP BY date")
    wanted = c.sql(f"SELECT i.*,d.target,d.known,d.next_date,0.5::DOUBLE AS weight FROM read_parquet('{ROOT}/market_inputs.parquet') i JOIN daily d USING(date) ORDER BY date,exchange").df()
    t = pd.read_parquet(path/'market_training.parquet')
    pd.testing.assert_frame_equal(t, wanted, check_dtype=False, check_exact=True)
    assert len(t) == m['rows'] == 482 and t.date.nunique() == m['days'] == 241
    assert (m['training_start'], m['training_end']) == (start, end) and m['last_observation'] == t.next_date.max() < end
    c.register('training', wanted)
    x = c.sql('SELECT ' + ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in NAMES) +
              ' FROM training ORDER BY date,exchange').df().to_numpy('int32')
    np.testing.assert_array_equal(x, encode_market(t))
    y, w = t.target.to_numpy(), t.weight.to_numpy()
    np.testing.assert_allclose(m['bias'], np.average(y, weights=w), rtol=0, atol=2e-12)
    score = np.full(len(t), m['bias']); checks = 0; minimum_days = 241
    for tree in m['trees']:
        residual = y - score; masks = {0: np.ones(len(t), dtype=bool)}; levels = {0: 0}
        leaves = np.empty(len(t), dtype='int32')
        for node, left in enumerate(tree['children_left']):
            mask = masks[node]; ww = w[mask]
            mean = np.average(residual[mask], weights=ww)
            variance = np.average((residual[mask]-mean)**2, weights=ww)
            assert mask.sum() == tree['n_node_samples'][node] and levels[node] <= 2
            np.testing.assert_allclose(ww.sum(), tree['weighted_n_node_samples'][node], rtol=0, atol=2e-10)
            np.testing.assert_allclose(mean, tree['value'][node], rtol=0, atol=2e-10)
            np.testing.assert_allclose(variance, tree['impurity'][node], rtol=0, atol=2e-10)
            right = tree['children_right'][node]
            if left < 0:
                days = t.loc[mask,'date'].nunique(); minimum_days = min(minimum_days, days)
                assert right < 0 and mask.sum() >= 40 and days >= 20
                leaves[mask] = node
            else:
                assert 0 <= tree['feature'][node] < 3
                cut = x[:,tree['feature'][node]] <= tree['threshold'][node]
                masks[left], masks[right] = mask & cut, mask & ~cut
                levels[left] = levels[right] = levels[node]+1
            checks += 1
        np.testing.assert_array_equal(leaves, base.leaf_indices(x, tree))
        score += .05*np.asarray(tree['value'])[leaves]
    np.testing.assert_allclose(score, base.predict(x, m), rtol=0, atol=2e-12)
    enc = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS X{i:02d}' for i,n in enumerate(NAMES,1))
    c.execute(f"CREATE VIEW encoded AS SELECT *,{enc} FROM read_parquet('{ROOT}/market_inputs.parquet')")
    expression = format(m['bias'],'.17e') + '+' + '+'.join(market_tree_sql(tree) for tree in m['trees'])
    scores = c.sql('SELECT date,exchange,'+','.join(NAMES)+','+expression+' AS market_score FROM encoded ORDER BY date,exchange').df(); c.close()
    actual = pd.read_parquet(path/'market_scores.parquet')
    pd.testing.assert_frame_equal(actual.drop(columns='market_score'), scores.drop(columns='market_score'), check_exact=True)
    np.testing.assert_allclose(actual.market_score, scores.market_score, rtol=0, atol=2e-12)
    proof = dict(passed=True, market_model_report_sha256=sha(path/'market_model_report.json'),
        rows=len(t), days=t.date.nunique(), node_checks=checks, minimum_leaf_distinct_days=minimum_days,
        all_date_baselines_weights_encodings_nodes_and_index_scores_independently_rebuilt=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(path/'market_model_verification.json', proof); return proof


def components(fold):
    root, stock = stock_component(fold)
    market = json.loads((base.ROOT/'market_model_report.json').read_text())
    v = json.loads((base.ROOT/'market_model_verification.json').read_text())
    assert v['passed'] and v['market_model_report_sha256'] == sha(base.ROOT/'market_model_report.json')
    assert market['protocol_sha256'] == sha(base.PROTOCOL)
    return root, stock, market


def merged_trees(stock, market):
    trees = deepcopy(stock['trees'])
    mapping = [stock['feature_names'].index(name) for name in market['feature_names']]
    for tree in market['trees']:
        item = deepcopy(tree)
        item['feature'] = [mapping[index] if index >= 0 else index for index in tree['feature']]
        trees.append(item)
    return trees


def model(fold):
    root, stock, market = components(fold); path = base.ROOT
    assert not (path/'model_report.json').exists()
    cfg = json.loads(base.PROTOCOL.read_text())
    t = base.training(start=cfg['training_start'], end=cfg['training_end'])
    assert len(t) == cfg['expected_rows'] and t.date.nunique() == 241
    m = dict(protocol_sha256=sha(base.PROTOCOL), inputs_protocol_sha256=sha(MASTER),
        feature_report_sha256=sha(original.ROOT/'feature_report.json'), label_report_sha256=sha(SOURCE/'full_label_report.json'),
        input_reuse_verification_sha256=sha(ROOT/'input_reuse_verification.json'),
        stock_model_report_sha256=sha(root/'model_report.json'), stock_model_verification_sha256=sha(root/'model_verification.json'),
        market_model_report_sha256=sha(path/'market_model_report.json'), market_model_verification_sha256=sha(path/'market_model_verification.json'),
        feature_names=list(original.EXPRESSIONS), variant='relative_plus_predicted_market',
        parameters=cfg['parameters'], learning_rate=.05, bias=stock['bias']+market['bias'], trees=merged_trees(stock, market),
        rows=len(t), days=t.date.nunique(), rows_are_threshold_calibration_only=True,
        training_start=cfg['training_start'], training_end=cfg['training_end'],
        last_observation=max(stock['last_observation'], market['last_observation'], t.next_date.max()),
        node_training_metadata_is_component_metadata=True, stock_model_refitted=False,
        construction='unchanged_stock_64_plus_separately_fitted_date_market_32',
        new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()), new_2025H2_score_groups_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    score = base.predict(base.encode(t), m)
    expected = base.predict(base.encode(t), stock)+base.predict(encode_market(t), market)
    np.testing.assert_allclose(score, expected, rtol=0, atol=2e-12)
    m['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(score,q))) for i,q in enumerate(base.QUANTILES)]
    save_json(path/'model_report.json', m)
    return {k:v for k,v in m.items() if k != 'trees'}


def verify_model(fold):
    root, stock, market = components(fold); path = base.ROOT
    m = json.loads((path/'model_report.json').read_text()); cfg = json.loads(base.PROTOCOL.read_text())
    for key, source in [('protocol', base.PROTOCOL), ('inputs_protocol', MASTER),
        ('feature_report', original.ROOT/'feature_report.json'), ('label_report', SOURCE/'full_label_report.json'),
        ('input_reuse_verification', ROOT/'input_reuse_verification.json'),
        ('stock_model_report', root/'model_report.json'), ('stock_model_verification', root/'model_verification.json'),
        ('market_model_report', path/'market_model_report.json'), ('market_model_verification', path/'market_model_verification.json')]:
        assert m[key+'_sha256'] == sha(source)
    assert m['parameters'] == cfg['parameters'] and m['feature_names'] == list(original.EXPRESSIONS)
    assert m['variant'] == 'relative_plus_predicted_market' and m['bias'] == stock['bias']+market['bias']
    assert len(m['trees']) == 96 and m['trees'][:64] == stock['trees']
    for old, new in zip(market['trees'], m['trees'][64:]):
        assert {k:v for k,v in old.items() if k != 'feature'} == {k:v for k,v in new.items() if k != 'feature'}
        for before, after in zip(old['feature'], new['feature']):
            assert (m['feature_names'][after] == market['feature_names'][before]) if before >= 0 else (before == after)
    t = base.training(start=cfg['training_start'], end=cfg['training_end'])
    assert len(t) == m['rows'] == cfg['expected_rows'] and t.date.nunique() == m['days'] == 241
    assert m['training_start'] == cfg['training_start'] and m['training_end'] == cfg['training_end']
    assert m['last_observation'] == max(t.next_date.max(), stock['last_observation'], market['last_observation']) < cfg['training_end']
    x = base.encode(t); score = base.predict(x,m)
    mapping = [m['feature_names'].index(name) for name in NAMES]
    expected = base.predict(x,stock)+base.predict(x[:,mapping],market)
    np.testing.assert_allclose(score,expected,rtol=0,atol=2e-12)
    for index,(q,cut) in enumerate(zip(base.QUANTILES,m['thresholds'])):
        assert cut['id'] == index and cut['training_quantile'] == q
        assert cut['threshold'] == float(np.quantile(score,q))
    old_cut = json.loads((root/'selection_report.json').read_text())['chosen_threshold']['threshold']
    assert base.native_core(stock,old_cut,original.EXPRESSIONS,original.HEADER) == (root/'frozen_numeric_core.tdx').read_text()
    core = base.native_core(m,m['thresholds'][3]['threshold'],original.EXPRESSIONS,original.HEADER)
    assert all(core.count(f'T{i:02d}:=') == 1 for i in range(1,97))
    proof = dict(passed=True,model_report_sha256=sha(path/'model_report.json'),rows=len(t),days=t.date.nunique(),
        node_checks=sum(len(tree['feature']) for tree in m['trees']),
        original_stock_node_proofs_reused=True, all_new_market_nodes_separately_verified=True,
        all_96_component_structures_values_index_mappings_and_calibration_quantiles_verified=True,
        original_stock_native_core_unchanged=True, max_sum_rounding_difference=float(np.max(np.abs(score-expected))),
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(path/'model_verification.json',proof); return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['inputs','market_model','verify_market','model','verify_model','scores','verify_scores','freeze','verify','analyze'])
    p.add_argument('--fold',choices=['2024','recent','combined'],default='2024')
    args=p.parse_args()
    if args.stage == 'inputs':
        result=inputs()
    else:
        setup(args.fold)
        if args.stage == 'analyze':
            assert args.fold == 'combined'
            result=evaluation.analyze(linkage.COMBINED,linkage.PROTOCOL)
        elif args.fold == 'combined':
            assert args.stage in ['freeze','verify']
            result=linkage.combine() if args.stage == 'freeze' else linkage.verify_combined()
        elif args.stage in ['model','verify_model']:
            result=globals()[args.stage](args.fold)
        elif args.stage in ['market_model','verify_market','verify_scores']:
            result=globals()[args.stage]()
        elif args.stage == 'scores':
            result=base.scores()
        else:
            result=getattr(study,args.stage)()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
