"""Matched-capacity phase predictions, fixed lists, and prospective evaluation."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from trade_research import tail_formula_phase_split as study
from trade_research import tail_formula_additive as base
from trade_research.research_io import check_sources, save_json, sha
from find_existing_tail_formula_models import find
from verify_tail_formula_additive import tree_sql
from trade_research import tail_formula_boundary_evaluation as evaluation
from tail_formula_reports import checked_selection, checked_analysis
import tail_formula_analysis_reuse as reuse
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared_compare
from audit_tail_formula_reference_coverage import audit


EXECUTION = Path('config/tail_formula_phase_split_model_protocol.json')
KEYS = ['date', 'code', 'half', 'board', 'decision_shares']


def candidates(out):
    eligible = out.formula_input_valid & out.score.gt(0)
    picked = out.loc[eligible, ['date', 'code', 'score']].copy()
    picked['integer_score'] = np.floor(picked.score * 1e6 + .5).astype('int64')
    picked['rank'] = picked.groupby('date').integer_score.rank(method='min', ascending=False)
    picked['flag'] = picked['rank'].le(5)
    counts = picked.loc[picked.flag].groupby('date').size()
    picked.loc[picked.date.isin(counts.loc[counts.gt(5)].index), 'flag'] = False
    return picked


def checked():
    p = study.checked()
    e = json.loads(EXECUTION.read_text())
    assert subprocess.check_output(['git', 'show', f'HEAD:{EXECUTION}']) == EXECUTION.read_bytes()
    assert e['input_protocol_sha256'] == sha(study.PROTOCOL)
    assert e['source_gate_sha256'] == sha(study.ROOT / 'source_gate.json')
    check_sources(e['source_hashes'])
    gate = json.loads((study.ROOT / 'source_gate.json').read_text())
    assert gate['passed'] and gate['targets_sha256'] == sha(study.ROOT / 'targets.parquet')
    base.EXPRESSIONS = study.CONTROL
    return p, e, gate


def lookup():
    p, _, gate = checked()
    assert not (study.ROOT / 'prefit_lookup.json').exists()
    records = []
    for fold, scope in p['folds'].items():
        for target, estimators in [('net', 64), ('tail_with_cost', 32), ('after_close', 32)]:
            config = dict(input_protocol_sha256=sha(study.PROTOCOL),
                execution_protocol_sha256=sha(EXECUTION), target=target, fold=fold, **scope,
                parameters=dict(p['parameters'], n_estimators=estimators),
                feature_names=list(study.CONTROL), expected_training_rows=gate['folds'][fold]['rows'],
                expected_training_days=gate['folds'][fold]['days'])
            path = Path('config/tail_formula_phase_split') / (fold + '_' + target + '.json')
            assert not path.exists()
            path.parent.mkdir(exist_ok=True)
            save_json(path, config)
            match = find(path)
            assert not match['matches'], 'Audit the matching old target and full inputs before any fit'
            records.append(dict(fold=fold, target=target, config_sha256=sha(path), lookup=match))
    save_json(study.ROOT / 'prefit_lookup.json', dict(passed=True, records=records,
        all_twelve_lookups_before_fitting=True, new_fits_performed=0,
        maximum_new_fits=12, new_2026_prices_read=False))
    return dict(passed=True, records=len(records), sha256=sha(study.ROOT / 'prefit_lookup.json'))


def training(fold):
    p, _, gate = checked()
    spec = p['folds'][fold]
    file = Path(p['features_root']) / 'features.parquet'
    f = pd.read_parquet(file, columns=[*study.META, *study.CONTROL],
        filters=[('date', '>=', spec['training_start']), ('date', '<', spec['training_end'])])
    l = pd.read_parquet(study.ROOT / 'targets.parquet',
        filters=[('date', '>=', spec['training_start']), ('date', '<', spec['training_end'])])
    d = l.loc[l.target_valid & l.date.ge(spec['training_start']) & l.next_date.lt(spec['training_end'])]
    train = f.loc[f.formula_input_valid].merge(d, on=['date', 'code'], validate='one_to_one')
    train = train.sort_values(['date', 'code']).reset_index(drop=True)
    assert len(train) == gate['folds'][fold]['rows']
    assert train.next_date.max() == gate['folds'][fold]['last_observation'] < spec['evaluation_start']
    con = base.conn()
    con.register('f', f)
    con.register('l', l)
    columns = ','.join(f'floor(least(greatest(100*{name}+10000+.000001,0),999999))::INT AS {name}'
                       for name in study.CONTROL)
    sql = con.sql(f'''SELECT f.date,f.code,next_date,net,tail_with_cost,after_close,
        1./count(*) OVER(PARTITION BY f.date) AS weight,{columns} FROM f JOIN l USING(date,code)
        WHERE formula_input_valid AND target_valid AND next_date<'{spec['training_end']}'
        ORDER BY f.date,f.code''').df()
    con.close()
    pd.testing.assert_frame_equal(train[['date', 'code', 'next_date']], sql[['date', 'code', 'next_date']])
    np.testing.assert_array_equal(base.encode(train), sql[list(study.CONTROL)].to_numpy(dtype='int32'))
    np.testing.assert_allclose(train[study.TARGETS], sql[study.TARGETS], rtol=0, atol=2e-12)
    weights = 1 / train.groupby('date').code.transform('size').to_numpy()
    np.testing.assert_array_equal(weights, sql.weight)
    return train, weights


def verify_nodes(model, x, y, weights):
    score = np.full(len(x), model['bias'])
    np.testing.assert_allclose(score[0], np.average(y, weights=weights), rtol=0, atol=2e-12)
    checks = 0
    for tree in model['trees']:
        assert len(tree['feature']) <= 15
        masks = {0: np.ones(len(x), bool)}
        terminal = np.empty(len(x))
        residual = y - score
        for node, feature in enumerate(tree['feature']):
            mask = masks[node]
            w = weights[mask]
            mean = np.average(residual[mask], weights=w)
            variance = np.average((residual[mask] - mean)**2, weights=w)
            assert int(mask.sum()) == tree['n_node_samples'][node]
            np.testing.assert_allclose(w.sum(), tree['weighted_n_node_samples'][node], rtol=0, atol=1e-8)
            np.testing.assert_allclose(mean, tree['value'][node], rtol=0, atol=2e-10)
            np.testing.assert_allclose(variance, tree['impurity'][node], rtol=0, atol=2e-10)
            left, right = tree['children_left'][node], tree['children_right'][node]
            if left < 0:
                assert mask.sum() >= 300
                terminal[mask] = mean
            else:
                lower = x[:, feature] <= tree['threshold'][node]
                masks[left], masks[right] = mask & lower, mask & ~lower
            checks += 1
        score += .05 * terminal
    np.testing.assert_allclose(score, base.predict(x, model), rtol=0, atol=2e-10)
    return checks


def fit(fold, target):
    p, _, gate = checked()
    config_path = Path('config/tail_formula_phase_split') / (fold + '_' + target + '.json')
    q = json.loads(config_path.read_text())
    pre = json.loads((study.ROOT / 'prefit_lookup.json').read_text())
    record = next(r for r in pre['records'] if r['fold'] == fold and r['target'] == target)
    assert pre['passed'] and record['config_sha256'] == sha(config_path)
    assert not record['lookup']['matches'] and q['execution_protocol_sha256'] == sha(EXECUTION)
    root = study.ROOT / 'models' / fold / target
    assert not root.exists(), 'Do not repeat a model fit'
    train, weights = training(fold)
    x = base.encode(train)
    y = train[target].to_numpy()
    estimator = GradientBoostingRegressor(**q['parameters'])
    estimator.fit(x, y, sample_weight=weights)
    trees = []
    for item in estimator.estimators_.ravel():
        t = item.tree_
        trees.append({**{k: getattr(t, k).tolist() for k in ['feature', 'threshold',
            'children_left', 'children_right', 'n_node_samples', 'weighted_n_node_samples', 'impurity']},
            'value': t.value.reshape(-1).tolist()})
    model = dict(protocol_sha256=sha(config_path), execution_protocol_sha256=sha(EXECUTION),
        target=target, rows=len(train), days=int(train.date.nunique()), last_observation=train.next_date.max(),
        training_start=q['training_start'], training_end=q['training_end'], parameters=estimator.get_params(),
        feature_names=list(study.CONTROL), bias=float(estimator.init_.constant_.ravel()[0]),
        learning_rate=.05, trees=trees, new_2026_prices_read=False, no_exit_rules=True)
    np.testing.assert_allclose(base.predict(x, model), estimator.predict(x), rtol=0, atol=2e-12)
    checks = verify_nodes(model, x, y, weights)
    root.mkdir(parents=True)
    save_json(root / 'model_report.json', model)
    save_json(root / 'model_verification.json', dict(passed=True, model_report_sha256=sha(root / 'model_report.json'),
        rows=len(train), node_checks=checks, all_targets_inputs_weights_and_node_residuals_rebuilt=True,
        target=target, new_2026_prices_read=False))
    return dict(fold=fold, target=target, rows=len(train), trees=len(trees), node_checks=checks)


def freeze():
    p, e, gate = checked()
    joint_file = study.ROOT / 'joint_selection_freeze.json'
    assert not joint_file.exists()
    f = pd.read_parquet(Path(p['features_root']) / 'features.parquet',
        columns=[*study.META, *study.CONTROL], filters=[('date', '>=', '2024-01-01')])
    assert len(f) == 1258085 and f.date.lt('2026-01-01').all()
    flags = {(arm, year): np.zeros(len(f), bool) for arm in p['arms'] for year in ['2024', '2025']}
    receipts = {str(EXECUTION): sha(EXECUTION), str(study.PROTOCOL): sha(study.PROTOCOL)}
    models = []
    con = base.conn()
    for fold, scope in p['folds'].items():
        components = {}
        for target in study.TARGETS:
            root = study.ROOT / 'models' / fold / target
            m = json.loads((root / 'model_report.json').read_text())
            v = json.loads((root / 'model_verification.json').read_text())
            assert v['passed'] and v['model_report_sha256'] == sha(root / 'model_report.json')
            assert m['rows'] == gate['folds'][fold]['rows'] and m['last_observation'] < scope['evaluation_start']
            assert m['feature_names'] == list(study.CONTROL)
            config_path = Path('config/tail_formula_phase_split') / (fold + '_' + target + '.json')
            q = json.loads(config_path.read_text())
            assert m['protocol_sha256'] == sha(config_path)
            assert m['execution_protocol_sha256'] == sha(EXECUTION) and m['target'] == target
            assert m['days'] == gate['folds'][fold]['days']
            assert all(m['parameters'][key] == value for key, value in q['parameters'].items())
            assert len(m['trees']) == (64 if target == 'net' else 32)
            components[target] = m
            for file in ['model_report.json', 'model_verification.json']:
                receipts[str(root / file)] = sha(root / file)
            receipts[str(config_path)] = sha(config_path)
            models.append(dict(fold=fold, target=target, root=str(root), trees=len(m['trees'])))
        mask = f.date.ge(scope['evaluation_start']) & f.date.lt(scope['evaluation_end'])
        d = f.loc[mask].copy().reset_index(drop=True)
        con.register('features', d)
        encoded = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS X{i:02d}'
                           for i, n in enumerate(study.CONTROL, 1))
        con.sql('SELECT date,code,' + encoded + ' FROM features WHERE formula_input_valid').create_view('encoded', replace=True)
        for arm in p['arms']:
            m = dict(components['net'])
            if arm == 'split':
                m['bias'] = sum(components[t]['bias'] for t in ['tail_with_cost', 'after_close'])
                m['trees'] = [t for name in ['tail_with_cost', 'after_close'] for t in components[name]['trees']]
            out = d[study.META].copy()
            out['score'] = np.nan
            out.loc[d.formula_input_valid, 'score'] = base.predict(base.encode(d.loc[d.formula_input_valid]), m)
            equation = format(m['bias'], '.17e') + '+' + '+'.join(tree_sql(t) for t in m['trees'])
            rebuilt = con.sql('SELECT date,code,' + equation + ' AS score FROM encoded').df()
            expected = out[['date', 'code']].merge(rebuilt, on=['date', 'code'], how='left', validate='one_to_one')
            np.testing.assert_allclose(out.score, expected.score, rtol=0, atol=2e-11, equal_nan=True)
            picked = candidates(out)
            con.register('rebuilt', expected)
            sql_flags = con.sql('''WITH eligible AS(SELECT date,code,floor(score*1000000+.5) AS integer_score
                FROM rebuilt WHERE score>0), ranked AS(SELECT *,rank() OVER(PARTITION BY date ORDER BY integer_score DESC) AS r FROM eligible),
                counted AS(SELECT *,count(*) FILTER(WHERE r<=5) OVER(PARTITION BY date) AS n FROM ranked)
                SELECT date,code,integer_score,r<=5 AND n<=5 AS flag FROM counted ORDER BY date,code''').df()
            py_flags = picked.sort_values(['date', 'code']).reset_index(drop=True)
            pd.testing.assert_frame_equal(py_flags[['date', 'code', 'integer_score', 'flag']], sql_flags, check_dtype=False)
            out = out.merge(picked[['date', 'code', 'flag']], on=['date', 'code'], how='left', validate='one_to_one')
            out['selected'] = out.pop('flag').fillna(False).astype(bool)
            assert not (out.selected & ~out.formula_input_valid).any()
            assert out.loc[out.selected].groupby('date').size().le(5).all()
            flags[(arm, fold[:4])][mask] = out.selected.to_numpy()
            root = study.ROOT / 'scores' / arm / fold
            assert not root.exists()
            root.mkdir(parents=True)
            out.to_parquet(root / 'scores.parquet', index=False, compression='zstd')
            numeric = base.native_core(m, 0, study.CONTROL, study.HEADER)
            (root / 'frozen_numeric_core.tdx').write_text(numeric)
            save_json(root / 'score_verification.json', dict(passed=True, scores_sha256=sha(root / 'scores.parquet'),
                full_scores_positive_gate_integer_rank_and_boundary_guard_sql_rebuilt=True,
                numeric_core_sha256=sha(root / 'frozen_numeric_core.tdx'), software_compilation_verified=False))
            for file in ['scores.parquet', 'frozen_numeric_core.tdx', 'score_verification.json']:
                receipts[str(root / file)] = sha(root / file)
    con.close()
    selections = []
    for (arm, year), selected in flags.items():
        root = study.ROOT / (arm + year)
        assert not root.exists()
        root.mkdir()
        out = f[KEYS].copy()
        out['selected'] = selected
        assert out.loc[selected, 'date'].str.startswith(year).all()
        out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
        save_json(root / 'selection_report.json', dict(protocol_sha256=sha(EXECUTION), group=arm + year,
            selection_sha256=sha(root / 'selection.parquet'), rows=len(out), selected=int(selected.sum()),
            days=int(out.loc[selected, 'date'].nunique()), no_future_label_or_fill_filter=True,
            new_2026_prices_read=False, no_exit_rules=True))
        save_json(root / 'selection_verification.json', dict(passed=True,
            selection_report_sha256=sha(root / 'selection_report.json'),
            all_metadata_fold_flags_and_native_integer_math_rebuilt=True))
        selections.append(dict(group=arm + year, root=str(root), selected=int(selected.sum()),
                               days=int(out.loc[selected, 'date'].nunique())))
        for file in ['selection.parquet', 'selection_report.json', 'selection_verification.json']:
            receipts[str(root / file)] = sha(root / file)
    save_json(joint_file, dict(passed=True, input_protocol_sha256=sha(study.PROTOCOL),
        execution_protocol_sha256=sha(EXECUTION), source_hashes=receipts, models=models, selections=selections,
        all_twelve_models_and_four_complete_lists_frozen_together=True, new_2026_prices_read=False,
        new_economic_groups_read=False, native_ranking_parity_verified=False, no_exit_rules=True))
    return dict(joint_sha256=sha(joint_file), selections=selections)


def checked_joint():
    p, e, _ = checked()
    path = study.ROOT / 'joint_selection_freeze.json'
    j = json.loads(path.read_text())
    assert j['passed'] and j['execution_protocol_sha256'] == sha(EXECUTION)
    committed = subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    assert sha(path) in committed, 'Jointly commit all models and lists before evaluation'
    check_sources(j['source_hashes'])
    evaluation.source.ROOT = Path(p['label_roots'][1])
    reuse.PROTOCOL = EXECUTION
    return p, e, j


def analyze():
    p, e, joint = checked_joint()
    previous = [Path(v) for v in e['prior_analysis_roots']]
    records = []
    for item in joint['selections']:
        root = Path(item['root'])
        assert not (root / 'analysis_report.json').exists()
        frame = checked_selection(root)
        selected = int(frame.selected.sum())
        parent = None
        for old in previous:
            sr = json.loads((old / 'selection_report.json').read_text())
            ar = json.loads((old / 'analysis_report.json').read_text())
            if sr.get('selected') != selected or ar.get('reference_label') != '09:59':
                continue
            if frame.equals(checked_selection(old)):
                checked_analysis(old)
                parent = old
                break
        if parent is not None:
            reuse.reuse_analysis(root, parent)
        else:
            evaluation.analyze(root, EXECUTION)
            with (root / 'analysis_check_command.log').open('w') as log:
                subprocess.run(['.venv/bin/python', 'scripts/verify_tail_formula_before1000.py',
                                'analysis', '--root', str(root)], stdout=log, check=True)
        checked_analysis(root)
        previous.append(root)
        records.append(dict(group=item['group'], root=str(root), reused_source=str(parent) if parent else None))
        print(json.dumps(dict(completed=item['group'], reused=parent is not None)), flush=True)
    save_json(study.ROOT / 'analysis_dispatch_verification.json', dict(passed=True,
        joint_sha256=sha(study.ROOT / 'joint_selection_freeze.json'), records=records,
        complete_frames_and_joined_labels_equal_before_reuse=True, new_2026_prices_read=False))
    return records


def finish():
    p, e, joint = checked_joint()
    output = study.ROOT / 'complete_results_manifest.json'
    assert not output.exists()
    roots = {x['group']: Path(x['root']) for x in joint['selections']}
    roots.update({key: Path(value) for key, value in e['controls'].items()})
    cache = {name: checked_analysis(root) for name, root in roots.items()}
    pairs = []
    receipts = dict(joint['source_hashes'])
    for root in roots.values():
        if not (root / 'reference_coverage_verification.json').exists():
            audit(root, [2024, 2025])
        ref = json.loads((root / 'reference_coverage_verification.json').read_text())
        assert ref['passed'] and ref['analysis_report_sha256'] == sha(root / 'analysis_report.json')

    def equal(name, other_frame, other_report):
        frame, report = cache[name]
        return frame.equals(other_frame) and all(report[k] == other_report[k] for k in
            ['summaries', 'daily_summary_sha256', 'label_report_sha256'])

    prior_pairs = []
    for file, digest in e['prior_completion_manifests'].items():
        manifest = Path(file)
        assert sha(manifest) == digest
        old = json.loads(manifest.read_text())
        if not old.get('passed'):
            continue
        for pair in old.get('comparisons', []):
            outputs = pair.get('outputs', [])
            if len(outputs) != 2 or not all(Path(f).exists() for f in outputs):
                continue
            same, shared = [json.loads(Path(f).read_text()) for f in outputs]
            if not same.get('passed') or not shared.get('passed') or not same.get('intersection_only'):
                continue
            endpoints = same.get('inputs', {})
            if not all(side in endpoints for side in ['left', 'right']):
                continue
            left_root, right_root = [Path(endpoints[side]['root']) for side in ['left', 'right']]
            if shared.get('comparison', {}).get('left') != str(left_root) or shared.get('comparison', {}).get('right') != str(right_root):
                continue
            prior_pairs.append(dict(left_root=left_root, right_root=right_root,
                outputs=outputs, manifest=str(manifest), manifest_sha256=digest))
    prior_cache = {}

    def equal_prior(name, root):
        if root not in prior_cache:
            # A count match is only a lookup; full metadata and label/statistic
            # identity are still required before reusing a numerical comparison.
            sr = json.loads((root / 'selection_report.json').read_text())
            if sr.get('selected') != int(cache[name][0].selected.sum()):
                return False
            ar = json.loads((root / 'analysis_report.json').read_text())
            if ar.get('reference_label') != '09:59':
                return False
            prior_cache[root] = checked_analysis(root)
        return equal(name, *prior_cache[root])

    for year in ['2024', '2025']:
        for left, right in [('split' + year, 'direct' + year),
                            ('direct' + year, 'original' + year), ('split' + year, 'original' + year)]:
            existing = next((v for v in pairs if equal(left, *cache[v['left']])
                            and equal(right, *cache[v['right']])), None)
            periods = [year + 'H1', year + 'H2', year]
            cached = next((v for v in prior_pairs if equal_prior(left, v['left_root'])
                and equal_prior(right, v['right_root'])
                and {s['period'] for s in json.loads(Path(v['outputs'][0]).read_text())['summaries']} == set(periods)
                and {s['period'] for s in json.loads(Path(v['outputs'][1]).read_text())['summaries']} == set(periods)), None) if existing is None else None
            if existing:
                files = existing['outputs']
            elif cached:
                files = cached['outputs']
                receipts[cached['manifest']] = cached['manifest_sha256']
            else:
                name = left + '_minus_' + right
                files = [str(study.ROOT / ('same_dates_' + name + '.json')),
                         str(study.ROOT / ('shared_unknowns_' + name + '.json'))]
                assert not any(Path(f).exists() for f in files)
                compare(roots[left], roots[right], Path(files[0]), periods, intersection_only=True)
                spec = dict(left=str(roots[left]), right=str(roots[right]), output=files[1],
                    left_analysis_sha256=sha(roots[left] / 'analysis_report.json'),
                    right_analysis_sha256=sha(roots[right] / 'analysis_report.json'))
                receipt = dict(protocol_sha256=sha(EXECUTION),
                    labels_sha256=sha(evaluation.source.ROOT / 'full_labels.parquet'), periods=periods,
                    signal_range=['2024-01-01', '2024-12-31'] if year == '2024' else None)
                shared_compare(spec, receipt)
            pairs.append(dict(left=left, right=right, outputs=files,
                reused_pair=existing is not None or cached is not None,
                prior_manifest=cached['manifest'] if cached else None))
            for file in files:
                assert json.loads(Path(file).read_text())['passed']
                receipts[file] = sha(Path(file))
    for root in roots.values():
        for file in ['selection_report.json', 'selection_verification.json', 'analysis_report.json',
                     'analysis_verification.json', 'full_label_report.json', 'full_label_verification.json',
                     'selection.parquet', 'daily_summary.parquet', 'reference_coverage_verification.json']:
            receipts[str(root / file)] = sha(root / file)
    def main(arm, period):
        return next(s for s in cache[arm + period[:4]][1]['summaries'] if s['arm'] == 'formula'
            and s['bps'] == 15 and not s['sensitive'] and s['period'] == period)
    criteria = {}
    for arm in p['arms']:
        annual = [main(arm, year) for year in ['2024', '2025']]
        half = [main(arm, year + part) for year in ['2024', '2025'] for part in ['H1', 'H2']]
        criteria[arm] = dict(both_annual_reference_positive=all(s['mean_reference'] is not None and s['mean_reference'] > 0 for s in annual),
            all_four_half_reference_positive=all(s['mean_reference'] is not None and s['mean_reference'] > 0 for s in half),
            all_four_half_at_least_20_days=all(s['days'] >= 20 for s in half),
            both_year_opportunity_above_original_and_bad3_not_above=all(
                main(arm, year)['rate'] is not None and main(arm, year)['bad3'] is not None
                and main(arm, year)['rate'] > main('original', year)['rate']
                and main(arm, year)['bad3'] <= main('original', year)['bad3'] for year in ['2024', '2025']))
    incremental = []
    for year in ['2024', '2025']:
        pair = next(v for v in pairs if v['left'] == 'split' + year and v['right'] == 'direct' + year)
        shared = json.loads(Path(pair['outputs'][1]).read_text())
        incremental.append(next(s for s in shared['summaries'] if s['bps'] == 15 and not s['sensitive'] and s['period'] == year))
    gate_file = study.ROOT / 'selection_gate.json'
    save_json(gate_file, dict(passed=True, criteria=criteria,
        each_arm_supports_further_validation={arm: all(c.values()) for arm, c in criteria.items()},
        split_minus_direct_shared_unknowns=incremental,
        split_increment_lower_ci_both_years_strict_positive=all(s['lower_ci'] is not None and s['lower_ci'][0] > 0 for s in incremental),
        no_publish_claim=True, no_automatic_2026_evaluation=True, no_exit_rules=True))
    for file in [gate_file, study.ROOT / 'prefit_lookup.json', study.ROOT / 'analysis_dispatch_verification.json']:
        receipts[str(file)] = sha(file)
    result = dict(passed=True, joint_sha256=sha(study.ROOT / 'joint_selection_freeze.json'),
        source_hashes=receipts, comparisons=pairs, models_fitted=12,
        complete_2024_and_2025_results=True, new_2026_prices_read=False, no_exit_rules=True,
        criteria=criteria, software_compilation_verified=False, native_ranking_parity_verified=False)
    save_json(output, result)
    return dict(passed=True, complete_sha256=sha(output), comparisons=len(pairs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare', 'lookup', 'fit', 'freeze', 'analyze', 'finish'])
    parser.add_argument('--fold')
    parser.add_argument('--target')
    args = parser.parse_args()
    result = (fit(args.fold, args.target) if args.phase == 'fit' else
              {'prepare': study.prepare, 'lookup': lookup, 'freeze': freeze,
               'analyze': analyze, 'finish': finish}[args.phase]())
    print(json.dumps(result, ensure_ascii=False))
