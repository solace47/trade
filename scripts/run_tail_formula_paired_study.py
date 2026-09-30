"""Shared two-fold input-extension runner; immutable study protocols drive each run."""
import argparse
import importlib
import json
from pathlib import Path
import re
import subprocess

import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_relative as relative
from trade_research import tail_formula_prior_bar_model as model
from trade_research.corporate_cash import save_json, sha
from trade_research.tail_formula_offset_logit48 import verify_scores
from freeze_tail_formula_bipower_gap import checked_selection
import evaluate_tail_formula_pool_scope as common

study = None
EVALUATION_PROTOCOL = None


def configure(stem):
    global study, EVALUATION_PROTOCOL
    assert re.fullmatch(r'tail_formula_[a-z0-9_]+', stem)
    study = importlib.import_module('trade_research.' + stem)
    model.study = study
    model.PROTOCOL = Path('config') / (stem + '_model_protocol.json')
    model.OLD = {f: study.prior.ROOT / 'morning' / f for f in ['2025h1', '2025h2']}
    model.fold_protocol = lambda arm, fold: Path('config') / stem / (arm + '_' + fold + '.json')
    EVALUATION_PROTOCOL = Path('config') / (stem + '_evaluation_protocol.json')


def protocols():
    p = model.checked()
    assert p['candidate_arm'] != 'control' and set(study.ARMS) == {'control', p['candidate_arm']}
    base.FEATURES = base.SOURCE = study.INPUTS
    base.EXPRESSIONS = study.ARMS['control']
    counts = {}
    for fold, spec in p['folds'].items():
        t = base.training(start=spec['training_start'], end=spec['training_end'])
        assert t.next_date.max() < spec['evaluation_start']
        counts[fold] = dict(rows=len(t), days=t.date.nunique(), last_observation=t.next_date.max())
    receipts = {str(study.INPUTS / f): sha(study.INPUTS / f) for f in
                ['feature_report.json', 'feature_verification.json', 'native_input_verification.json',
                 'full_label_report.json', 'full_label_verification.json']}
    records = []
    for arm, expressions in study.ARMS.items():
        for fold, spec in p['folds'].items():
            c = counts[fold]
            q = dict(master_protocol_sha256=sha(model.PROTOCOL), arm=arm, fold=fold, **spec,
                     expected_training_rows=c['rows'], expected_training_days=c['days'],
                     expected_last_observation=c['last_observation'], expected_features=len(expressions),
                     feature_names=list(expressions), parameters=p['parameters'], model_max_depth=3,
                     threshold=.995, target='relative', input_receipts=receipts,
                     no_training_period_selection=True, new_2026_prices_allowed=False, no_exit_rules=True)
            path = model.fold_protocol(arm, fold)
            assert not path.exists()
            path.parent.mkdir(parents=True, exist_ok=True)
            save_json(path, q)
            result = subprocess.run(['.venv/bin/python', 'scripts/find_existing_tail_formula_models.py',
                                     '--protocol', str(path), '--variant', 'relative'],
                                    capture_output=True, text=True, check=True)
            records.append(dict(arm=arm, fold=fold, protocol_sha256=sha(path), lookup=json.loads(result.stdout)))
    assert not any(r['lookup']['matches'] for r in records if r['arm'] == p['candidate_arm']), 'Reuse equivalent candidates before fitting'
    out = dict(passed=True, master_protocol_sha256=sha(model.PROTOCOL), counts=counts, records=records,
               exactly_two_new_models_allowed=True, two_old_controls_to_reuse_after_exact_audit=True,
               all_four_protocols_before_fitting=True, no_new_group_outcomes_read=True,
               new_2026_prices_read=False, no_exit_rules=True)
    save_json(study.ROOT / 'prefit_lookup_verification.json', out)
    return dict(prefit_sha256=sha(study.ROOT / 'prefit_lookup_verification.json'), counts=counts,
                candidate_metadata_matches=0)


def freeze():
    p = model.checked()
    assert not (study.ROOT / 'joint_selection_freeze.json').exists()
    receipts = {str(model.PROTOCOL): sha(model.PROTOCOL)}
    configs, models = {}, []
    for arm in study.ARMS:
        for fold in p['folds']:
            q = model.setup(arm, fold); root = base.ROOT
            m = json.loads((root / 'model_report.json').read_text())
            r = json.loads((root / 'score_report.json').read_text())
            assert m['protocol_sha256'] == r['protocol_sha256'] == sha(base.PROTOCOL)
            assert r['model_report_sha256'] == sha(root / 'model_report.json')
            assert r['scores_sha256'] == sha(root / 'scores.parquet')
            assert m['feature_names'] == list(study.ARMS[arm]) and m['variant'] == 'relative'
            assert m['feature_report_sha256'] == r['feature_report_sha256'] == sha(study.INPUTS / 'feature_report.json')
            assert m['label_report_sha256'] == sha(study.INPUTS / 'full_label_report.json')
            assert m['rows'] == q['expected_training_rows'] and m['days'] == q['expected_training_days']
            assert m['last_observation'] == q['expected_last_observation'] < q['evaluation_start']
            assert all(m['parameters'][k] == v for k, v in p['parameters'].items())
            assert m['thresholds'][3]['training_quantile'] == .995
            for kind in ['model', 'score']:
                v = json.loads((root / (kind + '_verification.json')).read_text())
                assert v['passed'] and v[kind + '_report_sha256'] == sha(root / (kind + '_report.json'))
            if arm == 'control':
                v = json.loads((root / 'control_reuse_verification.json').read_text())
                assert v['passed'] and v['master_protocol_sha256'] == sha(model.PROTOCOL) and m['no_model_fit_performed']
                for file, digest in v['source_hashes'].items():
                    assert sha(Path(file)) == digest
                receipts[str(root / 'control_reuse_verification.json')] = sha(root / 'control_reuse_verification.json')
            else:
                assert not m.get('no_model_fit_performed', False)
            configs[(arm, fold)] = (q, m)
            for file in ['model_report.json', 'model_verification.json', 'score_report.json', 'score_verification.json', 'scores.parquet']:
                receipts[str(root / file)] = sha(root / file)
            receipts[str(base.PROTOCOL)] = sha(base.PROTOCOL)
    f = pd.read_parquet(study.INPUTS / 'features.parquet', columns=study.META)
    meta = study.META[:-1]
    keys = f.loc[f.date.ge('2024-01-01'), meta].reset_index(drop=True)
    assert len(keys) == 1258085 and keys.date.lt('2026-01-01').all()
    selections, equality = [], []
    for arm in study.ARMS:
        flags, queries = [], []
        for fold in p['folds']:
            q, m = configs[(arm, fold)]
            root = study.ROOT / arm / fold; cut = m['thresholds'][3]['threshold']
            d = pd.read_parquet(root / 'scores.parquet', filters=[('date', '>=', q['evaluation_start']), ('date', '<', q['evaluation_end'])])
            d['selected'] = d.formula_input_valid & d.score.gt(cut)
            flags.append(d[['date', 'code', 'selected']])
            pd.testing.assert_frame_equal(d[meta].reset_index(drop=True),
                keys.loc[keys.date.ge(q['evaluation_start']) & keys.date.lt(q['evaluation_end'])].reset_index(drop=True), check_exact=True)
            queries.append(f'''SELECT date,code,coalesce(formula_input_valid AND score>{cut:.17e},false) AS selected
                FROM read_parquet('{root}/scores.parquet') WHERE date>='{q['evaluation_start']}' AND date<'{q['evaluation_end']}' ''')
            header = study.prior.HEADER if arm == 'control' else study.HEADER
            text = base.native_core(m, cut, study.ARMS[arm], header)
            if arm != 'control':
                assert text.count('CORE:SC>') == 1
                text = text.replace('CORE:SC>', 'CORE:' + study.CORE_GATE + ' AND SC>')
            core = root / 'frozen_numeric_core.tdx'; core.write_text(text)
            receipts[str(core)] = sha(core)
            models.append(dict(arm=arm, fold=fold, rows=m['rows'], days=m['days'], last_observation=m['last_observation'],
                selected=int(d.selected.sum()), signal_days=d.loc[d.selected, 'date'].nunique(),
                new_feature_split_nodes={name: sum(t['feature'].count(i) for t in m['trees'])
                    for i, name in enumerate(m['feature_names']) if name not in study.ARMS['control']},
                new_fit=arm != 'control', software_compilation_verified=False, native_source_parity_verified=False))
        out = keys.merge(pd.concat(flags, ignore_index=True), on=['date', 'code'], how='left', validate='one_to_one')
        out['selected'] = out.selected.eq(True)
        c = base.conn(); c.register('keys', keys); c.sql(' UNION ALL '.join(queries)).create_view('flags')
        expected = c.sql('SELECT k.*,coalesce(selected,false) AS selected FROM keys k LEFT JOIN flags USING(date,code) ORDER BY date,code').df(); c.close()
        pd.testing.assert_frame_equal(out, expected, check_exact=True)
        assert out.loc[out.selected, 'date'].ge('2025-01-01').all()
        group = p['control_group'] if arm == 'control' else p['candidate_group']
        root = study.ROOT / group; root.mkdir(parents=True, exist_ok=True)
        assert not (root / 'selection_report.json').exists()
        out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
        r = dict(protocol_sha256=sha(model.PROTOCOL), group=group, selection_sha256=sha(root / 'selection.parquet'),
                 rows=len(out), selected=int(out.selected.sum()), days=out.loc[out.selected, 'date'].nunique(), source_hashes=receipts.copy(),
                 no_new_group_evaluation=True, new_2026_prices_read=False, no_exit_rules=True)
        save_json(root / 'selection_report.json', r)
        save_json(root / 'selection_verification.json', dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'),
                  rows=len(out), all_original_pool_metadata_and_half_flags_sql_verified=True,
                  no_training_period_selection=True, new_2026_prices_read=False, no_exit_rules=True))
        selections.append(dict(group=group, root=str(root), selected=r['selected'], days=r['days'], rows=len(out)))
        for name, path in p['controls'].items():
            equality.append(dict(left=group, right=name, full_frame_equal=out.equals(checked_selection(Path(path)))))
        for file in ['selection.parquet', 'selection_report.json', 'selection_verification.json']:
            receipts[str(root / file)] = sha(root / file)
    assert checked_selection(study.ROOT / p['control_group']).equals(checked_selection(Path(p['controls'][p['control_group']])))
    joint = dict(passed=True, model_protocol_sha256=sha(model.PROTOCOL), source_hashes=receipts, models=models,
                 selections=selections, equality=equality, all_two_new_models_two_reused_controls_and_two_full_lists_jointly_frozen=True,
                 no_parent_refitting_or_prediction=True, no_new_raw_extraction=True, year_2025_is_exploratory=True,
                 no_new_group_evaluation=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(study.ROOT / 'joint_selection_freeze.json', joint)
    return dict(joint_sha256=sha(study.ROOT / 'joint_selection_freeze.json'), models=models, selections=selections, equality=equality)


def checked_joint():
    p = model.checked(); ep = json.loads(EVALUATION_PROTOCOL.read_text())
    assert ep['model_protocol_sha256'] == sha(model.PROTOCOL) and ep['planned_comparisons'] == p['planned_comparisons']
    for file, digest in ep['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    path = study.ROOT / 'joint_selection_freeze.json'; joint = json.loads(path.read_text())
    assert joint['passed'] and joint['model_protocol_sha256'] == sha(model.PROTOCOL)
    committed = subprocess.run(['git','show','HEAD:docs/selection-formula.md'],capture_output=True,text=True,check=True).stdout
    assert sha(path) in committed
    for file, digest in joint['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    for item in joint['selections']:
        checked_selection(Path(item['root']))
    common.PROTOCOL = EVALUATION_PROTOCOL
    return p, joint, ep


def analyze():
    p, joint, _ = checked_joint()
    previous = [Path(r) for r in p['controls'].values()]
    records = []
    for item in joint['selections']:
        root = Path(item['root']); frame = checked_selection(root)
        parent = next((r for r in previous if frame.equals(checked_selection(r))), None)
        existed = (root / 'analysis_report.json').exists()
        if not existed:
            if parent is not None:
                common.reuse_analysis(root, parent)
            else:
                common.evaluation.analyze(root, EVALUATION_PROTOCOL)
                with (root / 'analysis_check_command.log').open('w') as log:
                    subprocess.run(['.venv/bin/python','scripts/verify_tail_formula_before1000.py','analysis','--root',str(root)], stdout=log, check=True)
        common.checked_analysis(root); previous.append(root)
        records.append(dict(group=item['group'], reused_source=str(parent) if parent else None,
            new_economic_aggregation_run=not existed and parent is None, analysis_report_sha256=sha(root / 'analysis_report.json')))
        print(json.dumps(dict(completed=item['group'])),flush=True)
    save_json(study.ROOT / 'analysis_dispatch_verification.json', dict(passed=True,
        joint_sha256=sha(study.ROOT / 'joint_selection_freeze.json'), records=records,
        all_full_frames_and_all_applicable_labels_equal_before_reuse=True, no_duplicate_half_year_aggregation=True,
        no_new_raw_extraction=True, new_2026_prices_read=False, no_exit_rules=True))
    return records


def control_view(name, parent):
    root = study.ROOT / 'comparison_views' / name; root.mkdir(parents=True, exist_ok=True)
    for file in ['selection.parquet','selection_report.json','selection_verification.json']:
        target = root / file
        if not target.exists():
            target.symlink_to((parent / file).resolve())
        assert sha(target) == sha(parent / file)
    if not (root / 'analysis_report.json').exists():
        common.reuse_analysis(root, parent)
    common.checked_analysis(root)
    return root


def finish():
    p, joint, ep = checked_joint()
    roots = {r['group']: Path(r['root']) for r in joint['selections']}
    for name, path in p['controls'].items():
        if name in roots:
            a, ar = common.checked_analysis(roots[name]); b, br = common.checked_analysis(Path(path))
            assert a.equals(b) and ar['summaries'] == br['summaries'] and ar['daily_summary_sha256'] == br['daily_summary_sha256']
        else:
            roots[name] = control_view(name, Path(path))
    prior_pairs, receipts = [], {}
    for name in ep['prior_completed_comparison_roots']:
        root = Path(name); path = root / 'complete_results_manifest.json'; complete = json.loads(path.read_text())
        assert complete['passed'] and ep['prior_comparison_completion_hashes'][str(path)] == sha(path)
        for file, digest in complete['source_hashes'].items():
            assert sha(Path(file)) == digest
        receipts[str(path)] = sha(path)
        for pair in complete['comparisons']:
            assert all(json.loads(Path(f).read_text())['passed'] for f in pair['outputs'])
            prior_pairs.append(dict(**pair, left_root=root / pair['left'],
                right_root=root / pair['right'] if (root / pair['right']).exists() else root / 'comparison_views' / pair['right']))
    cache = {}
    def get(root):
        root = Path(root)
        if root not in cache:
            cache[root] = common.checked_analysis(root)
        return cache[root]
    def equivalent(a, b):
        x, xr = get(a); y, yr = get(b)
        return (x.equals(y) and xr['summaries'] == yr['summaries']
                and xr['daily_summary_sha256'] == yr['daily_summary_sha256']
                and xr['label_report_sha256'] == yr['label_report_sha256'])
    for root in roots.values():
        get(root); common.reference(root)
        for file in ['selection_report.json','selection_verification.json','analysis_report.json','analysis_verification.json',
                     'daily_summary.parquet','reference_coverage_verification.json','full_label_report.json','full_label_verification.json']:
            receipts[str(root / file)] = sha(root / file)
    done, aliases = [], []
    for left, right in p['planned_comparisons']:
        a, b = roots[left], roots[right]; pair = left + '_minus_' + right
        paths = [study.ROOT / ('same_dates_' + pair + '.json'), study.ROOT / ('shared_unknowns_' + pair + '.json')]
        candidates = [dict(**old, left_root=roots[old['left']], right_root=roots[old['right']]) for old in done] + prior_pairs
        alias = next((old for old in candidates if equivalent(a, old['left_root']) and equivalent(b, old['right_root'])), None)
        if alias is not None:
            paths = [Path(f) for f in alias['outputs']]
            aliases.append(dict(left=left,right=right,reused_pair=[alias['left'],alias['right']],
                complete_frames_all_statistics_daily_and_labels_equal=True,
                prior_completed_study=alias in prior_pairs))
        else:
            if not paths[0].exists():
                common.compare(a,b,paths[0],periods=ep['periods'],intersection_only=True)
            if not paths[1].exists():
                spec = dict(left=str(a),right=str(b),output=str(paths[1]),
                    left_analysis_sha256=sha(a / 'analysis_report.json'),right_analysis_sha256=sha(b / 'analysis_report.json'))
                receipt = dict(protocol_sha256=sha(EVALUATION_PROTOCOL),
                    labels_sha256=sha(common.evaluation.source.ROOT / 'full_labels.parquet'),periods=ep['periods'])
                common.shared(spec,receipt)
        for file in paths:
            assert json.loads(file.read_text())['passed']; receipts[str(file)] = sha(file)
        done.append(dict(left=left,right=right,outputs=[str(f) for f in paths],reused=alias is not None))
        print(json.dumps(dict(compared=pair)),flush=True)
    def main(name, period='2025'):
        r = json.loads((roots[name] / 'analysis_report.json').read_text())
        return next(s for s in r['summaries'] if s['arm']=='formula' and s['bps']==15 and not s['sensitive'] and s['period']==period)
    name = p['candidate_group']; new, old, other = main(name), main(p['control_group']), main(p['screen_comparison_group'])
    pair = next(r for r in done if r['right'] == p['control_group'])
    s = next(s for s in json.loads(Path(pair['outputs'][1]).read_text())['summaries']
             if s['period']=='2025' and s['bps']==15 and not s['sensitive'])
    criteria = dict(both_half_reference_positive=all((main(name,h)['mean_reference'] or -1)>0 for h in ['2025H1','2025H2']),
        annual_opportunity_above_both_controls=all(x['rate'] is not None for x in [new,old,other]) and new['rate']>max(old['rate'],other['rate']),
        annual_bad3_not_above_control=new['bad3'] is not None and old['bad3'] is not None and new['bad3']<=old['bad3'],
        shared_unknown_lower_ci_strict_positive=s['lower_ci'] is not None and s['lower_ci'][0]>0)
    gate = dict(passed=True,protocol_sha256=sha(EVALUATION_PROTOCOL),criteria=criteria,
        supports_2024_extension=all(criteria.values()),no_publish_claim=True,new_2026_prices_read=False,no_exit_rules=True)
    gatefile = study.ROOT / (study.STEM.removeprefix('tail_formula_') + '_gate.json'); save_json(gatefile,gate)
    receipts[str(gatefile)] = sha(gatefile)
    receipts[str(study.ROOT / 'analysis_dispatch_verification.json')] = sha(study.ROOT / 'analysis_dispatch_verification.json')
    out = dict(passed=True,protocol_sha256=sha(EVALUATION_PROTOCOL),joint_sha256=sha(study.ROOT / 'joint_selection_freeze.json'),
        source_hashes=receipts,comparisons=done,comparison_reuse=aliases,
        all_two_groups_six_predefined_comparisons_and_reference_gaps_preserved=True,
        no_parent_refit_prediction_or_economic_aggregation=True,no_new_raw_extraction=True,
        no_publish_claim=True,year_2025_is_exploratory=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(study.ROOT / 'complete_results_manifest.json',out)
    return dict(completion_sha256=sha(study.ROOT / 'complete_results_manifest.json'),gate=gate,
                new_numerical_comparisons=sum(not r['reused'] for r in done))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stem', required=True)
    parser.add_argument('stage',choices=['protocols','reuse','model','verify_model','scores','verify_scores','freeze','analyze','finish'])
    parser.add_argument('--fold',choices=['2025h1','2025h2'])
    args = parser.parse_args(); configure(args.stem)
    if args.stage in ['protocols','freeze','analyze','finish']:
        result = globals()[args.stage]()
    elif args.stage == 'reuse':
        assert args.fold; result = model.reuse(args.fold)
    else:
        assert args.fold
        p = model.checked(); model.setup(p['candidate_arm'],args.fold)
        if args.stage == 'model': result = relative.model('relative')
        elif args.stage == 'verify_model': result = relative.verify_model('relative')
        elif args.stage == 'scores': result = base.scores()
        else: result = verify_scores(expected_expressions=base.EXPRESSIONS)
    print(json.dumps(result,ensure_ascii=False,indent=2))
