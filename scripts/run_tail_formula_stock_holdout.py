"""Four shared stock-group models; complete cross-group and own-group controls."""
import argparse
import json
from pathlib import Path
import re

import pandas as pd

from trade_research import tail_formula_stock_holdout as study
from trade_research import tail_formula_chrono_logit48 as binary
from trade_research.corporate_cash import save_json, sha
from find_existing_tail_formula_models import find
from run_tail_formula_absolute_morning import verify_model
import run_tail_formula_paired_study as shared


def protocols():
    p = shared.model.checked()
    metadata = json.loads((study.ROOT / 'metadata_lookup_verification.json').read_text())
    receipts = {str(study.INPUTS / file): sha(study.INPUTS / file) for file in
        ['feature_report.json', 'feature_verification.json', 'native_input_verification.json',
         'full_label_report.json', 'full_label_verification.json']}
    records = []
    for fold, dates in p['folds'].items():
        for group in [0, 1]:
            counts = next(r for r in metadata['records'] if r['fold'] == fold and r['group'] == group)
            arm = 'group' + str(group)
            q = dict(master_protocol_sha256=sha(shared.model.PROTOCOL), arm=arm, fold=fold, **dates,
                training_group=group, calibration_group=1-group,
                expected_training_rows=counts['training_rows'], expected_training_days=counts['training_days'],
                expected_last_observation=counts['last_observation'], expected_features=50,
                feature_names=list(study.ARMS['cross']), parameters=p['parameters'], model_max_depth=3,
                threshold=.995, variant='absolute_logistic', target='own stock group known15 opportunity15, equal dates',
                projected_label_directory=str(study.model_root(fold, group) / 'labels'),
                input_receipts=receipts, no_training_period_selection=True,
                new_2026_prices_allowed=False, no_exit_rules=True)
            path = shared.model.fold_protocol(arm, fold)
            assert not path.exists()
            path.parent.mkdir(parents=True, exist_ok=True); save_json(path, q)
            lookup = find(path, variant='absolute_logistic')
            assert not lookup['matches'], 'Audit and reuse matching stock-group fits before fitting'
            records.append(dict(arm=arm, fold=fold, protocol_sha256=sha(path), lookup=lookup))
    report = dict(passed=True, master_protocol_sha256=sha(shared.model.PROTOCOL), records=records,
        exactly_four_shared_models_allowed=True, same_model_control_never_refit=True,
        metadata_lookup_sha256=sha(study.ROOT / 'metadata_lookup_verification.json'),
        no_price_score_or_outcome_values_read=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(study.ROOT / 'prefit_lookup_verification.json', report)
    return dict(prefit_sha256=sha(study.ROOT / 'prefit_lookup_verification.json'), candidate_metadata_matches=0)


def setup(fold, group):
    p = shared.model.checked(); arm = 'group' + str(group)
    shared.base.ROOT = study.model_root(fold, group)
    shared.base.FEATURES = study.INPUTS
    shared.base.SOURCE = shared.base.ROOT / 'labels'
    shared.base.EXPRESSIONS = study.ARMS['cross']; shared.base.HEADER = study.HEADER
    shared.base.PROTOCOL = shared.model.fold_protocol(arm, fold)
    q = json.loads(shared.base.PROTOCOL.read_text())
    pre = json.loads((study.ROOT / 'prefit_lookup_verification.json').read_text())
    assert pre['passed'] and pre['master_protocol_sha256'] == q['master_protocol_sha256'] == sha(shared.model.PROTOCOL)
    assert q['training_group'] == group and q['calibration_group'] == 1-group
    assert q['parameters'] == p['parameters'] and q['feature_names'] == list(shared.base.EXPRESSIONS)
    assert q['projected_label_directory'] == str(shared.base.SOURCE)
    assert all(q[k] == value for k, value in p['folds'][fold].items())
    assert any(r['arm'] == arm and r['fold'] == fold and r['protocol_sha256'] == sha(shared.base.PROTOCOL) for r in pre['records'])
    for file, digest in q['input_receipts'].items():
        assert sha(Path(file)) == digest, file
    return q


def fit():
    q = json.loads(shared.base.PROTOCOL.read_text()); root = shared.base.ROOT
    v = json.loads((shared.base.SOURCE / 'full_label_verification.json').read_text())
    r = json.loads((shared.base.SOURCE / 'full_label_report.json').read_text())
    assert v['passed'] and v['label_report_sha256'] == sha(shared.base.SOURCE / 'full_label_report.json')
    assert r['labels_sha256'] == sha(shared.base.SOURCE / 'full_labels.parquet')
    assert r['original_labels_sha256'] == sha(study.INPUTS / 'full_labels.parquet')
    assert r['training_group'] == q['training_group'] and r['fold'] == q['fold']
    binary.model()
    m = json.loads((root / 'model_report.json').read_text())
    m.update(training_group=q['training_group'], original_label_report_sha256=sha(study.INPUTS / 'full_label_report.json'),
        stock_group_label_projection_verified=True, year_2025_is_exploratory=True,
        uses_exposed_2025_training_labels=q['training_end'] > '2025-01-01')
    save_json(root / 'model_report.json', m)
    return {k: value for k, value in m.items() if k != 'trees'}


def checked_scores(fold, group):
    q = setup(fold, group); root = shared.base.ROOT
    m = json.loads((root / 'model_report.json').read_text())
    s = json.loads((root / 'score_report.json').read_text())
    assert m['protocol_sha256'] == s['protocol_sha256'] == sha(shared.base.PROTOCOL)
    assert m['variant'] == 'absolute_logistic' and m['training_group'] == group
    assert m['feature_names'] == q['feature_names'] and len(m['feature_names']) == 50
    assert m['feature_report_sha256'] == s['feature_report_sha256'] == sha(study.INPUTS / 'feature_report.json')
    assert m['label_report_sha256'] == sha(shared.base.SOURCE / 'full_label_report.json')
    assert m['rows'] == q['expected_training_rows'] and m['days'] == q['expected_training_days']
    assert m['last_observation'] == q['expected_last_observation'] < q['evaluation_start']
    assert all(m['parameters'][k] == value for k, value in q['parameters'].items())
    assert s['model_report_sha256'] == sha(root / 'model_report.json') and s['scores_sha256'] == sha(root / 'scores.parquet')
    for kind in ['model', 'score', 'calibration']:
        v = json.loads((root / (kind + '_verification.json')).read_text())
        assert v['passed'] and v[kind + '_report_sha256'] == sha(root / (kind + '_report.json'))
    r = json.loads((root / 'calibration_report.json').read_text())
    assert r['model_report_sha256'] == sha(root / 'model_report.json') and r['scores_sha256'] == sha(root / 'scores.parquet')
    assert r['quantile'] == .995 and r['training_group'] == group and r['calibration_group'] == 1-group
    return q, m, r


def native_core(models, cuts, cross):
    cores = [shared.base.native_core(m, cut, study.ARMS['cross'], study.HEADER) for m, cut in zip(models, cuts)]
    header = cores[0].split('T01:=', 1)[0]
    assert cores[1].split('T01:=', 1)[0] == header
    bodies = []
    for group, core in enumerate(cores):
        body = 'T01:=' + core.split('T01:=', 1)[1].split('CORE:', 1)[0]
        body = re.sub(r'\b(T\d{2}|SC)\b', lambda match: 'H' + str(group) + match[0], body)
        bodies.append(body)
    zero, one = (1, 0) if cross else (0, 1)
    result = header + ''.join(bodies) + 'HG:=STR2CON(SUBSTR(CODE,6,1))>=5;\n'
    return result + f'CORE:{study.CORE_GATE} AND IF(HG,H{one}SC>{cuts[one]:.17g},H{zero}SC>{cuts[zero]:.17g});\n'


def freeze():
    p = shared.model.checked()
    assert not (study.ROOT / 'joint_selection_freeze.json').exists()
    receipts = {str(shared.model.PROTOCOL): sha(shared.model.PROTOCOL),
        str(study.ROOT / 'prefit_lookup_verification.json'): sha(study.ROOT / 'prefit_lookup_verification.json')}
    configurations, models = {}, []
    for fold in p['folds']:
        configurations[fold] = []
        for group in [0, 1]:
            q, m, cut = checked_scores(fold, group); configurations[fold].append((q, m, cut))
            root = study.model_root(fold, group)
            for file in ['model_report.json', 'model_verification.json', 'score_report.json', 'score_verification.json',
                         'scores.parquet', 'calibration_report.json', 'calibration_verification.json',
                         'labels/full_labels.parquet', 'labels/full_label_report.json', 'labels/full_label_verification.json']:
                receipts[str(root / file)] = sha(root / file)
            receipts[str(shared.base.PROTOCOL)] = sha(shared.base.PROTOCOL)
            models.append(dict(fold=fold, training_group=group, root=str(root), rows=m['rows'], days=m['days'],
                last_observation=m['last_observation'], threshold=cut['threshold'],
                calibration_group=1-group, no_opposite_group_label_target=True,
                software_compilation_verified=False, native_source_parity_verified=False))
    f = pd.read_parquet(study.INPUTS / 'features.parquet', columns=study.META)
    assert len(f) == 1258085 and f.formula_input_valid.sum() == 1117397
    keys = f[study.META[:-1]]; selections, equality = [], []
    for arm in study.ARMS:
        flags, queries = [], []; cross = arm == 'cross'
        for fold, components in configurations.items():
            q = components[0][0]; cuts = [item[2]['threshold'] for item in components]
            frames = [pd.read_parquet(study.model_root(fold, group) / 'scores.parquet',
                filters=[('date', '>=', q['evaluation_start']), ('date', '<', q['evaluation_end'])]) for group in [0, 1]]
            meta = f.loc[f.date.ge(q['evaluation_start']) & f.date.lt(q['evaluation_end'])].reset_index(drop=True)
            for frame in frames:
                pd.testing.assert_frame_equal(frame[study.META], meta, check_exact=True)
            selected = study.route_flags(meta, frames[0].score, frames[1].score, cuts, cross)
            d = meta[['date', 'code']].copy(); d['selected'] = selected; flags.append(d)
            use0 = '>=5' if cross else '<5'
            expression = f'CASE WHEN cast(substr(a.code,9,1) AS INT){use0} THEN a.score>{cuts[0]:.17e} ELSE b.score>{cuts[1]:.17e} END'
            queries.append(f"SELECT a.date,a.code,coalesce(a.formula_input_valid AND ({expression}),false) AS selected FROM read_parquet('{study.model_root(fold,0)}/scores.parquet') a JOIN read_parquet('{study.model_root(fold,1)}/scores.parquet') b USING(date,code) WHERE a.date>='{q['evaluation_start']}' AND a.date<'{q['evaluation_end']}'")
            core = study.ROOT / (fold + '_' + arm + '_frozen_numeric_core.tdx')
            core.write_text(native_core([item[1] for item in components], cuts, cross)); receipts[str(core)] = sha(core)
        out = keys.merge(pd.concat(flags, ignore_index=True), on=['date', 'code'], how='left', validate='one_to_one')
        out['selected'] = out.selected.eq(True)
        c = shared.base.conn(); c.register('keys', keys); c.sql(' UNION ALL '.join(queries)).create_view('flags')
        expected = c.sql('SELECT k.*,coalesce(selected,false) AS selected FROM keys k LEFT JOIN flags USING(date,code) ORDER BY date,code').df()
        c.close(); pd.testing.assert_frame_equal(out, expected, check_exact=True)
        assert out.loc[out.selected, 'date'].ge('2025-01-01').all() and out.date.lt('2026-01-01').all()
        group = p['same_model_control_group'] if arm == 'control' else p['candidate_group']
        root = study.ROOT / group; root.mkdir(parents=True, exist_ok=True)
        assert not (root / 'selection_report.json').exists()
        out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
        r = dict(protocol_sha256=sha(shared.model.PROTOCOL), group=group, selection_sha256=sha(root / 'selection.parquet'),
            rows=len(out), selected=int(out.selected.sum()), days=out.loc[out.selected, 'date'].nunique(),
            source_hashes=receipts.copy(), no_new_group_evaluation=True, new_2026_prices_read=False, no_exit_rules=True)
        save_json(root / 'selection_report.json', r)
        save_json(root / 'selection_verification.json', dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'),
            rows=len(out), all_original_metadata_cross_own_flags_and_code_routing_sql_verified=True,
            no_training_or_calibration_period_selection=True, new_2026_prices_read=False, no_exit_rules=True))
        selections.append(dict(group=group, root=str(root), selected=r['selected'], days=r['days'], rows=len(out)))
        for name, path in p['controls'].items():
            equality.append(dict(left=group, right=name, full_frame_equal=out.equals(shared.checked_selection(Path(path)))))
        for file in ['selection.parquet', 'selection_report.json', 'selection_verification.json']:
            receipts[str(root / file)] = sha(root / file)
    equality.append(dict(left=p['candidate_group'], right=p['same_model_control_group'],
        full_frame_equal=shared.checked_selection(study.ROOT / p['candidate_group']).equals(
            shared.checked_selection(study.ROOT / p['same_model_control_group']))))
    joint = dict(passed=True, model_protocol_sha256=sha(shared.model.PROTOCOL), source_hashes=receipts,
        models=models, selections=selections, equality=equality, exactly_four_shared_models_four_fixed_cutoffs=True,
        both_full_annual_selection_frames_frozen_before_economics=True, no_calibration_outcome_based_threshold_choice=True,
        no_new_raw_extraction=True, no_new_group_evaluation=True, year_2025_is_exploratory=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(study.ROOT / 'joint_selection_freeze.json', joint)
    return dict(joint_sha256=sha(study.ROOT / 'joint_selection_freeze.json'), models=models, selections=selections, equality=equality)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['protocols', 'project', 'model', 'verify_model', 'scores', 'verify_scores',
        'calibrate', 'verify_calibration', 'freeze', 'analyze', 'finish'])
    parser.add_argument('--fold', choices=['2025h1', '2025h2']); parser.add_argument('--group', type=int, choices=[0, 1])
    args = parser.parse_args(); shared.configure(study.STEM)
    if args.stage in ['protocols', 'freeze']:
        result = globals()[args.stage]()
    elif args.stage in ['analyze', 'finish']:
        for fold in ['2025h1', '2025h2']:
            for group in [0, 1]:
                checked_scores(fold, group)
        result = getattr(shared, args.stage)()
    else:
        assert args.fold and args.group is not None; setup(args.fold, args.group)
        if args.stage == 'project': result = study.project_labels(args.fold, args.group)
        elif args.stage == 'model': result = fit()
        elif args.stage == 'verify_model': result = verify_model()
        elif args.stage == 'scores': result = shared.base.scores()
        elif args.stage == 'verify_scores': result = shared.verify_scores(expected_expressions=shared.base.EXPRESSIONS)
        else: result = getattr(study, args.stage)(args.fold, args.group)
    print(json.dumps(result, ensure_ascii=False, indent=2))
