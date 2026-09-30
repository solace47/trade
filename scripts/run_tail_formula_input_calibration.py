"""Two shared models, two fixed cutoffs, then the existing complete-frame evaluation."""
import argparse
import json
from pathlib import Path

import pandas as pd

import run_tail_formula_paired_study as shared
from find_existing_tail_formula_models import find
from trade_research import tail_formula_input_calibration as study
from trade_research.corporate_cash import save_json, sha


def protocols():
    p = shared.model.checked()
    records = []
    receipts = {str(study.INPUTS / file): sha(study.INPUTS / file) for file in
        ['feature_report.json', 'feature_verification.json', 'native_input_verification.json',
         'full_label_report.json', 'full_label_verification.json']}
    metadata = json.loads((study.ROOT / 'metadata_lookup_verification.json').read_text())
    for fold, spec in p['folds'].items():
        counts = next(x for x in metadata['records'] if x['fold'] == fold)
        q = dict(master_protocol_sha256=sha(shared.model.PROTOCOL), arm='calibrated', fold=fold, **spec,
            expected_training_rows=counts['training_rows'], expected_training_days=counts['training_days'],
            expected_last_observation=counts['last_observation'], expected_features=50,
            feature_names=list(study.ARMS['calibrated']), parameters=p['parameters'], model_max_depth=3,
            threshold=.995, target='relative', input_receipts=receipts,
            no_training_period_selection=True, new_2026_prices_allowed=False, no_exit_rules=True)
        file = shared.model.fold_protocol('calibrated', fold)
        assert not file.exists()
        file.parent.mkdir(parents=True, exist_ok=True)
        save_json(file, q)
        lookup = find(file, variant='relative')
        assert not lookup['matches'], 'Audit and reuse matching models instead of fitting again'
        records.append(dict(arm='calibrated', fold=fold, protocol_sha256=sha(file), lookup=lookup))
    r = dict(passed=True, master_protocol_sha256=sha(shared.model.PROTOCOL), records=records,
        exactly_two_new_models_allowed=True, same_model_control_never_refit=True,
        metadata_lookup_sha256=sha(study.ROOT / 'metadata_lookup_verification.json'),
        no_new_group_outcomes_read=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(study.ROOT / 'prefit_lookup_verification.json', r)
    return dict(prefit_sha256=sha(study.ROOT / 'prefit_lookup_verification.json'), candidate_metadata_matches=0)


def setup(fold):
    q = shared.model.setup('calibrated', fold)
    shared.base.ROOT = study.ROOT / 'models' / fold
    return q


def checked_scores(fold):
    q = setup(fold)
    root = shared.base.ROOT
    m = json.loads((root / 'model_report.json').read_text())
    r = json.loads((root / 'score_report.json').read_text())
    assert m['protocol_sha256'] == r['protocol_sha256'] == sha(shared.base.PROTOCOL)
    assert m['feature_names'] == list(study.ARMS['calibrated']) and m['variant'] == 'relative'
    assert m['feature_report_sha256'] == r['feature_report_sha256'] == sha(study.INPUTS / 'feature_report.json')
    assert m['label_report_sha256'] == sha(study.INPUTS / 'full_label_report.json')
    assert m['rows'] == q['expected_training_rows'] and m['days'] == q['expected_training_days']
    assert m['last_observation'] == q['expected_last_observation'] < q['calibration_start']
    assert all(m['parameters'][k] == v for k, v in q['parameters'].items())
    assert m['thresholds'][3]['training_quantile'] == .995 and not m.get('no_model_fit_performed', False)
    assert r['model_report_sha256'] == sha(root / 'model_report.json')
    assert r['scores_sha256'] == sha(root / 'scores.parquet')
    for kind in ['model', 'score']:
        v = json.loads((root / (kind + '_verification.json')).read_text())
        assert v['passed'] and v[kind + '_report_sha256'] == sha(root / (kind + '_report.json'))
    _, report, verification = study.calibration_paths(fold)
    c = json.loads(report.read_text())
    v = json.loads(verification.read_text())
    assert v['passed'] and v['calibration_report_sha256'] == sha(report)
    assert c['model_report_sha256'] == sha(root / 'model_report.json')
    assert c['scores_sha256'] == sha(root / 'scores.parquet') and c['quantile'] == .995
    assert c['calibration_end'] == q['evaluation_start']
    return q, m, c


def freeze():
    p = shared.model.checked()
    assert not (study.ROOT / 'joint_selection_freeze.json').exists()
    receipts = {str(shared.model.PROTOCOL): sha(shared.model.PROTOCOL),
        str(study.ROOT / 'prefit_lookup_verification.json'): sha(study.ROOT / 'prefit_lookup_verification.json')}
    configurations = {}
    for fold in p['folds']:
        q, m, c = checked_scores(fold)
        configurations[fold] = (q, m, c)
        root = shared.base.ROOT
        for file in ['model_report.json', 'model_verification.json', 'score_report.json',
                     'score_verification.json', 'scores.parquet', 'calibration_report.json', 'calibration_verification.json']:
            receipts[str(root / file)] = sha(root / file)
        receipts[str(shared.base.PROTOCOL)] = sha(shared.base.PROTOCOL)
    f = pd.read_parquet(study.INPUTS / 'features.parquet', columns=study.META)
    assert len(f) == 1258085 and f.formula_input_valid.sum() == 1117397
    assert f.date.ge('2024-01-01').all() and f.date.lt('2026-01-01').all()
    meta = study.META[:-1]
    keys = f[meta]
    selections, models, equality = [], [], []
    for arm in study.ARMS:
        flags, queries = [], []
        for fold, (q, m, calibration) in configurations.items():
            root = study.ROOT / 'models' / fold
            cut = m['thresholds'][3]['threshold'] if arm == 'control' else calibration['threshold']
            d = pd.read_parquet(root / 'scores.parquet',
                filters=[('date', '>=', q['evaluation_start']), ('date', '<', q['evaluation_end'])])
            pd.testing.assert_frame_equal(d[study.META].reset_index(drop=True),
                f.loc[f.date.ge(q['evaluation_start']) & f.date.lt(q['evaluation_end'])].reset_index(drop=True), check_exact=True)
            d['selected'] = d.formula_input_valid & d.score.gt(cut)
            flags.append(d[['date', 'code', 'selected']])
            queries.append(f"SELECT date,code,coalesce(formula_input_valid AND score>{cut:.17e},false) AS selected FROM read_parquet('{root}/scores.parquet') WHERE date>='{q['evaluation_start']}' AND date<'{q['evaluation_end']}'")
            core = root / (arm + '_frozen_numeric_core.tdx')
            text = shared.base.native_core(m, cut, study.ARMS[arm], study.HEADER)
            assert text.count('CORE:SC>') == 1
            core.write_text(text.replace('CORE:SC>', 'CORE:' + study.CORE_GATE + ' AND SC>'))
            receipts[str(core)] = sha(core)
            models.append(dict(arm=arm, fold=fold, rows=m['rows'], days=m['days'],
                last_observation=m['last_observation'], threshold=cut,
                cutoff_source='training' if arm == 'control' else 'post_training_visible_inputs',
                selected=int(d.selected.sum()), signal_days=d.loc[d.selected, 'date'].nunique(),
                shared_model_root=str(root), software_compilation_verified=False, native_source_parity_verified=False))
        out = keys.merge(pd.concat(flags, ignore_index=True), on=['date', 'code'], how='left', validate='one_to_one')
        out['selected'] = out.selected.eq(True)
        c = shared.base.conn()
        c.register('keys', keys)
        c.sql(' UNION ALL '.join(queries)).create_view('flags')
        expected = c.sql('SELECT k.*,coalesce(selected,false) AS selected FROM keys k LEFT JOIN flags USING(date,code) ORDER BY date,code').df()
        c.close()
        pd.testing.assert_frame_equal(out, expected, check_exact=True)
        assert out.loc[out.selected, 'date'].ge('2025-01-01').all()
        group = p['same_model_control_group'] if arm == 'control' else p['candidate_group']
        root = study.ROOT / group
        root.mkdir(parents=True, exist_ok=True)
        assert not (root / 'selection_report.json').exists()
        out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
        r = dict(protocol_sha256=sha(shared.model.PROTOCOL), group=group,
            selection_sha256=sha(root / 'selection.parquet'), rows=len(out), selected=int(out.selected.sum()),
            days=out.loc[out.selected, 'date'].nunique(), source_hashes=receipts.copy(),
            no_new_group_evaluation=True, new_2026_prices_read=False, no_exit_rules=True)
        save_json(root / 'selection_report.json', r)
        save_json(root / 'selection_verification.json', dict(passed=True,
            selection_report_sha256=sha(root / 'selection_report.json'), rows=len(out),
            all_original_pool_metadata_and_full_threshold_flags_sql_verified=True,
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
        models=models, selections=selections, equality=equality, exactly_two_shared_models_four_fixed_cutoffs=True,
        both_full_annual_selection_frames_frozen_before_economics=True,
        no_calibration_outcome_based_threshold_choice=True, no_new_raw_extraction=True,
        no_new_group_evaluation=True, year_2025_is_exploratory=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(study.ROOT / 'joint_selection_freeze.json', joint)
    return dict(joint_sha256=sha(study.ROOT / 'joint_selection_freeze.json'), models=models, selections=selections, equality=equality)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['protocols', 'model', 'verify_model', 'scores', 'verify_scores',
        'calibrate', 'verify_calibration', 'freeze', 'analyze', 'finish'])
    parser.add_argument('--fold', choices=['2025h1', '2025h2'])
    args = parser.parse_args()
    shared.configure(study.STEM)
    if args.stage in ['protocols', 'freeze']:
        result = globals()[args.stage]()
    elif args.stage in ['analyze', 'finish']:
        for fold in ['2025h1', '2025h2']:
            checked_scores(fold)
        result = getattr(shared, args.stage)()
    else:
        assert args.fold
        setup(args.fold)
        if args.stage in ['model', 'verify_model']:
            result = getattr(shared.relative, args.stage)('relative')
        elif args.stage == 'scores':
            result = shared.base.scores()
        elif args.stage == 'verify_scores':
            result = shared.verify_scores(expected_expressions=shared.base.EXPRESSIONS)
        else:
            result = getattr(study, args.stage)(args.fold)
    print(json.dumps(result, ensure_ascii=False, indent=2))
