"""Two absolute-opportunity fits; reuse the established score/economic verifiers."""
import argparse
import json
from pathlib import Path

import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_chrono_logit48 as binary
from trade_research.corporate_cash import save_json, sha
from trade_research.tail_formula_offset_logit48 import verify_scores
from find_existing_tail_formula_models import find
from freeze_tail_formula_bipower_gap import checked_selection
import run_tail_formula_paired_study as shared

STEM = 'tail_formula_absolute_morning'


def protocols():
    p = shared.model.checked(); study = shared.study
    base.FEATURES = base.SOURCE = study.INPUTS
    base.EXPRESSIONS = study.ARMS['absolute']
    receipts = {str(study.INPUTS / f): sha(study.INPUTS / f) for f in
                ['feature_report.json', 'feature_verification.json', 'native_input_verification.json',
                 'full_label_report.json', 'full_label_verification.json']}
    records, counts = [], {}
    for fold, spec in p['folds'].items():
        t = base.training(start=spec['training_start'], end=spec['training_end'])
        a = dict(rows=len(t), days=t.date.nunique(), last_observation=t.next_date.max())
        assert a == p['expected_training'][fold] and a['last_observation'] < spec['evaluation_start']
        q = dict(master_protocol_sha256=sha(shared.model.PROTOCOL), arm='absolute', fold=fold,
            **spec, expected_training_rows=a['rows'], expected_training_days=a['days'],
            expected_last_observation=a['last_observation'], expected_features=50,
            feature_names=list(base.EXPRESSIONS), parameters=p['parameters'],
            variant='absolute_logistic', target='known15 opportunity15, binary log-loss, dates equally weighted',
            threshold=.995, input_receipts=receipts, no_training_period_selection=True,
            new_2026_prices_allowed=False, no_exit_rules=True)
        file = shared.model.fold_protocol('absolute', fold)
        assert not file.exists(); save_json(file, q)
        lookup = find(file, 'absolute_logistic')
        assert not lookup['matches'], 'Audit matching inputs, labels and weights before reuse'
        counts[fold] = a
        records.append(dict(arm='absolute', fold=fold, protocol_sha256=sha(file), lookup=lookup))
    out = dict(passed=True, master_protocol_sha256=sha(shared.model.PROTOCOL), counts=counts,
        records=records, exactly_two_new_models_allowed=True, original_relative_controls_referenced_directly=True,
        all_two_protocols_before_fitting=True, no_new_group_outcomes_read=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(study.ROOT / 'prefit_lookup_verification.json', out)
    return dict(prefit_sha256=sha(study.ROOT / 'prefit_lookup_verification.json'), counts=counts, candidate_metadata_matches=0)


def setup(fold):
    return shared.model.setup('absolute', fold)


def verify_model():
    q = json.loads(base.PROTOCOL.read_text()); root = base.ROOT
    m = json.loads((root / 'model_report.json').read_text())
    assert m['feature_names'] == q['feature_names'] == list(base.EXPRESSIONS)
    assert len(m['feature_names']) == 50 and m['variant'] == q['variant'] == 'absolute_logistic'
    assert all(m['parameters'][k] == v for k, v in q['parameters'].items())
    assert all(m[k] == q[k] for k in ['training_start', 'training_end'])
    c = base.conn()
    keys = c.execute('''SELECT l.date,l.code,l.next_date FROM read_parquet(?) l
        JOIN read_parquet(?) f USING(date,code) WHERE l.date>=? AND l.next_date<?
        AND l.known15 AND f.formula_input_valid ORDER BY date,code''',
        [str(base.SOURCE / 'full_labels.parquet'), str(base.FEATURES / 'features.parquet'),
         q['training_start'], q['training_end']]).df(); c.close()
    assert len(keys) == m['rows'] == q['expected_training_rows']
    assert keys.date.nunique() == m['days'] == q['expected_training_days']
    assert keys.next_date.max() == m['last_observation'] == q['expected_last_observation'] < q['evaluation_start']
    for tree in m['trees']:
        depths = {0: 0}
        for node, left in enumerate(tree['children_left']):
            assert depths[node] <= q['parameters']['max_depth']
            if left >= 0:
                depths[left] = depths[tree['children_right'][node]] = depths[node] + 1
    original_loader = base.training
    def scoped_loader():
        t = original_loader(start=q['training_start'], end=q['training_end'])
        assert t[['date', 'code', 'next_date']].equals(keys)
        return t
    base.training = scoped_loader
    try:
        proof = base.verify_model()
    finally:
        base.training = original_loader
    proof.update(training_start=q['training_start'], training_end=q['training_end'],
        all_training_keys_and_depth_bounds_rebuilt=True, original_50_inputs_unchanged=True,
        absolute_binary_loss_without_day_centering=True)
    save_json(root / 'model_verification.json', proof)
    return proof


def freeze():
    p = shared.model.checked(); study = shared.study
    assert not (study.ROOT / 'joint_selection_freeze.json').exists()
    receipts = {str(shared.model.PROTOCOL): sha(shared.model.PROTOCOL)}
    models, flags, queries = [], [], []
    meta = study.META[:-1]
    keys = pd.read_parquet(study.INPUTS / 'features.parquet', columns=meta)
    assert len(keys) == 1258085 and keys.date.ge('2024-01-01').all() and keys.date.lt('2026-01-01').all()
    for fold in p['folds']:
        q = setup(fold); root = base.ROOT
        for arm, parent, expected_variant, parameters in [
                ('absolute', root, 'absolute_logistic', p['parameters']),
                ('control', shared.model.OLD[fold], 'relative', p['control_parameters'])]:
            m = json.loads((parent / 'model_report.json').read_text())
            r = json.loads((parent / 'score_report.json').read_text())
            assert m['variant'] == expected_variant and m['feature_names'] == q['feature_names']
            assert m['feature_report_sha256'] == r['feature_report_sha256'] == sha(study.INPUTS / 'feature_report.json')
            assert m['label_report_sha256'] == sha(study.INPUTS / 'full_label_report.json')
            assert r['model_report_sha256'] == sha(parent / 'model_report.json') and r['scores_sha256'] == sha(parent / 'scores.parquet')
            assert all(m['parameters'][k] == v for k, v in parameters.items())
            assert m['rows'] == q['expected_training_rows'] and m['days'] == q['expected_training_days']
            assert m['last_observation'] == q['expected_last_observation'] < q['evaluation_start']
            assert all(m[k] == q[k] for k in ['training_start', 'training_end'])
            assert m['thresholds'][3]['training_quantile'] == .995
            for kind in ['model', 'score']:
                v = json.loads((parent / (kind + '_verification.json')).read_text())
                assert v['passed'] and v[kind + '_report_sha256'] == sha(parent / (kind + '_report.json'))
            if arm == 'absolute':
                assert m['protocol_sha256'] == r['protocol_sha256'] == sha(base.PROTOCOL)
            else:
                # The same physical feature/label files, dates, metadata, parameters
                # and arithmetic proofs are referenced; no control copy or prediction.
                assert r['rows'] == p['expected_full_score_rows'] and r['valid'] == p['expected_valid_score_rows']
            s = pd.read_parquet(parent / 'scores.parquet', columns=meta)
            pd.testing.assert_frame_equal(s, keys, check_exact=True)
            models.append(dict(arm=arm, fold=fold, root=str(parent), rows=m['rows'], days=m['days'],
                last_observation=m['last_observation'], variant=m['variant'], new_fit=arm == 'absolute',
                exact_original_input_and_label_files=True, original_control_proofs_reused=arm == 'control'))
            for file in ['model_report.json', 'model_verification.json', 'score_report.json', 'score_verification.json', 'scores.parquet']:
                receipts[str(parent / file)] = sha(parent / file)
        receipts[str(base.PROTOCOL)] = sha(base.PROTOCOL)
        m = json.loads((root / 'model_report.json').read_text()); cut = m['thresholds'][3]['threshold']
        d = pd.read_parquet(root / 'scores.parquet', filters=[('date', '>=', q['evaluation_start']), ('date', '<', q['evaluation_end'])])
        d['selected'] = d.formula_input_valid & d.score.gt(cut)
        flags.append(d[['date', 'code', 'selected']])
        queries.append(f'''SELECT date,code,coalesce(formula_input_valid AND score>{cut:.17e},false) AS selected
            FROM read_parquet('{root}/scores.parquet') WHERE date>='{q['evaluation_start']}' AND date<'{q['evaluation_end']}' ''')
        text = base.native_core(m, cut, study.ARMS['absolute'], study.HEADER)
        assert text.count('CORE:SC>') == 1
        core = root / 'frozen_numeric_core.tdx'
        core.write_text(text.replace('CORE:SC>', 'CORE:' + study.CORE_GATE + ' AND SC>'))
        receipts[str(core)] = sha(core)
        models[-2].update(selected=int(d.selected.sum()), signal_days=d.loc[d.selected, 'date'].nunique(),
            software_compilation_verified=False, native_source_parity_verified=False)
    out = keys.merge(pd.concat(flags, ignore_index=True), on=['date', 'code'], how='left', validate='one_to_one')
    out['selected'] = out.selected.eq(True)
    c = base.conn(); c.register('keys', keys); c.sql(' UNION ALL '.join(queries)).create_view('flags')
    expected = c.sql('SELECT k.*,coalesce(selected,false) AS selected FROM keys k LEFT JOIN flags USING(date,code) ORDER BY date,code').df(); c.close()
    pd.testing.assert_frame_equal(out, expected, check_exact=True)
    assert out.loc[out.selected, 'date'].ge('2025-01-01').all()
    root = study.ROOT / p['candidate_group']; root.mkdir(parents=True, exist_ok=True)
    assert not (root / 'selection_report.json').exists()
    out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(shared.model.PROTOCOL), group=p['candidate_group'],
        selection_sha256=sha(root / 'selection.parquet'), rows=len(out), selected=int(out.selected.sum()),
        days=out.loc[out.selected, 'date'].nunique(), source_hashes=receipts.copy(),
        no_new_group_evaluation=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'selection_report.json', r)
    save_json(root / 'selection_verification.json', dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'),
        rows=len(out), all_original_pool_metadata_and_half_flags_sql_verified=True,
        no_training_period_selection=True, new_2026_prices_read=False, no_exit_rules=True))
    control = Path(p['controls'][p['control_group']]); original = checked_selection(control)
    pd.testing.assert_frame_equal(original[meta], keys, check_exact=True)
    selections = [dict(group=p['control_group'], root=str(control), rows=len(original), selected=int(original.selected.sum()),
        days=original.loc[original.selected, 'date'].nunique()),
        dict(group=p['candidate_group'], root=str(root), rows=r['rows'], selected=r['selected'], days=r['days'])]
    equality = [dict(left=p['candidate_group'], right=name, full_frame_equal=out.equals(checked_selection(Path(path))))
                for name, path in p['controls'].items()]
    for parent in [root, control]:
        for file in ['selection.parquet', 'selection_report.json', 'selection_verification.json']:
            receipts[str(parent / file)] = sha(parent / file)
    joint = dict(passed=True, model_protocol_sha256=sha(shared.model.PROTOCOL), source_hashes=receipts,
        models=models, selections=selections, equality=equality,
        all_two_new_models_two_reused_controls_and_two_full_lists_jointly_frozen=True,
        no_parent_refitting_or_prediction=True, no_new_raw_extraction=True, year_2025_is_exploratory=True,
        no_new_group_evaluation=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(study.ROOT / 'joint_selection_freeze.json', joint)
    return dict(joint_sha256=sha(study.ROOT / 'joint_selection_freeze.json'), models=models, selections=selections, equality=equality)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['protocols', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'analyze', 'finish'])
    parser.add_argument('--fold', choices=['2025h1', '2025h2']); args = parser.parse_args()
    shared.configure(STEM)
    if args.stage in ['analyze', 'finish']:
        result = getattr(shared, args.stage)()
    elif args.stage in ['protocols', 'freeze']:
        result = globals()[args.stage]()
    else:
        assert args.fold; setup(args.fold)
        if args.stage == 'model': result = binary.model()
        elif args.stage == 'verify_model': result = verify_model()
        elif args.stage == 'scores': result = base.scores()
        else: result = verify_scores(expected_expressions=base.EXPRESSIONS)
    print(json.dumps(result, ensure_ascii=False, indent=2))
