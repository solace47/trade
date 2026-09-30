"""Two expanded training fits, exact reuse of two parents and five fixed pools."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_relative as relative
from . import tail_formula_pool_scope_inputs as inputs
from . import tail_formula_pool_scope_labels as labels
from .corporate_cash import save_json, sha
from .tail_formula_offset_logit48 import verify_scores

PROTOCOL = Path('config/tail_formula_pool_scope_model_protocol.json')
META = ['date', 'code', 'half', 'board', 'decision_shares']
SCORE_META = [*META, 'formula_input_valid']
PARENT_INPUTS = Path('data/research/tail_formula_dense_bars/inputs')
PARENTS = {fold: Path('data/research/tail_formula_dense_bars/control')/fold for fold in ['2025h1', '2025h2']}
GROUPS = {'old_original': ('old', 'original'), 'old_full': ('old', 'full'),
          'expanded_original': ('expanded', 'original'), 'expanded_full': ('expanded', 'full'),
          'expanded_extra': ('expanded', 'extra')}


def checked():
    p = json.loads(PROTOCOL.read_text()); labels.checked('training')
    assert p['expressions'] == inputs.EXPRESSIONS and p['native_header'] == inputs.HEADER
    assert p['threshold'] == .995 and p['groups'] == {k: list(v) for k, v in GROUPS.items()}
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    root = labels.directory('training')
    r = json.loads((root/'full_label_report.json').read_text())
    v = json.loads((root/'full_label_verification.json').read_text())
    assert v['passed'] and v['label_report_sha256'] == sha(root/'full_label_report.json')
    assert r['labels_sha256'] == sha(root/'full_labels.parquet')
    assert r['last_observation'] < '2025-07-01' and not p['new_2026_prices_allowed']
    return p


def protocol_path(arm, fold):
    return Path('config')/f'tail_formula_pool_scope_{arm}_{fold}_protocol.json'


def configure(arm, fold):
    base.ROOT = inputs.ROOT/arm/fold; base.FEATURES = inputs.OUT
    base.SOURCE = labels.directory('training') if arm == 'expanded' else labels.original.ROOT
    base.EXPRESSIONS = inputs.EXPRESSIONS; base.HEADER = inputs.HEADER
    base.PROTOCOL = relative.PROTOCOL = protocol_path(arm, fold)


def protocols():
    master = checked(); records = []; counts = {}
    for arm in ['old', 'expanded']:
        for fold, spec in master['folds'].items():
            configure(arm, fold)
            t = base.training(start=spec['training_start'], end=spec['training_end'])
            assert t.next_date.max() < spec['evaluation_start']
            p = dict(master_protocol_sha256=sha(PROTOCOL), arm=arm, fold=fold, **spec,
                expected_training_rows=len(t), expected_training_days=int(t.date.nunique()),
                expected_last_observation=t.next_date.max(), expected_features=48,
                parameters=master['parameters'], model_max_depth=3, feature_names=list(inputs.EXPRESSIONS),
                threshold=.995, target='relative', no_training_period_selection=True,
                label_source=str(base.SOURCE), new_2026_prices_allowed=False, no_exit_rules=True)
            assert not base.PROTOCOL.exists(); save_json(base.PROTOCOL, p)
            result = subprocess.run(['.venv/bin/python', 'scripts/find_existing_tail_formula_models.py',
                '--protocol', str(base.PROTOCOL), '--variant', 'relative'],
                capture_output=True, text=True, check=True)
            lookup = json.loads(result.stdout)
            records.append(dict(arm=arm, fold=fold, protocol_sha256=sha(base.PROTOCOL), lookup=lookup))
            counts[arm+'_'+fold] = dict(rows=len(t), days=int(t.date.nunique()), last_observation=t.next_date.max())
    assert not any(r['lookup']['matches'] for r in records if r['arm'] == 'expanded'), 'Audit an existing candidate instead of refitting'
    report = dict(passed=True, master_protocol_sha256=sha(PROTOCOL), counts=counts, records=records,
        all_four_protocols_before_fitting=True, only_two_expanded_training_fits_allowed=True,
        old_parents_require_full_value_training_audit=True, no_evaluation_group_statistics_computed=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(inputs.ROOT/'prefit_lookup_verification.json', report)
    return dict(counts=counts, prefit_sha256=sha(inputs.ROOT/'prefit_lookup_verification.json'))


def setup(arm, fold):
    master = checked(); configure(arm, fold); p = json.loads(base.PROTOCOL.read_text())
    assert p['master_protocol_sha256'] == sha(PROTOCOL) and p['arm'] == arm and p['fold'] == fold
    assert all(p[k] == v for k, v in master['folds'][fold].items())
    assert p['feature_names'] == list(inputs.EXPRESSIONS) and p['parameters'] == master['parameters']
    pre = json.loads((inputs.ROOT/'prefit_lookup_verification.json').read_text())
    assert pre['passed'] and pre['master_protocol_sha256'] == sha(PROTOCOL)
    assert any(r['arm'] == arm and r['fold'] == fold and r['protocol_sha256'] == sha(base.PROTOCOL) for r in pre['records'])
    return p


def reuse_parent(fold):
    p = setup('old', fold); root = base.ROOT; parent = PARENTS[fold]
    assert not (root/'model_report.json').exists() and not (root/'scores.parquet').exists()
    columns = [*SCORE_META, *inputs.EXPRESSIONS]
    f = base.feature_inputs()[columns]
    membership = pd.read_parquet(inputs.OUT/'membership.parquet')
    old_mask = membership.original_pool; original = f.loc[old_mask].reset_index(drop=True)
    parent_inputs = pd.read_parquet(PARENT_INPUTS/'features.parquet', columns=columns)
    pd.testing.assert_frame_equal(original, parent_inputs, check_exact=True)
    m = json.loads((parent/'model_report.json').read_text())
    mv = json.loads((parent/'model_verification.json').read_text())
    sr = json.loads((parent/'score_report.json').read_text())
    sv = json.loads((parent/'score_verification.json').read_text())
    for kind, proof in [('model', mv), ('score', sv)]:
        assert proof['passed'] and proof[kind+'_report_sha256'] == sha(parent/(kind+'_report.json'))
    assert sr['model_report_sha256'] == sha(parent/'model_report.json')
    assert sr['scores_sha256'] == sha(parent/'scores.parquet')
    assert m['feature_report_sha256'] == sha(PARENT_INPUTS/'feature_report.json')
    assert m['label_report_sha256'] == sha(PARENT_INPUTS/'full_label_report.json')
    assert m['feature_names'] == p['feature_names'] and m['variant'] == 'relative'
    assert all(m['parameters'][k] == v for k, v in p['parameters'].items())
    # Compare every available parent training label, including the full-pool
    # daily baseline before applying input validity, without 2023 outcomes.
    cols = ['date', 'code', 'next_date', 'known15', 'opportunity15', 'known_no_trade', 'adverse_return15']
    c = base.conn()
    projected = c.execute('SELECT '+','.join(cols)+' FROM read_parquet(?) WHERE date>=? AND next_date<? ORDER BY date,code',
        [str(PARENT_INPUTS/'full_labels.parquet'), p['training_start'], p['training_end']]).df()
    current = c.execute('SELECT '+','.join(cols)+' FROM read_parquet(?) WHERE date>=? AND next_date<? ORDER BY date,code',
        [str(base.SOURCE/'full_labels.parquet'), p['training_start'], p['training_end']]).df()
    pd.testing.assert_frame_equal(current, projected, check_exact=True, check_dtype=False)
    t = relative.training('relative'); c.register('parent_features', parent_inputs)
    original_training = c.sql(f'''WITH l AS(SELECT date,code,next_date,opportunity15,
        opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target
        FROM read_parquet('{PARENT_INPUTS}/full_labels.parquet')
        WHERE known15 AND date>='{p['training_start']}' AND next_date<'{p['training_end']}')
        SELECT l.date,l.code,l.next_date,l.opportunity15,l.target,1./count(*) OVER(PARTITION BY l.date) AS w,
        {','.join('f.'+n for n in inputs.EXPRESSIONS)} FROM parent_features f JOIN l USING(date,code)
        WHERE formula_input_valid ORDER BY l.date,l.code''').df(); c.close()
    cols = ['date', 'code', 'next_date', 'opportunity15', *inputs.EXPRESSIONS]
    pd.testing.assert_frame_equal(t[cols], original_training[cols], check_exact=True, check_dtype=False)
    np.testing.assert_allclose(t.target, original_training.target, rtol=0, atol=2e-12)
    np.testing.assert_array_equal(1/t.groupby('date').code.transform('size'), original_training.w)
    assert len(t) == p['expected_training_rows'] == m['rows']
    assert t.date.nunique() == p['expected_training_days'] == m['days']
    assert t.next_date.max() == p['expected_last_observation'] == m['last_observation']
    report = dict(m)
    report.update(protocol_sha256=sha(base.PROTOCOL), feature_report_sha256=sha(inputs.OUT/'feature_report.json'),
        label_report_sha256=sha(base.SOURCE/'full_label_report.json'), reused_source_root=str(parent),
        reused_model_report_sha256=sha(parent/'model_report.json'), no_model_fit_performed=True)
    root.mkdir(parents=True, exist_ok=True); save_json(root/'model_report.json', report)
    for field in ['feature_names', 'parameters', 'trees', 'bias', 'learning_rate', 'thresholds', 'rows', 'days',
                  'last_observation', 'training_start', 'training_end']:
        assert report[field] == m[field]
    mv = dict(passed=True, model_report_sha256=sha(root/'model_report.json'),
        reused_model_verification_sha256=sha(parent/'model_verification.json'), rows=len(t),
        node_checks=mv['node_checks'], original_node_arithmetic_verification_reused=True,
        all_parent_inputs_labels_targets_weights_and_equations_exactly_preserved=True,
        no_model_fit_performed=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root/'model_verification.json', mv)
    cached = pd.read_parquet(parent/'scores.parquet')
    pd.testing.assert_frame_equal(cached[SCORE_META], original[SCORE_META], check_exact=True)
    added = f.loc[~old_mask, SCORE_META].copy(); added['score'] = np.nan
    valid = f.formula_input_valid & ~old_mask
    added.loc[valid.loc[~old_mask], 'score'] = base.predict(base.encode(f.loc[valid]), report)
    scores = pd.concat([cached, added], ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(scores[SCORE_META], f[SCORE_META], check_exact=True)
    pd.testing.assert_frame_equal(scores.loc[old_mask].reset_index(drop=True), cached, check_exact=True)
    scores.to_parquet(root/'scores.parquet', index=False, compression='zstd')
    sr.update(protocol_sha256=sha(base.PROTOCOL), model_report_sha256=sha(root/'model_report.json'),
        feature_report_sha256=sha(inputs.OUT/'feature_report.json'), scores_sha256=sha(root/'scores.parquet'),
        rows=len(scores), valid=int(f.formula_input_valid.sum()), reused_score_report_sha256=sha(parent/'score_report.json'),
        original_scores_exactly_reused=True, only_extra_rows_predicted=True)
    save_json(root/'score_report.json', sr)
    proof = dict(passed=True, fold=fold, source_root=str(parent), parent_model_report_sha256=sha(parent/'model_report.json'),
        parent_score_report_sha256=sha(parent/'score_report.json'), model_report_sha256=sha(root/'model_report.json'),
        original_score_rows=len(cached), added_score_rows=len(added), training_rows=len(t),
        all_parent_inputs_known_pool_baselines_targets_weights_models_and_quantiles_unchanged=True,
        no_parent_fit_or_original_row_prediction=True, no_evaluation_group_statistics_computed=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(root/'parent_reuse_verification.json', proof); return proof


def verify_model(p):
    m = json.loads((base.ROOT/'model_report.json').read_text())
    assert m['feature_names'] == p['feature_names'] and m['variant'] == 'relative'
    assert m['rows'] == p['expected_training_rows'] and m['days'] == p['expected_training_days']
    assert m['last_observation'] == p['expected_last_observation'] < p['evaluation_start']
    assert all(m['parameters'][k] == v for k, v in p['parameters'].items())
    return relative.verify_model('relative')


def freeze():
    master = checked(); path = inputs.ROOT/'joint_selection_freeze.json'; assert not path.exists()
    receipts = {str(PROTOCOL): sha(PROTOCOL),
        str(inputs.ROOT/'prefit_lookup_verification.json'): sha(inputs.ROOT/'prefit_lookup_verification.json')}
    configs = {}; models = []
    for arm in ['old', 'expanded']:
        for fold in master['folds']:
            p = setup(arm, fold); root = base.ROOT
            m = json.loads((root/'model_report.json').read_text())
            sr = json.loads((root/'score_report.json').read_text())
            for kind in ['model', 'score']:
                v = json.loads((root/(kind+'_verification.json')).read_text())
                assert v['passed'] and v[kind+'_report_sha256'] == sha(root/(kind+'_report.json'))
            assert m['protocol_sha256'] == sr['protocol_sha256'] == sha(base.PROTOCOL)
            assert sr['scores_sha256'] == sha(root/'scores.parquet')
            assert m['rows'] == p['expected_training_rows'] and m['days'] == p['expected_training_days']
            assert m['last_observation'] == p['expected_last_observation'] < p['evaluation_start']
            assert m['feature_names'] == list(inputs.EXPRESSIONS) and m['thresholds'][3]['training_quantile'] == .995
            core = root/'frozen_numeric_core.tdx'
            core.write_text(base.native_core(m, m['thresholds'][3]['threshold'], inputs.EXPRESSIONS, inputs.HEADER))
            files = [base.PROTOCOL, core, *(root/n for n in ['model_report.json', 'model_verification.json',
                'score_report.json', 'score_verification.json', 'scores.parquet'])]
            if arm == 'old':
                files.append(root/'parent_reuse_verification.json')
            receipts.update({str(f): sha(f) for f in files}); configs[(arm, fold)] = (p, m)
            models.append(dict(arm=arm, fold=fold, rows=m['rows'], days=m['days'],
                last_observation=m['last_observation'], new_fit=arm == 'expanded'))
    f = base.feature_inputs(); membership = pd.read_parquet(inputs.OUT/'membership.parquet')
    pd.testing.assert_frame_equal(f[['date', 'code']], membership[['date', 'code']], check_exact=True)
    records = []; selections = {}; equality = []
    for group, (arm, pool) in GROUPS.items():
        keys = f.loc[membership.original_pool if pool == 'original' else np.ones(len(f), dtype=bool), META].reset_index(drop=True)
        flags = []; sql = []
        for fold in master['folds']:
            p, m = configs[(arm, fold)]; root = inputs.ROOT/arm/fold; cut = m['thresholds'][3]['threshold']
            d = pd.read_parquet(root/'scores.parquet', filters=[('date', '>=', p['evaluation_start']), ('date', '<', p['evaluation_end'])])
            d['selected'] = d.formula_input_valid & d.score.gt(cut)
            flags.append(d[['date', 'code', 'selected']])
            sql.append(f'''SELECT date,code,coalesce(formula_input_valid AND score>{cut:.17e},false) AS selected
                FROM read_parquet('{root}/scores.parquet') WHERE date>='{p['evaluation_start']}' AND date<'{p['evaluation_end']}' ''')
        out = keys.merge(pd.concat(flags, ignore_index=True), on=['date', 'code'], how='left', validate='one_to_one')
        out['selected'] = out.selected.eq(True)
        c = base.conn(); c.register('keys', keys); c.register('membership', membership)
        c.sql(' UNION ALL '.join(sql)).create_view('flags')
        predicate = ' AND NOT original_pool' if pool == 'extra' else ''
        expected = c.sql('SELECT k.*,coalesce(selected,false)'+predicate+' AS selected FROM keys k '
            'JOIN membership USING(date,code) LEFT JOIN flags USING(date,code) ORDER BY date,code').df(); c.close()
        if pool == 'extra':
            eligible = keys[['date', 'code']].merge(membership, on=['date', 'code'], validate='one_to_one')
            out['selected'] &= ~eligible.original_pool
        pd.testing.assert_frame_equal(out, expected, check_exact=True)
        assert out.loc[out.selected, 'date'].ge('2025-01-01').all() and out.date.lt('2026-01-01').all()
        root = inputs.ROOT/group; root.mkdir(parents=True, exist_ok=True)
        assert not (root/'selection_report.json').exists()
        out.to_parquet(root/'selection.parquet', index=False, compression='zstd')
        r = dict(protocol_sha256=sha(PROTOCOL), group=group, model_arm=arm, pool=pool,
            selection_sha256=sha(root/'selection.parquet'), selected=int(out.selected.sum()),
            days=int(out.loc[out.selected, 'date'].nunique()), source_hashes=receipts.copy(),
            complete_metadata_rows=len(out), year_2025_is_exploratory=True,
            no_evaluation_group_statistics_computed=True, new_2026_prices_read=False, no_exit_rules=True)
        save_json(root/'selection_report.json', r)
        save_json(root/'selection_verification.json', dict(passed=True,
            selection_report_sha256=sha(root/'selection_report.json'), rows=len(out),
            all_complete_pool_metadata_and_unselected_history_rows_preserved=True,
            all_half_flags_and_membership_independently_sql_verified=True,
            no_training_period_selection=True, new_2026_prices_read=False, no_exit_rules=True))
        records.append(dict(group=group, root=str(root), selected=r['selected'], days=r['days'],
            rows=len(out), selection_report_sha256=sha(root/'selection_report.json'),
            selection_verification_sha256=sha(root/'selection_verification.json')))
        selections[group] = out
    # The original-pool control must exactly reuse the already verified result.
    old_control = Path(master['original_control'])
    control = pd.read_parquet(old_control/'selection.parquet')
    assert selections['old_original'].equals(control)
    equality.append(dict(left='old_original', right=str(old_control), full_frame_equal=True))
    from itertools import combinations
    for left, right in combinations(GROUPS, 2):
        equality.append(dict(left=left, right=right, full_frame_equal=selections[left].equals(selections[right])))
    for item in records:
        root = Path(item['root'])
        for name in ['selection.parquet', 'selection_report.json', 'selection_verification.json']:
            receipts[str(root/name)] = sha(root/name)
    joint = dict(passed=True, protocol_sha256=sha(PROTOCOL), source_hashes=receipts,
        models=models, selections=records, equality=equality,
        all_four_models_and_five_complete_lists_frozen_together=True,
        exactly_two_new_fits=True, no_evaluation_group_statistics_computed=True,
        year_2025_is_exploratory=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(path, joint); return dict(joint_sha256=sha(path), models=models, selections=records, equality=equality)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['protocols', 'reuse_parent', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze'])
    p.add_argument('--arm', choices=['old', 'expanded']); p.add_argument('--fold', choices=['2025h1', '2025h2']); a = p.parse_args()
    if a.stage in ['protocols', 'freeze']:
        result = globals()[a.stage]()
    elif a.stage == 'reuse_parent':
        assert a.arm == 'old' and a.fold; result = reuse_parent(a.fold)
    else:
        assert a.arm and a.fold; spec = setup(a.arm, a.fold)
        if a.stage == 'model':
            assert a.arm == 'expanded'; result = relative.model('relative')
        elif a.stage == 'verify_model':
            assert a.arm == 'expanded'; result = verify_model(spec)
        elif a.stage == 'scores':
            assert a.arm == 'expanded'; result = base.scores()
        else:
            result = verify_scores(expected_expressions=inputs.EXPRESSIONS)
    print(json.dumps(result, ensure_ascii=False, indent=2))
