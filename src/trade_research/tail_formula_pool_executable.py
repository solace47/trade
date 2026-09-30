"""Executable-event learning in the audited expanded pool; cached parents intact."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_executable_opportunity as recipe
from . import tail_formula_pool_scope_inputs as inputs
from . import tail_formula_pool_scope_labels as labels
from . import tail_formula_pool_scope_model as parent
from .corporate_cash import save_json, sha

ROOT = Path('data/research/tail_formula_pool_executable')
PROTOCOL = Path('config/tail_formula_pool_executable_model_protocol.json')
AUDIT_PROTOCOL = Path('config/tail_formula_pool_executable_audit_protocol.json')
GROUPS = {'exec_original': 'original', 'exec_full': 'full', 'exec_extra': 'extra'}


def checked():
    p = json.loads(PROTOCOL.read_text()); parent.checked()
    assert p['expressions'] == inputs.EXPRESSIONS and p['native_header'] == inputs.HEADER
    assert p['threshold'] == .995 and p['expected_features'] == 48
    assert not p['new_2026_prices_allowed'] and p['parameters']['max_depth'] == 3
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    gate = json.loads((inputs.ROOT/'expansion_training_gate.json').read_text())
    complete = json.loads((inputs.ROOT/'complete_results_manifest.json').read_text())
    assert gate['passed'] and not gate['supports_followup'] and complete['passed']
    for file, digest in complete['source_hashes'].items():
        assert sha(Path(file)) == digest
    audit = json.loads((ROOT/'training_audit_verification.json').read_text())
    assert audit['passed'] and audit['protocol_sha256'] == sha(AUDIT_PROTOCOL)
    for file, digest in audit['input_artifacts'].items():
        assert sha(Path(file)) == digest
    assert all(not f['existing_model_candidates'] for f in audit['folds'])
    return p, audit


def fold_protocol(fold):
    return Path('config')/f'tail_formula_pool_executable_{fold}_protocol.json'


def configure(fold):
    # These adapters have module-level roots. Each CLI stage runs in a fresh
    # process, and guards run before this strictly local binding operation.
    base.ROOT = ROOT/fold; base.FEATURES = inputs.OUT; base.SOURCE = labels.directory('training')
    base.PROTOCOL = fold_protocol(fold); base.EXPRESSIONS = inputs.EXPRESSIONS; base.HEADER = inputs.HEADER
    recipe.PROTOCOL = PROTOCOL
    recipe.inputs.ROOT = inputs.OUT; recipe.inputs.EXPRESSIONS = inputs.EXPRESSIONS
    recipe.inputs.HEADER = inputs.HEADER; recipe.labels.ROOT = labels.directory('training')


def protocols():
    p, audit = checked(); records = []
    for fold, spec in p['folds'].items():
        a = next(a for a in audit['folds'] if a['fold'] == fold)
        q = dict(master_protocol_sha256=sha(PROTOCOL), fold=fold, **spec,
            expected_training_rows=a['rows'], expected_training_days=a['days'],
            expected_last_observation=a['last_observation'], expected_features=48,
            feature_names=list(inputs.EXPRESSIONS), parameters=p['parameters'], threshold=.995,
            training_event='known_or_confirmed_non_entry_opportunity',
            training_audit_sha256=sha(ROOT/'training_audit_verification.json'),
            no_training_period_selection=True, new_2026_prices_allowed=False, no_exit_rules=True)
        file = fold_protocol(fold); assert not file.exists(); save_json(file, q)
        result = subprocess.run(['.venv/bin/python', 'scripts/find_existing_tail_formula_models.py',
            '--protocol', str(file), '--variant', 'relative_executable_opportunity'],
            text=True, capture_output=True, check=True)
        lookup = json.loads(result.stdout); assert not lookup['matches'], 'Audit and reuse a matching fit'
        records.append(dict(fold=fold, protocol_sha256=sha(file), lookup=lookup))
    out = dict(passed=True, master_protocol_sha256=sha(PROTOCOL), records=records,
        exactly_two_new_expanded_event_models_allowed=True, all_parent_models_scores_and_statistics_reused=True,
        no_original_pool_refitting=True, no_new_group_evaluation=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT/'prefit_lookup_verification.json', out); return dict(prefit_sha256=sha(ROOT/'prefit_lookup_verification.json'))


def setup(fold):
    p, audit = checked(); file = fold_protocol(fold); q = json.loads(file.read_text())
    assert q['master_protocol_sha256'] == sha(PROTOCOL) and q['fold'] == fold
    assert all(q[k] == v for k, v in p['folds'][fold].items())
    assert q['parameters'] == p['parameters'] and q['feature_names'] == list(inputs.EXPRESSIONS)
    pre = json.loads((ROOT/'prefit_lookup_verification.json').read_text())
    assert pre['passed'] and pre['master_protocol_sha256'] == sha(PROTOCOL)
    assert any(r['fold'] == fold and r['protocol_sha256'] == sha(file) for r in pre['records'])
    configure(fold); return p, q, next(a for a in audit['folds'] if a['fold'] == fold)


def training_verification(q, a):
    t = recipe.training()
    old = pd.read_parquet(base.ROOT/'training_keys.parquet')
    fields = ['date', 'code', 'next_date', 'known15', 'known_no_trade']
    pd.testing.assert_frame_equal(t[fields], old[fields], check_exact=True)
    np.testing.assert_allclose(t[['utility', 'target', 'w']], old[['utility', 'target', 'w']], rtol=0, atol=2e-12)
    assert len(t) == a['rows'] == q['expected_training_rows']
    assert int(t.known_no_trade.sum()) == a['added_no_trade_rows']
    assert t.date.nunique() == q['expected_training_days'] == 241
    assert t.next_date.max() == q['expected_last_observation'] < q['evaluation_start']
    r = dict(passed=True, protocol_sha256=sha(base.PROTOCOL), master_protocol_sha256=sha(PROTOCOL),
        training_audit_sha256=sha(ROOT/'training_audit_verification.json'),
        training_keys_sha256=sha(base.ROOT/'training_keys.parquet'),
        feature_report_sha256=sha(inputs.OUT/'feature_report.json'),
        label_report_sha256=sha(base.SOURCE/'full_label_report.json'), **{k: v for k, v in a.items() if k != 'fold'},
        independent_sql_audit_reused_after_all_complete_training_keys_targets_and_weights_equal=True,
        all_original_known_rows_retained=True, unknowns_not_converted_to_failures=True,
        new_2026_prices_read=False, no_exit_rules=True)
    assert not (base.ROOT/'training_verification.json').exists()
    save_json(base.ROOT/'training_verification.json', r); return r


def freeze():
    p, _ = checked(); assert not (ROOT/'joint_selection_freeze.json').exists()
    receipts = {str(PROTOCOL): sha(PROTOCOL), str(ROOT/'prefit_lookup_verification.json'): sha(ROOT/'prefit_lookup_verification.json')}
    models = []; configs = {}
    for fold in p['folds']:
        q = json.loads(fold_protocol(fold).read_text()); configure(fold); root = base.ROOT
        m = json.loads((root/'model_report.json').read_text()); sr = json.loads((root/'score_report.json').read_text())
        assert m['protocol_sha256'] == sr['protocol_sha256'] == sha(base.PROTOCOL)
        assert m['master_protocol_sha256'] == sha(PROTOCOL) and m['variant'] == 'relative_executable_opportunity'
        assert m['training_includes_known_no_trade'] and m['feature_names'] == list(inputs.EXPRESSIONS)
        assert m['rows'] == q['expected_training_rows'] and m['days'] == q['expected_training_days']
        assert m['last_observation'] == q['expected_last_observation'] < q['evaluation_start']
        assert m['thresholds'][3]['training_quantile'] == .995
        for kind in ['model', 'score']:
            v = json.loads((root/(kind+'_verification.json')).read_text())
            assert v['passed'] and v[kind+'_report_sha256'] == sha(root/(kind+'_report.json'))
        assert sr['scores_sha256'] == sha(root/'scores.parquet')
        core = root/'frozen_numeric_core.tdx'; core.write_text(base.native_core(m, m['thresholds'][3]['threshold'], inputs.EXPRESSIONS, inputs.HEADER))
        for f in [base.PROTOCOL, core, *(root/n for n in ['training_verification.json', 'model_report.json',
                'model_verification.json', 'score_report.json', 'score_verification.json', 'scores.parquet'])]:
            receipts[str(f)] = sha(f)
        models.append(dict(fold=fold, rows=m['rows'], days=m['days'], last_observation=m['last_observation']))
        configs[fold] = (q, m)
    f = base.feature_inputs(); membership = pd.read_parquet(inputs.OUT/'membership.parquet')
    pd.testing.assert_frame_equal(f[['date', 'code']], membership[['date', 'code']], check_exact=True)
    selections = []; equality = []
    for group, pool in GROUPS.items():
        mask = membership.original_pool if pool == 'original' else np.ones(len(f), dtype=bool)
        keys = f.loc[mask, parent.META].reset_index(drop=True); flags = []; sql = []
        for fold, (q, m) in configs.items():
            root = ROOT/fold; cut = m['thresholds'][3]['threshold']
            d = pd.read_parquet(root/'scores.parquet', filters=[('date', '>=', q['evaluation_start']), ('date', '<', q['evaluation_end'])])
            d['selected'] = d.formula_input_valid & d.score.gt(cut); flags.append(d[['date', 'code', 'selected']])
            sql.append(f'''SELECT date,code,coalesce(formula_input_valid AND score>{cut:.17e},false) AS selected
                FROM read_parquet('{root}/scores.parquet') WHERE date>='{q['evaluation_start']}' AND date<'{q['evaluation_end']}' ''')
        out = keys.merge(pd.concat(flags, ignore_index=True), on=['date', 'code'], how='left', validate='one_to_one')
        out['selected'] = out.selected.eq(True)
        if pool == 'extra':
            out['selected'] &= ~membership.original_pool
        c = base.conn(); c.register('keys', keys); c.register('membership', membership)
        c.sql(' UNION ALL '.join(sql)).create_view('flags')
        extra = ' AND NOT original_pool' if pool == 'extra' else ''
        expected = c.sql('SELECT k.*,coalesce(selected,false)'+extra+' AS selected FROM keys k JOIN membership USING(date,code) '
            'LEFT JOIN flags USING(date,code) ORDER BY date,code').df(); c.close()
        pd.testing.assert_frame_equal(out, expected, check_exact=True)
        assert out.loc[out.selected, 'date'].ge('2025-01-01').all() and out.date.lt('2026-01-01').all()
        root = ROOT/group; root.mkdir(exist_ok=True); assert not (root/'selection_report.json').exists()
        out.to_parquet(root/'selection.parquet', index=False, compression='zstd')
        r = dict(protocol_sha256=sha(PROTOCOL), group=group, pool=pool, selection_sha256=sha(root/'selection.parquet'),
            selected=int(out.selected.sum()), days=int(out.loc[out.selected, 'date'].nunique()), source_hashes=receipts.copy(),
            no_new_group_evaluation=True, new_2026_prices_read=False, no_exit_rules=True)
        save_json(root/'selection_report.json', r)
        save_json(root/'selection_verification.json', dict(passed=True, selection_report_sha256=sha(root/'selection_report.json'),
            rows=len(out), all_pool_metadata_history_unselected_and_half_flags_sql_verified=True,
            no_training_period_selection=True, new_2026_prices_read=False, no_exit_rules=True))
        selections.append(dict(group=group, root=str(root), selected=r['selected'], days=r['days'], rows=len(out),
            selection_report_sha256=sha(root/'selection_report.json'), selection_verification_sha256=sha(root/'selection_verification.json')))
        for name, path in p['controls'].items():
            equality.append(dict(left=group, right=name, full_frame_equal=out.equals(pd.read_parquet(Path(path)/'selection.parquet'))))
        for file in ['selection.parquet', 'selection_report.json', 'selection_verification.json']:
            receipts[str(root/file)] = sha(root/file)
    out = dict(passed=True, protocol_sha256=sha(PROTOCOL), models=models, selections=selections,
        source_hashes=receipts, equality=equality, all_two_new_models_and_three_complete_lists_jointly_frozen=True,
        all_parent_models_scores_labels_and_statistics_reused=True, no_new_group_evaluation=True,
        year_2025_is_exploratory=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT/'joint_selection_freeze.json', out)
    return dict(joint_sha256=sha(ROOT/'joint_selection_freeze.json'), models=models, selections=selections, equality=equality)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['protocols', 'training_verification', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze'])
    p.add_argument('--fold', choices=['2025h1', '2025h2']); a = p.parse_args()
    if a.stage in ['protocols', 'freeze']:
        result = globals()[a.stage]()
    else:
        assert a.fold; master, q, audit = setup(a.fold)
        if a.stage == 'training_verification': result = training_verification(q, audit)
        elif a.stage == 'model': result = recipe.model()
        elif a.stage == 'verify_model': result = recipe.verify_model()
        elif a.stage == 'scores': result = base.scores()
        else: result = recipe.verify_scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))
