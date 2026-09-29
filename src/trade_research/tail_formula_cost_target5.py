"""Change only the training opportunity cost, keeping the known15 population."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_before1000_model as baseline
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_relative as relative
from . import tail_formula_recent as selection
from .corporate_cash import save_json, sha

STEM = 'tail_formula_cost_target5'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM+'_protocol.json')
ORIGINAL_TRAINING = relative.training
FOLD = None


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for path, digest in p['references'].items(): assert sha(Path(path)) == digest
    for root, names in [(baseline.inputs.ROOT, ['feature']), (baseline.labels.ROOT, ['full_label'])]:
        for name in names:
            r = json.loads((root/(name+'_report.json')).read_text())
            v = json.loads((root/(name+'_verification.json')).read_text())
            key = 'label_report_sha256' if name == 'full_label' else 'feature_report_sha256'
            assert v['passed'] and v[key] == sha(root/(name+'_report.json'))
    assert p['training_cost_bps'] == 5 and p['primary_evaluation_cost_bps'] == 15
    return p


def setup(fold, inputs=False):
    global FOLD
    checked_sources(); FOLD = fold; baseline.STEM = STEM; baseline.setup(fold)
    for name in ['2024', 'recent', 'combined']:
        p = json.loads((Path('config')/(STEM+'_'+name+'_protocol.json')).read_text())
        assert p['inputs_protocol_sha256'] == sha(PROTOCOL)
    relative.training = ORIGINAL_TRAINING if inputs else training


def apply_targets(t, targets):
    pd.testing.assert_frame_equal(t[['date', 'code']], targets[['date', 'code']], check_exact=True)
    out = t.copy(); out['target'] = targets.target.to_numpy()
    pd.testing.assert_frame_equal(out.drop(columns='target'), t.drop(columns='target'), check_exact=True)
    return out


def inputs():
    master = checked_sources(); p = json.loads(base.PROTOCOL.read_text()); root = ROOT/FOLD
    assert not (root/'input_verification.json').exists()
    root.mkdir(parents=True, exist_ok=True)
    t = ORIGINAL_TRAINING('relative')
    expected = next(v for v in master['training'] if v['fold'] == FOLD)
    assert len(t) == expected['rows'] and t.date.nunique() == expected['days'] == 241
    _, _, scope = relative.training_scope(); c = base.conn()
    l = c.execute(f'''SELECT date,code,known5,opportunity5,opportunity15 FROM read_parquet(?)
        WHERE known15 AND {scope} ORDER BY date,code''', [str(base.SOURCE/'full_labels.parquet')]).df()
    conditions = dict(known15_records=len(l), missing_cost5=int((~l.known5).sum()),
        nonbinary_cost5=int((~l.opportunity5.isin([0, 1])).sum()),
        nonmonotone_opportunities=int((l.opportunity5 < l.opportunity15).sum()))
    save_json(root/'target_coverage.json', dict(protocol_sha256=sha(PROTOCOL), **conditions))
    assert conditions['missing_cost5'] == conditions['nonbinary_cost5'] == conditions['nonmonotone_opportunities'] == 0
    l['target'] = l.opportunity5-l.groupby('date').opportunity5.transform('mean')
    targets = t[['date', 'code']].merge(l[['date', 'code', 'target']], on=['date', 'code'], validate='one_to_one')
    out = apply_targets(t, targets)
    c.register('keys', t[['date', 'code']])
    rebuilt = c.execute(f'''WITH population AS(
        SELECT date,code,opportunity5-avg(opportunity5) OVER(PARTITION BY date) AS target
        FROM read_parquet(?) WHERE known15 AND {scope})
        SELECT k.date,k.code,p.target FROM keys k JOIN population p USING(date,code) ORDER BY k.date,k.code''',
        [str(base.SOURCE/'full_labels.parquet')]).df(); c.close()
    pd.testing.assert_frame_equal(targets[['date', 'code']], rebuilt[['date', 'code']], check_exact=True)
    np.testing.assert_allclose(targets.target, rebuilt.target, rtol=0, atol=2e-12)
    np.testing.assert_array_equal(base.encode(out), base.encode(t))
    assert out.groupby('date').size().equals(t.groupby('date').size())
    control = Path('data/research')/('tail_formula_before1000_model_'+FOLD)
    r = json.loads((control/'model_report.json').read_text()); v = json.loads((control/'model_verification.json').read_text())
    assert v['passed'] and v['model_report_sha256'] == sha(control/'model_report.json')
    assert r['rows'] == len(t) and r['days'] == t.date.nunique()
    assert r['feature_names'] == list(base.EXPRESSIONS) and all(r['parameters'][k] == val for k,val in p['parameters'].items())
    for key in ['feature_report_sha256', 'label_report_sha256', 'training_start', 'training_end']: assert r[key] == p[key]
    targets.to_parquet(root/'targets.parquet', index=False, compression='zstd')
    training_labels = t[['date', 'code']].merge(l[['date', 'code', 'opportunity5', 'opportunity15']], on=['date','code'], validate='one_to_one')
    report = dict(protocol_sha256=sha(PROTOCOL), fold_protocol_sha256=sha(base.PROTOCOL),
        feature_report_sha256=sha(base.FEATURES/'feature_report.json'), label_report_sha256=sha(base.SOURCE/'full_label_report.json'),
        targets_sha256=sha(root/'targets.parquet'), rows=len(t), days=t.date.nunique(), **conditions,
        base_labels_changed=int(l.opportunity5.ne(l.opportunity15).sum()),
        training_labels_changed=int(training_labels.opportunity5.ne(training_labels.opportunity15).sum()),
        training_targets_changed=int(np.abs(out.target-t.target).gt(1e-12).sum()),
        original_control_model_sha256=sha(control/'model_report.json'), all_original_keys_values_encodings_and_weights_unchanged=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root/'input_report.json', report)
    proof = dict(passed=True, input_report_sha256=sha(root/'input_report.json'), rows=len(t), days=241,
        independent_sql_all_targets_and_population_means_verified=True, all_original_training_rows_and_weights_preserved=True,
        original_cost15_control_reusable=True, lower_cost_opportunity_monotonicity_verified=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root/'input_verification.json', proof); return {**proof, **conditions,
        'training_labels_changed': report['training_labels_changed'], 'training_targets_changed': report['training_targets_changed']}


def training(variant):
    assert variant == 'relative' and FOLD in ['2024', 'recent']
    root = ROOT/FOLD; r = json.loads((root/'input_report.json').read_text()); v = json.loads((root/'input_verification.json').read_text())
    assert v['passed'] and v['input_report_sha256'] == sha(root/'input_report.json')
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['fold_protocol_sha256'] == sha(base.PROTOCOL)
    assert r['feature_report_sha256'] == sha(base.FEATURES/'feature_report.json')
    assert r['label_report_sha256'] == sha(base.SOURCE/'full_label_report.json')
    assert r['targets_sha256'] == sha(root/'targets.parquet')
    return apply_targets(ORIGINAL_TRAINING('relative'), pd.read_parquet(root/'targets.parquet'))


def verify_model():
    p = json.loads(base.PROTOCOL.read_text()); r = json.loads((base.ROOT/'model_report.json').read_text())
    assert r['days'] == p['expected_training_days'] == 241
    assert r['feature_names'] == list(base.EXPRESSIONS)
    assert all(r['parameters'][k] == v for k,v in p['parameters'].items())
    return relative.verify_model('relative', training_cost_bps=5)


def main():
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['inputs', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024'); a = p.parse_args()
    setup(a.fold, inputs=a.stage=='inputs')
    if a.stage == 'analyze':
        joint = json.loads((ROOT/'joint_selection_freeze.json').read_text())
        assert joint['passed'] and joint['protocol_sha256'] == sha(PROTOCOL) and len(joint['selections']) == 3
        result = evaluation.analyze(linkage.COMBINED if a.fold=='combined' else base.ROOT, linkage.PROTOCOL if a.fold=='combined' else base.PROTOCOL)
    elif a.fold == 'combined':
        assert a.stage in ['freeze','verify']; result = linkage.combine() if a.stage=='freeze' else linkage.verify_combined()
    elif a.stage == 'inputs': result = inputs()
    elif a.stage == 'model': result = relative.model('relative')
    elif a.stage == 'verify_model': result = verify_model()
    elif a.stage == 'verify_scores': result = verify_scores()
    elif a.stage in ['freeze', 'verify']: result = getattr(selection, a.stage)()
    else: result = base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__=='__main__':main()
