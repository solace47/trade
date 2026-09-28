"""One fixed node-feature sampling setting on the unchanged 48-input method."""
import argparse
from functools import partial
import json
from pathlib import Path

import sklearn
from sklearn.ensemble import GradientBoostingRegressor

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as inputs
from . import tail_formula_prior_day as adapter
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_subspace48'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')


def setup(fold):
    master = json.loads(PROTOCOL.read_text())
    for key, path in [
        ('feature_report_sha256', inputs.ROOT / 'feature_report.json'),
        ('feature_verification_sha256', inputs.ROOT / 'feature_verification.json'),
        ('features_sha256', inputs.ROOT / 'features.parquet'),
        ('boundary_protocol_sha256', labels.PROTOCOL),
        ('label_report_sha256', labels.ROOT / 'full_label_report.json'),
        ('label_verification_sha256', labels.ROOT / 'full_label_verification.json'),
        ('labels_sha256', labels.ROOT / 'full_labels.parquet'),
        ('implementation_sha256', Path(master['implementation_file'])),
    ]:
        assert master[key] == sha(path)
    assert sklearn.__version__ == master['sklearn_version']
    adapter.STEM = STEM
    adapter.ROOT = inputs.ROOT
    adapter.EXPRESSIONS = inputs.EXPRESSIONS
    adapter.HEADER = inputs.HEADER
    adapter.COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')
    adapter.setup(fold)
    base.SOURCE = labels.ROOT
    for name in ['2024', 'recent']:
        p = json.loads((Path('config') / (STEM + '_' + name + '_protocol.json')).read_text())
        assert p['master_protocol_sha256'] == sha(PROTOCOL)
        assert p['parameters'] == master['parameters'] and p['parameters']['max_features'] == 6
        assert p['expected_features'] == len(inputs.EXPRESSIONS) == 48


def model():
    # Fresh process per CLI stage: change only this constructor argument;
    # all training records, targets, weights and thresholds stay in the old engine.
    relative.GradientBoostingRegressor = partial(GradientBoostingRegressor, max_features=6)
    return relative.model('relative')


def verify_model():
    p = json.loads(base.PROTOCOL.read_text())
    r = json.loads((base.ROOT / 'model_report.json').read_text())
    assert r['days'] == p['expected_training_days'] == 241
    assert r['rows'] == p['expected_training_rows']
    assert r['feature_names'] == list(inputs.EXPRESSIONS)
    assert all(r['parameters'][k] == v for k, v in p['parameters'].items())
    return relative.verify_model('relative')


def selection_gate(write=False):
    reports = []
    for fold in ['2024', 'recent', '2025']:
        root = Path('data/research') / (STEM + '_' + fold)
        r = json.loads((root / 'selection_report.json').read_text())
        v = json.loads((root / 'selection_verification.json').read_text())
        assert v['passed'] and v['selection_report_sha256'] == sha(root / 'selection_report.json')
        assert r['selection_sha256'] == sha(root / 'selection.parquet')
        if write:
            assert not (root / 'analysis_report.json').exists()
        if fold != '2025':
            m = json.loads((root / 'model_report.json').read_text())
            old = json.loads((Path('data/research') / ('tail_formula_before1000_model_' + fold) / 'model_report.json').read_text())
            for key in ['rows', 'days', 'last_observation', 'training_start', 'training_end', 'feature_report_sha256', 'label_report_sha256']:
                assert m[key] == old[key]
            changed = {k for k in m['parameters'] if m['parameters'][k] != old['parameters'][k]}
            assert changed == {'max_features'} and old['parameters']['max_features'] is None
        reports.append(dict(root=str(root), selected=r['selected'],
            selection_report_sha256=sha(root / 'selection_report.json'),
            selection_verification_sha256=sha(root / 'selection_verification.json')))
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), selections=reports,
        only_estimator_parameter_changed_is_max_features=True,
        both_folds_and_whole_year_frozen_together=True, new_group_outcomes_read=False,
        year_2025_is_exploratory=True, new_2026_prices_read=False, no_exit_rules=True)
    path = ROOT / 'joint_selection_freeze.json'
    if write:
        assert not path.exists()
        ROOT.mkdir(parents=True, exist_ok=True)
        save_json(path, proof)
    else:
        assert json.loads(path.read_text()) == proof
    return proof


if __name__ == '__main__':
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'gate', 'analyze'])
    p.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    a = p.parse_args()
    setup(a.fold)
    if a.stage == 'analyze':
        selection_gate()
        result = evaluation.analyze(linkage.COMBINED if a.fold == 'combined' else base.ROOT,
            linkage.PROTOCOL if a.fold == 'combined' else base.PROTOCOL)
    elif a.stage == 'gate':
        result = selection_gate(write=True)
    elif a.fold == 'combined':
        assert a.stage in ['freeze', 'verify']
        result = linkage.combine() if a.stage == 'freeze' else linkage.verify_combined()
    elif a.stage in ['model', 'verify_model']:
        result = globals()[a.stage]()
    elif a.stage == 'verify_scores':
        result = verify_scores()
    elif a.stage == 'freeze':
        result = study.freeze()
    elif a.stage == 'verify':
        m = json.loads((base.ROOT / 'model_report.json').read_text())
        assert (base.ROOT / 'frozen_numeric_core.tdx').read_text() == base.native_core(m, m['thresholds'][3]['threshold'], base.EXPRESSIONS, base.HEADER)
        result = study.verify()
    else:
        result = base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))
