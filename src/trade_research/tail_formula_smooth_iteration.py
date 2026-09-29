"""Complete the fixed smooth model's optimization with one larger budget."""
import argparse
import json
from pathlib import Path

import numpy as np
import sklearn

from . import tail_formula_smooth_network as original
from .corporate_cash import save_json, sha

STEM = 'tail_formula_smooth_iteration'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
FOLD = None


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for path, digest in {**p['references'], **p['local_primary_source_hashes']}.items():
        assert sha(Path(path)) == digest
    assert sklearn.__version__ == p['sklearn_version'] == '1.7.2'
    assert p['arms'] == ['smooth'] and p['ridge_alpha'] == original.ALPHA
    previous = json.loads(Path('config/tail_formula_smooth_network_protocol.json').read_text())
    changes = {k: (previous['parameters'][k], v) for k, v in p['parameters'].items()
               if v != previous['parameters'][k]}
    assert changes == {'max_iter': (200, 2000)}
    assert p['parameters'].keys() == previous['parameters'].keys()
    for key in ['training', 'preprocessing', 'target', 'ridge_alpha', 'training_quantile']:
        assert p[key] == previous[key]
    r = json.loads((original.original.ROOT / 'feature_report.json').read_text())
    v = json.loads((original.original.ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(original.original.ROOT / 'feature_report.json')
    assert r['features_sha256'] == sha(original.original.ROOT / 'features.parquet')
    return p


def setup(fold):
    global FOLD
    FOLD = fold
    original.STEM = STEM
    original.ROOT = ROOT
    original.PROTOCOL = PROTOCOL
    original.checked_sources = checked_sources
    original.setup('smooth', fold)
    # The reused routines operate in a fresh process for each stage. This is a
    # single new arm; the two old comparison arms are never trained again.
    original.base.ROOT = ROOT.parent / (STEM + '_' + ('2025' if fold == 'combined' else fold))
    original.base.PROTOCOL = Path('config') / (STEM + '_' + fold + '_protocol.json')
    original.relative.PROTOCOL = original.base.PROTOCOL
    if fold == 'combined':
        original.linkage.ROOT = ROOT.parent / (STEM + '_2024')
        original.linkage.H2 = ROOT.parent / (STEM + '_recent')
        original.linkage.COMBINED = original.base.ROOT
        original.linkage.PROTOCOL = original.base.PROTOCOL


def verify_model():
    proof = original.verify_model()
    root = original.base.ROOT
    old = ROOT.parent / ('tail_formula_smooth_network_smooth_' + FOLD)
    m = json.loads((root / 'model_report.json').read_text())
    previous = json.loads((old / 'model_report.json').read_text())
    previous_proof = json.loads((old / 'model_verification.json').read_text())
    assert previous_proof['passed'] and previous_proof['model_report_sha256'] == sha(old / 'model_report.json')
    for key in ['input_means', 'input_scales', 'input_constant_indices', 'standardized_clip',
                'scale_floor', 'ridge_alpha', 'weights_sum', 'feature_names', 'rows', 'days',
                'training_start', 'training_end', 'last_observation', 'feature_report_sha256',
                'label_report_sha256']:
        assert m[key] == previous[key], key
    assert {k: v for k, v in m['parameters'].items() if k != 'max_iter'} == {
        k: v for k, v in previous['parameters'].items() if k != 'max_iter'}
    assert np.isfinite(m['optimizer_loss']) and np.isfinite(proof['max_gradient'])
    proof.update(previous_200_model_sha256=sha(old / 'model_report.json'),
        previous_200_verification_sha256=sha(old / 'model_verification.json'),
        all_training_metadata_and_preprocessing_exactly_unchanged=True,
        only_parameter_change_is_max_iter_200_to_2000=True,
        previous_200_loss=previous['optimizer_loss'], current_loss=m['optimizer_loss'],
        previous_200_max_gradient=previous_proof['max_gradient'],
        previous_200_iterations=previous['iterations'], current_iterations=m['iterations'],
        optimizer_trajectory_prefix_not_recorded=True, global_optimum_not_claimed=True)
    save_json(root / 'model_verification.json', proof)
    return proof


def analyze():
    joint = json.loads((ROOT / 'joint_selection_freeze.json').read_text())
    assert joint['passed'] and joint['protocol_sha256'] == sha(PROTOCOL)
    assert len(joint['selections']) == 3
    record = next(x for x in joint['selections'] if x['root'] == str(original.base.ROOT))
    assert record['selection_report_sha256'] == sha(original.base.ROOT / 'selection_report.json')
    return original.reuse.evaluation.analyze(original.base.ROOT, original.base.PROTOCOL)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    a = p.parse_args(); setup(a.fold)
    if a.fold == 'combined' and a.stage != 'analyze':
        assert a.stage in ['freeze', 'verify']
        result = getattr(original.linkage, 'combine' if a.stage == 'freeze' else 'verify_combined')()
    elif a.stage == 'analyze':
        result = analyze()
    elif a.stage in ['scores', 'freeze', 'verify']:
        result = getattr(original.reuse, a.stage)()
    elif a.stage == 'verify_model':
        result = verify_model()
    else:
        result = getattr(original, a.stage)()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
