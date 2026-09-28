"""Original shallow models using either of the two frozen coordinate systems."""
import argparse
import json
from pathlib import Path

from . import tail_formula_pca_axes as parent
from . import tail_formula_additive as base
from . import tail_formula_relative as relative
from . import tail_formula_recent as study
from . import tail_formula_context_2024 as linkage
from . import tail_formula_boundary_evaluation as evaluation
from .tail_formula_offset_logit48 import verify_scores
from .corporate_cash import sha


def setup(arm, fold):
    assert arm in ['axis', 'pca'] and fold in ['2024', 'recent', 'combined']
    parent.checked_sources()
    stem = parent.STEM+'_'+arm
    master = Path('config') / (stem+'_protocol.json')
    child = json.loads(master.read_text())
    assert child['parent_protocol_sha256'] == sha(parent.MASTER) and child['arm'] == arm
    if fold == 'combined':
        linkage.ROOT = Path('data/research') / (stem+'_2024')
        linkage.H2 = Path('data/research') / (stem+'_recent')
        linkage.COMBINED = Path('data/research') / (stem+'_2025')
        linkage.PROTOCOL = Path('config') / (stem+'_combined_protocol.json')
        p = json.loads(linkage.PROTOCOL.read_text())
        assert p['inputs_protocol_sha256'] == sha(parent.MASTER)
        paths = [Path('config') / (stem+'_'+name+'_protocol.json') for name in ['2024', 'recent']]
        assert p['fold_protocols'] == [str(x) for x in paths]
        for path, name in zip(paths, ['2024', 'recent']):
            root = Path('data/research') / (stem+'_'+name)
            for item in ['model', 'selection']:
                assert json.loads((root / (item+'_report.json')).read_text())['protocol_sha256'] == sha(path)
        linkage.H2_SELECTION_SHA = None
        linkage.H2_MODEL_SHA = None
        linkage.H2_OUTCOMES_PREVIOUSLY_SEEN = False
        base.SOURCE = Path('data/research/tail_formula_before1000')
        return
    parent.checked_transform(fold)
    features = parent.ROOT / fold / arm
    r = json.loads((features / 'feature_report.json').read_text())
    v = json.loads((features / 'feature_verification.json').read_text())
    n = json.loads((features / 'native_input_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(features / 'feature_report.json')
    assert v['effective_input_intersection_unchanged'] and r['newly_invalid'] == 0
    assert r['features_sha256'] == sha(features / 'features.parquet')
    assert r['protocol_sha256'] == n['protocol_sha256'] == sha(parent.MASTER)
    assert n['passed'] and n['feature_report_sha256'] == sha(features / 'feature_report.json')
    assert n['feature_verification_sha256'] == sha(features / 'feature_verification.json')
    base.FEATURES = features
    base.EXPRESSIONS = r['expressions']
    base.HEADER = r['native_header']
    assert list(base.EXPRESSIONS) == parent.FIELDS
    base.ROOT = Path('data/research') / (stem+'_'+fold)
    base.PROTOCOL = Path('config') / (stem+'_'+fold+'_protocol.json')
    relative.PROTOCOL = base.PROTOCOL
    study.ROOT, study.PROTOCOL = base.ROOT, base.PROTOCOL
    p = json.loads(base.PROTOCOL.read_text())
    assert p['inputs_protocol_sha256'] == sha(parent.MASTER) and p['arm_protocol_sha256'] == sha(master)
    for name in ['feature_report', 'feature_verification', 'native_input_verification']:
        assert p[name+'_sha256'] == sha(features / (name+'.json'))
    assert p['label_report_sha256'] == sha(base.SOURCE / 'full_label_report.json')
    assert p['expected_features'] == 48 and p['expected_training_days'] == 241


def main(arm):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    parser.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    args = parser.parse_args()
    setup(arm, args.fold)
    if args.stage == 'analyze':
        assert args.fold == 'combined', 'Aggregate the annual report once; it contains both halves'
        result = evaluation.analyze(linkage.COMBINED, linkage.PROTOCOL)
    elif args.fold == 'combined':
        assert args.stage in ['freeze', 'verify']
        result = linkage.combine() if args.stage == 'freeze' else linkage.verify_combined()
    elif args.stage == 'model':
        result = relative.model('relative')
    elif args.stage == 'verify_model':
        p = json.loads(base.PROTOCOL.read_text())
        m = json.loads((base.ROOT / 'model_report.json').read_text())
        assert m['feature_names'] == parent.FIELDS and m['rows'] == p['expected_rows'] and m['days'] == 241
        assert all(m['parameters'][key] == value for key, value in p['parameters'].items())
        result = relative.verify_model('relative')
    elif args.stage == 'verify_scores':
        result = verify_scores()
    elif args.stage == 'scores':
        result = base.scores()
    elif args.stage == 'freeze':
        result = study.freeze()
    else:
        m = json.loads((base.ROOT / 'model_report.json').read_text())
        assert (base.ROOT / 'frozen_numeric_core.tdx').read_text() == base.native_core(m, m['thresholds'][3]['threshold'], base.EXPRESSIONS, base.HEADER)
        result = study.verify()
    print(json.dumps(result, ensure_ascii=False, indent=2))
