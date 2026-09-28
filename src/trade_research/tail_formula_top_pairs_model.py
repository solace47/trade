"""Center fixed top-pair scores and freeze every list before evaluation."""
import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd

from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_float as inputs
from . import tail_formula_score_center as center
from . import tail_formula_top_pairs_raw as raw
from .corporate_cash import save_json, sha
from .tail_formula_pairwise_normalization import verify_center_native

STEM = 'tail_formula_top_pairs'
ROOT = Path('data/research') / STEM
MASTER = Path('config') / (STEM + '_protocol.json')


def config():
    raw.checked_sources()
    p = json.loads(MASTER.read_text())
    assert p['training_quantile'] == .995 and not p['new_2026_prices_allowed']
    return p


def source(fold):
    config()
    protocol = Path('config') / (STEM + '_raw_' + fold + '_protocol.json')
    p = json.loads(protocol.read_text())
    path = Path('data/research') / (STEM + '_raw_' + fold)
    for stage in ['model', 'score']:
        v = json.loads((path / (stage + '_verification.json')).read_text())
        assert v['passed'] and v[stage + '_report_sha256'] == sha(path / (stage + '_report.json'))
    m = json.loads((path / 'model_report.json').read_text())
    s = json.loads((path / 'score_report.json').read_text())
    assert m['protocol_sha256'] == sha(protocol) and m['inputs_protocol_sha256'] == sha(MASTER)
    assert s['scores_sha256'] == sha(path / 'scores.parquet')
    assert s['model_report_sha256'] == sha(path / 'model_report.json')
    assert m['feature_names'] == list(inputs.EXPRESSIONS) and len(m['trees']) == 64
    assert m['parameters'] == raw.PARAMETERS
    assert (m['training_start'], m['training_end']) == (p['training_start'], p['training_end'])
    assert m['last_observation'] < p['training_end'] == p['evaluation_start']
    assert p['evaluation_end'] <= '2026-01-01'
    return p, path, m


def setup(fold):
    center.STEM = STEM
    center.PROTOCOL = MASTER
    center.config = config
    center.source = source
    center.ACTIVE_FOLD = fold


def joint_freeze():
    setup('2025')
    assert not (ROOT / 'joint_selection_freeze.json').exists()
    # Every raw model/score and both centered lists must already be verified.
    for fold in ['2024', 'recent']:
        source(fold)
        path = center.root(fold)
        for stage in ['score', 'selection']:
            v = json.loads((path / (stage + '_verification.json')).read_text())
            assert v['passed'] and v[stage + '_report_sha256'] == sha(path / (stage + '_report.json'))
        v = json.loads((path / 'native_center_verification.json').read_text())
        assert v['passed'] and v['score_report_sha256'] == sha(path / 'score_report.json')
    center.freeze('2025')
    center.verify('2025')
    records = []; comparisons = []
    for fold in ['2024', 'recent', '2025']:
        path = center.root(fold)
        sr = json.loads((path / 'selection_report.json').read_text())
        sp = json.loads((path / 'selection_verification.json').read_text())
        assert sr['selection_sha256'] == sha(path / 'selection.parquet')
        assert sp['passed'] and sp['selection_report_sha256'] == sha(path / 'selection_report.json')
        f = pd.read_parquet(path / 'selection.parquet')
        record = dict(root=str(path), fold=fold, selected=int(f.selected.sum()), days=f.loc[f.selected,'date'].nunique(),
            selection_report_sha256=sha(path / 'selection_report.json'),
            selection_verification_sha256=sha(path / 'selection_verification.json'))
        if fold != '2025':
            _, model_root, _ = source(fold)
            record['model_root'] = str(model_root)
            for file in ['model_report.json', 'model_verification.json', 'score_report.json', 'score_verification.json']:
                record['raw_' + file.removesuffix('.json') + '_sha256'] = sha(model_root / file)
            for file in ['score_report.json', 'score_verification.json', 'native_center_verification.json']:
                record[file.removesuffix('.json') + '_sha256'] = sha(path / file)
        records.append(record)
        controls = [Path('data/research') / (name + '_' + fold)
                    for name in ['tail_formula_pairwise_center', 'tail_formula_before1000_model']]
        if fold == '2025':
            controls.append(Path('data/research/tail_formula_before1000/evaluation/tail_formula_float_2025'))
        for control in controls:
            original = pd.read_parquet(control / 'selection.parquet')
            comparisons.append(dict(left=str(path), right=str(control), full_dataframe_equal=f.equals(original),
                left_selection_sha256=sha(path / 'selection.parquet'), right_selection_sha256=sha(control / 'selection.parquet')))
    receipt = dict(passed=True, protocol_sha256=sha(MASTER), selections=records, full_selection_prechecks=comparisons,
        all_two_models_and_three_centered_selections_frozen_together=True,
        raw_threshold_lists_not_evaluated=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'joint_selection_freeze.json', receipt)
    return dict(joint_sha256=sha(ROOT / 'joint_selection_freeze.json'), **receipt)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['scores', 'verify_scores', 'freeze', 'verify', 'joint', 'analyze'])
    p.add_argument('--fold', choices=['2024', 'recent', '2025', 'combined'], default='2024')
    args = p.parse_args()
    fold = '2025' if args.fold == 'combined' else args.fold
    setup(fold)
    if args.stage == 'joint':
        result = joint_freeze()
    elif args.stage == 'analyze':
        assert fold == '2025'
        receipt = ROOT / 'joint_selection_freeze.json'
        joint = json.loads(receipt.read_text())
        assert joint['passed'] and joint['protocol_sha256'] == sha(MASTER)
        committed = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'],
            capture_output=True, text=True, check=True).stdout
        assert sha(receipt) in committed
        for row in joint['selections']:
            path = Path(row['root'])
            assert row['selection_report_sha256'] == sha(path / 'selection_report.json')
            assert row['selection_verification_sha256'] == sha(path / 'selection_verification.json')
        result = evaluation.analyze(center.root(fold), Path('config') / (STEM + '_combined_protocol.json'))
    else:
        assert fold in ['2024', 'recent'], 'Annual freeze must follow the joint workflow'
        result = getattr(center, args.stage)(fold)
        if args.stage == 'verify_scores':
            result = verify_center_native(center)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
