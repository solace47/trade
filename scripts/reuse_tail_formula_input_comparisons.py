"""Reuse verified pair comparisons only after exact full-list equality."""
import argparse
import json
from pathlib import Path
import re
import subprocess

import pandas as pd

from trade_research.corporate_cash import save_json, sha


def checked(root):
    for stage in ['selection', 'analysis']:
        report = json.loads((root / (stage + '_report.json')).read_text())
        proof = json.loads((root / (stage + '_verification.json')).read_text())
        assert proof['passed'] and proof[stage + '_report_sha256'] == sha(root / (stage + '_report.json'))
    selection = json.loads((root / 'selection_report.json').read_text())
    assert selection['selection_sha256'] == sha(root / 'selection.parquet')
    assert report['selection_report_sha256'] == sha(root / 'selection_report.json')
    assert report['daily_summary_sha256'] == sha(root / 'daily_summary.parquet')
    assert report['reference_label'] == '09:59'
    label = json.loads((root / 'full_label_report.json').read_text())
    proof = json.loads((root / 'full_label_verification.json').read_text())
    assert proof['passed'] and proof['label_report_sha256'] == sha(root / 'full_label_report.json')
    assert report['label_report_sha256'] == sha(root / 'full_label_report.json')
    assert label['labels_sha256'] == sha(root / 'full_labels.parquet')
    frame = pd.read_parquet(root / 'selection.parquet')
    assert frame.loc[frame.selected, 'date'].str.startswith('2025').all()
    return frame, report


def finish(stem, source):
    assert re.fullmatch(r'tail_formula_[a-z0-9_]+', stem)
    root = Path('data/research') / stem; target = root.parent / (stem + '_2025')
    assert source.resolve() != target.resolve()
    protocol = Path('config') / (stem + '_protocol.json')
    joint = root / 'joint_selection_freeze.json'; freeze = json.loads(joint.read_text())
    assert freeze['passed'] and freeze['protocol_sha256'] == sha(protocol)
    committed = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'], capture_output=True, text=True, check=True).stdout
    assert sha(joint) in committed
    dispatch = json.loads((root / 'analysis_dispatch_verification.json').read_text())
    assert dispatch['passed'] and dispatch['annual_only'] and dispatch['joint_selection_freeze_sha256'] == sha(joint)
    a, ar = checked(target); b, br = checked(source)
    pd.testing.assert_frame_equal(a, b, check_exact=True)
    assert ar['summaries'] == br['summaries']
    assert ar['daily_summary_sha256'] == br['daily_summary_sha256']
    assert ar['label_report_sha256'] == br['label_report_sha256']
    assert sha(target / 'full_labels.parquet') == sha(source / 'full_labels.parquet')
    reports = {str(target / file): sha(target / file) for file in ['analysis_report.json', 'analysis_verification.json']}
    records = []
    controls = dict(corrected48=Path('data/research/tail_formula_before1000_model_2025'),
                    fixed48=Path('data/research/tail_formula_before1000/evaluation/tail_formula_float_2025'))
    for name, right in controls.items():
        checked(right)
        paired = source / ('same_dates_' + name + '.json')
        shared = source / ('shared_unknowns_' + name + '.json')
        pr = json.loads(paired.read_text()); sr = json.loads(shared.read_text())
        assert pr['passed'] and pr['intersection_only'] and sr['passed']
        assert pr['inputs']['left']['root'] == str(source) and pr['inputs']['right']['root'] == str(right)
        assert pr['inputs']['left']['analysis_report_sha256'] == sha(source / 'analysis_report.json')
        assert pr['inputs']['right']['analysis_report_sha256'] == sha(right / 'analysis_report.json')
        assert sr['comparison']['left'] == str(source) and sr['comparison']['right'] == str(right)
        assert sr['comparison']['left_analysis_sha256'] == sha(source / 'analysis_report.json')
        assert sr['comparison']['right_analysis_sha256'] == sha(right / 'analysis_report.json')
        assert sr['labels_sha256'] == sha(target / 'full_labels.parquet') == sha(right / 'full_labels.parquet')
        for file in [paired, shared]:
            reports[str(file)] = sha(file)
        records.append(dict(comparison=name, target_left=str(target), source_left=str(source), unchanged_right=str(right),
            source_same_dates_report=str(paired), source_shared_unknowns_report=str(shared),
            left_all_selection_fields_daily_summaries_and_labels_identical=True,
            verified_numerical_results_reused_without_new_comparison=True))
    reference = source / 'reference_coverage_verification.json'; rv = json.loads(reference.read_text())
    assert rv['passed'] and rv['analysis_report_sha256'] == sha(source / 'analysis_report.json')
    assert rv['selection_report_sha256'] == sha(source / 'selection_report.json')
    assert rv['label_report_sha256'] == sha(target / 'full_label_report.json')
    reports[str(reference)] = sha(reference)
    proof = root / 'comparison_reuse_verification.json'; assert not proof.exists()
    save_json(proof, dict(passed=True, verifier_sha256=sha(Path(__file__)), target=str(target), source=str(source),
        target_selection_report_sha256=sha(target / 'selection_report.json'),
        source_selection_report_sha256=sha(source / 'selection_report.json'), records=records,
        reference_coverage_reused=str(reference), numerical_results_recomputed=False,
        new_2026_prices_read=False, no_exit_rules=True))
    reports[str(proof)] = sha(proof)
    complete = root / 'complete_results_manifest.json'; assert not complete.exists()
    save_json(complete, dict(passed=True, protocol_sha256=sha(protocol), joint_sha256=sha(joint), reports=reports,
        comparison_names=list(controls), annual_only=True, identical_full_list_control_comparisons_reused=True,
        new_2026_prices_read=False, no_exit_rules=True))
    return dict(completion_sha256=sha(complete), comparisons_reused_from=str(source))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stem', required=True); p.add_argument('--source', required=True, type=Path)
    a = p.parse_args(); print(json.dumps(finish(a.stem, a.source), ensure_ascii=False, indent=2))
