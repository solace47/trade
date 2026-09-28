"""Jointly freeze and complete the two predefined coordinate-system studies."""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import pandas as pd

from trade_research import tail_formula_pca_axes as study
from trade_research.corporate_cash import save_json, sha
from analyze_tail_formula_input_study import run as analyze_study
from finish_tail_formula_input_study import finish as finish_study
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared


def freeze():
    study.checked_sources()
    output = study.ROOT / 'joint_selection_freeze.json'
    assert not output.exists()
    input_receipt = study.ROOT / 'joint_input_verification.json'
    assert json.loads(input_receipt.read_text())['passed']
    assert not any((Path('data/research') / (study.STEM+'_'+arm+'_'+fold) / 'analysis_report.json').exists()
                   for arm in ['axis', 'pca'] for fold in ['2024', 'recent', '2025'])
    # Require all four complete models and lists before creating either annual list.
    for arm in ['axis', 'pca']:
        for fold in ['2024', 'recent']:
            root = Path('data/research') / (study.STEM+'_'+arm+'_'+fold)
            for kind in ['model', 'score', 'selection']:
                r = json.loads((root / (kind+'_verification.json')).read_text())
                assert r['passed'] and r[kind+'_report_sha256'] == sha(root / (kind+'_report.json'))
    for arm in ['axis', 'pca']:
        stem = study.STEM+'_'+arm
        for stage in ['freeze', 'verify']:
            with (study.ROOT / (arm+'_combined_'+stage+'.log')).open('w') as log:
                subprocess.run(['.venv/bin/python', '-m', 'trade_research.'+stem+'_model', stage, '--fold', 'combined'], stdout=log, check=True)
    arms, all_checks = [], []
    for arm in ['axis', 'pca']:
        stem = study.STEM+'_'+arm
        child = Path('data/research') / stem
        child.mkdir(exist_ok=True)
        records, checks = [], []
        for fold in ['2024', 'recent', '2025']:
            root = Path('data/research') / (stem+'_'+fold)
            r = json.loads((root / 'selection_report.json').read_text())
            v = json.loads((root / 'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == sha(root / 'selection_report.json')
            assert r['selection_sha256'] == sha(root / 'selection.parquet')
            f = pd.read_parquet(root / 'selection.parquet')
            chosen = f.loc[f.selected]
            row = dict(root=str(root), fold=fold, selected=len(chosen), days=chosen.date.nunique(),
                selection_report_sha256=sha(root / 'selection_report.json'), selection_verification_sha256=sha(root / 'selection_verification.json'))
            if fold != '2025':
                m = json.loads((root / 'model_verification.json').read_text())
                row.update(model_report_sha256=sha(root / 'model_report.json'), node_checks=m['node_checks'])
            records.append(row)
            other = study.STEM+'_'+('pca' if arm == 'axis' else 'axis')
            controls = [Path('data/research') / ('tail_formula_before1000_model_'+fold),
                        Path('data/research/tail_formula_before1000/evaluation') / ('tail_formula_float_'+fold),
                        Path('data/research') / (other+'_'+fold)]
            for control in controls:
                if (control / 'selection.parquet').exists():
                    checks.append(dict(root=str(root), control=str(control),
                        full_frame_equal=f.equals(pd.read_parquet(control / 'selection.parquet'))))
        save_json(child / 'full_frame_precheck.json', dict(passed=True, checks=checks, new_group_outcomes_read=False))
        save_json(child / 'joint_selection_freeze.json', dict(passed=True,
            protocol_sha256=sha(Path('config') / (stem+'_protocol.json')), selections=records,
            all_three_selections_frozen_together=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True))
        arms.append(dict(arm=arm, root=str(child), joint_sha256=sha(child / 'joint_selection_freeze.json'),
                         full_frame_precheck_sha256=sha(child / 'full_frame_precheck.json'), selections=records))
        all_checks.extend(checks)
    save_json(output, dict(passed=True, protocol_sha256=sha(study.MASTER),
        joint_input_verification_sha256=sha(input_receipt), arms=arms, full_frame_checks=all_checks,
        all_four_models_and_six_lists_frozen_together=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True))
    return dict(joint_sha256=sha(output), arms=arms, full_frame_checks=all_checks)


def checked_joint():
    study.checked_sources()
    path = study.ROOT / 'joint_selection_freeze.json'
    r = json.loads(path.read_text())
    assert r['passed'] and r['protocol_sha256'] == sha(study.MASTER)
    assert r['joint_input_verification_sha256'] == sha(study.ROOT / 'joint_input_verification.json')
    committed = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'], capture_output=True, text=True, check=True).stdout
    assert sha(path) in committed
    for item in r['arms']:
        assert item['joint_sha256'] == sha(Path(item['root']) / 'joint_selection_freeze.json')
        assert item['joint_sha256'] in committed
    return r


def analyze():
    checked_joint()
    for arm in ['axis', 'pca']:
        analyze_study(study.STEM+'_'+arm, annual_only=True)
    return dict(passed=True, both_annual_reports_include_their_two_halves=True)


def finish():
    joint = checked_joint()
    for arm in ['axis', 'pca']:
        path = Path('data/research') / (study.STEM+'_'+arm) / 'complete_results_manifest.json'
        if not path.exists():
            finish_study(study.STEM+'_'+arm, [])
        else:
            r = json.loads(path.read_text())
            assert r['passed']
            for file, digest in r['reports'].items():
                assert sha(Path(file)) == digest
    left = Path('data/research') / (study.STEM+'_pca_2025')
    right = Path('data/research') / (study.STEM+'_axis_2025')
    protocol = json.loads(Path('config/tail_formula_weekday_shared_unknowns_protocol.json').read_text())
    spec = copy.deepcopy(protocol['comparisons'][0])
    spec.update(left=str(left), right=str(right), left_analysis_sha256=sha(left / 'analysis_report.json'),
        right_analysis_sha256=sha(right / 'analysis_report.json'), output=str(study.ROOT / 'shared_unknowns_pca_minus_axis.json'))
    protocol.update(objective='事前完整48维旋转与同预处理原轴的直接比较，保留全部各自与独有日期。', comparisons=[spec])
    protocol_path = Path('config') / (study.STEM+'_shared_unknowns_protocol.json')
    assert not protocol_path.exists()
    save_json(protocol_path, protocol)
    protocol['protocol_sha256'] = sha(protocol_path)
    paired = study.ROOT / 'same_dates_pca_minus_axis.json'
    compare(left, right, paired, intersection_only=True)
    shared(spec, protocol)
    output = study.ROOT / 'complete_results_manifest.json'
    assert not output.exists()
    reports = {str(path): sha(path) for path in [paired, Path(spec['output'])]}
    for arm in ['axis', 'pca']:
        path = Path('data/research') / (study.STEM+'_'+arm) / 'complete_results_manifest.json'
        reports[str(path)] = sha(path)
    save_json(output, dict(passed=True, protocol_sha256=sha(study.MASTER),
        joint_selection_freeze_sha256=sha(study.ROOT / 'joint_selection_freeze.json'), reports=reports,
        total_comparisons=5, both_arms_complete=True, annual_only=True, new_2026_prices_read=False, no_exit_rules=True))
    return dict(passed=True, completion_sha256=sha(output), total_comparisons=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'analyze', 'finish'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
