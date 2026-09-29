"""Jointly freeze and complete the dual-unit and same-dimension constant studies."""
import argparse
import copy
import json
from pathlib import Path
import subprocess

import pandas as pd

from trade_research import tail_formula_dual_units as study
from trade_research.corporate_cash import save_json, sha
from analyze_tail_formula_input_study import run as analyze_study
from finish_tail_formula_input_study import finish as finish_study
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared
from reuse_tail_formula_input_comparisons import finish as reuse_comparisons, checked as checked_analysis
from freeze_tail_formula_bipower_gap import checked_selection


def freeze():
    study.checked_sources()
    output = study.ROOT / 'joint_selection_freeze.json'
    assert not output.exists()
    input_receipt = study.ROOT / 'native_input_verification.json'
    assert json.loads(input_receipt.read_text())['passed']
    assert not any((Path('data/research') / (study.STEM+'_'+arm+'_'+fold) / 'analysis_report.json').exists()
                   for arm in ['constant', 'dual'] for fold in ['2024', 'recent', '2025'])
    # Require all four complete models and lists before creating either annual list.
    for arm in ['constant', 'dual']:
        for fold in ['2024', 'recent']:
            root = Path('data/research') / (study.STEM+'_'+arm+'_'+fold)
            for kind in ['model', 'score', 'selection']:
                r = json.loads((root / (kind+'_verification.json')).read_text())
                assert r['passed'] and r[kind+'_report_sha256'] == sha(root / (kind+'_report.json'))
    for arm in ['constant', 'dual']:
        stem = study.STEM+'_'+arm
        for stage in ['freeze', 'verify']:
            with (study.ROOT / (arm+'_combined_'+stage+'.log')).open('w') as log:
                subprocess.run(['.venv/bin/python', '-m', 'trade_research.'+stem+'_model', stage, '--fold', 'combined'], stdout=log, check=True)
    arms, all_checks = [], []
    for arm in ['constant', 'dual']:
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
                model = json.loads((root / 'model_report.json').read_text())
                expected = [*study.previous.EXPRESSIONS, *study.ARM_FIELDS[arm]]
                assert model['feature_names'] == expected and len(expected) == 67
                if arm == 'constant':
                    assert m['nineteen_constant_columns_unused_by_all_nodes']
                row.update(model_report_sha256=sha(root / 'model_report.json'),
                    model_verification_sha256=sha(root / 'model_verification.json'),
                    score_report_sha256=sha(root / 'score_report.json'),
                    score_verification_sha256=sha(root / 'score_verification.json'),
                    numeric_core_sha256=sha(root / 'frozen_numeric_core.tdx'),
                    added_unit_node_uses=sum(i >= 48 for t in model['trees'] for i in t['feature']),
                    node_checks=m['node_checks'])
            records.append(row)
            other = study.STEM+'_'+('dual' if arm == 'constant' else 'constant')
            controls = [Path('data/research') / ('tail_formula_before1000_model_'+fold),
                        Path('data/research/tail_formula_before1000/evaluation') / ('tail_formula_float_'+fold),
                        Path('data/research') / (other+'_'+fold),
                        Path('data/research') / ('tail_formula_unscaled_prices_'+fold)]
            for control in controls:
                if (control / 'selection.parquet').exists():
                    checks.append(dict(root=str(root), control=str(control),
                        selection_report_sha256=sha(root / 'selection_report.json'),
                        control_selection_report_sha256=sha(control / 'selection_report.json'),
                        full_frame_equal=f.equals(checked_selection(control))))
        save_json(child / 'full_frame_precheck.json', dict(passed=True, checks=checks, new_group_outcomes_read=False))
        save_json(child / 'joint_selection_freeze.json', dict(passed=True,
            protocol_sha256=sha(Path('config') / (stem+'_protocol.json')), selections=records,
            all_three_selections_frozen_together=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True))
        arms.append(dict(arm=arm, root=str(child), joint_sha256=sha(child / 'joint_selection_freeze.json'),
                         full_frame_precheck_sha256=sha(child / 'full_frame_precheck.json'), selections=records))
        all_checks.extend(checks)
    save_json(output, dict(passed=True, protocol_sha256=sha(study.PROTOCOL),
        native_input_verification_sha256=sha(input_receipt), arms=arms, selections=[row for arm in arms for row in arm['selections']], full_frame_checks=all_checks,
        all_four_models_and_six_lists_frozen_together=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True))
    return dict(joint_sha256=sha(output), arms=arms, full_frame_checks=all_checks)


def checked_joint():
    study.checked_sources()
    path = study.ROOT / 'joint_selection_freeze.json'
    r = json.loads(path.read_text())
    assert r['passed'] and r['protocol_sha256'] == sha(study.PROTOCOL)
    assert r['native_input_verification_sha256'] == sha(study.ROOT / 'native_input_verification.json')
    committed = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'], capture_output=True, text=True, check=True).stdout
    assert sha(path) in committed
    for item in r['arms']:
        assert item['joint_sha256'] == sha(Path(item['root']) / 'joint_selection_freeze.json')
        assert item['joint_sha256'] in committed
    return r


def analyze():
    checked_joint()
    for arm in ['constant', 'dual']:
        analyze_study(study.STEM+'_'+arm, annual_only=True)
    return dict(passed=True, both_annual_reports_include_their_two_halves=True)


def reuse_constant_comparison(left, right):
    """Do not recompute dual versus canonical when the constant arm is canonical."""
    canonical = Path('data/research/tail_formula_before1000_model_2025')
    b, br = checked_analysis(right); c, cr = checked_analysis(canonical)
    if not b.equals(c):
        return None
    assert br['summaries'] == cr['summaries'] and br['daily_summary_sha256'] == cr['daily_summary_sha256']
    assert br['label_report_sha256'] == cr['label_report_sha256']
    source_left = left
    if not (left / 'same_dates_corrected48.json').exists():
        proof = json.loads((left.parent / (study.STEM + '_dual') / 'comparison_reuse_verification.json').read_text())
        assert proof['passed'] and proof['target'] == str(left)
        source_left = Path(proof['source'])
    a, ar = checked_analysis(left); s, sr = checked_analysis(source_left)
    pd.testing.assert_frame_equal(a, s, check_exact=True)
    assert ar['summaries'] == sr['summaries'] and ar['daily_summary_sha256'] == sr['daily_summary_sha256']
    assert ar['label_report_sha256'] == sr['label_report_sha256']
    paired = source_left / 'same_dates_corrected48.json'
    shared_path = source_left / 'shared_unknowns_corrected48.json'
    pr = json.loads(paired.read_text()); ur = json.loads(shared_path.read_text())
    assert pr['passed'] and ur['passed']
    assert pr['inputs']['left']['root'] == str(source_left) and pr['inputs']['right']['root'] == str(canonical)
    assert pr['inputs']['left']['analysis_report_sha256'] == sha(source_left / 'analysis_report.json')
    assert pr['inputs']['right']['analysis_report_sha256'] == sha(canonical / 'analysis_report.json')
    assert ur['comparison']['left'] == str(source_left) and ur['comparison']['right'] == str(canonical)
    assert ur['comparison']['left_analysis_sha256'] == sha(source_left / 'analysis_report.json')
    assert ur['comparison']['right_analysis_sha256'] == sha(canonical / 'analysis_report.json')
    assert ur['labels_sha256'] == sha(left / 'full_labels.parquet') == sha(right / 'full_labels.parquet')
    output = study.ROOT / 'constant_comparison_reuse_verification.json'
    assert not output.exists()
    save_json(output, dict(passed=True, left=str(left), right=str(right), source_left=str(source_left),
        source_right=str(canonical), source_same_dates_report=str(paired), source_shared_unknowns_report=str(shared_path),
        left_analysis_report_sha256=sha(left / 'analysis_report.json'),
        right_analysis_report_sha256=sha(right / 'analysis_report.json'),
        source_same_dates_sha256=sha(paired), source_shared_unknowns_sha256=sha(shared_path),
        both_complete_selection_tables_daily_summaries_and_labels_equal=True,
        same_dates_summaries=pr['summaries'], shared_unknowns_summaries=ur['summaries'],
        numerical_results_recomputed=False, new_2026_prices_read=False, no_exit_rules=True))
    return {str(path): sha(path) for path in [output, paired, shared_path]}


def finish():
    joint = checked_joint()
    for arm in ['constant', 'dual']:
        path = Path('data/research') / (study.STEM+'_'+arm) / 'complete_results_manifest.json'
        if not path.exists():
            candidate = Path('data/research') / (study.STEM+'_'+arm+'_2025')
            reusable = Path('data/research/tail_formula_exchange_context_constant_2025')
            if checked_selection(candidate).equals(checked_selection(reusable)):
                reuse_comparisons(study.STEM+'_'+arm, reusable)
            else:
                finish_study(study.STEM+'_'+arm, [])
        else:
            r = json.loads(path.read_text())
            assert r['passed']
            for file, digest in r['reports'].items():
                assert sha(Path(file)) == digest
    left = Path('data/research') / (study.STEM+'_dual_2025')
    protocol = json.loads(Path('config/tail_formula_weekday_shared_unknowns_protocol.json').read_text())
    comparisons = []
    for label, right in [('constant', Path('data/research') / (study.STEM+'_constant_2025')),
                         ('unscaled48', Path('data/research/tail_formula_unscaled_prices_2025'))]:
        if label == 'unscaled48':
            master = json.loads(study.PROTOCOL.read_text())
            assert master['references'][str(right / 'analysis_report.json')] == sha(right / 'analysis_report.json')
        spec = copy.deepcopy(protocol['comparisons'][0])
        spec.update(left=str(left), right=str(right), left_analysis_sha256=sha(left / 'analysis_report.json'),
            right_analysis_sha256=sha(right / 'analysis_report.json'), output=str(study.ROOT / ('shared_unknowns_dual_minus_'+label+'.json')))
        comparisons.append((label, spec))
    protocol.update(objective='同一归一化48基础上追加19原百分比，对67维常量控制和直接替换19项的48版；保留完整日期与未知。',
                    comparisons=[s for _, s in comparisons])
    protocol_path = Path('config') / (study.STEM+'_shared_unknowns_protocol.json')
    assert not protocol_path.exists()
    save_json(protocol_path, protocol); protocol['protocol_sha256'] = sha(protocol_path)
    reports = {}
    for label, spec in comparisons:
        if label == 'constant':
            reused = reuse_constant_comparison(left, Path(spec['right']))
            if reused is not None:
                reports.update(reused)
                continue
        paired = study.ROOT / ('same_dates_dual_minus_'+label+'.json')
        compare(left, Path(spec['right']), paired, intersection_only=True)
        shared(spec, protocol)
        for path in [paired, Path(spec['output'])]:
            assert json.loads(path.read_text())['passed']
            reports[str(path)] = sha(path)
    output = study.ROOT / 'complete_results_manifest.json'
    assert not output.exists()
    for arm in ['constant', 'dual']:
        path = Path('data/research') / (study.STEM+'_'+arm) / 'complete_results_manifest.json'
        reports[str(path)] = sha(path)
    save_json(output, dict(passed=True, protocol_sha256=sha(study.PROTOCOL),
        joint_selection_freeze_sha256=sha(study.ROOT / 'joint_selection_freeze.json'), reports=reports,
        total_comparisons=6, both_arms_complete=True, annual_only=True, new_2026_prices_read=False, no_exit_rules=True))
    return dict(passed=True, completion_sha256=sha(output), total_comparisons=6)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'analyze', 'finish'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
