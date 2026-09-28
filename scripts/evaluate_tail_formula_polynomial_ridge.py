"""Jointly freeze six lists, then analyze distinct selections and all five controls."""
import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd

from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research.corporate_cash import save_json, sha
from trade_research.tail_formula_polynomial_ridge import MASTER, ROOT, STEM
from reuse_tail_formula_selected_analysis import reuse

DATA = Path('data/research')


def root_for(arm, fold):
    return DATA / f'{STEM}_{arm}_{fold}'


def command(arm, fold, stage):
    return ['.venv/bin/python', '-m', 'trade_research.tail_formula_polynomial_ridge',
            stage, '--arm', arm, '--fold', 'combined' if fold == '2025' else fold]


def run_log(cmd, path):
    with path.open('w') as out:
        subprocess.run(cmd, stdout=out, check=True)


def freeze():
    assert not (ROOT / 'joint_selection_freeze.json').exists()
    ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for arm in ['linear', 'quadratic']:
        combined = root_for(arm, '2025')
        combined.mkdir(parents=True, exist_ok=True)
        stages = ['verify'] if (combined / 'selection_report.json').exists() else ['freeze', 'verify']
        for stage in stages:
            run_log(command(arm, '2025', stage), combined / (stage + '_command.log'))
        for fold in ['2024', 'recent', '2025']:
            root = root_for(arm, fold)
            assert not (root / 'analysis_report.json').exists()
            r = json.loads((root / 'selection_report.json').read_text())
            v = json.loads((root / 'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == sha(root / 'selection_report.json')
            assert r['selection_sha256'] == sha(root / 'selection.parquet')
            selection = pd.read_parquet(root / 'selection.parquet')
            selected = selection.loc[selection.selected]
            controls = [DATA / ('tail_formula_before1000_model_' + fold),
                        DATA / 'tail_formula_before1000/evaluation' / ('tail_formula_float_' + fold)]
            if arm == 'quadratic':
                controls.insert(0, root_for('linear', fold))
            identical = []
            for control in controls:
                if (control / 'selection.parquet').exists() and selection.equals(pd.read_parquet(control / 'selection.parquet')):
                    identical.append(str(control))
            record = dict(arm=arm, fold=fold, root=str(root), selected=len(selected), days=selected.date.nunique(),
                          selection_report_sha256=sha(root / 'selection_report.json'),
                          selection_verification_sha256=sha(root / 'selection_verification.json'),
                          identical_full_selection_sources=identical)
            if fold != '2025':
                m = json.loads((root / 'model_report.json').read_text())
                proof = json.loads((root / 'model_verification.json').read_text())
                assert proof['passed'] and proof['model_report_sha256'] == sha(root / 'model_report.json')
                if arm == 'quadratic':
                    linear = json.loads((root_for('linear', fold) / 'model_report.json').read_text())
                    for key in ['input_means', 'input_scales', 'input_constant_indices', 'rows', 'days', 'last_observation']:
                        assert m[key] == linear[key]
                record.update(model_report_sha256=sha(root / 'model_report.json'), terms=len(m['terms']))
            records.append(record)
    save_json(ROOT / 'joint_selection_freeze.json', dict(passed=True, protocol_sha256=sha(MASTER), selections=records,
        all_four_models_and_six_lists_frozen_together=True, identical_input_transforms_between_arms=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True))
    return dict(joint_sha256=sha(ROOT / 'joint_selection_freeze.json'), selections=records)


def analyze():
    path = ROOT / 'joint_selection_freeze.json'
    joint = json.loads(path.read_text())
    assert joint['passed'] and joint['protocol_sha256'] == sha(MASTER) and len(joint['selections']) == 6
    assert not (ROOT / 'analysis_dispatch_verification.json').exists()
    records = []
    for frozen in joint['selections']:
        root = Path(frozen['root'])
        assert frozen['selection_report_sha256'] == sha(root / 'selection_report.json')
        assert frozen['selection_verification_sha256'] == sha(root / 'selection_verification.json')
        assert not (root / 'analysis_report.json').exists()
        protocol = Path('config') / f'{STEM}_{frozen["arm"]}_{"combined" if frozen["fold"] == "2025" else frozen["fold"]}_protocol.json'
        sources = frozen['identical_full_selection_sources']
        if sources:
            source = Path(sources[0])
            reuse(root, source)
            evaluation.attach_labels(root)
            assert sha(root / 'full_labels.parquet') == sha(source / 'full_labels.parquet')
            report = json.loads((source / 'analysis_report.json').read_text())
            assert report['reference_label'] == '09:59'
            report.update(protocol_sha256=sha(protocol), selection_report_sha256=sha(root / 'selection_report.json'),
                          label_report_sha256=sha(root / 'full_label_report.json'),
                          reused_analysis_report_sha256=sha(source / 'analysis_report.json'))
            (root / 'daily_summary.parquet').symlink_to((source / 'daily_summary.parquet').resolve())
            save_json(root / 'analysis_report.json', report)
        else:
            run_log(command(frozen['arm'], frozen['fold'], 'analyze'), root / 'analysis_command.log')
        run_log(['.venv/bin/python', 'scripts/verify_tail_formula_before1000.py', 'analysis', '--root', str(root)],
                root / 'analysis_check_command.log')
        proof = json.loads((root / 'analysis_verification.json').read_text())
        assert proof['passed'] and proof['analysis_report_sha256'] == sha(root / 'analysis_report.json')
        report = json.loads((root / 'analysis_report.json').read_text())
        if sources:
            assert report['summaries'] == json.loads((Path(sources[0]) / 'analysis_report.json').read_text())['summaries']
        record = dict(root=str(root), reused_source=sources[0] if sources else None,
                      analysis_report_sha256=sha(root / 'analysis_report.json'))
        records.append(record)
        primary = next(x for x in report['summaries'] if x['arm'] == 'formula' and x['bps'] == 15
                       and not x['sensitive'] and x['period'] == '2025')
        print(json.dumps(dict(**record, primary=primary), ensure_ascii=False), flush=True)
    out = dict(passed=True, joint_selection_freeze_sha256=sha(path), records=records,
               full_selection_equality_checked_before_evaluation=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'analysis_dispatch_verification.json', out)
    return out


def finish():
    from compare_tail_formula_same_dates import compare
    from compare_tail_formula_shared_unknowns import compare as shared
    from audit_tail_formula_reference_coverage import audit
    assert not (ROOT / 'complete_results_manifest.json').exists()
    dispatch = json.loads((ROOT / 'analysis_dispatch_verification.json').read_text())
    assert dispatch['passed'] and dispatch['joint_selection_freeze_sha256'] == sha(ROOT / 'joint_selection_freeze.json')
    reports = {}
    specs = []
    for arm in ['linear', 'quadratic']:
        for fold in ['2024', 'recent', '2025']:
            root = root_for(arm, fold)
            proof = json.loads((root / 'analysis_verification.json').read_text())
            assert proof['passed'] and proof['analysis_report_sha256'] == sha(root / 'analysis_report.json')
            for name in ['analysis_report.json', 'analysis_verification.json']:
                reports[str(root / name)] = sha(root / name)
        root = root_for(arm, '2025')
        controls = [('corrected48', DATA / 'tail_formula_before1000_model_2025'),
                    ('fixed48', DATA / 'tail_formula_before1000/evaluation/tail_formula_float_2025')]
        if arm == 'quadratic':
            controls.insert(0, ('matched_linear', root_for('linear', '2025')))
        for name, right in controls:
            specs.append(dict(left=str(root), right=str(right), output=str(root / ('shared_unknowns_' + name + '.json')),
                              left_analysis_sha256=sha(root / 'analysis_report.json'),
                              right_analysis_sha256=sha(right / 'analysis_report.json')))
    p = json.loads(Path('config/tail_formula_weekday_shared_unknowns_protocol.json').read_text())
    p.update(objective='固定完整二阶相对同预处理线性，两臂各对两套48树，五组比较全部报告。', comparisons=specs)
    protocol = Path('config') / (STEM + '_shared_unknowns_protocol.json')
    assert not protocol.exists()
    save_json(protocol, p)
    p['protocol_sha256'] = sha(protocol)
    for spec in specs:
        left = Path(spec['left'])
        right = Path(spec['right'])
        target = Path(spec['output'])
        name = target.stem.removeprefix('shared_unknowns_')
        other = left / ('same_dates_' + name + '.json')
        compare(left, right, other, intersection_only=True)
        shared(spec, p)
        for file in [target, other]:
            assert json.loads(file.read_text())['passed']
            reports[str(file)] = sha(file)
    for arm in ['linear', 'quadratic']:
        root = root_for(arm, '2025')
        audit(root)
        proof = root / 'reference_coverage_verification.json'
        assert json.loads(proof.read_text())['passed']
        reports[str(proof)] = sha(proof)
    save_json(ROOT / 'complete_results_manifest.json', dict(passed=True, protocol_sha256=sha(MASTER), reports=reports,
        all_five_controls_included=True, new_2026_prices_read=False, no_exit_rules=True))
    return dict(complete_sha256=sha(ROOT / 'complete_results_manifest.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'analyze', 'finish'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
