"""Freeze the exhaustive support partition, then evaluate both sides and controls."""
import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd

from trade_research import tail_formula_training_support as study
from trade_research.corporate_cash import save_json, sha


def freeze():
    study.checked()
    root = study.ROOT
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / 'joint_selection_freeze.json').exists()
    for fold in ['2024', 'recent']:
        part = json.loads((study.support_root(fold) / 'partition_verification.json').read_text())
        assert part['passed'] and part['mutually_exclusive_and_exhaustive']
        assert part['original_selection_sha256'] == sha(study.control(fold) / 'selection_report.json')
        for arm in ['supported', 'rejected']:
            assert part[arm + '_selection_sha256'] == sha(study.selection_root(arm, fold) / 'selection_report.json')
    records = []
    for arm in ['supported', 'rejected']:
        if not (study.selection_root(arm, '2025') / 'selection_report.json').exists():
            study.combined(arm)
        study.combined(arm, True)
        for fold in ['2024', 'recent', '2025']:
            target = study.selection_root(arm, fold)
            assert not (target / 'analysis_report.json').exists()
            r = json.loads((target / 'selection_report.json').read_text())
            v = json.loads((target / 'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == sha(target / 'selection_report.json')
            assert r['selection_sha256'] == sha(target / 'selection.parquet')
            selection = pd.read_parquet(target / 'selection.parquet')
            selected = selection.loc[selection.selected]
            candidates = [study.control(fold), study.DATA / 'tail_formula_before1000/evaluation' / ('tail_formula_float_' + fold)]
            candidates += [Path(record['root']) for record in records]
            same = [str(path) for path in candidates if (path / 'selection.parquet').exists()
                    and selection.equals(pd.read_parquet(path / 'selection.parquet'))]
            records.append(dict(arm=arm, fold=fold, root=str(target), selected=len(selected), days=selected.date.nunique(),
                selection_report_sha256=sha(target / 'selection_report.json'),
                selection_verification_sha256=sha(target / 'selection_verification.json'), identical_full_selection_sources=same))
    a = pd.read_parquet(study.selection_root('supported', '2025') / 'selection.parquet')
    b = pd.read_parquet(study.selection_root('rejected', '2025') / 'selection.parquet')
    original = pd.read_parquet(study.control('2025') / 'selection.parquet')
    assert not (a.selected & b.selected).any()
    combined = a.copy()
    combined['selected'] = a.selected | b.selected
    pd.testing.assert_frame_equal(combined, original, check_exact=True)
    save_json(root / 'joint_selection_freeze.json', dict(passed=True, protocol_sha256=sha(study.MASTER), selections=records,
        six_lists_frozen_together=True, mutually_exclusive_and_exhaustive=True, original_tree_models_unchanged=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True))
    return dict(joint_sha256=sha(root / 'joint_selection_freeze.json'), selections=records)


def analyze():
    # The generic six-record dispatcher checks full selection equality, reuses canonical
    # statistics when equal, and independently verifies every resulting report.
    import evaluate_tail_formula_polynomial_ridge as workflow
    workflow.ROOT = study.ROOT
    workflow.MASTER = study.MASTER
    workflow.STEM = study.STEM
    workflow.command = lambda arm, fold, stage: ['.venv/bin/python', '-m', 'trade_research.tail_formula_training_support',
                                               stage, '--arm', arm, '--fold', fold]
    return workflow.analyze()


def finish():
    from compare_tail_formula_same_dates import compare
    from compare_tail_formula_shared_unknowns import compare as shared
    from audit_tail_formula_reference_coverage import audit
    root = study.ROOT
    assert not (root / 'complete_results_manifest.json').exists()
    dispatch = json.loads((root / 'analysis_dispatch_verification.json').read_text())
    assert dispatch['passed'] and dispatch['joint_selection_freeze_sha256'] == sha(root / 'joint_selection_freeze.json')
    reports = {}
    for arm in ['supported', 'rejected']:
        for fold in ['2024', 'recent', '2025']:
            target = study.selection_root(arm, fold)
            proof = json.loads((target / 'analysis_verification.json').read_text())
            assert proof['passed'] and proof['analysis_report_sha256'] == sha(target / 'analysis_report.json')
            for name in ['analysis_report.json', 'analysis_verification.json']:
                reports[str(target / name)] = sha(target / name)
    a = study.selection_root('supported', '2025')
    b = study.selection_root('rejected', '2025')
    specs = []
    for left, right, name in [(a, study.control('2025'), 'corrected48'),
                              (a, study.DATA / 'tail_formula_before1000/evaluation/tail_formula_float_2025', 'fixed48'),
                              (a, b, 'rejected'), (b, study.control('2025'), 'corrected48')]:
        specs.append(dict(left=str(left), right=str(right), output=str(left / ('shared_unknowns_' + name + '.json')),
                          left_analysis_sha256=sha(left / 'analysis_report.json'), right_analysis_sha256=sha(right / 'analysis_report.json')))
    p = json.loads(Path('config/tail_formula_weekday_shared_unknowns_protocol.json').read_text())
    p.update(objective='支持范围保留组对原版、旧固定及过滤组；过滤组对原版，四组完整报告。', comparisons=specs)
    protocol = Path('config') / (study.STEM + '_shared_unknowns_protocol.json')
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
    for arm in ['supported', 'rejected']:
        target = study.selection_root(arm, '2025')
        audit(target)
        proof = target / 'reference_coverage_verification.json'
        assert json.loads(proof.read_text())['passed']
        reports[str(proof)] = sha(proof)
    save_json(root / 'complete_results_manifest.json', dict(passed=True, protocol_sha256=sha(study.MASTER), reports=reports,
        all_four_controls_included=True, new_2026_prices_read=False, no_exit_rules=True))
    return dict(complete_sha256=sha(root / 'complete_results_manifest.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'analyze', 'finish'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
