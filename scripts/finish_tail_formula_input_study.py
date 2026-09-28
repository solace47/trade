"""Complete annual input-study comparisons without repeating half-year analyses."""
import argparse
import copy
import json
from pathlib import Path
import re

from trade_research.corporate_cash import save_json, sha
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared
from audit_tail_formula_reference_coverage import audit


def finish(stem, matched_controls):
    assert re.fullmatch(r'tail_formula_[a-z0-9_]+', stem)
    root = Path('data/research') / stem
    annual = Path('data/research') / (stem + '_2025')
    master_path = Path('config') / (stem + '_protocol.json')
    master = json.loads(master_path.read_text())
    dispatch = json.loads((root / 'analysis_dispatch_verification.json').read_text())
    assert dispatch['passed'] and dispatch['annual_only']
    assert dispatch['joint_selection_freeze_sha256'] == sha(root / 'joint_selection_freeze.json')
    proof = json.loads((annual / 'analysis_verification.json').read_text())
    assert proof['passed'] and proof['analysis_report_sha256'] == sha(annual / 'analysis_report.json')
    protocol = json.loads(Path('config/tail_formula_weekday_shared_unknowns_protocol.json').read_text())
    protocol['objective'] = '固定' + stem + '的全年名单及两半年，与全部事前对照比较，保留未知、未买入及独有日期。'
    comparisons = []
    for text in matched_controls:
        name, path = text.split('=', 1)
        assert re.fullmatch(r'[a-z0-9_]+', name) and name not in ['corrected48', 'fixed48']
        right = Path(path)
        report = right / 'analysis_report.json'
        assert master['references'][str(report)] == sha(report), 'Additional control must be fixed in the master protocol'
        spec = copy.deepcopy(protocol['comparisons'][0])
        spec.update(right=str(right), right_analysis_sha256=sha(report))
        comparisons.append((name, spec))
    comparisons += list(zip(['corrected48', 'fixed48'], protocol['comparisons']))
    assert len({name for name, _ in comparisons}) == len(comparisons)
    for name, spec in comparisons:
        spec.update(left=str(annual), left_analysis_sha256=sha(annual / 'analysis_report.json'),
                    output=str(annual / ('shared_unknowns_' + name + '.json')))
    protocol['comparisons'] = [spec for _, spec in comparisons]
    path = Path('config') / (stem + '_shared_unknowns_protocol.json')
    assert not path.exists()
    save_json(path, protocol)
    protocol['protocol_sha256'] = sha(path)
    reports = {str(annual / name): sha(annual / name)
               for name in ['analysis_report.json', 'analysis_verification.json']}
    for name, spec in comparisons:
        paired = annual / ('same_dates_' + name + '.json')
        compare(annual, Path(spec['right']), paired, intersection_only=True)
        shared(spec, protocol)
        for report in [paired, Path(spec['output'])]:
            assert json.loads(report.read_text())['passed']
            reports[str(report)] = sha(report)
    audit(annual)
    reference = annual / 'reference_coverage_verification.json'
    assert json.loads(reference.read_text())['passed']
    reports[str(reference)] = sha(reference)
    path = root / 'complete_results_manifest.json'
    assert not path.exists()
    save_json(path, dict(passed=True, protocol_sha256=sha(master_path), reports=reports,
        comparison_names=[name for name, _ in comparisons], annual_only=True,
        new_2026_prices_read=False, no_exit_rules=True))
    print(json.dumps(dict(stem=stem, completion_sha256=sha(path)), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stem', required=True)
    parser.add_argument('--matched-control', action='append', default=[], metavar='NAME=ROOT')
    args = parser.parse_args()
    finish(args.stem, args.matched_control)
