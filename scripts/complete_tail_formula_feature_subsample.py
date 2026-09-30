"""Complete the fixed comparisons with corrected historical-control key lookup.

The committed pre-outcome evaluator remains unchanged and hash-verifiable.
Only two mistaken dictionary key names in its completion stage are corrected.
"""
import json
from pathlib import Path
from trade_research.corporate_cash import save_json, sha
from evaluate_tail_formula_feature_subsample import (
    checked_joint, checked_analysis, equivalent, ROOT, study, historical, audit, compare, shared)

ERRATUM = Path('config/tail_formula_feature_subsample_completion_erratum.json')


def finish():
    joint = checked_joint(); master = study.checked(); reports = {}; refs = []; reused = []; done = []
    roots = {**{k: Path(v) for k, v in master['controls'].items()},
             **{i['group']: Path(i['root']) for i in joint['selections']}}
    for right, suffix in [('norm_full2025', 'corrected48'), ('fixed2025', 'fixed48')]:
        same_file = historical.OLD_ANNUAL / ('same_dates_' + suffix + '.json')
        shared_file = historical.OLD_ANNUAL / ('shared_unknowns_' + suffix + '.json')
        for path in [same_file, shared_file]:
            assert json.loads(path.read_text())['passed']; reports[str(path)] = sha(path)
        done.append(dict(year=2025, left=historical.OLD_ANNUAL, right=roots[right],
            same=same_file, shared=shared_file, historical=True))
    for item in joint['selections']:
        root = Path(item['root']); year = item['year']; checked_analysis(root, year)
        previous = [Path(v) for k, v in master['controls'].items() if k.endswith(str(year))]
        if year == 2025:
            previous.insert(0, historical.OLD_ANNUAL)
        previous += [Path(i['root']) for i in joint['selections'] if i['year'] == year and i['group'] != item['group']
                     and (Path(i['root']) / 'reference_coverage_verification.json').exists()]
        source = next((r for r in previous if equivalent(root, r, year)), root)
        file = source / 'reference_coverage_verification.json'
        if not file.exists():
            audit(source, years=(year,))
        r = json.loads(file.read_text()); assert r['passed']
        assert r['analysis_report_sha256'] == sha(source / 'analysis_report.json')
        reports[str(file)] = sha(file)
        refs.append(dict(group=item['group'], source=str(source), reused=source != root))
        for name in ['analysis_report.json', 'analysis_verification.json']:
            reports[str(root / name)] = sha(root / name)
    for year in [2024, 2025]:
        protocol = json.loads(Path('config/tail_formula_weekday_shared_unknowns_protocol.json').read_text())
        protocol.update(objective='随机变量候选两臂跨年完整比较；原数据、日期和未知全部保留。',
            periods=[str(year) + 'H1', str(year) + 'H2', str(year)])
        if year == 2024:
            protocol['signal_range'] = ['2024-01-01', '2024-12-31']
        pairs = [(a, b) for a, b in master['comparisons'] if a.endswith(str(year))]
        specs = [dict(left=str(roots[a]), right=str(roots[b]),
            left_analysis_sha256=sha(roots[a] / 'analysis_report.json'), right_analysis_sha256=sha(roots[b] / 'analysis_report.json'),
            output=str(ROOT / ('shared_unknowns_' + a + '_minus_' + b + '.json'))) for a, b in pairs]
        protocol['comparisons'] = specs
        file = Path('config') / (study.STEM + '_' + str(year) + '_shared_unknowns_protocol.json')
        if file.exists():
            assert json.loads(file.read_text()) == protocol
        else:
            save_json(file, protocol)
        protocol['protocol_sha256'] = sha(file)
        for (a, b), spec in zip(pairs, specs):
            left, right = roots[a], roots[b]
            match = next((d for d in done if d['year'] == year and equivalent(left, d['left'], year)
                           and equivalent(right, d['right'], year)), None)
            if match is not None:
                reused.append(dict(left=str(left), right=str(right), source_same_dates=str(match['same']),
                    source_shared=str(match['shared']), all_complete_frames_summaries_and_labels_equal=True,
                    numerical_results_recomputed=False))
                continue
            same_file = ROOT / ('same_dates_' + a + '_minus_' + b + '.json'); shared_file = Path(spec['output'])
            if not same_file.exists():
                compare(left, right, same_file, periods=protocol['periods'], intersection_only=True)
            if not shared_file.exists():
                shared(spec, protocol)
            for path in [same_file, shared_file]:
                assert json.loads(path.read_text())['passed']; reports[str(path)] = sha(path)
            done.append(dict(year=year, left=left, right=right, same=same_file, shared=shared_file))
    proof = ROOT / 'comparison_reuse_verification.json'
    save_json(proof, dict(passed=True, reference_sources=refs, comparisons=reused,
        logical_comparisons=10, numerical_comparisons=sum(not d.get('historical', False) for d in done),
        historical_comparisons_available=2, new_2026_prices_read=False, no_exit_rules=True))
    reports[str(proof)] = sha(proof)
    reports[str(ERRATUM)] = sha(ERRATUM)
    reports[str(Path(__file__))] = sha(Path(__file__))
    path = ROOT / 'complete_results_manifest.json'; assert not path.exists()
    save_json(path, dict(passed=True, protocol_sha256=sha(study.PROTOCOL),
        joint_sha256=sha(ROOT / 'joint_selection_freeze.json'), reports=reports,
        all_four_annual_groups_and_ten_comparisons_complete=True, year_2024_and_2025_are_exploratory=True,
        new_2026_prices_read=False, no_exit_rules=True))
    return dict(completion_sha256=sha(path), logical_comparisons=10,
        numerical_comparisons=sum(not d.get('historical', False) for d in done))


if __name__ == '__main__':
    p = json.loads(ERRATUM.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    print(json.dumps(finish(), ensure_ascii=False, indent=2))
