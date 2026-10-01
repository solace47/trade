"""Evaluate the frozen-bank profit guard with both unchanged controls."""
import argparse
import json
from pathlib import Path

import run_tail_formula_rule_reference_gate as fit
import finish_tail_formula_rule_search as shared
from trade_research.research_io import check_sources, save_json, sha


ROOT = fit.ROOT
EXECUTION = Path('config/tail_formula_rule_reference_gate_evaluation.json')
shared.fit = fit
shared.ROOT = ROOT
shared.EXECUTION = EXECUTION


def finish():
    _, execution, joint = shared.checked_joint()
    destination = ROOT / 'complete_results_manifest.json'
    assert not destination.exists(), 'Never replace a completed comparison'
    prior_pairs = shared.prior_pairs(execution)
    receipts, pairs = {}, []
    old_cache = {}
    for year in ['2024', '2025']:
        left = ROOT / ('rule' + year)
        right = Path(execution['prior_rule_controls'][year])
        current, prior = shared.checked_analysis(left), shared.checked_analysis(right)
        periods = [year + 'H1', year + 'H2', year]

        def same(expected, path):
            if path not in old_cache:
                sr = json.loads((path / 'selection_report.json').read_text())
                ar = json.loads((path / 'analysis_report.json').read_text())
                if sr['selected'] != int(expected[0].selected.sum()) or ar.get('reference_label') != '09:59':
                    return False
                old_cache[path] = shared.checked_analysis(path)
            frame, report = old_cache[path]
            return expected[0].equals(frame) and all(expected[1][key] == report[key]
                for key in ['summaries', 'daily_summary_sha256', 'label_report_sha256'])

        old = next((item for item in prior_pairs if item['periods'] == set(periods)
            and same(current, item['roots'][0]) and same(prior, item['roots'][1])), None)
        outputs = old['outputs'] if old else [str(ROOT / ('prior_rule_same_dates_' + year + '.json')),
                                              str(ROOT / ('prior_rule_shared_unknowns_' + year + '.json'))]
        if old:
            receipts[old['manifest']] = old['manifest_sha256']
        else:
            shared.compare(left, right, Path(outputs[0]), periods, intersection_only=True)
            shared.shared_compare(dict(left=str(left), right=str(right), output=outputs[1],
                left_analysis_sha256=sha(left / 'analysis_report.json'),
                right_analysis_sha256=sha(right / 'analysis_report.json')),
                dict(protocol_sha256=sha(EXECUTION),
                    labels_sha256=sha(shared.evaluation.source.ROOT / 'full_labels.parquet'), periods=periods))
        for name in outputs:
            assert json.loads(Path(name).read_text())['passed']
            receipts[name] = sha(Path(name))
        for name in ['selection.parquet', 'selection_report.json', 'selection_verification.json',
                     'analysis_report.json', 'analysis_verification.json', 'daily_summary.parquet',
                     'full_label_report.json', 'full_label_verification.json']:
            receipts[str(right / name)] = sha(right / name)
        pairs.append(dict(left='rule' + year, right='prior_rule' + year, outputs=outputs,
                          reused_pair=old is not None))
        print(json.dumps(dict(prior_rule_comparison_completed=year, reused=old is not None)), flush=True)
    shared.finish()
    manifest = json.loads(destination.read_text())
    manifest['source_hashes'].update(receipts)
    manifest['comparisons'].extend(pairs)
    manifest.update(selector_fits=4, new_tree_fits=0, new_candidate_conditions=0,
                    original_and_prior_rule_controls_unchanged=True,
                    complete_two_year_and_four_half_comparisons_with_both_controls=True)
    check_sources(manifest['source_hashes'])
    save_json(destination, manifest)
    return dict(complete_sha256=sha(destination), criteria=manifest['criteria'],
                fingerprint_count=len(manifest['source_hashes']), comparisons=len(manifest['comparisons']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'analyze', 'finish'])
    stage = parser.parse_args().stage
    print(json.dumps(finish() if stage == 'finish' else getattr(shared, stage)(), ensure_ascii=False))
