"""Evaluate profit-directed paths against three frozen controls, with exact reuse."""
import argparse
import json
from pathlib import Path
import subprocess

import run_tail_formula_profit_rule_search as fit
import finish_tail_formula_rule_search as shared
from trade_research.research_io import check_sources, save_json, sha


ROOT = fit.ROOT
EXECUTION = Path('config/tail_formula_profit_rule_search_evaluation.json')
shared.fit, shared.ROOT, shared.EXECUTION = fit, ROOT, EXECUTION


def checked():
    p, _ = fit.checked()
    e = json.loads(EXECUTION.read_text())
    assert e['input_protocol_sha256'] == sha(fit.PROTOCOL)
    check_sources(e['source_hashes'])
    assert subprocess.check_output(['git','show',f'HEAD:{EXECUTION}']) == EXECUTION.read_bytes()
    registry = json.loads(Path(e['prior_registry']).read_text())
    assert len(registry['prior_analysis_roots']) == e['prior_registry_analysis_count']
    assert len(registry['prior_completion_manifests']) == e['prior_registry_manifest_count']
    e['prior_analysis_roots'] = registry['prior_analysis_roots'] + e['additional_prior_analysis_roots']
    e['prior_completion_manifests'] = dict(registry['prior_completion_manifests'], **e['additional_prior_completion_manifests'])
    e['controls'] = registry['controls']
    completed = json.loads((ROOT/'all_models_verified.json').read_text())
    assert completed['passed'] and len(completed['folds']) == 4
    for fold in completed['folds']:
        path = ROOT/fold['fold']
        assert fold['model_report_sha256'] == sha(path/'model_report.json')
        report = json.loads((path/'model_report.json').read_text())
        verification = json.loads((path/'model_verification.json').read_text())
        assert verification['passed'] and verification['model_report_sha256'] == sha(path/'model_report.json')
        assert report['search_trace_sha256'] == sha(path/'search_trace.json')
    return p, e


shared.checked = checked


def finish():
    _, execution, _ = shared.checked_joint()
    destination = ROOT/'complete_results_manifest.json'
    assert not destination.exists()
    old_pairs, old_cache = shared.prior_pairs(execution), {}
    receipts, pairs = {}, []
    def same(expected, path):
        if path not in old_cache:
            sr = json.loads((path/'selection_report.json').read_text())
            ar = json.loads((path/'analysis_report.json').read_text())
            if sr['selected'] != int(expected[0].selected.sum()) or ar.get('reference_label') != '09:59':
                return False
            old_cache[path] = shared.checked_analysis(path)
        frame, report = old_cache[path]
        return expected[0].equals(frame) and all(expected[1][key] == report[key]
            for key in ['summaries','daily_summary_sha256','label_report_sha256'])
    for arm, controls in execution['additional_controls'].items():
        for year in ['2024','2025']:
            left, right = ROOT/('rule'+year), Path(controls[year])
            current, prior = shared.checked_analysis(left), shared.checked_analysis(right)
            periods = [year+'H1',year+'H2',year]
            old = next((item for item in old_pairs if item['periods'] == set(periods)
                and same(current,item['roots'][0]) and same(prior,item['roots'][1])),None)
            files = old['outputs'] if old else [str(ROOT/(arm+'_same_dates_'+year+'.json')),
                                                str(ROOT/(arm+'_shared_unknowns_'+year+'.json'))]
            if old:
                receipts[old['manifest']] = old['manifest_sha256']
            else:
                shared.compare(left,right,Path(files[0]),periods,intersection_only=True)
                shared.shared_compare(dict(left=str(left),right=str(right),output=files[1],
                    left_analysis_sha256=sha(left/'analysis_report.json'),right_analysis_sha256=sha(right/'analysis_report.json')),
                    dict(protocol_sha256=sha(EXECUTION),labels_sha256=sha(shared.evaluation.source.ROOT/'full_labels.parquet'),periods=periods))
            for name in files:
                assert json.loads(Path(name).read_text())['passed']
                receipts[name] = sha(Path(name))
            for name in ['selection.parquet','selection_report.json','selection_verification.json','analysis_report.json',
                         'analysis_verification.json','daily_summary.parquet','full_label_report.json','full_label_verification.json']:
                receipts[str(right/name)] = sha(right/name)
            pairs.append(dict(left='rule'+year,right=arm+year,outputs=files,reused_pair=old is not None))
            print(json.dumps(dict(additional_comparison=arm+year,reused=old is not None)),flush=True)
    shared.finish()
    manifest = json.loads(destination.read_text())
    manifest['source_hashes'].update(receipts)
    manifest['comparisons'].extend(pairs)
    manifest.update(selector_fits=4,new_tree_fits=0,new_input_features=0,new_atomic_conditions=0,
                    complete_two_year_four_half_comparisons_with_all_three_controls=True)
    check_sources(manifest['source_hashes'])
    save_json(destination,manifest)
    return dict(complete_sha256=sha(destination),criteria=manifest['criteria'],
                fingerprint_count=len(manifest['source_hashes']),comparisons=len(manifest['comparisons']))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','analyze','finish'])
    stage=parser.parse_args().stage
    print(json.dumps(finish() if stage=='finish' else getattr(shared,stage)(),ensure_ascii=False))
