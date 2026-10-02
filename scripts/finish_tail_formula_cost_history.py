"""Require improvement against the independently fitted same-domain control."""
import json
from pathlib import Path

import run_tail_formula_cost_history as study
from trade_research.research_io import save_json, sha
from tail_formula_reports import checked_analysis


def finish():
    study.finish()
    root = study.ROOT
    gate_path = root/'selection_gate.json'; gate = json.loads(gate_path.read_text())
    matched = []; above = []
    for year in ['2024','2025']:
        d = json.loads((root/('matched_shared_unknowns_'+year+'.json')).read_text())
        matched.append(next(s for s in d['summaries'] if s['period']==year and s['bps']==15 and not s['sensitive']))
        def main(group):
            _,report = checked_analysis(root/(group+year))
            return next(s for s in report['summaries'] if s['arm']=='formula' and s['period']==year and s['bps']==15 and not s['sensitive'])
        a,b = main('rule'),main('matched')
        above.append(a['rate'] is not None and a['bad3'] is not None and b['rate'] is not None and b['bad3'] is not None
                     and a['rate']>b['rate'] and a['bad3']<=b['bad3'])
    gate['criteria'].update(both_year_opportunity_above_matched_and_bad3_not_above=all(above),
        shared_unknown_lower_ci_both_years_above_matched=all(s['lower_ci'] is not None and s['lower_ci'][0]>0 for s in matched))
    gate['supports_further_validation'] = all(gate['criteria'].values())
    gate['matched_control_increment'] = matched; save_json(gate_path,gate)
    path = root/'complete_results_manifest.json'; manifest = json.loads(path.read_text())
    manifest['source_hashes'][str(gate_path)] = sha(gate_path); manifest['criteria'] = gate['criteria']
    save_json(path,manifest)
    return dict(complete_sha256=sha(path),criteria=manifest['criteria'],comparisons=len(manifest['comparisons']),
                fingerprints=len(manifest['source_hashes']),new_tree_fits=8)


if __name__ == '__main__':
    print(json.dumps(finish(),ensure_ascii=False))
