"""One fixed recurrence condition; no fitting, search, or threshold adjustment."""
import argparse
import json
from functools import lru_cache
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as numeric
from trade_research.research_io import check_runtime, check_sources, save_json, sha
from tail_formula_reports import checked_selection, checked_analysis
import finish_tail_formula_rule_search as shared
import finish_tail_formula_profit_rule_search as comparisons

ROOT = Path('data/research/tail_formula_cost_history_literal')
PROTOCOL = Path('config/tail_formula_cost_history_literal_protocol.json')
KEYS = shared.KEYS


@lru_cache(maxsize=1)
def checked():
    check_runtime()
    assert subprocess.check_output(['git', 'show', f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    assert sha(PROTOCOL) in subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    p = json.loads(PROTOCOL.read_text()); check_sources(p['source_hashes'])
    assert p['selector_fits'] == p['new_tree_fits'] == 0
    assert p['condition'] == 'CM01>0 AND CM02>50'
    assert not p['new_2026_prices_allowed']
    registry = json.loads(Path(p['prior_registry']).read_text())
    e = dict(p, input_protocol_sha256=sha(PROTOCOL),
        prior_analysis_roots=registry['prior_analysis_roots'] + p['additional_prior_analysis_roots'],
        prior_completion_manifests=dict(registry['prior_completion_manifests'],
                                       **p['additional_prior_completion_manifests']))
    return p, e


def freeze():
    p, e = checked(); destination = ROOT/'joint_selection_freeze.json'
    assert not destination.exists()
    f = pd.read_parquet(p['features'], columns=[*KEYS, 'formula_input_valid', 'V01', 'CM01', 'CM02'])
    assert f.date.lt('2026-01-01').all() and not f.duplicated(['date', 'code']).any()
    assert f.loc[f.formula_input_valid, 'V01'].gt(0).all()
    flags = f.formula_input_valid & f.CM01.gt(0) & f.CM02.gt(50)
    c = numeric.conn(); c.register('f', f)
    c.read_parquet(p['atoms']).create_view('atoms')
    expected = c.sql('''WITH r AS(SELECT date,code,
        count(proxy_mark) OVER w AS marks,count(proxy_positive) OVER w AS positives,
        sum(proxy_mark) OVER w AS signed_sum,sum(proxy_positive) OVER w AS positive_count,
        avg(proxy_mark) OVER w AS mean5,100*avg(proxy_positive) OVER w AS frequency5
        FROM atoms WINDOW w AS(PARTITION BY code ORDER BY date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW))
        SELECT f.date,f.code,coalesce(f.formula_input_valid AND r.marks=5 AND r.positives=5
        AND r.signed_sum>0 AND r.positive_count>=3,false) AS selected,r.mean5,r.frequency5
        FROM f LEFT JOIN r USING(date,code) ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(f[['date', 'code']], expected[['date', 'code']], check_exact=True)
    v = f.formula_input_valid
    np.testing.assert_allclose(f.loc[v, 'CM01'] * f.loc[v, 'V01'], expected.loc[v, 'mean5'], rtol=0, atol=2e-10)
    np.testing.assert_allclose(f.loc[v, 'CM02'], expected.loc[v, 'frequency5'], rtol=0, atol=2e-10)
    np.testing.assert_array_equal(flags, expected.selected)
    scope = f.date.ge('2024-01-01'); keys = f.loc[scope, KEYS].reset_index(drop=True)
    assert len(keys) == 1258085
    receipts = dict(e['source_hashes'], **{str(PROTOCOL):sha(PROTOCOL)})
    lists, lookup = [], []
    for year in ['2024', '2025']:
        root = ROOT/('rule'+year); assert not root.exists(); root.mkdir()
        out = keys.copy(); out['selected'] = (flags.loc[scope] & f.loc[scope, 'date'].str.startswith(year)).to_numpy()
        out.to_parquet(root/'selection.parquet', index=False, compression='zstd')
        chosen = out.loc[out.selected]; sizes = chosen.groupby('date').size()
        equivalent = []
        for old in e['prior_analysis_roots']:
            old = Path(old); record = json.loads((old/'selection_report.json').read_text())
            if record['selected'] == len(chosen) and checked_selection(old).equals(out):
                equivalent.append(str(old))
        lookup.append(dict(year=year, prior_selection_roots_examined=len(e['prior_analysis_roots']),
                           exact_equivalent_complete_lists=equivalent, no_economic_statistics_read=True))
        save_json(root/'selection_report.json', dict(protocol_sha256=sha(PROTOCOL),
            selection_sha256=sha(root/'selection.parquet'), rows=len(out), selected=len(chosen), days=len(sizes),
            half_counts=chosen.groupby('half').agg(rows=('code','size'), days=('date','nunique')).reset_index().to_dict('records'),
            median_daily=float(sizes.median()) if len(sizes) else None, max_daily=int(sizes.max()) if len(sizes) else None,
            largest_day_fraction=float(sizes.max()/len(chosen)) if len(chosen) else None,
            condition=p['condition'], no_outcome_or_fill_filter=True, new_2026_prices_read=False, no_exit_rules=True))
        save_json(root/'selection_verification.json', dict(passed=True,
            selection_report_sha256=sha(root/'selection_report.json'),
            all_five_day_counts_signed_mean_flags_and_metadata_SQL_rebuilt=True,
            all_original_input_qualification_preserved=True, native_client_parity_verified=False))
        lists.append(dict(group='rule'+year, root=str(root), selected=len(chosen), days=len(sizes)))
        for name in ['selection.parquet','selection_report.json','selection_verification.json']:
            receipts[str(root/name)] = sha(root/name)
        for family, controls in dict(control=e['controls'], **e['additional_controls']).items():
            old = Path(controls[year]); old_frame = checked_selection(old)
            pd.testing.assert_frame_equal(keys, old_frame[KEYS], check_exact=True)
            if family == 'control':
                lists.append(dict(group=family+year, root=str(old), original_unchanged=True))
            for name in ['selection.parquet','selection_report.json','selection_verification.json']:
                receipts[str(old/name)] = sha(old/name)
    save_json(ROOT/'selection_equivalence_lookup.json', dict(passed=True, records=lookup,
        exact_conditions_not_previously_used_in_indexed_protocol_or_definition_sources=True,
        keyword_lookup_is_not_complete_semantic_proof=True, new_model_fits=0))
    receipts[str(ROOT/'selection_equivalence_lookup.json')] = sha(ROOT/'selection_equivalence_lookup.json')
    save_json(destination, dict(passed=True, input_protocol_sha256=sha(PROTOCOL), execution_protocol_sha256=sha(PROTOCOL),
        source_hashes=receipts, selections=lists, fits_completed=0,
        all_two_year_complete_lists_fixed_together_before_economics=True,
        new_economic_outcomes_read=False, new_2026_prices_read=False))
    return dict(joint_sha256=sha(destination), selections=lists, equivalence=lookup)


def finish():
    comparisons.finish()
    _, execution = checked()
    gate_path = ROOT/'selection_gate.json'; gate = json.loads(gate_path.read_text())
    increments = {}
    for family in execution['additional_controls']:
        support, bounds = [], []
        for year in ['2024', '2025']:
            def main(root):
                _, report = checked_analysis(root)
                return next(s for s in report['summaries'] if s['arm']=='formula' and s['period']==year
                            and s['bps']==15 and not s['sensitive'])
            a = main(ROOT/('rule'+year)); b = main(Path(execution['additional_controls'][family][year]))
            support.append(a['rate'] is not None and b['rate'] is not None and a['bad3'] is not None
                           and b['bad3'] is not None and a['rate']>b['rate'] and a['bad3']<=b['bad3'])
            d = json.loads((ROOT/(family+'_shared_unknowns_'+year+'.json')).read_text())
            bounds.append(next(s for s in d['summaries'] if s['period']==year and s['bps']==15 and not s['sensitive']))
        gate['criteria']['both_year_opportunity_above_'+family+'_and_bad3_not_above'] = all(support)
        gate['criteria']['shared_unknown_lower_ci_both_years_above_'+family] = all(s['lower_ci'] is not None and s['lower_ci'][0]>0 for s in bounds)
        increments[family] = bounds
    gate['supports_further_validation'] = all(gate['criteria'].values())
    gate['additional_control_increment'] = increments; save_json(gate_path, gate)
    path = ROOT/'complete_results_manifest.json'; manifest = json.loads(path.read_text())
    manifest['source_hashes'][str(gate_path)] = sha(gate_path); manifest['criteria'] = gate['criteria']
    manifest.update(new_tree_fits=0, selector_fits=0, fixed_literal_conditions=2,
        new_input_features=0, no_parameter_or_window_search=True)
    check_sources(manifest['source_hashes']); save_json(path, manifest)
    return dict(complete_sha256=sha(path), criteria=manifest['criteria'],
                comparisons=len(manifest['comparisons']), fingerprints=len(manifest['source_hashes']), new_fits=0)


shared.ROOT = comparisons.ROOT = ROOT
shared.EXECUTION = comparisons.EXECUTION = PROTOCOL
shared.fit = SimpleNamespace(PROTOCOL=PROTOCOL, EXECUTION=PROTOCOL)
shared.checked = comparisons.checked = checked

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'analyze', 'finish'])
    stage = parser.parse_args().stage
    print(json.dumps(shared.analyze() if stage=='analyze' else globals()[stage](), ensure_ascii=False), flush=True)
