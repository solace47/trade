"""Freeze every learned rule together, then evaluate without retuning."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

import run_tail_formula_rule_search as fit
from trade_research import tail_formula_baseline as original
from trade_research import tail_formula_additive as numeric
from trade_research import tail_formula_rule_search as rules
from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research.research_io import check_sources, save_json, sha
from tail_formula_reports import checked_selection, checked_analysis
from verify_tail_formula_before1000 import analysis as verify_analysis
from audit_tail_formula_reference_coverage import audit
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared_compare
from finish_tail_formula_external_pattern import prior_pairs
import tail_formula_analysis_reuse as reuse


ROOT = fit.ROOT
EXECUTION = Path('config/tail_formula_rule_search_evaluation.json')
KEYS = ['date', 'code', 'half', 'board', 'decision_shares']


def checked():
    p, _ = fit.checked()
    e = json.loads(EXECUTION.read_text())
    assert e['input_protocol_sha256'] == sha(fit.PROTOCOL)
    check_sources(e['source_hashes'])
    assert subprocess.check_output(['git', 'show', f'HEAD:{EXECUTION}']) == EXECUTION.read_bytes()
    completed = json.loads((ROOT / 'all_models_verified.json').read_text())
    assert completed['passed'] and len(completed['folds']) == 4
    for fold in completed['folds']:
        path = ROOT / fold['fold']
        assert fold['model_report_sha256'] == sha(path / 'model_report.json')
        report = json.loads((path / 'model_report.json').read_text())
        v = json.loads((path / 'model_verification.json').read_text())
        assert v['passed'] and v['model_report_sha256'] == sha(path / 'model_report.json')
        assert report['search_trace_sha256'] == sha(path / 'search_trace.json')
    return p, e


def freeze():
    p, e = checked()
    destination = ROOT / 'joint_selection_freeze.json'
    assert not destination.exists()
    f = original.original().loc[lambda x: x.date.ge('2024-01-01')].reset_index(drop=True)
    assert len(f) == 1258085 and f.date.lt('2026-01-01').all()
    valid = f.formula_input_valid.to_numpy()
    x = numeric.encode(f.loc[valid])
    flags = np.zeros(len(f), dtype=bool)
    receipts = {str(EXECUTION): sha(EXECUTION), str(fit.PROTOCOL): sha(fit.PROTOCOL),
                str(fit.EXECUTION): sha(fit.EXECUTION)}
    con = numeric.conn()
    con.register('visible', f[['date', 'code', 'formula_input_valid', *p['feature_names']]])
    encoding = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in p['feature_names'])
    con.execute('CREATE TABLE encoded AS SELECT date,code,' + encoding + ' FROM visible WHERE formula_input_valid')
    expressions, formula_definitions = [], []
    for fold in p['folds']:
        path = ROOT / fold['id']
        model = json.loads((path / 'model_report.json').read_text())
        chosen = model['chosen_rule']
        scope = f.loc[valid, 'date'].ge(fold['evaluation_start']) & f.loc[valid, 'date'].lt(fold['evaluation_end'])
        if chosen:
            conditions = fit.condition_key(chosen['conditions'])
            flags[np.flatnonzero(valid)] |= scope.to_numpy() & rules.masks(x, conditions)
            terms = [f"{p['feature_names'][j]} {'<=' if op == 0 else '>'} {threshold}" for j, op, threshold in conditions]
            expressions.append(f"(date>='{fold['evaluation_start']}' AND date<'{fold['evaluation_end']}' AND " + ' AND '.join(terms) + ')')
            formula_definitions.append(dict(fold=fold['id'], encoded_conditions=terms,
                raw_definitions={p['feature_names'][j]: original.EXPRESSIONS[p['feature_names'][j]] for j, _, _ in conditions}))
        for name in ['model_report.json', 'model_verification.json', 'search_trace.json']:
            receipts[str(path / name)] = sha(path / name)
    expression = ' OR '.join(expressions) if expressions else 'false'
    expected = con.sql(f'''SELECT v.date,v.code,coalesce(e.selected,false) AS selected FROM visible v
        LEFT JOIN (SELECT date,code,({expression}) AS selected FROM encoded) e USING(date,code) ORDER BY date,code''').df()
    con.close()
    pd.testing.assert_frame_equal(expected[['date', 'code']], f[['date', 'code']], check_exact=True)
    np.testing.assert_array_equal(flags, expected.selected)
    save_json(ROOT / 'frozen_rule_definitions.json', dict(rules=formula_definitions,
        native_encoding=p['encoding'], client_compiled=False, native_numeric_parity_verified=False))
    lists = []
    for year in ['2024', '2025']:
        path = ROOT / ('rule' + year)
        assert not path.exists()
        path.mkdir()
        out = f[KEYS].copy()
        out['selected'] = flags & f.date.str.startswith(year)
        out.to_parquet(path / 'selection.parquet', index=False, compression='zstd')
        selected = out.loc[out.selected]
        sizes = selected.groupby('date').size()
        save_json(path / 'selection_report.json', dict(protocol_sha256=sha(EXECUTION),
            selection_sha256=sha(path / 'selection.parquet'), rows=len(out), selected=len(selected),
            days=len(sizes), half_counts=selected.groupby('half').agg(rows=('code', 'size'), days=('date', 'nunique')).reset_index().to_dict('records'),
            median_daily=float(sizes.median()) if len(sizes) else None,
            max_daily=int(sizes.max()) if len(sizes) else None,
            largest_day_fraction=float(sizes.max()/len(selected)) if len(selected) else None,
            no_outcome_or_fill_filter=True, new_2026_prices_read=False, no_exit_rules=True))
        save_json(path / 'selection_verification.json', dict(passed=True,
            selection_report_sha256=sha(path / 'selection_report.json'),
            all_conditions_qualification_scopes_flags_and_metadata_SQL_rebuilt=True))
        lists.append(dict(group='rule'+year, root=str(path), selected=len(selected), days=len(sizes)))
        for name in ['selection.parquet', 'selection_report.json', 'selection_verification.json']:
            receipts[str(path / name)] = sha(path / name)
        control = Path(e['controls'][year])
        old = checked_selection(control)
        pd.testing.assert_frame_equal(out.drop(columns='selected'), old.drop(columns='selected'), check_exact=True)
        lists.append(dict(group='control'+year, root=str(control), original_unchanged=True,
            selected=int(old.selected.sum()), days=old.loc[old.selected, 'date'].nunique()))
        for name in ['selection.parquet', 'selection_report.json', 'selection_verification.json']:
            receipts[str(control / name)] = sha(control / name)
    for path in [ROOT / 'all_models_verified.json', ROOT / 'frozen_rule_definitions.json']:
        receipts[str(path)] = sha(path)
    save_json(destination, dict(passed=True, input_protocol_sha256=sha(fit.PROTOCOL),
        execution_protocol_sha256=sha(EXECUTION), source_hashes=receipts, selections=lists,
        all_four_learned_rules_and_complete_annual_lists_frozen_together=True,
        fits_completed=4, new_economic_outcomes_read=False, new_2026_prices_read=False))
    return dict(joint_sha256=sha(destination), selections=lists)


def checked_joint():
    p, e = checked()
    path = ROOT / 'joint_selection_freeze.json'
    j = json.loads(path.read_text())
    assert j['passed'] and j['execution_protocol_sha256'] == sha(EXECUTION)
    check_sources(j['source_hashes'])
    text = subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    assert sha(path) in text, 'Commit complete annual lists before economics'
    return p, e, j


def analyze():
    _, e, joint = checked_joint()
    previous = [Path(x) for x in e['prior_analysis_roots']]
    records = []
    reuse.PROTOCOL = EXECUTION
    for item in joint['selections']:
        if item['group'].startswith('control'):
            checked_analysis(Path(item['root']))
            continue
        root = Path(item['root'])
        assert not (root / 'analysis_report.json').exists()
        frame, parent = checked_selection(root), None
        for old in previous:
            sr = json.loads((old / 'selection_report.json').read_text())
            ar = json.loads((old / 'analysis_report.json').read_text())
            if sr.get('selected') == int(frame.selected.sum()) and ar.get('reference_label') == '09:59' and frame.equals(checked_selection(old)):
                checked_analysis(old)
                parent = old
                break
        if parent:
            reuse.reuse_analysis(root, parent)
        else:
            evaluation.analyze(root, EXECUTION)
            verify_analysis(root)
        checked_analysis(root)
        audit(root, [2024, 2025])
        previous.append(root)
        records.append(dict(group=item['group'], reused_source=str(parent) if parent else None,
            analysis_report_sha256=sha(root / 'analysis_report.json')))
        print(json.dumps(dict(completed=item['group'], reused=parent is not None)), flush=True)
    save_json(ROOT / 'analysis_dispatch_verification.json', dict(passed=True,
        joint_sha256=sha(ROOT / 'joint_selection_freeze.json'), records=records,
        controls_unchanged_and_reused=True, full_frames_and_applicable_labels_checked=True))
    return records


def finish():
    _, e, joint = checked_joint()
    destination = ROOT / 'complete_results_manifest.json'
    assert not destination.exists()
    roots = {r['group']: Path(r['root']) for r in joint['selections']}
    cache = {g: checked_analysis(r) for g, r in roots.items()}
    old_pairs, old_cache = prior_pairs(e), {}
    receipts = dict(joint['source_hashes'])

    def same(group, path):
        if path not in old_cache:
            sr = json.loads((path / 'selection_report.json').read_text())
            ar = json.loads((path / 'analysis_report.json').read_text())
            if sr['selected'] != int(cache[group][0].selected.sum()) or ar.get('reference_label') != '09:59':
                return False
            old_cache[path] = checked_analysis(path)
        f, report = old_cache[path]
        return cache[group][0].equals(f) and all(cache[group][1][k] == report[k]
            for k in ['summaries', 'daily_summary_sha256', 'label_report_sha256'])

    pairs, increments = [], []
    for year in ['2024', '2025']:
        left, right = 'rule'+year, 'control'+year
        periods = [year+'H1', year+'H2', year]
        old = next((r for r in old_pairs if r['periods'] == set(periods)
                    and same(left, r['roots'][0]) and same(right, r['roots'][1])), None)
        files = old['outputs'] if old else [str(ROOT / ('same_dates_'+year+'.json')),
                                            str(ROOT / ('shared_unknowns_'+year+'.json'))]
        if old:
            receipts[old['manifest']] = old['manifest_sha256']
        else:
            compare(roots[left], roots[right], Path(files[0]), periods, intersection_only=True)
            shared_compare(dict(left=str(roots[left]), right=str(roots[right]), output=files[1],
                left_analysis_sha256=sha(roots[left]/'analysis_report.json'),
                right_analysis_sha256=sha(roots[right]/'analysis_report.json')),
                dict(protocol_sha256=sha(EXECUTION), labels_sha256=sha(evaluation.source.ROOT/'full_labels.parquet'), periods=periods))
        pairs.append(dict(left=left, right=right, outputs=files, reused_pair=old is not None))
        for name in files:
            assert json.loads(Path(name).read_text())['passed']
            receipts[name] = sha(Path(name))
        d = json.loads(Path(files[1]).read_text())
        increments.append(next(s for s in d['summaries'] if s['bps']==15 and not s['sensitive'] and s['period']==year))
        print(json.dumps(dict(comparison_completed=year, reused=old is not None)), flush=True)
    def main(arm, period):
        return next(s for s in cache[arm+period[:4]][1]['summaries'] if s['arm']=='formula'
            and s['bps']==15 and not s['sensitive'] and s['period']==period)
    criteria = dict(
        both_annual_reference_positive=all(main('rule', y)['mean_reference'] is not None and main('rule', y)['mean_reference']>0 for y in ['2024','2025']),
        all_four_half_reference_positive=all(main('rule', y+h)['mean_reference'] is not None and main('rule', y+h)['mean_reference']>0 for y in ['2024','2025'] for h in ['H1','H2']),
        all_four_half_at_least_20_days=all(main('rule', y+h)['days']>=20 for y in ['2024','2025'] for h in ['H1','H2']),
        both_year_opportunity_above_original_and_bad3_not_above=all(main('rule', y)['rate'] is not None and main('rule', y)['bad3'] is not None and main('rule', y)['rate']>main('control', y)['rate'] and main('rule', y)['bad3']<=main('control', y)['bad3'] for y in ['2024','2025']),
        shared_unknown_lower_ci_both_years_strict_positive=all(s['lower_ci'] is not None and s['lower_ci'][0]>0 for s in increments))
    save_json(ROOT / 'selection_gate.json', dict(passed=True, criteria=criteria,
        supports_further_validation=all(criteria.values()), shared_unknown_increment=increments,
        no_publish_claim=True, native_parity_verified=False))
    for root in roots.values():
        for name in ['analysis_report.json','analysis_verification.json','daily_summary.parquet','full_label_report.json','full_label_verification.json']:
            receipts[str(root/name)] = sha(root/name)
    for name in ['selection_gate.json','analysis_dispatch_verification.json']:
        receipts[str(ROOT/name)] = sha(ROOT/name)
    check_sources(receipts)
    save_json(destination, dict(passed=True, input_protocol_sha256=sha(fit.PROTOCOL),
        execution_protocol_sha256=sha(EXECUTION), joint_sha256=sha(ROOT/'joint_selection_freeze.json'),
        source_hashes=receipts, comparisons=pairs, criteria=criteria, fits_completed=4,
        complete_2024_and_2025_results=True, new_2026_prices_read=False, no_exit_rules=True))
    return dict(complete_sha256=sha(destination), criteria=criteria, fingerprint_count=len(receipts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze','analyze','finish'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False))
