"""Jointly freeze the external rule and controls before any economic evaluation."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

import run_tail_formula_external_pattern as inputs
from trade_research.research_io import check_sources, save_json, sha
from trade_research import tail_formula_boundary_evaluation as evaluation
from tail_formula_reports import checked_selection, checked_analysis
import tail_formula_analysis_reuse as reuse
from verify_tail_formula_before1000 import analysis as verify_analysis
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared_compare
from audit_tail_formula_reference_coverage import audit


EXECUTION = Path('config/tail_formula_external_pattern_execution.json')
ROOT = inputs.ROOT
KEYS = ['date', 'code', 'half', 'board', 'decision_shares']


def checked():
    p = inputs.checked()
    e = json.loads(EXECUTION.read_text())
    assert subprocess.check_output(['git', 'show', f'HEAD:{EXECUTION}']) == EXECUTION.read_bytes()
    assert e['input_protocol_sha256'] == sha(inputs.PROTOCOL)
    check_sources(e['source_hashes'])
    gate = json.loads((ROOT / 'source_gate.json').read_text())
    assert gate['passed'] and gate['protocol_sha256'] == sha(inputs.PROTOCOL)
    assert gate['features_sha256'] == sha(ROOT / 'prefix_features.parquet')
    assert gate['summary_sha256'] == sha(ROOT / 'prefix_summary.parquet')
    check_sources(gate['input_receipts'])
    return p, e, gate


def freeze():
    p, e, gate = checked()
    destination = ROOT / 'joint_selection_freeze.json'
    assert not destination.exists(), 'Do not replace complete frozen lists'
    f = pd.read_parquet(ROOT / 'prefix_features.parquet')
    assert len(f) == gate['keys'] == 1258085
    assert not f.duplicated(['date', 'code']).any() and f.date.lt('2026-01-01').all()
    quality = f.formula_input_valid & f.prefix_quality_valid
    con = inputs.conn()
    con.register('features', f)
    literal = con.sql('''SELECT date,code,formula_input_valid AND prefix_quality_valid
        AND branch IN (4,5,6,7,8) AS selected FROM features ORDER BY date,code''').df()
    pd.testing.assert_frame_equal(literal[['date', 'code']], f[['date', 'code']])
    np.testing.assert_array_equal(literal.selected, f.literal_buy)
    receipts = {str(EXECUTION): sha(EXECUTION), str(inputs.PROTOCOL): sha(inputs.PROTOCOL),
        str(ROOT / 'source_gate.json'): sha(ROOT / 'source_gate.json'),
        str(ROOT / 'prefix_features.parquet'): sha(ROOT / 'prefix_features.parquet'),
        str(ROOT / 'prefix_summary.parquet'): sha(ROOT / 'prefix_summary.parquet')}
    lists = []
    for year, old_root in zip(['2024', '2025'], inputs.ORIGINALS):
        old = checked_selection(Path(old_root))
        con.register('old', old)
        rebuilt = con.sql(f'''SELECT f.date,f.code,
            f.formula_input_valid AND f.prefix_quality_valid AND coalesce(o.selected,false)
            AND starts_with(f.date,'{year}') AS selected
            FROM features f LEFT JOIN old o USING(date,code) ORDER BY f.date,f.code''').df()
        matched = f[KEYS].merge(old, on=['date', 'code'], how='left', validate='one_to_one', suffixes=('', '_old'))
        for name in KEYS[2:]:
            pd.testing.assert_series_equal(matched.loc[matched[name + '_old'].notna(), name].reset_index(drop=True),
                matched.loc[matched[name + '_old'].notna(), name + '_old'].reset_index(drop=True),
                check_names=False, check_dtype=False)
        control = quality & matched.selected.eq(True) & f.date.str.startswith(year)
        np.testing.assert_array_equal(control, rebuilt.selected)
        for arm, flags in [('literal', f.literal_buy & f.date.str.startswith(year)), ('control', control)]:
            root = ROOT / (arm + year)
            assert not root.exists()
            root.mkdir()
            out = f[KEYS].copy()
            out['selected'] = flags
            assert not (flags & ~quality).any()
            out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
            selected = out.loc[flags]
            sizes = selected.groupby('date').size()
            save_json(root / 'selection_report.json', dict(protocol_sha256=sha(EXECUTION),
                group=arm + year, selection_sha256=sha(root / 'selection.parquet'), rows=len(out),
                selected=len(selected), days=int(selected.date.nunique()),
                half_counts=selected.groupby('half').agg(rows=('code', 'size'), days=('date', 'nunique')).reset_index().to_dict('records'),
                median_daily=float(sizes.median()) if len(sizes) else None,
                max_daily=int(sizes.max()) if len(sizes) else None,
                largest_day_fraction=float(sizes.max() / len(selected)) if len(selected) else None,
                no_fill_or_outcome_filter=True, new_fits=0, new_2026_prices_read=False, no_exit_rules=True))
            save_json(root / 'selection_verification.json', dict(passed=True,
                selection_report_sha256=sha(root / 'selection_report.json'),
                all_flags_quality_and_original_metadata_SQL_rebuilt=True))
            lists.append(dict(group=arm + year, root=str(root), selected=len(selected), days=len(sizes)))
            for name in ['selection.parquet', 'selection_report.json', 'selection_verification.json']:
                receipts[str(root / name)] = sha(root / name)
    con.close()
    save_json(destination, dict(passed=True, input_protocol_sha256=sha(inputs.PROTOCOL),
        execution_protocol_sha256=sha(EXECUTION), source_hashes=receipts, selections=lists,
        all_four_complete_lists_frozen_together=True, new_economic_outcomes_read=False,
        new_fits=0, new_2026_prices_read=False))
    return dict(joint_sha256=sha(destination), selections=lists)


def checked_joint():
    p, e, gate = checked()
    path = ROOT / 'joint_selection_freeze.json'
    j = json.loads(path.read_text())
    assert j['passed'] and j['execution_protocol_sha256'] == sha(EXECUTION)
    document = subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    assert sha(path) in document, 'Commit all complete lists before economics'
    check_sources(j['source_hashes'])
    reuse.PROTOCOL = EXECUTION
    return p, e, j


def analyze():
    p, e, joint = checked_joint()
    dispatch, previous = [], [Path(x) for x in p['prior_analysis_roots']]
    for item in joint['selections']:
        root = Path(item['root'])
        assert not (root / 'analysis_report.json').exists()
        frame = checked_selection(root)
        parent, examined = None, 0
        for old in previous:
            sr = json.loads((old / 'selection_report.json').read_text())
            ar = json.loads((old / 'analysis_report.json').read_text())
            examined += 1
            if sr.get('selected') != int(frame.selected.sum()) or ar.get('reference_label') != '09:59':
                continue
            if frame.equals(checked_selection(old)):
                checked_analysis(old)
                parent = old
                break
        if parent is not None:
            reuse.reuse_analysis(root, parent)
        else:
            evaluation.analyze(root, EXECUTION)
            verify_analysis(root)
        checked_analysis(root)
        audit(root, [2024, 2025])
        previous.append(root)
        dispatch.append(dict(group=item['group'], reused_source=str(parent) if parent else None,
            prior_metadata_examined=examined, analysis_report_sha256=sha(root / 'analysis_report.json')))
        print(json.dumps(dict(completed=item['group'], reused=parent is not None)), flush=True)
    save_json(ROOT / 'analysis_dispatch_verification.json', dict(passed=True,
        joint_sha256=sha(ROOT / 'joint_selection_freeze.json'), records=dispatch,
        all_complete_frames_and_applicable_labels_equal_before_reuse=True,
        new_fits=0, new_2026_prices_read=False))
    return dispatch


def prior_pairs(p):
    records, fingerprints = [], {}
    audit_path = Path('data/research/tail_formula_phase_split/prior_comparison_receipts_verified.json')
    old_audit = json.loads(audit_path.read_text())
    assert old_audit['passed']
    check_sources(old_audit['source_hashes'])
    fingerprints.update(old_audit['source_hashes'])
    for name, digest in p['prior_completion_manifests'].items():
        path = Path(name)
        assert sha(path) == digest
        manifest = json.loads(path.read_text())
        assert manifest['passed']
        fingerprints.update(manifest.get('source_hashes', {}))
        for pair in manifest.get('comparisons', []):
            files = pair.get('outputs', [])
            if len(files) != 2 or not all(Path(x).exists() for x in files):
                continue
            a, b = [json.loads(Path(x).read_text()) for x in files]
            if not a.get('passed') or not b.get('passed') or not a.get('intersection_only'):
                continue
            ends = a.get('inputs', {})
            if not all(x in ends for x in ['left', 'right']):
                continue
            check_sources({x: fingerprints[x] for x in files})
            roots = [Path(ends[x]['root']) for x in ['left', 'right']]
            assert b['comparison']['left'] == str(roots[0]) and b['comparison']['right'] == str(roots[1])
            records.append(dict(roots=roots, outputs=files, periods={x['period'] for x in a['summaries']},
                manifest=name, manifest_sha256=digest))
    return records


def finish():
    p, e, joint = checked_joint()
    destination = ROOT / 'complete_results_manifest.json'
    assert not destination.exists()
    roots = {x['group']: Path(x['root']) for x in joint['selections']}
    cache = {name: checked_analysis(root) for name, root in roots.items()}
    old_pairs, old_cache = prior_pairs(p), {}
    receipts = dict(joint['source_hashes'])
    pairs = []

    def same(name, old):
        if old not in old_cache:
            sr = json.loads((old / 'selection_report.json').read_text())
            ar = json.loads((old / 'analysis_report.json').read_text())
            if sr['selected'] != int(cache[name][0].selected.sum()) or ar.get('reference_label') != '09:59':
                return False
            old_cache[old] = checked_analysis(old)
        frame, report = old_cache[old]
        return cache[name][0].equals(frame) and all(cache[name][1][key] == report[key]
            for key in ['summaries', 'daily_summary_sha256', 'label_report_sha256'])

    for year in ['2024', '2025']:
        left, right = 'literal' + year, 'control' + year
        periods = [year + 'H1', year + 'H2', year]
        reused = next((x for x in old_pairs if x['periods'] == set(periods)
            and same(left, x['roots'][0]) and same(right, x['roots'][1])), None)
        files = reused['outputs'] if reused else [str(ROOT / ('same_dates_' + year + '.json')),
                                                str(ROOT / ('shared_unknowns_' + year + '.json'))]
        if reused:
            receipts[reused['manifest']] = reused['manifest_sha256']
        else:
            compare(roots[left], roots[right], Path(files[0]), periods, intersection_only=True)
            spec = dict(left=str(roots[left]), right=str(roots[right]), output=files[1],
                left_analysis_sha256=sha(roots[left] / 'analysis_report.json'),
                right_analysis_sha256=sha(roots[right] / 'analysis_report.json'))
            shared_compare(spec, dict(protocol_sha256=sha(EXECUTION),
                labels_sha256=sha(evaluation.source.ROOT / 'full_labels.parquet'), periods=periods))
        pairs.append(dict(left=left, right=right, outputs=files, reused_pair=reused is not None))
        for name in files:
            assert json.loads(Path(name).read_text())['passed']
            receipts[name] = sha(Path(name))
        print(json.dumps(dict(comparison_completed=year, reused=reused is not None)), flush=True)
    def main(arm, period):
        return next(s for s in cache[arm + period[:4]][1]['summaries'] if s['arm'] == 'formula'
            and s['bps'] == 15 and not s['sensitive'] and s['period'] == period)
    annual = [main('literal', y) for y in ['2024', '2025']]
    half = [main('literal', y + part) for y in ['2024', '2025'] for part in ['H1', 'H2']]
    incremental = []
    for year, pair in zip(['2024', '2025'], pairs):
        d = json.loads(Path(pair['outputs'][1]).read_text())
        incremental.append(next(s for s in d['summaries'] if s['bps'] == 15
            and not s['sensitive'] and s['period'] == year))
    criteria = dict(
        both_annual_reference_positive=all(s['mean_reference'] is not None and s['mean_reference'] > 0 for s in annual),
        all_four_half_reference_positive=all(s['mean_reference'] is not None and s['mean_reference'] > 0 for s in half),
        all_four_half_at_least_20_days=all(s['days'] >= 20 for s in half),
        both_year_opportunity_above_same_quality_control_and_bad3_not_above=all(
            main('literal', year)['rate'] is not None and main('control', year)['rate'] is not None
            and main('literal', year)['bad3'] is not None and main('control', year)['bad3'] is not None
            and main('literal', year)['rate'] > main('control', year)['rate']
            and main('literal', year)['bad3'] <= main('control', year)['bad3'] for year in ['2024', '2025']),
        shared_unknown_lower_ci_both_years_strict_positive=all(
            s['lower_ci'] is not None and s['lower_ci'][0] > 0 for s in incremental))
    gate_path = ROOT / 'selection_gate.json'
    save_json(gate_path, dict(passed=True, criteria=criteria, supports_further_validation=all(criteria.values()),
        shared_unknown_increment=incremental, no_publish_claim=True, no_exit_rules=True, new_fits=0))
    for root in roots.values():
        for name in ['selection.parquet', 'selection_report.json', 'selection_verification.json',
            'analysis_report.json', 'analysis_verification.json', 'daily_summary.parquet',
            'full_label_report.json', 'full_label_verification.json', 'reference_coverage_verification.json']:
            receipts[str(root / name)] = sha(root / name)
    for path in [gate_path, ROOT / 'analysis_dispatch_verification.json']:
        receipts[str(path)] = sha(path)
    check_sources(receipts)
    save_json(destination, dict(passed=True, input_protocol_sha256=sha(inputs.PROTOCOL),
        execution_protocol_sha256=sha(EXECUTION), joint_sha256=sha(ROOT / 'joint_selection_freeze.json'),
        source_hashes=receipts, comparisons=pairs, criteria=criteria, new_fits=0,
        complete_2024_and_2025_results=True, new_2026_prices_read=False, no_exit_rules=True,
        software_compilation_verified=False, native_client_numeric_parity_verified=False))
    return dict(complete_sha256=sha(destination), criteria=criteria, fingerprint_count=len(receipts))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['freeze', 'analyze', 'finish'])
    args = parser.parse_args()
    print(json.dumps(globals()[args.phase](), ensure_ascii=False))
