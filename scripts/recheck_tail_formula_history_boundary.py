"""Evaluate unchanged legacy history selections at the strict 09:59 boundary."""
import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as numeric
from trade_research import tail_formula_baseline as baseline
from trade_research.research_io import check_runtime, check_sources, save_json, sha
from tail_formula_reports import checked_analysis, checked_selection
import finish_tail_formula_rule_search as shared
import finish_tail_formula_profit_rule_search as comparisons

ROOT = Path('data/research/tail_formula_history_boundary_recheck')
PROTOCOL = Path('config/tail_formula_history_boundary_recheck_protocol.json')
EXECUTION = Path('config/tail_formula_history_boundary_recheck_evaluation.json')
KEYS = shared.KEYS


def committed(path):
    assert subprocess.check_output(['git', 'show', f'HEAD:{path}']) == path.read_bytes()
    assert sha(path) in subprocess.check_output(
        ['git', 'show', 'HEAD:docs/selection-formula.md']).decode()


def preparation_protocol():
    check_runtime()
    committed(PROTOCOL)
    p = json.loads(PROTOCOL.read_text())
    check_sources(p['source_hashes'])
    assert p['selector_fits'] == 0 and p['new_tree_fits'] == 0
    return p


def verify_cached_model(root, features, scope):
    model = json.loads((root / 'model_report.json').read_text())
    mv = json.loads((root / 'model_verification.json').read_text())
    sr = json.loads((root / 'score_report.json').read_text())
    sv = json.loads((root / 'score_verification.json').read_text())
    selection_report = json.loads((root / 'selection_report.json').read_text())
    selected = checked_selection(root)
    assert mv['passed'] and mv['model_report_sha256'] == sha(root / 'model_report.json')
    assert sv['passed'] and sv['score_report_sha256'] == sha(root / 'score_report.json')
    assert sv['all_integer_encodings_tree_scores_and_training_quantiles_rebuilt']
    assert sr['model_report_sha256'] == selection_report['model_report_sha256'] == sha(root / 'model_report.json')
    assert sr['scores_sha256'] == sha(root / 'scores.parquet')
    assert selection_report['score_report_sha256'] == sha(root / 'score_report.json')
    assert selection_report['core_sha256'] == sha(root / 'frozen_numeric_core.tdx')
    assert model['training_end'] == scope['start'] and model['last_observation'] < scope['start']
    threshold = selection_report['chosen_threshold']
    assert threshold in model['thresholds'] and threshold['training_quantile'] == .995
    scores = pd.read_parquet(root / 'scores.parquet')
    pd.testing.assert_frame_equal(scores[KEYS + ['formula_input_valid']],
                                  features[KEYS + ['formula_input_valid']], check_exact=True)
    pd.testing.assert_frame_equal(selected[KEYS], features[KEYS], check_exact=True)
    assert np.isfinite(scores.loc[scores.formula_input_valid, 'score']).all()
    c = numeric.conn()
    c.register('scores', scores)
    expected = c.execute('''SELECT date,code,coalesce(formula_input_valid AND score>? AND date>=?
        AND date<?,false) AS selected FROM scores ORDER BY date,code''',
        [threshold['threshold'], scope['start'], scope['end']]).df()
    c.close()
    pd.testing.assert_frame_equal(selected[['date', 'code', 'selected']], expected, check_exact=True)
    return model, selected


def prepare():
    p = preparation_protocol()
    destination = ROOT / 'source_alignment_verified.json'
    assert not destination.exists()
    ROOT.mkdir(exist_ok=True)
    old = Path(p['feature_root'])
    fr = json.loads((old / 'feature_report.json').read_text())
    fv = json.loads((old / 'feature_verification.json').read_text())
    assert fv['passed'] and fv['feature_report_sha256'] == sha(old / 'feature_report.json')
    assert fr['features_sha256'] == sha(old / 'features.parquet')
    assert fv['all_stock_day_lags_twenty_morning_windows_scalars_encodings_and_validity_rebuilt']
    f = pd.read_parquet(old / 'features.parquet')
    current = baseline.original().loc[lambda x: x.date.ge('2024-01-01')].reset_index(drop=True)
    columns = KEYS + ['formula_input_valid'] + p['base_48_features']
    pd.testing.assert_frame_equal(f[columns], current[columns], check_exact=True)
    base_root = Path(p['base_feature_root'])
    base_report = json.loads((base_root / 'feature_report.json').read_text())
    base_proof = json.loads((base_root / 'feature_verification.json').read_text())
    assert base_proof['passed'] and base_proof['feature_report_sha256'] == sha(base_root / 'feature_report.json')
    assert base_report['features_sha256'] == sha(base_root / 'features.parquet')
    assert fr['previous_feature_report_sha256'] == sha(base_root / 'feature_report.json')
    pd.testing.assert_frame_equal(f[columns], pd.read_parquet(base_root / 'features.parquet', columns=columns), check_exact=True)
    assert len(f) == 1258085 and int(f.formula_input_valid.sum()) == 1117397
    assert f.date.lt('2026-01-01').all()
    assert np.isfinite(f.loc[f.formula_input_valid, p['history_50_features']].to_numpy()).all()
    history_flags = np.zeros(len(f), dtype=bool)
    control_flags = np.zeros(len(f), dtype=bool)
    for scope in p['legacy_folds']:
        hm, hs = verify_cached_model(Path(scope['history_root']), f, scope)
        bm, bs = verify_cached_model(Path(scope['base_root']), f, scope)
        assert hm['parameters'] == bm['parameters'] and hm['variant'] == bm['variant']
        assert hm['training_start'] == bm['training_start'] and hm['training_end'] == bm['training_end']
        assert hm['feature_names'] == p['history_50_features'] and bm['feature_names'] == p['base_48_features']
        assert hm['feature_report_sha256'] == sha(old / 'feature_report.json')
        assert bm['feature_report_sha256'] == sha(base_root / 'feature_report.json')
        history_flags |= hs.selected.to_numpy()
        control_flags |= bs.selected.to_numpy()
    history = checked_selection(Path(p['history_selection']))
    control = checked_selection(Path(p['legacy_control_selection']))
    modern_control, _ = checked_analysis(Path(p['strict_legacy_control']))
    for frame in [history, control, modern_control]:
        pd.testing.assert_frame_equal(frame[KEYS], current[KEYS], check_exact=True)
    np.testing.assert_array_equal(history_flags, history.selected)
    np.testing.assert_array_equal(control_flags, control.selected)
    pd.testing.assert_frame_equal(control, modern_control, check_exact=True)
    candidates = []
    # Inspect every on-disk report with the same selected count, then check full
    # flags and metadata; counts alone never establish prior evaluation reuse.
    for report_path in Path('data/research').rglob('analysis_report.json'):
        sr_path = report_path.parent / 'selection_report.json'
        if not sr_path.exists():
            continue
        s = json.loads(sr_path.read_text())
        if s.get('selected') != int(history.selected.sum()):
            continue
        ar = json.loads(report_path.read_text())
        identical = history.equals(checked_selection(report_path.parent))
        candidates.append(dict(root=str(report_path.parent), same_full_frame=identical,
                               reference_label=ar.get('reference_label')))
        assert not (identical and ar.get('reference_label') == '09:59'), 'Reuse existing strict result instead'
    save_json(destination, dict(passed=True, input_protocol_sha256=sha(PROTOCOL),
        source_hashes=p['source_hashes'], source_models_reused=2, new_fits=0,
        original_48_values_validity_and_all_metadata_identical=True,
        all_cached_score_threshold_scope_flags_SQL_rebuilt=True,
        both_legacy_models_same_parameters_and_training_ranges_as_48_control=True,
        old_full_history_list_exactly_two_unchanged_scopes=True,
        legacy_48_strict_control_full_frame_identical=True,
        existing_result_lookup=candidates, selected=int(history.selected.sum()),
        year_2024_model_available=False, native_parity_verified=False,
        new_economic_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True))
    return dict(source_alignment_sha256=sha(destination), selected=int(history.selected.sum()), new_fits=0)


def checked():
    p = preparation_protocol()
    committed(EXECUTION)
    e = json.loads(EXECUTION.read_text())
    assert e['input_protocol_sha256'] == sha(PROTOCOL)
    check_sources(e['source_hashes'])
    source = json.loads((ROOT / 'source_alignment_verified.json').read_text())
    assert source['passed'] and source['input_protocol_sha256'] == sha(PROTOCOL)
    registry = json.loads(Path(e['prior_registry']).read_text())
    assert len(registry['prior_analysis_roots']) == e['prior_registry_analysis_count']
    assert len(registry['prior_completion_manifests']) == e['prior_registry_manifest_count']
    e['prior_analysis_roots'] = registry['prior_analysis_roots'] + e['additional_prior_analysis_roots']
    e['prior_completion_manifests'] = dict(registry['prior_completion_manifests'],
                                          **e['additional_prior_completion_manifests'])
    e['controls'] = registry['controls']
    return p, e


def freeze():
    p, e = checked()
    destination = ROOT / 'joint_selection_freeze.json'
    assert not destination.exists()
    f = checked_selection(Path(p['history_selection']))
    receipts = dict(e['source_hashes'], **{str(PROTOCOL): sha(PROTOCOL), str(EXECUTION): sha(EXECUTION)})
    lists = []
    for year in ['2024', '2025']:
        root = ROOT / ('rule' + year)
        assert not root.exists()
        root.mkdir()
        out = f.copy()
        out['selected'] &= out.date.str.startswith(year)
        out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
        c = numeric.conn()
        c.register('legacy', f)
        expected = c.execute('SELECT date,code,half,board,decision_shares,selected AND starts_with(date,?) AS selected FROM legacy', [year]).df()
        c.close()
        pd.testing.assert_frame_equal(out, expected, check_exact=True)
        selected = out.loc[out.selected]
        sizes = selected.groupby('date').size()
        save_json(root / 'selection_report.json', dict(protocol_sha256=sha(EXECUTION),
            selection_sha256=sha(root / 'selection.parquet'), rows=len(out), selected=len(selected),
            days=len(sizes), half_counts=selected.groupby('half').agg(rows=('code','size'), days=('date','nunique')).reset_index().to_dict('records'),
            median_daily=float(sizes.median()) if len(sizes) else None,
            max_daily=int(sizes.max()) if len(sizes) else None,
            largest_day_fraction=float(sizes.max()/len(selected)) if len(selected) else None,
            original_legacy_flags_unchanged=True, year_2024_model_available=False,
            no_outcome_or_fill_filter=True, new_2026_prices_read=False, no_exit_rules=True))
        save_json(root / 'selection_verification.json', dict(passed=True,
            selection_report_sha256=sha(root / 'selection_report.json'),
            unchanged_legacy_flags_year_scopes_and_all_metadata_SQL_rebuilt=True))
        lists.append(dict(group='rule'+year, root=str(root), selected=len(selected), days=len(sizes)))
        control = Path(e['controls'][year])
        prior = checked_selection(control)
        pd.testing.assert_frame_equal(out[KEYS], prior[KEYS], check_exact=True)
        lists.append(dict(group='control'+year, root=str(control), original_unchanged=True))
        for path in [root, control]:
            for name in ['selection.parquet','selection_report.json','selection_verification.json']:
                receipts[str(path/name)] = sha(path/name)
    save_json(destination, dict(passed=True, input_protocol_sha256=sha(PROTOCOL),
        execution_protocol_sha256=sha(EXECUTION), source_hashes=receipts, selections=lists,
        complete_annual_lists_fixed_before_strict_economics=True, source_models_reused=2,
        fits_completed=0, year_2024_model_available=False, no_new_2024_strategy_evidence=True,
        new_economic_outcomes_read=False, new_2026_prices_read=False))
    return dict(joint_sha256=sha(destination), selections=lists)


def finish():
    comparisons.finish()
    destination = ROOT / 'complete_results_manifest.json'
    m = json.loads(destination.read_text())
    m.pop('complete_two_year_four_half_comparisons_with_all_three_controls', None)
    m.update(source_models_reused=2, source_models_refitted=False,
        old_thresholds_and_full_flags_unchanged=True, year_2024_model_available=False,
        empty_2024_is_model_unavailability_not_strategy_quality=True,
        complete_2025_and_two_half_comparison_with_original_50_and_matched_48=True,
        native_parity_verified=False, no_publish_claim=True)
    save_json(destination, m)
    return dict(complete_sha256=sha(destination), criteria=m['criteria'],
                fingerprint_count=len(m['source_hashes']), comparisons=len(m['comparisons']), new_fits=0)


shared.ROOT = comparisons.ROOT = ROOT
shared.EXECUTION = comparisons.EXECUTION = EXECUTION
shared.fit = SimpleNamespace(PROTOCOL=PROTOCOL, EXECUTION=PROTOCOL)
shared.checked = comparisons.checked = checked

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'freeze', 'analyze', 'finish'])
    stage = parser.parse_args().stage
    print(json.dumps(shared.analyze() if stage == 'analyze' else globals()[stage](), ensure_ascii=False))
