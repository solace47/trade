"""Four fixed certificate models; freeze full selections before economics."""
import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd

from trade_research import tail_formula_confirmed_success as study
from trade_research import tail_formula_additive as base
from trade_research import tail_formula_relative as relative
from trade_research import tail_formula_executable_opportunity as leaf_verifier
from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research.corporate_cash import save_json, sha
from find_existing_tail_formula_models import find
from freeze_tail_formula_bipower_gap import checked_selection
from audit_tail_formula_reference_coverage import audit
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared_compare
import run_tail_formula_market_reference as market
import run_tail_formula_volume_memory as aggregation

KEYS = study.META[:-1]


def protocols():
    p = study.checked(); path = study.ROOT / 'prefit_lookup_verification.json'; assert not path.exists()
    verification = json.loads((study.ROOT / 'target_verification.json').read_text())
    assert verification['passed'] and verification['protocol_sha256'] == sha(study.PROTOCOL)
    receipts = {str(study.INPUTS / f): sha(study.INPUTS / f)
                for f in ['feature_report.json', 'feature_verification.json']}
    receipts[str(study.ROOT / 'target_verification.json')] = sha(study.ROOT / 'target_verification.json')
    records = []
    for fold, spec in p['folds'].items():
        q = dict(master_protocol_sha256=sha(study.PROTOCOL), objective_identifier=study.OBJECTIVE,
                 fold=fold, **spec, expected_features=50, feature_names=list(study.EXPRESSIONS),
                 parameters=p['parameters'], model_max_depth=3, threshold=.995, target='relative',
                 input_receipts=receipts, training_verification_sha256=sha(study.ROOT / 'models' / fold / 'training_verification.json'),
                 no_training_period_selection=True, new_2026_prices_allowed=False, no_exit_rules=True)
        file = Path('config') / study.STEM / (fold + '.json')
        assert not file.exists(); file.parent.mkdir(parents=True, exist_ok=True); save_json(file, q)
        lookup = find(file, 'relative')
        assert not lookup['matches'], 'Audit equivalent mature-domain training before any fit'
        records.append(dict(fold=fold, protocol_sha256=sha(file), lookup=lookup))
    save_json(path, dict(passed=True, master_protocol_sha256=sha(study.PROTOCOL), records=records,
                        all_four_protocols_before_any_fit=True, maximum_new_fits=4,
                        full_target_verification_sha256=sha(study.ROOT / 'target_verification.json'),
                        original_control_models_reused_without_fit=True,
                        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True))
    return dict(prefit_sha256=sha(path), records=[dict(fold=r['fold'], protocol_sha256=r['protocol_sha256']) for r in records])


def setup(fold):
    p = study.checked(); assert fold in p['folds']
    base.ROOT, base.FEATURES, base.SOURCE = study.ROOT / 'models' / fold, study.INPUTS, study.LABELS
    base.EXPRESSIONS, base.HEADER = study.EXPRESSIONS, study.HEADER
    base.PROTOCOL = relative.PROTOCOL = Path('config') / study.STEM / (fold + '.json')
    q = json.loads(base.PROTOCOL.read_text())
    assert q['master_protocol_sha256'] == sha(study.PROTOCOL) and q['objective_identifier'] == study.OBJECTIVE
    assert q['fold'] == fold and q['parameters'] == p['parameters']
    assert all(q[k] == v for k, v in p['folds'][fold].items())
    for file, digest in q['input_receipts'].items(): assert sha(Path(file)) == digest, file
    assert q['training_verification_sha256'] == sha(base.ROOT / 'training_verification.json')
    pre = json.loads((study.ROOT / 'prefit_lookup_verification.json').read_text())
    assert pre['passed'] and pre['master_protocol_sha256'] == sha(study.PROTOCOL)
    record = next(r for r in pre['records'] if r['fold'] == fold)
    assert record['protocol_sha256'] == sha(base.PROTOCOL) and not record['lookup']['matches']
    committed = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'], text=True, capture_output=True, check=True).stdout
    assert sha(base.PROTOCOL) in committed and sha(study.ROOT / 'target_verification.json') in committed
    relative.training = lambda variant: study.training(q)
    # Reuse the unchanged node-verification algorithm, replacing its target
    # builder explicitly. Physical known flags and label tables stay untouched.
    leaf_verifier.PROTOCOL = study.PROTOCOL
    leaf_verifier.inputs = SimpleNamespace(ROOT=study.INPUTS, EXPRESSIONS=study.EXPRESSIONS)
    leaf_verifier.labels = SimpleNamespace(ROOT=study.LABELS)
    leaf_verifier.training = lambda independent=False: study.training(q, independent)
    return p, q


def verify_scores(q):
    root = base.ROOT; m = json.loads((root / 'model_report.json').read_text())
    r = json.loads((root / 'score_report.json').read_text())
    assert r['scores_sha256'] == sha(root / 'scores.parquet')
    c = base.conn()
    encoded = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in study.EXPRESSIONS)
    expression = format(m['bias'], '.17e') + '+' + '+'.join(leaf_verifier.tree_sql(t) for t in m['trees'])
    d = c.sql(f'''WITH encoded AS (SELECT {','.join(study.META)},{encoded}
        FROM read_parquet('{study.INPUTS}/features.parquet'))
        SELECT {','.join(study.META)},CASE WHEN formula_input_valid THEN {expression} ELSE NULL END AS score
        FROM encoded ORDER BY date,code''').df()
    actual = pd.read_parquet(root / 'scores.parquet')
    pd.testing.assert_frame_equal(actual[study.META], d[study.META], check_exact=True)
    np.testing.assert_allclose(actual.score, d.score, rtol=0, atol=2e-11, equal_nan=True)
    # All mature rows, including unknowns, define the fixed training q995.
    c.register('rebuilt_scores', d)
    s = c.sql(f'''SELECT s.score FROM rebuilt_scores s JOIN read_parquet('{study.LABELS}/full_labels.parquet') l USING(date,code)
        WHERE formula_input_valid AND s.date>='{q['training_start']}' AND s.date<'{q['training_end']}'
          AND next_date>s.date AND next_date<'{q['training_end']}' ORDER BY s.date,s.code''').df().score
    c.close(); assert len(s) == q['expected_training_rows']
    for quantile, cut in zip(base.QUANTILES, m['thresholds']):
        assert quantile == cut['training_quantile']
        np.testing.assert_allclose(np.quantile(s, quantile), cut['threshold'], rtol=0, atol=2e-11)
    proof = dict(passed=True, score_report_sha256=sha(root / 'score_report.json'),
                 rows=len(d), valid=int(d.formula_input_valid.sum()), training_rows=len(s),
                 all_original_metadata_scores_and_all_mature_quantiles_independent_sql_equal=True,
                 unknown_rows_in_training_quantile_domain=True, no_future_fill_filter=True,
                 new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'score_verification.json', proof)
    return proof


def fit(fold):
    p, q = setup(fold); assert not (base.ROOT / 'model_report.json').exists()
    study.verify_training(fold, q, save=False)
    relative.model('relative')
    path = base.ROOT / 'model_report.json'; m = json.loads(path.read_text())
    m.update(objective_identifier=study.OBJECTIVE, master_protocol_sha256=sha(study.PROTOCOL),
             training_verification_sha256=sha(base.ROOT / 'training_verification.json'),
             unknown_actual_events_and_returns_not_imputed=True, fitted_score_is_not_probability_bound=True)
    save_json(path, m)
    assert m['rows'] == q['expected_training_rows'] and m['days'] == q['expected_training_days']
    assert m['last_observation'] == q['expected_last_observation'] < q['evaluation_start']
    leaf_verifier.verify_model(); base.scores(); verify_scores(q)
    return dict(fold=fold, rows=m['rows'], days=m['days'], last_observation=m['last_observation'],
                model_sha256=sha(path), score_verification_sha256=sha(base.ROOT / 'score_verification.json'))


def freeze():
    p = study.checked(); joint = study.ROOT / 'joint_selection_freeze.json'; assert not joint.exists()
    f = pd.read_parquet(study.INPUTS / 'features.parquet', columns=study.META)
    keys = f.loc[f.date.ge('2024-01-01'), KEYS].reset_index(drop=True); assert len(keys) == 1258085
    flags = {year: np.zeros(len(keys), bool) for year in ['2024', '2025']}; receipts, models = {}, []
    for fold in p['folds']:
        _, q = setup(fold); root = base.ROOT; m = json.loads((root / 'model_report.json').read_text())
        assert m['protocol_sha256'] == sha(base.PROTOCOL) and m['objective_identifier'] == study.OBJECTIVE
        assert m['last_observation'] < q['evaluation_start'] and m['rows'] == q['expected_training_rows']
        for kind in ['model', 'score']:
            r = json.loads((root / (kind + '_report.json')).read_text())
            v = json.loads((root / (kind + '_verification.json')).read_text())
            assert v['passed'] and v[kind + '_report_sha256'] == sha(root / (kind + '_report.json'))
            assert r['protocol_sha256'] == sha(base.PROTOCOL)
        assert r['scores_sha256'] == sha(root / 'scores.parquet')
        d = pd.read_parquet(root / 'scores.parquet', filters=[('date', '>=', q['evaluation_start']), ('date', '<', q['evaluation_end'])])
        mask = keys.date.ge(q['evaluation_start']) & keys.date.lt(q['evaluation_end'])
        pd.testing.assert_frame_equal(d[KEYS].reset_index(drop=True), keys.loc[mask].reset_index(drop=True), check_exact=True)
        cut = m['thresholds'][3]; assert cut['training_quantile'] == .995
        values = d.formula_input_valid & d.score.gt(cut['threshold'])
        c = base.conn(); c.register('d', d)
        independent = c.sql(f'SELECT coalesce(formula_input_valid AND score>{cut["threshold"]:.17e},false) AS flag FROM d').df(); c.close()
        np.testing.assert_array_equal(values, independent.flag); flags[fold[:4]][mask] = values.to_numpy()
        core = root / 'frozen_numeric_core.tdx'; assert not core.exists()
        core.write_text(base.native_core(m, cut['threshold'], study.EXPRESSIONS, study.HEADER))
        models.append(dict(fold=fold, rows=m['rows'], days=m['days'], last_observation=m['last_observation'],
                           selected=int(values.sum()), signal_days=d.loc[values, 'date'].nunique()))
        for file in ['training_verification.json', 'model_report.json', 'model_verification.json',
                     'score_report.json', 'score_verification.json', 'scores.parquet', 'frozen_numeric_core.tdx']:
            receipts[str(root / file)] = sha(root / file)
        receipts[str(base.PROTOCOL)] = sha(base.PROTOCOL)
    selections = []
    for year, flag in flags.items():
        group = 'certificate' + year; root = study.ROOT / group; root.mkdir(parents=True, exist_ok=True)
        out = keys.copy(); out['selected'] = flag
        assert out.loc[out.selected, 'date'].str.startswith(year).all()
        out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
        r = dict(protocol_sha256=sha(study.PROTOCOL), selection_sha256=sha(root / 'selection.parquet'),
                 group=group, rows=len(out), selected=int(flag.sum()), days=out.loc[out.selected, 'date'].nunique(),
                 no_new_group_outcomes_read=True, no_future_fill_or_label_filtering=True, new_2026_prices_read=False, no_exit_rules=True)
        save_json(root / 'selection_report.json', r)
        save_json(root / 'selection_verification.json', dict(passed=True,
            selection_report_sha256=sha(root / 'selection_report.json'),
            full_original_metadata_all_flags_and_invalid_unselected_other_year_rows_independent_equal=True))
        selections.append(dict(group=group, root=str(root), rows=len(out), selected=r['selected'], days=r['days']))
        for file in ['selection.parquet', 'selection_report.json', 'selection_verification.json']: receipts[str(root / file)] = sha(root / file)
    for root in p['original_selection_roots'].values():
        checked_selection(Path(root))
        for file in ['selection.parquet', 'selection_report.json', 'selection_verification.json']:
            receipts[str(Path(root) / file)] = sha(Path(root) / file)
    save_json(joint, dict(passed=True, protocol_sha256=sha(study.PROTOCOL), source_hashes=receipts,
                         selections=selections, models=models, original_control_models_not_refitted=True,
                         all_four_models_and_two_full_annual_frames_fixed_together=True,
                         no_new_group_outcomes_read=True, new_2026_prices_read=False, no_exit_rules=True,
                         software_compilation_verified=False, native_source_parity_verified=False))
    return dict(joint_sha256=sha(joint), models=models, selections=selections)


def checked_joint():
    p = study.checked(); path = study.ROOT / 'joint_selection_freeze.json'; joint = json.loads(path.read_text())
    assert joint['passed'] and joint['protocol_sha256'] == sha(study.PROTOCOL)
    committed = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'], text=True, capture_output=True, check=True).stdout
    assert sha(path) in committed
    for file, digest in joint['source_hashes'].items(): assert sha(Path(file)) == digest, file
    for item in joint['selections']: checked_selection(Path(item['root']))
    evaluation.source.ROOT = Path(p['evaluation_label_root'])
    return p, p, joint


def analyze():
    # The common dispatcher requires exact full frames and all label/statistical
    # receipts before reusing an annual report; each annual includes its halves.
    aggregation.study = study; aggregation.EXECUTION = study.PROTOCOL
    aggregation.checked_joint = checked_joint
    return aggregation.analyze()


def finish():
    p, _, joint = checked_joint(); path = study.ROOT / 'complete_results_manifest.json'; assert not path.exists()
    roots = {s['group']: Path(s['root']) for s in joint['selections']}
    roots.update({'original' + y: Path(r) for y, r in p['original_selection_roots'].items()})
    cache, receipts, done, uncertainty = {}, {}, [], []
    for name, root in roots.items():
        cache[name] = market.checked_analysis(root)
        if not (root / 'reference_coverage_verification.json').exists(): audit(root, [2024, 2025])
        proof = json.loads((root / 'reference_coverage_verification.json').read_text())
        assert proof['passed'] and proof['analysis_report_sha256'] == sha(root / 'analysis_report.json')
        for file in ['selection.parquet', 'selection_report.json', 'selection_verification.json', 'analysis_report.json',
                     'analysis_verification.json', 'daily_summary.parquet', 'full_label_report.json',
                     'full_label_verification.json', 'reference_coverage_verification.json']: receipts[str(root / file)] = sha(root / file)
    # Reuse previous pair outputs only if both entire endpoint frames, all
    # economic labels, daily summaries and summaries match, including unknowns.
    prior_pairs = []
    matches = json.loads((study.ROOT / 'analysis_reuse_lookup.json').read_text())['full_frame_matches']
    if any(matches.values()):
        for file, digest in p['prior_completion_manifests'].items():
            manifest = Path(file); assert sha(manifest) == digest
            for pair in json.loads(manifest.read_text()).get('comparisons', []):
                endpoints = []
                for side in ['left', 'right']:
                    value = pair.get(side)
                    if not value: break
                    root = next((r for r in [manifest.parent / value, manifest.parent / 'comparison_views' / value]
                                 if (r / 'selection_report.json').exists()), None)
                    if root is None: break
                    endpoints.append(root)
                if len(endpoints) == 2 and all(Path(f).exists() for f in pair.get('outputs', [])):
                    prior_pairs.append(dict(endpoints=endpoints, outputs=pair['outputs'], manifest=str(manifest), sha=digest))
    def equivalent(name, root):
        frame, report = market.checked_analysis(root); own, own_report = cache[name]
        return own.equals(frame) and all(own_report[k] == report[k] for k in ['summaries', 'daily_summary_sha256', 'label_report_sha256'])
    for year in ['2024', '2025']:
        left, right = 'certificate' + year, 'original' + year
        old = next((r for r in prior_pairs if equivalent(left, r['endpoints'][0]) and equivalent(right, r['endpoints'][1])), None)
        files = [study.ROOT / ('same_dates_' + year + '.json'), study.ROOT / ('shared_unknowns_' + year + '.json')]
        periods = [year + 'H1', year + 'H2', year]
        if old:
            files = [Path(f) for f in old['outputs']]; receipts[old['manifest']] = old['sha']
        else:
            compare(roots[left], roots[right], files[0], periods, intersection_only=True)
            spec = dict(left=str(roots[left]), right=str(roots[right]), output=str(files[1]),
                        left_analysis_sha256=sha(roots[left] / 'analysis_report.json'),
                        right_analysis_sha256=sha(roots[right] / 'analysis_report.json'))
            shared_compare(spec, dict(protocol_sha256=sha(study.PROTOCOL),
                labels_sha256=sha(evaluation.source.ROOT / 'full_labels.parquet'), periods=periods,
                signal_range=['2024-01-01', '2024-12-31'] if year == '2024' else None))
        for file in files: assert json.loads(file.read_text())['passed']; receipts[str(file)] = sha(file)
        shared = json.loads(files[1].read_text())
        uncertainty.append(next(s for s in shared['summaries'] if s['bps'] == 15 and not s['sensitive'] and s['period'] == year))
        done.append(dict(left=left, right=right, outputs=[str(f) for f in files], reused=old is not None))
        print(json.dumps(dict(compared=year)), flush=True)
    def main(group, period):
        return next(s for s in cache[group][1]['summaries'] if s['arm'] == 'formula' and s['bps'] == 15
                    and not s['sensitive'] and s['period'] == period)
    criteria = dict(each_half_at_least_20_signal_days=all(main('certificate' + y, y + h)['days'] >= 20
                    for y in ['2024', '2025'] for h in ['H1', 'H2']),
        all_four_half_reference_means_positive=all((main('certificate' + y, y + h)['mean_reference'] or -1) > 0
                    for y in ['2024', '2025'] for h in ['H1', 'H2']),
        both_year_opportunity_above_original_and_bad3_not_above=all(
            main('certificate' + y, y)['rate'] is not None and main('certificate' + y, y)['bad3'] is not None
            and main('certificate' + y, y)['rate'] > main('original' + y, y)['rate']
            and main('certificate' + y, y)['bad3'] <= main('original' + y, y)['bad3'] for y in ['2024', '2025']),
        both_year_shared_unknown_lower_ci_strict_positive=all(s['lower_ci'] is not None and s['lower_ci'][0] > 0 for s in uncertainty))
    gate = study.ROOT / 'certificate_gate.json'
    save_json(gate, dict(passed=True, criteria=criteria, supports_further_validation=all(criteria.values()),
        evidence_target_package_comparison_not_isolated_unknown_causal_effect=True,
        no_automatic_2026_evaluation=True, no_publish_claim=True, new_2026_prices_read=False, no_exit_rules=True))
    for file in [gate, study.ROOT / 'analysis_dispatch_verification.json', study.ROOT / 'analysis_reuse_lookup.json']:
        receipts[str(file)] = sha(file)
    save_json(path, dict(passed=True, protocol_sha256=sha(study.PROTOCOL), joint_sha256=sha(study.ROOT / 'joint_selection_freeze.json'),
        source_hashes=receipts, comparisons=done, all_two_full_annual_groups_and_two_predefined_comparisons_complete=True,
        all_full_frames_labels_daily_statistics_equal_before_reuse=True, no_duplicate_half_year_aggregation=True,
        years_2024_and_2025_exploratory=True, no_publish_claim=True, new_2026_prices_read=False, no_exit_rules=True))
    return dict(completion_sha256=sha(path), criteria=criteria,
                periods={y + h: main('certificate' + y, y + h) for y in ['2024', '2025'] for h in ['H1', 'H2', '']})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'protocols', 'fit', 'freeze', 'analyze', 'finish'])
    parser.add_argument('--fold', choices=['2024h1', '2024h2', '2025h1', '2025h2']); args = parser.parse_args()
    result = study.prepare() if args.stage == 'prepare' else fit(args.fold) if args.stage == 'fit' else globals()[args.stage]()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
