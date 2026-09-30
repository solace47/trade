"""Jointly evaluate feature competition across both exposed research years."""
import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess

import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research import tail_formula_feature_subsample as study
from trade_research import tail_formula_feature_subsample_model as model
from trade_research.corporate_cash import save_json, sha
from audit_tail_formula_reference_coverage import audit
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared
from freeze_tail_formula_bipower_gap import checked_selection
from reuse_tail_formula_selected_analysis import reuse
import reuse_tail_formula_subsample_control as historical

ROOT = study.ROOT
META = study.META[:-1]


def checked_analysis(root, year):
    frame = checked_selection(root)
    assert frame.loc[frame.selected, 'date'].str.startswith(str(year)).all()
    assert frame.date.ge('2024-01-01').all() and frame.date.lt(str(year + 1) + '-01-01').all()
    r = json.loads((root / 'analysis_report.json').read_text())
    v = json.loads((root / 'analysis_verification.json').read_text())
    assert v['passed'] and v['analysis_report_sha256'] == sha(root / 'analysis_report.json')
    assert r['selection_report_sha256'] == sha(root / 'selection_report.json')
    assert r['daily_summary_sha256'] == sha(root / 'daily_summary.parquet') and r['reference_label'] == '09:59'
    label = json.loads((root / 'full_label_report.json').read_text())
    lv = json.loads((root / 'full_label_verification.json').read_text())
    assert lv['passed'] and lv['label_report_sha256'] == r['label_report_sha256'] == sha(root / 'full_label_report.json')
    assert label['labels_sha256'] == sha(root / 'full_labels.parquet')
    return frame, r


def checked_models():
    master = study.checked(); configs = {}; receipts = {}
    historical.checked_proof()
    for path in [historical.PROTOCOL, historical.PROOF]:
        receipts[str(path)] = sha(path)
    for arm in study.ARMS:
        for fold in master['folds']:
            p = model.setup(arm, fold); root = base.ROOT
            m = json.loads((root / 'model_report.json').read_text())
            sr = json.loads((root / 'score_report.json').read_text())
            for kind in ['model', 'score']:
                proof = json.loads((root / (kind + '_verification.json')).read_text())
                assert proof['passed'] and proof[kind + '_report_sha256'] == sha(root / (kind + '_report.json'))
            assert m['protocol_sha256'] == sr['protocol_sha256'] == sha(base.PROTOCOL)
            assert m['feature_names'] == list(study.ARMS[arm])
            assert m['last_observation'] == p['expected_last_observation'] < p['evaluation_start']
            assert m['rows'] == p['expected_training_rows'] and m['days'] == p['expected_training_days']
            assert all(m['parameters'][k] == v for k, v in master['parameters'].items())
            assert m['thresholds'][3]['training_quantile'] == .995
            assert m['feature_report_sha256'] == sr['feature_report_sha256'] == sha(study.INPUTS / 'feature_report.json')
            assert sr['model_report_sha256'] == sha(root / 'model_report.json')
            assert sr['scores_sha256'] == sha(root / 'scores.parquet')
            configs[(arm, fold)] = (p, m)
            for path in [base.PROTOCOL, *(root / n for n in ['model_report.json', 'model_verification.json',
                    'score_report.json', 'score_verification.json', 'scores.parquet'])]:
                receipts[str(path)] = sha(path)
    return master, configs, receipts


def freeze():
    master, configs, receipts = checked_models()
    assert not (ROOT / 'joint_selection_freeze.json').exists()
    assert not any((ROOT / (arm + '_' + str(year)) / 'analysis_report.json').exists()
                   for arm in study.ARMS for year in [2024, 2025])
    f = base.feature_inputs(); records = []; models = []; equality = []
    for year in [2024, 2025]:
        keys = f.loc[f.date.ge('2024-01-01') & f.date.lt(str(year + 1) + '-01-01'), META].reset_index(drop=True)
        assert len(keys) == (587276 if year == 2024 else 1258085)
        control_paths = [Path(v) for k, v in master['controls'].items() if k.endswith(str(year))]
        if year == 2025:
            control_paths.append(historical.OLD_ANNUAL)
        for control in control_paths:
            pd.testing.assert_frame_equal(keys, checked_selection(control)[META], check_exact=True)
        for arm, expressions in study.ARMS.items():
            parts = []; queries = []
            for half in ['h1', 'h2']:
                fold = str(year) + half; p, m = configs[(arm, fold)]; root = ROOT / arm / fold
                score_root = historical.OLD_FOLDS[fold] if arm == 'norm' and year == 2025 else root
                d = pd.read_parquet(score_root / 'scores.parquet', filters=[('date', '>=', p['evaluation_start']),
                                                                    ('date', '<', p['evaluation_end'])])
                cut = m['thresholds'][3]['threshold']
                d['selected'] = d.formula_input_valid & d.score.gt(cut)
                pd.testing.assert_frame_equal(d[META].reset_index(drop=True),
                    keys.loc[keys.date.ge(p['evaluation_start']) & keys.date.lt(p['evaluation_end'])].reset_index(drop=True),
                    check_exact=True)
                parts.append(d[['date', 'code', 'selected']])
                queries.append(f'''SELECT date,code,coalesce(formula_input_valid AND score>{cut:.17e},false) AS selected
                    FROM read_parquet('{score_root}/scores.parquet')
                    WHERE date>='{p['evaluation_start']}' AND date<'{p['evaluation_end']}' ''')
                core = root / 'frozen_numeric_core.tdx'
                core.write_text(base.native_core(m, cut, expressions, study.HEADER)); receipts[str(core)] = sha(core)
                v = json.loads((root / 'model_verification.json').read_text())
                root_counts = Counter(m['feature_names'][t['feature'][0]] for t in m['trees'] if t['feature'][0] >= 0)
                models.append(dict(arm=arm, fold=fold, rows=m['rows'], days=m['days'],
                    last_observation=m['last_observation'], node_checks=v['node_checks'],
                    root_feature_counts=dict(root_counts), selected=int(d.selected.sum()),
                    signal_days=d.loc[d.selected, 'date'].nunique(), score_source=str(score_root),
                    independent_new_model_evidence=score_root == root))
            flags = pd.concat(parts, ignore_index=True)
            out = keys.merge(flags, on=['date', 'code'], how='left', validate='one_to_one')
            out['selected'] = out.selected.eq(True)
            c = base.conn(); c.register('keys', keys)
            c.sql(' UNION ALL '.join(queries)).create_view('flags')
            expected = c.sql('SELECT k.*,coalesce(selected,false) AS selected FROM keys k '
                             'LEFT JOIN flags USING(date,code) ORDER BY date,code').df()
            c.close(); pd.testing.assert_frame_equal(out, expected, check_exact=True)
            root = ROOT / (arm + '_' + str(year)); root.mkdir(parents=True, exist_ok=True)
            out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
            r = dict(protocol_sha256=sha(study.PROTOCOL), arm=arm, year=year,
                selection_sha256=sha(root / 'selection.parquet'), selected=int(out.selected.sum()),
                days=out.loc[out.selected, 'date'].nunique(), source_hashes=receipts.copy(),
                year_2024_and_2025_are_exploratory=True, new_group_outcomes_read=False,
                new_2026_prices_read=False, no_exit_rules=True, software_compilation_verified=False)
            save_json(root / 'selection_report.json', r)
            save_json(root / 'selection_verification.json', dict(passed=True,
                selection_report_sha256=sha(root / 'selection_report.json'), rows=len(out),
                all_metadata_and_original_pool_preserved=True, all_half_flags_and_unselected_history_rows_sql_verified=True,
                no_training_period_selection=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True))
            records.append(dict(group=arm + '_' + str(year), root=str(root), arm=arm, year=year,
                selected=r['selected'], days=r['days'], selection_report_sha256=sha(root / 'selection_report.json'),
                selection_verification_sha256=sha(root / 'selection_verification.json')))
            for control in control_paths:
                equality.append(dict(left=str(root), right=str(control), full_frame_equal=out.equals(checked_selection(control))))
        left, right = [ROOT / (arm + '_' + str(year)) for arm in study.ARMS]
        equality.append(dict(left=str(left), right=str(right),
            full_frame_equal=checked_selection(left).equals(checked_selection(right))))
    save_json(ROOT / 'joint_selection_freeze.json', dict(passed=True, protocol_sha256=sha(study.PROTOCOL),
        source_hashes=receipts, models=models, selections=records, equality=equality,
        all_eight_models_and_four_annual_lists_frozen_together=True, new_group_outcomes_read=False,
        six_new_models_and_two_reused_historical_models=True,
        repeated_2025_norm_fit_is_audit_only=True,
        year_2024_and_2025_are_exploratory=True, new_2026_prices_read=False, no_exit_rules=True))
    return dict(joint_sha256=sha(ROOT / 'joint_selection_freeze.json'), models=models, selections=records, equality=equality)


def checked_joint():
    study.checked(); historical.checked_proof()
    path = ROOT / 'joint_selection_freeze.json'; r = json.loads(path.read_text())
    assert r['passed'] and r['protocol_sha256'] == sha(study.PROTOCOL)
    committed = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'],
        capture_output=True, text=True, check=True).stdout
    assert sha(path) in committed
    for file, digest in r['source_hashes'].items():
        assert sha(Path(file)) == digest
    for item in r['selections']:
        root = Path(item['root']); checked_selection(root)
        for kind in ['selection_report', 'selection_verification']:
            assert item[kind + '_sha256'] == sha(root / (kind + '.json'))
    return r


def analyze():
    joint = checked_joint(); master = study.checked(); records = []
    controls = {year: [Path(v) for k, v in master['controls'].items() if k.endswith(str(year))] for year in [2024, 2025]}
    controls[2025].insert(0, historical.OLD_ANNUAL)
    for item in joint['selections']:
        root = Path(item['root']); year = item['year']; f = checked_selection(root)
        same = next((r for r in controls[year] if f.equals(checked_selection(r))), None)
        already = (root / 'analysis_report.json').exists()
        if not already:
            if same is not None:
                reuse(root, same); evaluation.attach_labels(root)
                r = json.loads((same / 'analysis_report.json').read_text())
                r.update(protocol_sha256=sha(study.PROTOCOL), selection_report_sha256=sha(root / 'selection_report.json'),
                    label_report_sha256=sha(root / 'full_label_report.json'), reused_analysis_report_sha256=sha(same / 'analysis_report.json'))
                (root / 'daily_summary.parquet').symlink_to((same / 'daily_summary.parquet').resolve())
                save_json(root / 'analysis_report.json', r)
            else:
                evaluation.analyze(root, study.PROTOCOL)
        if not (root / 'analysis_verification.json').exists():
            with (root / 'analysis_check_command.log').open('w') as log:
                subprocess.run(['.venv/bin/python', 'scripts/verify_tail_formula_before1000.py', 'analysis', '--root', str(root)],
                    stdout=log, check=True)
        checked_analysis(root, year); controls[year].append(root)
        records.append(dict(group=item['group'], reused_source=str(same) if same else None,
            new_economic_aggregation_run=not already and same is None, analysis_report_sha256=sha(root / 'analysis_report.json')))
        print(json.dumps(dict(completed=item['group'])), flush=True)
    save_json(ROOT / 'analysis_dispatch_verification.json', dict(passed=True, records=records,
        joint_sha256=sha(ROOT / 'joint_selection_freeze.json'), full_frame_equality_before_aggregation=True,
        no_duplicate_half_year_aggregation=True, new_2026_prices_read=False, no_exit_rules=True))
    return records


def equivalent(left, right, year):
    a, ar = checked_analysis(left, year); b, br = checked_analysis(right, year)
    if not a.equals(b):
        return False
    assert ar['summaries'] == br['summaries'] and ar['daily_summary_sha256'] == br['daily_summary_sha256']
    assert ar['label_report_sha256'] == br['label_report_sha256']
    return True


def finish():
    joint = checked_joint(); master = study.checked(); reports = {}; refs = []; reused = []; done = []
    roots = {**{k: Path(v) for k, v in master['controls'].items()},
             **{i['group']: Path(i['root']) for i in joint['selections']}}
    for right, suffix in [('norm_full_2025', 'corrected48'), ('fixed_2025', 'fixed48')]:
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
    path = ROOT / 'complete_results_manifest.json'; assert not path.exists()
    save_json(path, dict(passed=True, protocol_sha256=sha(study.PROTOCOL),
        joint_sha256=sha(ROOT / 'joint_selection_freeze.json'), reports=reports,
        all_four_annual_groups_and_ten_comparisons_complete=True, year_2024_and_2025_are_exploratory=True,
        new_2026_prices_read=False, no_exit_rules=True))
    return dict(completion_sha256=sha(path), logical_comparisons=10,
        numerical_comparisons=sum(not d.get('historical', False) for d in done))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['freeze', 'analyze', 'finish'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
