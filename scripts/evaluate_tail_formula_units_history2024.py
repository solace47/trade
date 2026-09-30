"""Freeze all six price-unit models before the complete 2024 comparisons."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess

import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research import tail_formula_units_history2024 as study
from trade_research import tail_formula_units_history2024_model as model
from trade_research.corporate_cash import save_json, sha
from audit_tail_formula_reference_coverage import audit
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared
from freeze_tail_formula_bipower_gap import checked_selection
from reuse_tail_formula_selected_analysis import reuse

ROOT = study.ROOT
PERIODS = ['2024H1', '2024H2', '2024']
META = study.META[:-1]


def checked_analysis(root):
    """Validate the 2024 evaluation explicitly; the existing reuse helper is 2025-only."""
    frame = checked_selection(root)
    assert frame.date.between('2024-01-01', '2024-12-31').all()
    report = json.loads((root / 'analysis_report.json').read_text())
    proof = json.loads((root / 'analysis_verification.json').read_text())
    assert proof['passed'] and proof['analysis_report_sha256'] == sha(root / 'analysis_report.json')
    assert report['selection_report_sha256'] == sha(root / 'selection_report.json')
    assert report['daily_summary_sha256'] == sha(root / 'daily_summary.parquet')
    assert report['reference_label'] == '09:59'
    labels = json.loads((root / 'full_label_report.json').read_text())
    label_proof = json.loads((root / 'full_label_verification.json').read_text())
    assert label_proof['passed'] and label_proof['label_report_sha256'] == sha(root / 'full_label_report.json')
    assert report['label_report_sha256'] == sha(root / 'full_label_report.json')
    assert labels['labels_sha256'] == sha(root / 'full_labels.parquet')
    return frame, report


def checked_models():
    master = study.checked(); configs = {}; receipts = {}
    for arm in study.ARMS:
        for fold in ['h1', 'h2']:
            p = model.setup(arm, fold); root = base.ROOT
            m = json.loads((root / 'model_report.json').read_text())
            r = json.loads((root / 'score_report.json').read_text())
            for stage in ['model', 'score']:
                path = root / (stage + '_report.json')
                v = json.loads((root / (stage + '_verification.json')).read_text())
                assert v['passed'] and v[stage + '_report_sha256'] == sha(path)
            assert r['protocol_sha256'] == m['protocol_sha256'] == sha(base.PROTOCOL)
            assert r['model_report_sha256'] == sha(root / 'model_report.json')
            assert r['scores_sha256'] == sha(root / 'scores.parquet')
            assert r['feature_report_sha256'] == m['feature_report_sha256'] == sha(study.INPUTS / 'feature_report.json')
            assert m['feature_names'] == list(study.ARMS[arm])
            assert m['training_start'] == p['training_start'] and m['training_end'] == p['training_end']
            assert m['last_observation'] == p['expected_last_observation'] < p['evaluation_start']
            assert m['thresholds'][3]['training_quantile'] == master['threshold'] == .995
            configs[(arm, fold)] = (p, m)
            for path in [base.PROTOCOL, *(root / name for name in [
                'model_report.json', 'model_verification.json', 'score_report.json',
                'score_verification.json', 'scores.parquet'])]:
                receipts[str(path)] = sha(path)
    return master, configs, receipts


def freeze():
    p, configs, receipts = checked_models()
    assert not (ROOT / 'joint_selection_freeze.json').exists()
    assert not any((ROOT / arm / 'analysis_report.json').exists() for arm in study.ARMS)
    f = base.feature_inputs()
    keys = f.loc[f.date.ge('2024-01-01') & f.date.lt('2025-01-01'), META].reset_index(drop=True)
    assert len(keys) == 587276
    control = checked_selection(study.CONTROL)
    pd.testing.assert_frame_equal(keys, control[META], check_exact=True)
    saved = {'full': control}; records = []; model_rows = []
    for arm, expressions in study.ARMS.items():
        frames = []; queries = []
        for fold in ['h1', 'h2']:
            spec, m = configs[(arm, fold)]; root = ROOT / arm / fold
            d = pd.read_parquet(root / 'scores.parquet')
            d = d.loc[d.date.ge(spec['evaluation_start']) & d.date.lt(spec['evaluation_end'])].copy()
            threshold = m['thresholds'][3]['threshold']
            d['selected'] = d.formula_input_valid & d.score.gt(threshold)
            frames.append(d[META + ['selected']])
            queries.append(f'''SELECT {','.join(META)},
                coalesce(formula_input_valid AND score>{threshold:.17e},false) AS selected
                FROM read_parquet('{root}/scores.parquet')
                WHERE date>='{spec['evaluation_start']}' AND date<'{spec['evaluation_end']}' ''')
            core = root / 'frozen_numeric_core.tdx'
            core.write_text(base.native_core(m, threshold, expressions, study.HEADER))
            receipts[str(core)] = sha(core)
            v = json.loads((root / 'model_verification.json').read_text())
            model_rows.append(dict(arm=arm, fold=fold, training_rows=m['rows'], training_days=m['days'],
                last_observation=m['last_observation'], node_checks=v['node_checks'],
                added_unit_node_uses=sum(i >= 48 for t in m['trees'] for i in t['feature'])
                    if arm != 'raw' else None,
                selected=int(d.selected.sum()), days=d.loc[d.selected, 'date'].nunique()))
        out = pd.concat(frames, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
        pd.testing.assert_frame_equal(out[META], keys, check_exact=True)
        c = base.conn()
        expected = c.sql('SELECT * FROM (' + ' UNION ALL '.join(queries) + ') ORDER BY date,code').df()
        c.close(); pd.testing.assert_frame_equal(out, expected, check_exact=True)
        root = ROOT / arm
        out.to_parquet(root / 'selection.parquet', index=False, compression='zstd'); saved[arm] = out
        report = dict(protocol_sha256=sha(study.PROTOCOL), arm=arm, source_hashes=receipts.copy(),
            selection_sha256=sha(root / 'selection.parquet'), selected=int(out.selected.sum()),
            days=out.loc[out.selected, 'date'].nunique(), year_2024_is_exploratory=True,
            rule_selected_after_2025_results=True, new_group_outcomes_read=False,
            software_compilation_verified=False, new_2026_prices_read=False, no_exit_rules=True)
        save_json(root / 'selection_report.json', report)
        proof = dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'), rows=len(out),
            entire_original_2024_universe_and_nonselection_fields_preserved=True,
            all_flags_sql_rebuilt_from_fixed_training_quantiles_and_chronological_scores=True,
            no_training_period_selection=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
        save_json(root / 'selection_verification.json', proof)
        records.append(dict(arm=arm, root=str(root), selected=report['selected'], days=report['days'],
            selection_report_sha256=sha(root / 'selection_report.json'),
            selection_verification_sha256=sha(root / 'selection_verification.json')))
    equality = [dict(left=a, right=b, full_frame_equal=saved[a].equals(saved[b]))
                for a, b in itertools.combinations(saved, 2)]
    joint = dict(passed=True, protocol_sha256=sha(study.PROTOCOL), source_hashes=receipts,
        models=model_rows, selections=records, equality=equality,
        all_six_models_and_three_annual_lists_frozen_together=True, new_group_outcomes_read=False,
        year_2024_is_exploratory=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'joint_selection_freeze.json', joint)
    return dict(joint_sha256=sha(ROOT / 'joint_selection_freeze.json'), models=model_rows,
                selections=records, equality=equality)


def checked_joint():
    study.checked(); path = ROOT / 'joint_selection_freeze.json'; joint = json.loads(path.read_text())
    assert joint['passed'] and joint['protocol_sha256'] == sha(study.PROTOCOL)
    committed = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'],
        capture_output=True, text=True, check=True).stdout
    assert sha(path) in committed
    for file, digest in joint['source_hashes'].items():
        assert sha(Path(file)) == digest
    for item in joint['selections']:
        root = Path(item['root']); checked_selection(root)
        for stage in ['selection_report', 'selection_verification']:
            assert item[stage + '_sha256'] == sha(root / (stage + '.json'))
    return joint


def analyze():
    joint = checked_joint(); controls = [study.CONTROL]; records = []
    for item in joint['selections']:
        root = Path(item['root']); f = checked_selection(root)
        same = next((r for r in controls if f.equals(checked_selection(r))), None)
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
        checked_analysis(root)
        records.append(dict(arm=item['arm'], root=str(root), reused_source=str(same) if same else None,
            new_economic_aggregation_run=not already and same is None, analysis_report_sha256=sha(root / 'analysis_report.json')))
        controls.append(root); print(json.dumps(dict(completed=item['arm'])), flush=True)
    save_json(ROOT / 'analysis_dispatch_verification.json', dict(passed=True,
        joint_sha256=sha(ROOT / 'joint_selection_freeze.json'), records=records,
        full_table_equality_before_aggregation=True, no_duplicate_half_year_aggregation=True,
        year_2024_is_exploratory=True, new_2026_prices_read=False, no_exit_rules=True))
    return records


def equivalent_analysis(left, right):
    a, ar = checked_analysis(left); b, br = checked_analysis(right)
    if not a.equals(b):
        return False
    assert ar['summaries'] == br['summaries'] and ar['daily_summary_sha256'] == br['daily_summary_sha256']
    assert ar['label_report_sha256'] == br['label_report_sha256']
    return True


def finish():
    checked_joint(); master = study.checked()
    roots = {'full': study.CONTROL, **{arm: ROOT / arm for arm in study.ARMS}}
    reports = {}; references = []; previous = [study.CONTROL]
    for arm in study.ARMS:
        root = roots[arm]; checked_analysis(root)
        source = next((r for r in previous if equivalent_analysis(root, r)), root)
        file = source / 'reference_coverage_verification.json'
        if not file.exists():
            audit(source, years=(2024,))
        assert json.loads(file.read_text())['passed']
        reports[str(file)] = sha(file)
        for name in ['analysis_report.json', 'analysis_verification.json']:
            reports[str(root / name)] = sha(root / name)
        references.append(dict(arm=arm, reference_source=str(source), reused=source != root,
            source_reference_sha256=sha(file)))
        previous.append(root)
    template = json.loads(Path('config/tail_formula_weekday_shared_unknowns_protocol.json').read_text())
    template.update(objective='三种价格单位2024完整比较，保留所有日期及共同未知。', periods=PERIODS,
        signal_range=['2024-01-01', '2024-12-31'], year_2024_is_exploratory=True)
    specs = []
    for left, right in master['comparisons']:
        specs.append(dict(left=str(roots[left]), right=str(roots[right]),
            left_analysis_sha256=sha(roots[left] / 'analysis_report.json'),
            right_analysis_sha256=sha(roots[right] / 'analysis_report.json'),
            output=str(ROOT / ('shared_unknowns_' + left + '_minus_' + right + '.json'))))
    template['comparisons'] = specs
    file = Path('config') / (study.STEM + '_shared_unknowns_protocol.json')
    if file.exists():
        assert json.loads(file.read_text()) == template
    else:
        save_json(file, template)
    template['protocol_sha256'] = sha(file)
    done = []; reuse_records = []
    for (left, right), spec in zip(master['comparisons'], specs):
        a, b = roots[left], roots[right]
        match = next((record for record in done
            if equivalent_analysis(a, record['left']) and equivalent_analysis(b, record['right'])), None)
        if match is not None:
            reuse_records.append(dict(left=str(a), right=str(b), source_left=str(match['left']),
                source_right=str(match['right']), same_dates_source=str(match['same']), shared_source=str(match['shared']),
                all_selection_fields_summaries_and_labels_exact=True, numerical_results_recomputed=False))
            continue
        same_file = ROOT / ('same_dates_' + left + '_minus_' + right + '.json')
        shared_file = Path(spec['output'])
        if not same_file.exists():
            compare(a, b, same_file, periods=PERIODS, intersection_only=True)
        if not shared_file.exists():
            shared(spec, template)
        for path in [same_file, shared_file]:
            assert json.loads(path.read_text())['passed']; reports[str(path)] = sha(path)
        done.append(dict(left=a, right=b, same=same_file, shared=shared_file))
    proof = dict(passed=True, references=references, comparisons=reuse_records,
        logical_comparisons=len(specs), numerical_comparisons=len(done),
        reused_sources_bound_in_completion_manifest=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'comparison_reuse_verification.json', proof)
    reports[str(ROOT / 'comparison_reuse_verification.json')] = sha(ROOT / 'comparison_reuse_verification.json')
    path = ROOT / 'complete_results_manifest.json'; assert not path.exists()
    save_json(path, dict(passed=True, protocol_sha256=sha(study.PROTOCOL),
        joint_sha256=sha(ROOT / 'joint_selection_freeze.json'), reports=reports,
        all_three_arms_and_six_comparisons_complete=True, year_2024_is_exploratory=True,
        rule_selected_after_2025_results=True, new_2026_prices_read=False, no_exit_rules=True))
    return dict(completion_sha256=sha(path), logical_comparisons=len(specs), numerical_comparisons=len(done))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'analyze', 'finish'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
