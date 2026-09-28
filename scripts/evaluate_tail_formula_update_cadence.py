"""Freeze every calendar segment before comparing monthly and quarterly updates."""
import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd

from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research import tail_formula_update_cadence as study
from trade_research.corporate_cash import save_json, sha
from reuse_tail_formula_selected_analysis import reuse


def parts(arm):
    assert arm in ['monthly', 'quarterly']
    return [f'm{i:02d}' for i in range(1, 13)] if arm == 'monthly' else [f'q{i}' for i in range(1, 5)]


def protocol(arm, part):
    return Path('config') / f'{study.STEM}_{arm}_{"combined" if part == "2025" else part}_protocol.json'


def root_for(arm, part):
    return study.ROOT / arm / part


def config(arm, part):
    p = json.loads(protocol(arm, part).read_text())
    assert p['inputs_protocol_sha256'] == sha(study.MASTER) and p['arm'] == arm and p['window_end'] == '09:59'
    return p


def freeze():
    study.checked()
    root = study.ROOT
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / 'joint_selection_freeze.json').exists()
    models = {}
    registry = []
    for month in range(1, 13):
        source, model, cut, desc = study.validate_source(month)
        core = study.base.native_core(model, cut['threshold'], study.inputs.EXPRESSIONS, study.inputs.HEADER)
        assert (source / 'frozen_numeric_core.tdx').read_text() == core
        models[month] = (source, model, cut, core)
        registry.append(dict(month=month, source=str(source), reused_without_refit=desc['reuse_existing'],
            training_start=model['training_start'], training_end=model['training_end'], last_observation=model['last_observation'],
            training_rows=model['rows'], training_days=model['days'],
            training_includes_2025H2=model['new_2025H2_score_groups_read'],
            **{name.removesuffix('.json') + '_sha256': sha(source / name)
               for name in ['model_report.json', 'model_verification.json', 'score_report.json', 'score_verification.json']}))
    save_json(root / 'models_manifest.json', dict(passed=True, protocol_sha256=sha(study.MASTER), models=registry,
        all_month_models_precede_their_evaluation=True, new_2026_prices_read=False, no_exit_rules=True))
    for arm in ['monthly', 'quarterly']:
        aggregate = None
        fold_records = []
        for part in parts(arm):
            target = root_for(arm, part)
            target.mkdir(parents=True, exist_ok=True)
            assert not (target / 'analysis_report.json').exists()
            p = config(arm, part)
            source, model, cut, core = models[p['model_month']]
            assert model['training_end'] == p['evaluation_start']
            scores = pd.read_parquet(source / 'scores.parquet')
            selected = scores[['date', 'code', 'half', 'board', 'decision_shares']].copy()
            selected['selected'] = (scores.formula_input_valid & scores.score.gt(cut['threshold'])
                & scores.date.ge(p['evaluation_start']) & scores.date.lt(p['evaluation_end']))
            path = target / 'selection_report.json'
            if not path.exists():
                selected.to_parquet(target / 'selection.parquet', index=False, compression='zstd')
                (target / 'frozen_numeric_core.tdx').write_text(core)
                report = dict(protocol_sha256=sha(protocol(arm, part)), source_root=str(source),
                    source_model_report_sha256=sha(source / 'model_report.json'),
                    source_score_report_sha256=sha(source / 'score_report.json'),
                    core_sha256=sha(target / 'frozen_numeric_core.tdx'), selection_sha256=sha(target / 'selection.parquet'),
                    chosen_threshold=cut, selected=int(selected.selected.sum()),
                    evaluation_start=p['evaluation_start'], evaluation_end=p['evaluation_end'], model_month=p['model_month'],
                    training_includes_2025H2=model['new_2025H2_score_groups_read'],
                    new_group_outcomes_read=False, year_2025_is_exploratory=True,
                    new_2026_prices_read=False, no_exit_rules=True, software_compilation_verified=False)
                save_json(path, report)
            else:
                # A partial pre-outcome freeze can resume without regenerating any list.
                pd.testing.assert_frame_equal(pd.read_parquet(target / 'selection.parquet'), selected, check_exact=True)
            verify_part(arm, part, models[p['model_month']])
            if aggregate is None:
                aggregate = selected.copy()
                aggregate['selected'] = False
            pd.testing.assert_frame_equal(aggregate.drop(columns='selected'), selected.drop(columns='selected'), check_exact=True)
            assert not (aggregate.selected & selected.selected).any()
            aggregate['selected'] |= selected.selected
            fold_records.append(dict(part=part, root=str(target), model_month=p['model_month'],
                start=p['evaluation_start'], end=p['evaluation_end'], selection_report_sha256=sha(path)))
        assert fold_records[0]['start'] == '2025-01-01' and fold_records[-1]['end'] == '2026-01-01'
        assert all(a['end'] == b['start'] for a, b in zip(fold_records, fold_records[1:]))
        combined = root_for(arm, '2025')
        combined.mkdir(parents=True, exist_ok=True)
        assert not (combined / 'analysis_report.json').exists()
        if not (combined / 'selection_report.json').exists():
            aggregate.to_parquet(combined / 'selection.parquet', index=False, compression='zstd')
            save_json(combined / 'selection_report.json', dict(protocol_sha256=sha(protocol(arm, '2025')),
                models_manifest_sha256=sha(root / 'models_manifest.json'), folds=fold_records,
                selection_sha256=sha(combined / 'selection.parquet'), selected=int(aggregate.selected.sum()),
                by_half=aggregate.groupby('half').selected.agg(['size', 'sum']).reset_index().to_dict('records'),
                updates_per_year=len(fold_records), identical_coefficients_all_year=False,
                all_evaluation_dates_after_their_model_training=True, new_group_outcomes_read=False,
                year_2025_is_exploratory=True, new_2026_prices_read=False, no_exit_rules=True,
                software_compilation_verified=False))
        verify_combined(arm, models)
    records = []
    for arm in ['monthly', 'quarterly']:
        for part in [*parts(arm), '2025']:
            target = root_for(arm, part)
            frame = pd.read_parquet(target / 'selection.parquet')
            chosen = frame.loc[frame.selected]
            same = []
            candidates = [Path(r['root']) for r in records]
            if part == '2025':
                candidates += [Path('data/research/tail_formula_before1000_model_2025'),
                    Path('data/research/tail_formula_before1000/evaluation/tail_formula_float_2025')]
            for candidate in candidates:
                if frame.equals(pd.read_parquet(candidate / 'selection.parquet')):
                    same.append(str(candidate))
            records.append(dict(arm=arm, part=part, root=str(target), protocol=str(protocol(arm, part)),
                selected=len(chosen), days=chosen.date.nunique(), identical_full_selection_sources=same,
                selection_report_sha256=sha(target / 'selection_report.json'),
                selection_verification_sha256=sha(target / 'selection_verification.json')))
    assert len(records) == 18
    save_json(root / 'joint_selection_freeze.json', dict(passed=True, protocol_sha256=sha(study.MASTER),
        models_manifest_sha256=sha(root / 'models_manifest.json'), selections=records,
        all_twelve_models_and_eighteen_lists_frozen_together=True, new_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True))
    return dict(joint_sha256=sha(root / 'joint_selection_freeze.json'), selections=records)


def verify_part(arm, part, model_data):
    target = root_for(arm, part)
    p = config(arm, part)
    source, model, cut, core = model_data
    r = json.loads((target / 'selection_report.json').read_text())
    for key, path in [('protocol_sha256', protocol(arm, part)), ('source_model_report_sha256', source / 'model_report.json'),
                      ('source_score_report_sha256', source / 'score_report.json'), ('selection_sha256', target / 'selection.parquet'),
                      ('core_sha256', target / 'frozen_numeric_core.tdx')]:
        assert r[key] == sha(path)
    assert r['source_root'] == str(source) and r['chosen_threshold'] == cut
    assert r['evaluation_start'] == p['evaluation_start'] == model['training_end']
    assert r['evaluation_end'] == p['evaluation_end'] and model['last_observation'] < p['evaluation_start']
    assert (target / 'frozen_numeric_core.tdx').read_text() == core
    c = study.base.conn()
    expected = c.sql(f"""SELECT date,code,half,board,decision_shares,
        formula_input_valid AND score>{cut['threshold']:.17e} AND date>='{p['evaluation_start']}'
        AND date<'{p['evaluation_end']}' AS selected FROM read_parquet('{source}/scores.parquet') ORDER BY date,code""").df()
    c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(target / 'selection.parquet'), expected, check_exact=True)
    assert int(expected.selected.sum()) == r['selected']
    save_json(target / 'selection_verification.json', dict(passed=True,
        selection_report_sha256=sha(target / 'selection_report.json'), rows=len(expected),
        all_calendar_boundaries_thresholds_and_flags_sql_verified=True,
        no_observation_at_or_after_evaluation_start_in_training=True,
        native_core_identical_to_its_frozen_model=True, new_2026_prices_read=False, no_exit_rules=True))


def verify_combined(arm, models):
    target = root_for(arm, '2025')
    r = json.loads((target / 'selection_report.json').read_text())
    assert r['protocol_sha256'] == sha(protocol(arm, '2025'))
    assert r['models_manifest_sha256'] == sha(study.ROOT / 'models_manifest.json')
    assert r['selection_sha256'] == sha(target / 'selection.parquet')
    queries = []
    for fold in r['folds']:
        p = config(arm, fold['part'])
        source, model, cut, _ = models[p['model_month']]
        assert fold['selection_report_sha256'] == sha(Path(fold['root']) / 'selection_report.json')
        assert fold['start'] == p['evaluation_start'] and fold['end'] == p['evaluation_end']
        assert model['last_observation'] < p['evaluation_start']
        queries.append(f"SELECT date,code FROM read_parquet('{source}/scores.parquet') WHERE formula_input_valid "
                       f"AND score>{cut['threshold']:.17e} AND date>='{p['evaluation_start']}' AND date<'{p['evaluation_end']}'")
    assert len(queries) == r['updates_per_year'] == len(parts(arm))
    c = study.base.conn()
    c.sql(' UNION ALL '.join(queries)).create_view('chosen')
    assert c.sql('SELECT count(*)-count(DISTINCT(date,code)) FROM chosen').fetchone()[0] == 0
    expected = c.sql(f"""SELECT k.date,k.code,k.half,k.board,k.decision_shares,s.code IS NOT NULL AS selected
        FROM read_parquet('{study.inputs.ROOT}/features.parquet') k LEFT JOIN chosen s USING(date,code)
        ORDER BY date,code""").df()
    c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(target / 'selection.parquet'), expected, check_exact=True)
    assert int(expected.selected.sum()) == r['selected']
    save_json(target / 'selection_verification.json', dict(passed=True,
        selection_report_sha256=sha(target / 'selection_report.json'), rows=len(expected),
        all_flags_rebuilt_directly_from_time_bound_model_scores=True, no_overlapping_or_missing_calendar_segments=True,
        all_evaluation_dates_after_training=True, new_2026_prices_read=False, no_exit_rules=True))


def analyze():
    root = study.ROOT
    joint = json.loads((root / 'joint_selection_freeze.json').read_text())
    assert joint['passed'] and joint['protocol_sha256'] == sha(study.MASTER) and len(joint['selections']) == 18
    assert joint['models_manifest_sha256'] == sha(root / 'models_manifest.json')
    assert not (root / 'analysis_dispatch_verification.json').exists()
    records = []
    for frozen in joint['selections']:
        target = Path(frozen['root'])
        assert frozen['selection_report_sha256'] == sha(target / 'selection_report.json')
        assert frozen['selection_verification_sha256'] == sha(target / 'selection_verification.json')
        if not (target / 'analysis_report.json').exists():
            candidates = frozen['identical_full_selection_sources']
            if candidates:
                source = Path(candidates[0])
                reuse(target, source)
                evaluation.attach_labels(target)
                assert sha(target / 'full_labels.parquet') == sha(source / 'full_labels.parquet')
                report = json.loads((source / 'analysis_report.json').read_text())
                assert report['reference_label'] == '09:59'
                report.update(protocol_sha256=sha(Path(frozen['protocol'])),
                    selection_report_sha256=sha(target / 'selection_report.json'),
                    label_report_sha256=sha(target / 'full_label_report.json'),
                    reused_analysis_report_sha256=sha(source / 'analysis_report.json'))
                (target / 'daily_summary.parquet').symlink_to((source / 'daily_summary.parquet').resolve())
                save_json(target / 'analysis_report.json', report)
            else:
                evaluation.analyze(target, Path(frozen['protocol']))
            with (target / 'analysis_check.log').open('w') as log:
                subprocess.run(['.venv/bin/python', 'scripts/verify_tail_formula_before1000.py', 'analysis', '--root', str(target)],
                               stdout=log, check=True)
        proof = json.loads((target / 'analysis_verification.json').read_text())
        assert proof['passed'] and proof['analysis_report_sha256'] == sha(target / 'analysis_report.json')
        report = json.loads((target / 'analysis_report.json').read_text())
        primary = next(x for x in report['summaries'] if x['arm'] == 'formula' and x['bps'] == 15
                       and not x['sensitive'] and x['period'] == '2025')
        record = dict(arm=frozen['arm'], part=frozen['part'], root=str(target),
            reused_sources=frozen['identical_full_selection_sources'], analysis_report_sha256=sha(target / 'analysis_report.json'))
        records.append(record)
        print(json.dumps(dict(**record, primary=primary), ensure_ascii=False), flush=True)
    save_json(root / 'analysis_dispatch_verification.json', dict(passed=True, records=records,
        joint_selection_freeze_sha256=sha(root / 'joint_selection_freeze.json'), all_eighteen_groups_included=True,
        new_2026_prices_read=False, no_exit_rules=True))
    return dict(dispatch_sha256=sha(root / 'analysis_dispatch_verification.json'))


def finish():
    from compare_tail_formula_same_dates import compare
    from compare_tail_formula_shared_unknowns import compare as shared
    from audit_tail_formula_reference_coverage import audit
    root = study.ROOT
    assert not (root / 'complete_results_manifest.json').exists()
    dispatch = json.loads((root / 'analysis_dispatch_verification.json').read_text())
    assert dispatch['passed'] and dispatch['joint_selection_freeze_sha256'] == sha(root / 'joint_selection_freeze.json')
    assert len(dispatch['records']) == 18
    reports = {}
    for record in dispatch['records']:
        target = Path(record['root'])
        proof = json.loads((target / 'analysis_verification.json').read_text())
        assert proof['passed'] and proof['analysis_report_sha256'] == record['analysis_report_sha256'] == sha(target / 'analysis_report.json')
        for name in ['analysis_report.json', 'analysis_verification.json']:
            reports[str(target / name)] = sha(target / name)
    specs = []
    for arm in ['monthly', 'quarterly']:
        left = root_for(arm, '2025')
        controls = [('corrected48', Path('data/research/tail_formula_before1000_model_2025')),
                    ('fixed48', Path('data/research/tail_formula_before1000/evaluation/tail_formula_float_2025'))]
        if arm == 'monthly':
            controls.insert(0, ('quarterly', root_for('quarterly', '2025')))
        for name, right in controls:
            specs.append(dict(left=str(left), right=str(right), output=str(left / ('shared_unknowns_' + name + '.json')),
                left_analysis_sha256=sha(left / 'analysis_report.json'), right_analysis_sha256=sha(right / 'analysis_report.json')))
    p = json.loads(Path('config/tail_formula_weekday_shared_unknowns_protocol.json').read_text())
    p.update(objective='固定逐月与同标签季度更新，两侧各对原半年和旧固定48；五组比较全部报告。', comparisons=specs)
    path = Path('config') / (study.STEM + '_shared_unknowns_protocol.json')
    assert not path.exists()
    save_json(path, p)
    p['protocol_sha256'] = sha(path)
    for spec in specs:
        left, right, target = Path(spec['left']), Path(spec['right']), Path(spec['output'])
        name = target.stem.removeprefix('shared_unknowns_')
        other = left / ('same_dates_' + name + '.json')
        compare(left, right, other, intersection_only=True)
        shared(spec, p)
        for file in [target, other]:
            assert json.loads(file.read_text())['passed']
            reports[str(file)] = sha(file)
    for arm in ['monthly', 'quarterly']:
        target = root_for(arm, '2025')
        audit(target)
        proof = target / 'reference_coverage_verification.json'
        assert json.loads(proof.read_text())['passed']
        reports[str(proof)] = sha(proof)
    save_json(root / 'complete_results_manifest.json', dict(passed=True, protocol_sha256=sha(study.MASTER), reports=reports,
        all_eighteen_groups_and_five_controls_included=True, new_2026_prices_read=False, no_exit_rules=True))
    return dict(complete_sha256=sha(root / 'complete_results_manifest.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'analyze', 'finish'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
