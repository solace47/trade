"""Jointly freeze five chronological 2024 lists before reading their outcomes."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess

import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research import tail_formula_quarter_history2024_model as model
from trade_research import tail_formula_quarter_history2024_scores as scoring
from trade_research.corporate_cash import save_json, sha
from audit_tail_formula_reference_coverage import audit
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared
from reuse_tail_formula_selected_analysis import reuse

ROOT = model.ROOT
PROTOCOL = model.PROTOCOL
GROUPS = ['full', 'minimum', 'both', 'full_only', 'minimum_only']
PERIODS = ['2024H1', '2024H2', '2024']
META = ['date', 'code', 'half', 'board', 'decision_shares']


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['fixed_groups'] == GROUPS and p['primary'] == 'both'
    assert p['comparisons'] == ['both_minus_full', 'both_minus_minimum']
    assert p['years']['evaluation_first'] == '2024-01-01' and p['years']['evaluation_end'] == '2025-01-01'
    assert p['year_2024_is_exploratory'] and not p['new_2026_prices_allowed']
    folds = {}; hashes = {str(PROTOCOL): sha(PROTOCOL), str(Path(__file__)): sha(Path(__file__))}
    for fold in ['h1', 'h2']:
        model.setup(fold); child, m = scoring.checked(); root = ROOT / fold
        r = json.loads((root / 'score_report.json').read_text())
        v = json.loads((root / 'score_verification.json').read_text())
        assert v['passed'] and v['score_report_sha256'] == sha(root / 'score_report.json')
        for key, file in [('scores_sha256', root / 'scores.parquet'),
            ('model_report_sha256', root / 'model_report.json'),
            ('model_verification_sha256', root / 'model_verification.json'),
            ('implementation_sha256', Path(scoring.__file__))]:
            assert r[key] == sha(file)
        assert child['evaluation_start'] == p['folds'][fold]['evaluation_start']
        assert child['evaluation_end'] == p['folds'][fold]['evaluation_end']
        for file, digest in {**m['component_hashes'], **r['native_cores']}.items():
            assert sha(Path(file)) == digest; hashes[file] = digest
        for file in ['model_report.json', 'model_verification.json', 'score_report.json',
                     'score_verification.json', 'scores.parquet']:
            hashes[str(root / file)] = sha(root / file)
        folds[fold] = dict(child=child, model=m)
    return p, folds, hashes


def freeze():
    p, folds, hashes = checked()
    assert not (ROOT / 'joint_selection_freeze.json').exists()
    assert not any((ROOT / group / 'analysis_report.json').exists() for group in GROUPS)
    frames = []; queries = []
    for fold, spec in folds.items():
        child, m = spec['child'], spec['model']; root = ROOT / fold
        d = pd.read_parquet(root / 'scores.parquet')
        d = d.loc[d.date.ge(child['evaluation_start']) & d.date.lt(child['evaluation_end'])].copy()
        for name in ['full', 'minimum']:
            d[name] = d.formula_input_valid & d[name + '_score'].gt(m['thresholds'][name]['threshold'])
        frames.append(d[META + ['full', 'minimum']])
        queries.append(f'''SELECT date,code,half,board,decision_shares,
            coalesce(formula_input_valid AND full_score>{m['thresholds']['full']['threshold']:.17e},false) AS "full",
            coalesce(formula_input_valid AND minimum_score>{m['thresholds']['minimum']['threshold']:.17e},false) AS minimum
            FROM read_parquet('{root}/scores.parquet')
            WHERE date>='{child['evaluation_start']}' AND date<'{child['evaluation_end']}' ''')
    parents = pd.concat(frames, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    assert not parents.duplicated(['date', 'code']).any()
    features = base.feature_inputs()
    keys = features.loc[features.date.ge(p['years']['evaluation_first']) & features.date.lt(p['years']['evaluation_end']), META]
    keys = keys.sort_values(['date', 'code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(parents[META], keys, check_exact=True)
    c = base.conn()
    c.sql(' UNION ALL '.join(queries)).create_view('parents')
    expected = c.sql('SELECT * FROM parents ORDER BY date,code').df()
    pd.testing.assert_frame_equal(parents, expected, check_exact=True)
    flags = dict(full=parents.full, minimum=parents.minimum, both=parents.full & parents.minimum,
                 full_only=parents.full & ~parents.minimum, minimum_only=parents.minimum & ~parents.full)
    assert ((flags['both'].astype(int) + flags['full_only'] + flags['minimum_only']) ==
            (flags['full'] | flags['minimum']).astype(int)).all()
    expressions = dict(full='"full"', minimum='minimum', both='"full" AND minimum',
                       full_only='"full" AND NOT minimum', minimum_only='minimum AND NOT "full"')
    records = []; saved = {}
    for group in GROUPS:
        root = ROOT / group; root.mkdir(parents=True, exist_ok=True)
        out = keys.copy(); out['selected'] = flags[group]
        check = c.sql('SELECT ' + ','.join(META) + ',' + expressions[group]
                      + ' AS selected FROM parents ORDER BY date,code').df()
        pd.testing.assert_frame_equal(out, check, check_exact=True)
        out.to_parquet(root / 'selection.parquet', index=False, compression='zstd'); saved[group] = out
        report = dict(protocol_sha256=sha(PROTOCOL), group=group, source_hashes=hashes,
            selection_sha256=sha(root / 'selection.parquet'), selected=int(out.selected.sum()),
            days=out.loc[out.selected, 'date'].nunique(), year_2024_is_exploratory=True,
            rule_selected_after_2025_results=True, new_group_outcomes_read=False,
            software_compilation_verified=False, new_2026_prices_read=False, no_exit_rules=True)
        save_json(root / 'selection_report.json', report)
        proof = dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'), rows=len(out),
            original_2024_universe_and_all_nonselection_fields_preserved=True,
            all_flags_independently_rebuilt_from_four_parent_scores_and_fixed_dates=True,
            three_remainders_disjoint_and_exhaust_parent_union=True,
            new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
        save_json(root / 'selection_verification.json', proof)
        records.append(dict(group=group, root=str(root), selected=report['selected'], days=report['days'],
            selection_report_sha256=sha(root / 'selection_report.json'),
            selection_verification_sha256=sha(root / 'selection_verification.json')))
    c.close()
    equality = [dict(left=a, right=b, all_fields_equal=saved[a].equals(saved[b]))
                for a, b in itertools.combinations(GROUPS, 2)]
    save_json(ROOT / 'selection_equality_verification.json', dict(passed=True,
        full_frame_equality_not_selected_key_only=True, comparisons=equality, before_economic_aggregation=True))
    joint = dict(passed=True, protocol_sha256=sha(PROTOCOL), source_hashes=hashes, selections=records,
        selection_equality_verification_sha256=sha(ROOT / 'selection_equality_verification.json'),
        all_ten_components_six_native_cores_and_five_annual_lists_frozen_together=True,
        year_2024_is_exploratory=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'joint_selection_freeze.json', joint)
    return dict(joint_sha256=sha(ROOT / 'joint_selection_freeze.json'), records=records, equality=equality)


def joint_checked():
    _, _, hashes = checked()
    joint = json.loads((ROOT / 'joint_selection_freeze.json').read_text())
    assert joint['passed'] and joint['protocol_sha256'] == sha(PROTOCOL) and joint['source_hashes'] == hashes
    assert joint['selection_equality_verification_sha256'] == sha(ROOT / 'selection_equality_verification.json')
    committed = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'], capture_output=True, text=True, check=True).stdout
    assert sha(ROOT / 'joint_selection_freeze.json') in committed
    for item in joint['selections']:
        root = Path(item['root'])
        for stage in ['selection']:
            assert item[stage + '_report_sha256'] == sha(root / (stage + '_report.json'))
            assert item[stage + '_verification_sha256'] == sha(root / (stage + '_verification.json'))
        r = json.loads((root / 'selection_report.json').read_text())
        assert r['selection_sha256'] == sha(root / 'selection.parquet')
    return joint


def analyze():
    joint = joint_checked(); controls = []; records = []
    for item in joint['selections']:
        root = Path(item['root']); selected = pd.read_parquet(root / 'selection.parquet')
        same = next((path for path in controls if selected.equals(pd.read_parquet(path / 'selection.parquet'))), None)
        already = (root / 'analysis_report.json').exists()
        if not already:
            if same is not None:
                reuse(root, same); evaluation.attach_labels(root)
                r = json.loads((same / 'analysis_report.json').read_text())
                r.update(protocol_sha256=sha(PROTOCOL), selection_report_sha256=sha(root / 'selection_report.json'),
                    label_report_sha256=sha(root / 'full_label_report.json'), reused_analysis_report_sha256=sha(same / 'analysis_report.json'))
                (root / 'daily_summary.parquet').symlink_to((same / 'daily_summary.parquet').resolve())
                save_json(root / 'analysis_report.json', r)
            else:
                evaluation.analyze(root, PROTOCOL)
        if not (root / 'analysis_verification.json').exists():
            with (root / 'analysis_check_command.log').open('w') as log:
                subprocess.run(['.venv/bin/python', 'scripts/verify_tail_formula_before1000.py', 'analysis', '--root', str(root)],
                               stdout=log, check=True)
        proof = json.loads((root / 'analysis_verification.json').read_text())
        assert proof['passed'] and proof['analysis_report_sha256'] == sha(root / 'analysis_report.json')
        assert json.loads((root / 'analysis_report.json').read_text())['selection_report_sha256'] == sha(root / 'selection_report.json')
        records.append(dict(group=item['group'], root=str(root), reused_source=str(same) if same else None,
            new_economic_aggregation_run=not already and same is None, analysis_report_sha256=sha(root / 'analysis_report.json')))
        controls.append(root); print(json.dumps(dict(completed=item['group'])), flush=True)
    save_json(ROOT / 'analysis_dispatch_verification.json', dict(passed=True, joint_sha256=sha(ROOT / 'joint_selection_freeze.json'),
        full_selection_equality_checked_before_evaluation=True, records=records, no_duplicate_half_year_aggregation=True,
        year_2024_is_exploratory=True, new_2026_prices_read=False, no_exit_rules=True))
    return records


def finish():
    joint_checked(); reports = {}
    for group in GROUPS:
        root = ROOT / group
        v = json.loads((root / 'analysis_verification.json').read_text())
        assert v['passed'] and v['analysis_report_sha256'] == sha(root / 'analysis_report.json')
        if not (root / 'reference_coverage_verification.json').exists():
            audit(root, years=(2024,))
        for file in ['analysis_report.json', 'analysis_verification.json', 'reference_coverage_verification.json']:
            reports[str(root / file)] = sha(root / file)
    protocol = json.loads(Path('config/tail_formula_weekday_shared_unknowns_protocol.json').read_text())
    protocol.update(objective='季度双重确认2024时间顺序扩展对两父版完整同日及共享未知比较；全部五组保留。',
                    periods=PERIODS, signal_range=['2024-01-01', '2024-12-31'], year_2024_is_exploratory=True)
    specs = []; out = ROOT / 'both'
    for name in ['full', 'minimum']:
        right = ROOT / name
        same_file, shared_file = out / ('same_dates_' + name + '.json'), out / ('shared_unknowns_' + name + '.json')
        if not same_file.exists():
            compare(out, right, same_file, periods=PERIODS, intersection_only=True)
        specs.append(dict(left=str(out), right=str(right), left_analysis_sha256=sha(out / 'analysis_report.json'),
            right_analysis_sha256=sha(right / 'analysis_report.json'), output=str(shared_file)))
        reports[str(same_file)] = sha(same_file)
    protocol['comparisons'] = specs
    file = Path('config') / (model.STEM + '_shared_unknowns_protocol.json')
    if file.exists():
        assert json.loads(file.read_text()) == protocol
    else:
        save_json(file, protocol)
    protocol['protocol_sha256'] = sha(file)
    for spec in specs:
        if not Path(spec['output']).exists():
            shared(spec, protocol)
        reports[spec['output']] = sha(Path(spec['output']))
    for file in reports:
        assert json.loads(Path(file).read_text()).get('passed', True)
    complete = ROOT / 'complete_results_manifest.json'; assert not complete.exists()
    save_json(complete, dict(passed=True, protocol_sha256=sha(PROTOCOL),
        joint_sha256=sha(ROOT / 'joint_selection_freeze.json'), reports=reports,
        five_groups_and_both_parent_comparisons_complete=True, year_2024_is_exploratory=True,
        rule_selected_after_2025_results=True, new_2026_prices_read=False, no_exit_rules=True))
    return dict(completion_sha256=sha(complete))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'analyze', 'finish'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
