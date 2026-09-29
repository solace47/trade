"""Freeze and evaluate the overlap and both remainders of two fixed formulas."""
import argparse
import json
from pathlib import Path
import re
import subprocess

import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research.corporate_cash import save_json, sha
from reuse_tail_formula_selected_analysis import reuse
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared
from audit_tail_formula_reference_coverage import audit

STEM = 'tail_formula_quarter_agreement'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
GROUPS = ['both', 'original_only', 'minimum_only']
CONTROL = Path('data/research/tail_formula_before1000/evaluation/tail_formula_float_2025')


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['groups'] == GROUPS and p['primary'] == 'both'
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for fold, specs in p['folds'].items():
        for side, folder in specs.items():
            root = Path(folder)
            stages = ['selection', 'analysis'] if fold == '2025' else ['selection', 'model', 'score']
            for stage in stages:
                v = json.loads((root / (stage + '_verification.json')).read_text())
                assert v['passed'] and v[stage + '_report_sha256'] == sha(root / (stage + '_report.json'))
            r = json.loads((root / 'selection_report.json').read_text())
            assert r['selection_sha256'] == sha(root / 'selection.parquet')
            if fold != '2025':
                m = json.loads((root / 'model_report.json').read_text())
                score = json.loads((root / 'score_report.json').read_text())
                assert r['core_sha256'] == sha(root / 'frozen_numeric_core.tdx')
                assert score['scores_sha256'] == sha(root / 'scores.parquet')
                assert score['model_report_sha256'] == r['model_report_sha256'] == sha(root / 'model_report.json')
                cut = m['thresholds'][3 if side == 'original' else 0]
                assert r['chosen_threshold'] == cut and cut['training_quantile'] == .995
                assert len(m['feature_names']) == 48 and m['last_observation'] < r['evaluation_start']
                if side == 'original':
                    assert len(m['trees']) == 64 and m['learning_rate'] == .05 and m['variant'] == 'relative'
                else:
                    assert m['model_family'] == 'minimum_of_four_frozen_quarter_models'
                    assert len(m['components']) == 4 and all(len(c['trees']) == 64 for c in m['components'])
                    assert not m['new_model_fitting_performed']
    assert not p['new_2026_prices_allowed']
    return p


def core_for_fold(p, fold):
    roots = [Path(p['folds'][fold][side]) for side in ['original', 'minimum']]
    sources = [(root / 'frozen_numeric_core.tdx').read_text() for root in roots]
    cuts = [json.loads((root / 'selection_report.json').read_text())['chosen_threshold']['threshold'] for root in roots]
    prefixes = []; bodies = []; endings = []
    for source, marker in zip(sources, ['T01:=', 'QMT001:=']):
        prefix, tail = source.split(marker, 1)
        body, ending = (marker + tail).rsplit('CORE:', 1)
        prefixes.append(prefix); bodies.append(body); endings.append('CORE:' + ending)
    assert prefixes[0] == prefixes[1], 'Both parents must use exactly the same 48 native inputs'
    first = re.sub(r'\bSC\b', 'ORSC', bodies[0])
    second = re.sub(r'\bSC\b', 'MISC', bodies[1])
    clause = f'CORE:(ORSC>{cuts[0]:.17e}) AND (MISC>{cuts[1]:.17e});\n'
    text = prefixes[0] + first + second + clause
    assert prefixes[0] + re.sub(r'\bORSC\b', 'SC', first) + endings[0] == sources[0]
    assert prefixes[1] + re.sub(r'\bMISC\b', 'SC', second) + endings[1] == sources[1]
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', text)
    assert len(names) == len({name.casefold() for name in names})
    assert len(re.findall(r'(?m)^T[0-9]{2}:=', first)) == 64
    assert len(re.findall(r'(?m)^QMT[0-9]{3}:=', second)) == 256
    assert 'MISC:=MIN(QMS1,MIN(QMS2,MIN(QMS3,QMS4)));' in second
    return text


def freeze():
    p = checked()
    assert not (ROOT / 'joint_selection_freeze.json').exists()
    assert not any((ROOT / g / 'analysis_report.json').exists() for g in GROUPS)
    a, b = [pd.read_parquet(Path(p['folds']['2025'][s]) / 'selection.parquet') for s in ['original', 'minimum']]
    pd.testing.assert_frame_equal(a.drop(columns='selected'), b.drop(columns='selected'), check_exact=True)
    flags = {'both': a.selected & b.selected, 'original_only': a.selected & ~b.selected, 'minimum_only': b.selected & ~a.selected}
    assert ((flags['both'].astype(int) + flags['original_only'] + flags['minimum_only']) == (a.selected | b.selected).astype(int)).all()
    c = base.conn()
    c.register('keys', a.drop(columns='selected'))
    parts = []
    for fold in ['2024', 'recent']:
        roots = [Path(p['folds'][fold][s]) for s in ['original', 'minimum']]
        reports = [json.loads((r / 'selection_report.json').read_text()) for r in roots]
        assert all(reports[0][k] == reports[1][k] for k in ['evaluation_start', 'evaluation_end'])
        cuts = [r['chosen_threshold']['threshold'] for r in reports]
        parts.append(f'''SELECT a.date,a.code,a.formula_input_valid AND a.score>{cuts[0]:.17e} AS original_flag,
            b.formula_input_valid AND b.score>{cuts[1]:.17e} AS minimum_flag
            FROM read_parquet('{roots[0]}/scores.parquet') a JOIN read_parquet('{roots[1]}/scores.parquet') b USING(date,code)
            WHERE a.date>='{reports[0]['evaluation_start']}' AND a.date<'{reports[0]['evaluation_end']}' ''')
    c.sql(' UNION ALL '.join(parts)).create_view('parent_flags')
    assert c.sql('SELECT count(*)=count(DISTINCT (date,code)) FROM parent_flags').fetchone()[0]
    expected = c.sql('''SELECT k.*,coalesce(original_flag,false) AS a,coalesce(minimum_flag,false) AS b
        FROM keys k LEFT JOIN parent_flags USING(date,code) ORDER BY date,code''').df()
    pd.testing.assert_series_equal(expected.a, a.selected, check_names=False, check_exact=True)
    pd.testing.assert_series_equal(expected.b, b.selected, check_names=False, check_exact=True)
    c.register('expected', expected)
    ROOT.mkdir(parents=True, exist_ok=True)
    cores = {}
    for fold in ['2024', 'recent']:
        path = ROOT / (fold + '_numeric_core.tdx')
        text = core_for_fold(p, fold)
        path.write_text(text)
        assert path.read_text() == text
        cores[str(path)] = sha(path)
    records = []
    for group, expression in [('both', 'a AND b'), ('original_only', 'a AND NOT b'), ('minimum_only', 'b AND NOT a')]:
        root = ROOT / group
        root.mkdir(parents=True, exist_ok=True)
        out = a.copy()
        out['selected'] = flags[group]
        check = c.sql(f'SELECT date,code,half,board,decision_shares,{expression} AS selected FROM expected ORDER BY date,code').df()
        pd.testing.assert_frame_equal(out, check, check_exact=True)
        out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
        report = dict(protocol_sha256=sha(PROTOCOL), group=group, parent_annual_selections=p['folds']['2025'],
            selection_sha256=sha(root / 'selection.parquet'), selected=int(out.selected.sum()),
            days=out.loc[out.selected, 'date'].nunique(), shared_intersection_cores=cores,
            new_model_fitted=False, parent_results_previously_seen=True, new_group_outcomes_read=False,
            year_2025_is_exploratory=True, new_2026_prices_read=False, no_exit_rules=True, software_compilation_verified=False)
        save_json(root / 'selection_report.json', report)
        proof = dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'), rows=len(out),
            all_flags_rebuilt_from_four_parent_scores_and_dates=True, all_nonselection_fields_unchanged=True,
            groups_are_disjoint_and_exhaust_parent_union=True, both_parent_cores_exactly_restored_and_original_minimum_preserved=True,
            new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
        save_json(root / 'selection_verification.json', proof)
        records.append(dict(group=group, root=str(root), selected=report['selected'], days=report['days'],
            selection_report_sha256=sha(root / 'selection_report.json'), selection_verification_sha256=sha(root / 'selection_verification.json')))
    c.close()
    joint = dict(passed=True, protocol_sha256=sha(PROTOCOL), selections=records, core_hashes=cores,
        all_three_annual_lists_and_both_fold_cores_frozen_together=True, new_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'joint_selection_freeze.json', joint)
    return dict(joint_sha256=sha(ROOT / 'joint_selection_freeze.json'), records=records, cores=cores)


def analyze():
    p = checked()
    joint = json.loads((ROOT / 'joint_selection_freeze.json').read_text())
    assert joint['passed'] and joint['protocol_sha256'] == sha(PROTOCOL)
    committed = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'], capture_output=True, text=True, check=True).stdout
    assert sha(ROOT / 'joint_selection_freeze.json') in committed
    controls = [Path(x) for x in p['folds']['2025'].values()] + [CONTROL]
    records = []
    for item in joint['selections']:
        root = Path(item['root'])
        assert item['selection_report_sha256'] == sha(root / 'selection_report.json')
        assert item['selection_verification_sha256'] == sha(root / 'selection_verification.json')
        selected = pd.read_parquet(root / 'selection.parquet')
        same = next((candidate for candidate in controls if selected.equals(pd.read_parquet(candidate / 'selection.parquet'))), None)
        already = (root / 'analysis_report.json').exists()
        if not already:
            if same is not None:
                reuse(root, same)
                evaluation.attach_labels(root)
                r = json.loads((same / 'analysis_report.json').read_text())
                r.update(protocol_sha256=sha(PROTOCOL), selection_report_sha256=sha(root / 'selection_report.json'),
                    label_report_sha256=sha(root / 'full_label_report.json'), reused_analysis_report_sha256=sha(same / 'analysis_report.json'))
                (root / 'daily_summary.parquet').symlink_to((same / 'daily_summary.parquet').resolve())
                save_json(root / 'analysis_report.json', r)
            else:
                evaluation.analyze(root, PROTOCOL)
            with (root / 'analysis_check_command.log').open('w') as log:
                subprocess.run(['.venv/bin/python', 'scripts/verify_tail_formula_before1000.py', 'analysis', '--root', str(root)], stdout=log, check=True)
        proof = json.loads((root / 'analysis_verification.json').read_text())
        assert proof['passed'] and proof['analysis_report_sha256'] == sha(root / 'analysis_report.json')
        assert json.loads((root / 'analysis_report.json').read_text())['selection_report_sha256'] == sha(root / 'selection_report.json')
        records.append(dict(group=item['group'], root=str(root), reused_source=str(same) if same else None,
            new_economic_aggregation_run=not already and same is None, analysis_report_sha256=sha(root / 'analysis_report.json')))
        controls.append(root)
    save_json(ROOT / 'analysis_dispatch_verification.json', dict(passed=True, joint_sha256=sha(ROOT / 'joint_selection_freeze.json'),
        full_selection_equality_checked_before_evaluation=True, records=records, no_duplicate_half_year_aggregation=True,
        new_2026_prices_read=False, no_exit_rules=True))
    return records


def finish():
    p = checked()
    reports = {}
    for group in GROUPS:
        root = ROOT / group
        v = json.loads((root / 'analysis_verification.json').read_text())
        assert v['passed'] and v['analysis_report_sha256'] == sha(root / 'analysis_report.json')
        audit(root)
        for file in ['analysis_report.json', 'analysis_verification.json', 'reference_coverage_verification.json']:
            reports[str(root / file)] = sha(root / file)
    shared_protocol = json.loads(Path('config/tail_formula_weekday_shared_unknowns_protocol.json').read_text())
    shared_protocol['objective'] = '双重确认与原48、季度最低分和旧固定48的完整同日及共享未知比较；两个独有组另完整保留。'
    specs = []
    out = ROOT / 'both'
    for name, right in [('original', Path(p['folds']['2025']['original'])), ('minimum', Path(p['folds']['2025']['minimum'])), ('fixed48', CONTROL)]:
        same_file, shared_file = out / ('same_dates_' + name + '.json'), out / ('shared_unknowns_' + name + '.json')
        compare(out, right, same_file, intersection_only=True)
        specs.append(dict(left=str(out), right=str(right), left_analysis_sha256=sha(out / 'analysis_report.json'),
            right_analysis_sha256=sha(right / 'analysis_report.json'), output=str(shared_file)))
        reports[str(same_file)] = sha(same_file)
    shared_protocol['comparisons'] = specs
    file = Path('config') / (STEM + '_shared_unknowns_protocol.json')
    assert not file.exists()
    save_json(file, shared_protocol)
    shared_protocol['protocol_sha256'] = sha(file)
    for spec in specs:
        shared(spec, shared_protocol)
        reports[spec['output']] = sha(Path(spec['output']))
    for file in reports:
        assert json.loads(Path(file).read_text()).get('passed', True)
    complete = ROOT / 'complete_results_manifest.json'
    assert not complete.exists()
    save_json(complete, dict(passed=True, protocol_sha256=sha(PROTOCOL), reports=reports,
        three_groups_and_three_parent_comparisons_complete=True, new_2026_prices_read=False, no_exit_rules=True))
    return dict(completion_sha256=sha(complete))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'analyze', 'finish'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
