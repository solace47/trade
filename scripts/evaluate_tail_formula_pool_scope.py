"""Compare expanded training and pool scope while reusing unchanged statistics."""
import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research import tail_formula_pool_scope_inputs as inputs
from trade_research import tail_formula_pool_scope_labels as labels
from trade_research import tail_formula_pool_scope_model as model
from trade_research.corporate_cash import save_json, sha
from audit_tail_formula_reference_coverage import audit
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared
from freeze_tail_formula_bipower_gap import checked_selection
from reuse_tail_formula_selected_analysis import reuse

PROTOCOL = Path('config/tail_formula_pool_scope_evaluation_protocol.json')


def checked_joint():
    master = model.checked(); p = json.loads(PROTOCOL.read_text())
    assert p['model_protocol_sha256'] == sha(model.PROTOCOL)
    assert p['planned_comparisons'] == master['planned_comparisons']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    path = inputs.ROOT/'joint_selection_freeze.json'; joint = json.loads(path.read_text())
    assert joint['passed'] and joint['protocol_sha256'] == sha(model.PROTOCOL)
    text = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'],
        text=True, capture_output=True, check=True).stdout
    assert sha(path) in text
    for file, digest in joint['source_hashes'].items():
        assert sha(Path(file)) == digest
    for item in joint['selections']:
        root = Path(item['root']); checked_selection(root)
        for name in ['selection_report', 'selection_verification']:
            assert item[name+'_sha256'] == sha(root/(name+'.json'))
    r = labels.directory('evaluation')
    v = json.loads((r/'full_label_verification.json').read_text())
    report = json.loads((r/'full_label_report.json').read_text())
    assert v['passed'] and v['label_report_sha256'] == sha(r/'full_label_report.json')
    assert report['labels_sha256'] == sha(r/'full_labels.parquet')
    evaluation.source.ROOT = r
    return master, joint


def checked_analysis(root):
    f = checked_selection(root)
    assert f.loc[f.selected, 'date'].ge('2025-01-01').all() and f.date.lt('2026-01-01').all()
    ar = json.loads((root/'analysis_report.json').read_text())
    av = json.loads((root/'analysis_verification.json').read_text())
    assert av['passed'] and av['analysis_report_sha256'] == sha(root/'analysis_report.json')
    assert ar['selection_report_sha256'] == sha(root/'selection_report.json')
    assert ar['daily_summary_sha256'] == sha(root/'daily_summary.parquet') and ar['reference_label'] == '09:59'
    lr = json.loads((root/'full_label_report.json').read_text())
    lv = json.loads((root/'full_label_verification.json').read_text())
    assert lv['passed'] and lv['label_report_sha256'] == ar['label_report_sha256'] == sha(root/'full_label_report.json')
    assert lr['labels_sha256'] == sha(root/'full_labels.parquet')
    return f, ar


def reuse_analysis(root, parent):
    """New full label file, but all labels joined by this complete list are equal."""
    frame = checked_selection(root); parent_frame, ar = checked_analysis(parent)
    assert frame.equals(parent_frame)
    evaluation.attach_labels(root)
    c = base.conn(); c.register('keys', frame[['date', 'code']])
    old = c.execute('SELECT o.* FROM read_parquet(?) o JOIN keys USING(date,code) ORDER BY date,code',
        [str(parent/'full_labels.parquet')]).df()
    new = c.execute('SELECT o.* FROM read_parquet(?) o JOIN keys USING(date,code) ORDER BY date,code',
        [str(root/'full_labels.parquet')]).df(); c.close()
    pd.testing.assert_frame_equal(new[old.columns], old, check_exact=True)
    assert len(new) == len(frame)
    proof = dict(passed=True, old_labels_sha256=sha(parent/'full_labels.parquet'),
        new_labels_sha256=sha(root/'full_labels.parquet'), rows=len(frame),
        every_complete_selection_key_and_applicable_label_field_exactly_identical=True,
        old_metadata_and_unknowns_preserved=True, no_new_economic_aggregation=True)
    save_json(root/'label_projection_reuse_verification.json', proof)
    reuse(root, parent)
    (root/'daily_summary.parquet').symlink_to((parent/'daily_summary.parquet').resolve())
    report = dict(ar)
    report.update(protocol_sha256=sha(PROTOCOL), selection_report_sha256=sha(root/'selection_report.json'),
        label_report_sha256=sha(root/'full_label_report.json'),
        reused_analysis_report_sha256=sha(parent/'analysis_report.json'),
        label_projection_reuse_verification_sha256=sha(root/'label_projection_reuse_verification.json'),
        all_joined_label_values_unchanged=True, no_new_economic_aggregation=True)
    save_json(root/'analysis_report.json', report)
    v = json.loads((parent/'analysis_verification.json').read_text())
    v.update(analysis_report_sha256=sha(root/'analysis_report.json'),
        reused_analysis_verification_sha256=sha(parent/'analysis_verification.json'),
        all_source_stats_reused_after_full_metadata_and_joined_labels_equal=True,
        no_new_economic_aggregation=True)
    save_json(root/'analysis_verification.json', v)
    return proof


def analyze():
    master, joint = checked_joint(); previous = [Path(master['original_control'])]
    previous.extend(Path(v) for v in master['controls'].values())
    records = []
    for item in joint['selections']:
        root = Path(item['root']); frame = checked_selection(root)
        parent = next((p for p in previous if frame.equals(checked_selection(p))), None)
        existed = (root/'analysis_report.json').exists()
        if not existed:
            if parent is not None:
                reuse_analysis(root, parent)
            else:
                evaluation.analyze(root, PROTOCOL)
                with (root/'analysis_check_command.log').open('w') as log:
                    subprocess.run(['.venv/bin/python', 'scripts/verify_tail_formula_before1000.py',
                        'analysis', '--root', str(root)], stdout=log, check=True)
        checked_analysis(root); previous.append(root)
        records.append(dict(group=item['group'], reused_source=str(parent) if parent else None,
            new_economic_aggregation_run=not existed and parent is None,
            analysis_report_sha256=sha(root/'analysis_report.json')))
        print(json.dumps(dict(completed=item['group'])), flush=True)
    result = dict(passed=True, joint_sha256=sha(inputs.ROOT/'joint_selection_freeze.json'), records=records,
        full_frame_equality_before_aggregation=True, no_duplicate_half_year_aggregation=True,
        all_joined_labels_proven_equal_before_any_reuse=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(inputs.ROOT/'analysis_dispatch_verification.json', result); return records


def view(name, parent):
    """Comparison-only view preserves the old list/stats and proves label equivalence."""
    root = inputs.ROOT/'comparison_views'/name; root.mkdir(parents=True, exist_ok=True)
    for file in ['selection.parquet', 'selection_report.json', 'selection_verification.json']:
        target = root/file
        if not target.exists():
            target.symlink_to((parent/file).resolve())
        assert sha(target) == sha(parent/file)
    if not (root/'analysis_report.json').exists():
        reuse_analysis(root, parent)
    checked_analysis(root); return root


def reference(root):
    checked_analysis(root)
    if (root/'reference_coverage_verification.json').exists():
        return json.loads((root/'reference_coverage_verification.json').read_text())
    reused = root/'analysis_reuse.json'
    if reused.exists():
        parent = Path(json.loads(reused.read_text())['source']); old = parent/'reference_coverage_verification.json'
        if old.exists():
            original = json.loads(old.read_text())
            assert original['passed'] and original['analysis_report_sha256'] == sha(parent/'analysis_report.json')
            assert original['selection_report_sha256'] == sha(parent/'selection_report.json')
            r = dict(original)
            r.update(analysis_report_sha256=sha(root/'analysis_report.json'),
                selection_report_sha256=sha(root/'selection_report.json'), label_report_sha256=sha(root/'full_label_report.json'),
                reused_reference_verification_sha256=sha(old), full_list_and_joined_label_values_equal=True,
                no_new_reference_aggregation=True)
            save_json(root/'reference_coverage_verification.json', r); return r
    return audit(root, [2025])


def finish():
    master, joint = checked_joint(); roots = {i['group']: Path(i['root']) for i in joint['selections']}
    roots.update({name: view(name, Path(parent)) for name, parent in master['controls'].items()})
    receipts = {}; done = []; equivalent = []
    for name, root in roots.items():
        checked_analysis(root); reference(root)
        for file in ['selection_report.json', 'selection_verification.json', 'analysis_report.json',
                     'analysis_verification.json', 'daily_summary.parquet', 'reference_coverage_verification.json']:
            receipts[str(root/file)] = sha(root/file)
    for left, right in master['planned_comparisons']:
        a, b = roots[left], roots[right]
        pair = left+'_minus_'+right
        paths = [inputs.ROOT/('same_dates_'+pair+'.json'), inputs.ROOT/('shared_unknowns_'+pair+'.json')]
        af, ar = checked_analysis(a); bf, br = checked_analysis(b)
        reused = None
        for old in done:
            ofa, ora = checked_analysis(roots[old['left']]); ofb, orb = checked_analysis(roots[old['right']])
            if (af.equals(ofa) and bf.equals(ofb) and ar['summaries'] == ora['summaries']
                and br['summaries'] == orb['summaries'] and ar['daily_summary_sha256'] == ora['daily_summary_sha256']
                and br['daily_summary_sha256'] == orb['daily_summary_sha256']):
                reused = old; break
        if reused:
            equivalent.append(dict(left=left, right=right, reused_pair=[reused['left'], reused['right']],
                both_complete_frames_stats_daily_and_labels_equal=True))
            paths = [Path(x) for x in reused['outputs']]
        else:
            if not paths[0].exists():
                compare(a, b, paths[0], periods=['2025H1', '2025H2', '2025'], intersection_only=True)
            if not paths[1].exists():
                spec = dict(left=str(a), right=str(b), output=str(paths[1]),
                    left_analysis_sha256=sha(a/'analysis_report.json'), right_analysis_sha256=sha(b/'analysis_report.json'))
                p = dict(protocol_sha256=sha(PROTOCOL), labels_sha256=sha(labels.directory('evaluation')/'full_labels.parquet'),
                    periods=['2025H1', '2025H2', '2025'])
                shared(spec, p)
        for file in paths:
            assert json.loads(file.read_text())['passed']; receipts[str(file)] = sha(file)
        done.append(dict(left=left, right=right, outputs=[str(x) for x in paths], reused=reused is not None))
        print(json.dumps(dict(compared=pair)), flush=True)
    def annual(name, period='2025'):
        r = json.loads((roots[name]/'analysis_report.json').read_text())
        return next(s for s in r['summaries'] if s['arm']=='formula' and s['bps']==15
                    and not s['sensitive'] and s['period']==period)
    new, old = annual('expanded_full'), annual('old_full')
    paired = next(r for r in done if r['left']=='expanded_full' and r['right']=='old_full')
    sr = json.loads(Path(paired['outputs'][1]).read_text())
    cs = next(s for s in sr['summaries'] if s['bps']==15 and not s['sensitive'] and s['period']=='2025')
    criteria = dict(both_half_reference_positive=all((annual('expanded_full', h)['mean_reference'] or -1)>0 for h in ['2025H1','2025H2']),
        annual_opportunity_above_old_full=new['rate'] is not None and old['rate'] is not None and new['rate']>old['rate'],
        annual_bad3_not_above_old_full=new['bad3'] is not None and old['bad3'] is not None and new['bad3']<=old['bad3'],
        shared_unknown_lower_ci_strict_positive=cs['lower_ci'] is not None and cs['lower_ci'][0]>0)
    gate = dict(passed=True, protocol_sha256=sha(PROTOCOL), criteria=criteria,
        supports_followup=all(criteria.values()), no_publish_claim=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(inputs.ROOT/'expansion_training_gate.json', gate)
    receipts[str(inputs.ROOT/'expansion_training_gate.json')] = sha(inputs.ROOT/'expansion_training_gate.json')
    receipts[str(labels.directory('evaluation')/'full_label_verification.json')] = sha(labels.directory('evaluation')/'full_label_verification.json')
    out = dict(passed=True, protocol_sha256=sha(PROTOCOL), joint_sha256=sha(inputs.ROOT/'joint_selection_freeze.json'),
        source_hashes=receipts, comparisons=done, comparison_reuse=equivalent,
        all_five_groups_eight_predefined_comparisons_and_reference_gaps_preserved=True,
        unchanged_old_annual_statistics_reused=True, no_duplicate_half_aggregation=True,
        no_publish_claim=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(inputs.ROOT/'complete_results_manifest.json', out)
    return dict(completion_sha256=sha(inputs.ROOT/'complete_results_manifest.json'), gate=gate,
        new_numerical_comparisons=sum(not r['reused'] for r in done))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['analyze', 'finish'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
