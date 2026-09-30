"""Evaluate the jointly frozen morning quote center and volume-weighted range; identical control statistics reused."""
import argparse
import json
from pathlib import Path
import subprocess

from trade_research import tail_formula_morning_distribution as study
from trade_research import tail_formula_morning_distribution_model as model
from trade_research.corporate_cash import save_json, sha
from freeze_tail_formula_bipower_gap import checked_selection
import evaluate_tail_formula_pool_scope as common

PROTOCOL = Path('config/tail_formula_morning_distribution_evaluation_protocol.json')


def checked_joint():
    master = model.checked(); p = json.loads(PROTOCOL.read_text())
    assert p['model_protocol_sha256']==sha(model.PROTOCOL) and p['planned_comparisons']==master['planned_comparisons']
    for f,d in p['source_hashes'].items():assert sha(Path(f))==d
    path = study.ROOT/'joint_selection_freeze.json'; joint = json.loads(path.read_text())
    assert joint['passed'] and joint['model_protocol_sha256']==sha(model.PROTOCOL)
    doc = subprocess.run(['git','show','HEAD:docs/selection-formula.md'],capture_output=True,text=True,check=True).stdout
    assert sha(path) in doc
    for f,d in joint['source_hashes'].items():assert sha(Path(f))==d
    for item in joint['selections']:checked_selection(Path(item['root']))
    common.PROTOCOL = PROTOCOL
    return master,joint


def analyze():
    master,joint = checked_joint(); previous = [Path(r) for r in master['controls'].values()]; records = []
    for item in joint['selections']:
        root = Path(item['root']); frame = checked_selection(root)
        parent = next((r for r in previous if frame.equals(checked_selection(r))),None)
        existed = (root/'analysis_report.json').exists()
        if not existed:
            if parent is not None:common.reuse_analysis(root,parent)
            else:
                common.evaluation.analyze(root,PROTOCOL)
                with (root/'analysis_check_command.log').open('w') as log:
                    subprocess.run(['.venv/bin/python','scripts/verify_tail_formula_before1000.py','analysis','--root',str(root)],stdout=log,check=True)
        common.checked_analysis(root); previous.append(root)
        records.append(dict(group=item['group'],reused_source=str(parent) if parent else None,
            new_economic_aggregation_run=not existed and parent is None,analysis_report_sha256=sha(root/'analysis_report.json')))
        print(json.dumps(dict(completed=item['group'])),flush=True)
    out = dict(passed=True,joint_sha256=sha(study.ROOT/'joint_selection_freeze.json'),records=records,
        all_full_frames_and_all_applicable_labels_equal_before_reuse=True,no_duplicate_half_year_aggregation=True,
        no_new_raw_extraction=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(study.ROOT/'analysis_dispatch_verification.json',out); return records


def control_view(name,parent):
    root = study.ROOT/'comparison_views'/name; root.mkdir(parents=True,exist_ok=True)
    for file in ['selection.parquet','selection_report.json','selection_verification.json']:
        target = root/file
        if not target.exists():target.symlink_to((parent/file).resolve())
        assert sha(target)==sha(parent/file)
    if not (root/'analysis_report.json').exists():common.reuse_analysis(root,parent)
    common.checked_analysis(root); return root


def finish():
    master,joint = checked_joint(); roots = {r['group']:Path(r['root']) for r in joint['selections']}
    for name,path in master['controls'].items():
        if name in roots:
            frame,ar = common.checked_analysis(roots[name]); old,prior = common.checked_analysis(Path(path))
            assert frame.equals(old) and ar['summaries']==prior['summaries']
            assert ar['daily_summary_sha256']==prior['daily_summary_sha256']
        else:roots[name]=control_view(name,Path(path))
    ep = json.loads(PROTOCOL.read_text()); prior_pairs = []; receipts = {}
    for parent_name in ep['prior_completed_comparison_roots']:
        parent_root=Path(parent_name);path=parent_root/'complete_results_manifest.json'
        complete=json.loads(path.read_text());assert complete['passed']
        assert ep['prior_comparison_completion_hashes'][str(path)]==sha(path)
        for f,d in complete['source_hashes'].items():assert sha(Path(f))==d
        receipts[str(path)]=sha(path)
        for pair in complete['comparisons']:
            assert all(json.loads(Path(f).read_text())['passed'] for f in pair['outputs'])
            prior_pairs.append(dict(**pair,left_root=parent_root/pair['left'],
                right_root=(parent_root/pair['right'] if (parent_root/pair['right']).exists()
                            else parent_root/'comparison_views'/pair['right'])))
    done = []; aliases = []
    for root in roots.values():
        common.checked_analysis(root); common.reference(root)
        for file in ['selection_report.json','selection_verification.json','analysis_report.json','analysis_verification.json',
                'daily_summary.parquet','reference_coverage_verification.json','full_label_report.json','full_label_verification.json']:
            receipts[str(root/file)]=sha(root/file)
    for left,right in master['planned_comparisons']:
        a,b = roots[left],roots[right]; pair = left+'_minus_'+right
        paths = [study.ROOT/('same_dates_'+pair+'.json'),study.ROOT/('shared_unknowns_'+pair+'.json')]
        af,ar = common.checked_analysis(a); bf,br = common.checked_analysis(b); alias = None
        for old in done:
            ofa,ora = common.checked_analysis(roots[old['left']]); ofb,orb = common.checked_analysis(roots[old['right']])
            if (af.equals(ofa) and bf.equals(ofb) and ar['summaries']==ora['summaries'] and br['summaries']==orb['summaries']
                and ar['daily_summary_sha256']==ora['daily_summary_sha256'] and br['daily_summary_sha256']==orb['daily_summary_sha256']
                and ar['label_report_sha256']==ora['label_report_sha256'] and br['label_report_sha256']==orb['label_report_sha256']):
                alias=old; break
        if alias is None:
            for old in prior_pairs:
                ofa,ora = common.checked_analysis(old['left_root']); ofb,orb = common.checked_analysis(old['right_root'])
                if (af.equals(ofa) and bf.equals(ofb) and ar['summaries']==ora['summaries'] and br['summaries']==orb['summaries']
                    and ar['daily_summary_sha256']==ora['daily_summary_sha256'] and br['daily_summary_sha256']==orb['daily_summary_sha256']
                    and ar['label_report_sha256']==ora['label_report_sha256'] and br['label_report_sha256']==orb['label_report_sha256']):
                    alias=old; break
        if alias:
            paths = [Path(f) for f in alias['outputs']]
            aliases.append(dict(left=left,right=right,reused_pair=[alias['left'],alias['right']],
                complete_frames_all_statistics_daily_and_labels_equal=True,
                prior_completed_study='left_root' in alias))
        else:
            if not paths[0].exists():common.compare(a,b,paths[0],periods=['2025H1','2025H2','2025'],intersection_only=True)
            if not paths[1].exists():
                spec = dict(left=str(a),right=str(b),output=str(paths[1]),left_analysis_sha256=sha(a/'analysis_report.json'),right_analysis_sha256=sha(b/'analysis_report.json'))
                r = dict(protocol_sha256=sha(PROTOCOL),labels_sha256=sha(common.evaluation.source.ROOT/'full_labels.parquet'),periods=['2025H1','2025H2','2025'])
                common.shared(spec,r)
        for file in paths:
            assert json.loads(file.read_text())['passed']; receipts[str(file)]=sha(file)
        done.append(dict(left=left,right=right,outputs=[str(f) for f in paths],reused=alias is not None))
        print(json.dumps(dict(compared=pair)),flush=True)
    def main(name,period='2025'):
        r = json.loads((roots[name]/'analysis_report.json').read_text())
        return next(s for s in r['summaries'] if s['arm']=='formula' and s['bps']==15 and not s['sensitive'] and s['period']==period)
    new,old,other = main('distribution2025'),main('morning2025'),main('low2025'); pair = next(r for r in done if r['right']=='morning2025')
    r = json.loads(Path(pair['outputs'][1]).read_text())
    s = next(s for s in r['summaries'] if s['period']=='2025' and s['bps']==15 and not s['sensitive'])
    criteria = dict(both_half_reference_positive=all((main('distribution2025',h)['mean_reference'] or -1)>0 for h in ['2025H1','2025H2']),
        annual_opportunity_above_both_controls=all(x['rate'] is not None for x in [new,old,other]) and new['rate']>max(old['rate'],other['rate']),
        annual_bad3_not_above_control=new['bad3'] is not None and old['bad3'] is not None and new['bad3']<=old['bad3'],
        shared_unknown_lower_ci_strict_positive=s['lower_ci'] is not None and s['lower_ci'][0]>0)
    gate = dict(passed=True,protocol_sha256=sha(PROTOCOL),criteria=criteria,supports_2024_extension=all(criteria.values()),
        no_publish_claim=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(study.ROOT/'morning_distribution_gate.json',gate); receipts[str(study.ROOT/'morning_distribution_gate.json')]=sha(study.ROOT/'morning_distribution_gate.json')
    receipts[str(study.ROOT/'analysis_dispatch_verification.json')]=sha(study.ROOT/'analysis_dispatch_verification.json')
    out = dict(passed=True,protocol_sha256=sha(PROTOCOL),joint_sha256=sha(study.ROOT/'joint_selection_freeze.json'),
        source_hashes=receipts,comparisons=done,comparison_reuse=aliases,
        all_two_groups_six_predefined_comparisons_and_reference_gaps_preserved=True,
        no_parent_refit_prediction_or_economic_aggregation=True,no_new_raw_extraction=True,
        no_publish_claim=True,year_2025_is_exploratory=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(study.ROOT/'complete_results_manifest.json',out)
    return dict(completion_sha256=sha(study.ROOT/'complete_results_manifest.json'),gate=gate,new_numerical_comparisons=sum(not r['reused'] for r in done))


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['analyze','finish'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
