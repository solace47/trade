"""Complete the frozen external-screen morning study, preserving every group."""
import json
from pathlib import Path

from trade_research import tail_formula_external_morning as study
from trade_research.corporate_cash import save_json, sha
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared


def main():
    study.checked()
    root=study.ROOT;left=root/'full';joint=json.loads((root/'joint_selection_freeze.json').read_text())
    assert joint['passed'] and joint['protocol_sha256']==sha(study.PROTOCOL)
    reports={}
    for item in joint['records']:
        out=Path(item['root'])
        for name in ['analysis','reference_coverage']:
            v=json.loads((out/(name+'_verification.json')).read_text())
            assert v['passed'] and v['analysis_report_sha256']==sha(out/'analysis_report.json')
        reports.update({str(out/name):sha(out/name) for name in ['analysis_report.json','analysis_verification.json','reference_coverage_verification.json']})
    controls=[('base',root/'base'),('eligible',root/'eligible'),
        ('corrected48',Path('data/research/tail_formula_before1000_model_2025')),
        ('fixed48',Path('data/research/tail_formula_before1000/evaluation/tail_formula_float_2025'))]
    protocols=[]
    for name,right in controls:
        periods=['2025H1','2025H2','2025'] if name.endswith('48') else ['2024H1','2024H2','2025H1','2025H2','2024','2025']
        spec=dict(left=str(left),right=str(right),left_analysis_sha256=sha(left/'analysis_report.json'),
            right_analysis_sha256=sha(right/'analysis_report.json'),output=str(root/('shared_unknowns_'+name+'.json')))
        protocol=dict(objective='原外部完整条件的当前次晨目标比较，保留全部原分组。',
            master_protocol_sha256=sha(study.PROTOCOL),labels_sha256=sha(left/'full_labels.parquet'),
            periods=periods,comparisons=[spec],new_2026_prices_allowed=False,no_exit_rule_research=True,
            coverage_note='Coverage retains the source selections entire date ranges; 48-input comparisons have common dates only in 2025.')
        path=Path('config')/(study.STEM+'_shared_'+name+'_protocol.json');assert not path.exists();save_json(path,protocol)
        protocol['protocol_sha256']=sha(path);protocols.append(str(path))
        output=root/('same_dates_'+name+'.json')
        compare(left,right,output,periods=periods,intersection_only=True)
        shared(spec,protocol)
        for q in [output,Path(spec['output'])]:
            assert json.loads(q.read_text())['passed'];reports[str(q)]=sha(q)
    q=root/'complete_results_manifest.json';assert not q.exists()
    save_json(q,dict(passed=True,protocol_sha256=sha(study.PROTOCOL),
        joint_selection_freeze_sha256=sha(root/'joint_selection_freeze.json'),reports=reports,
        all_eleven_original_groups_reported=True,primary_group='full',all_four_comparisons_completed=True,
        comparison_protocols=protocols,new_2026_prices_read=False,no_exit_rules=True))
    print(json.dumps(dict(passed=True,completion_sha256=sha(q)),ensure_ascii=False))


if __name__=='__main__':
    main()
