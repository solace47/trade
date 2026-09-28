from pathlib import Path
import json,sys
sys.path.insert(0,'scripts')
from trade_research.corporate_cash import sha,save_json
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared
root=Path('data/research/tail_formula_pairwise_normalization');left=Path('data/research/tail_formula_pairwise_scale_2025');right=Path('data/research/tail_formula_pairwise_center_2025')
spec=dict(left=str(left),right=str(right),left_analysis_sha256=sha(left/'analysis_report.json'),right_analysis_sha256=sha(right/'analysis_report.json'),output=str(root/'shared_scale_minus_center.json'))
p=dict(objective='固定两个配对评分变换的完整比较，不据结果选择第三种变换。',master_protocol_sha256=sha(Path('config/tail_formula_pairwise_normalization_protocol.json')),labels_sha256=sha(left/'full_labels.parquet'),comparisons=[spec],new_2026_prices_allowed=False,no_exit_rule_research=True)
path=Path('config/tail_formula_pairwise_normalization_shared_protocol.json');assert not path.exists();save_json(path,p);p['protocol_sha256']=sha(path)
compare(left,right,root/'same_dates_scale_minus_center.json',intersection_only=True);shared(spec,p)
reports={}
for arm in ['center','scale']:
 path=Path('data/research/tail_formula_pairwise_'+arm)/'complete_results_manifest.json';r=json.loads(path.read_text());assert r['passed'];reports[str(path)]=sha(path)
 for name,digest in r['reports'].items():assert sha(Path(name))==digest
for filename in ['same_dates_scale_minus_center.json','shared_scale_minus_center.json']:
 path=root/filename;assert json.loads(path.read_text())['passed'];reports[str(path)]=sha(path)
path=root/'complete_results_manifest.json';assert not path.exists();save_json(path,dict(passed=True,protocol_sha256=sha(Path('config/tail_formula_pairwise_normalization_protocol.json')),reports=reports,both_arms_all_eight_controls_and_direct_comparison_complete=True,new_2026_prices_read=False,no_exit_rules=True));print(sha(path))
