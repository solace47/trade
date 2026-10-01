"""Four ETF-activity fits and exact existing controls under fixed evaluation."""
import argparse
import json
from pathlib import Path

from trade_research import tail_formula_etf_activity as study
from trade_research.corporate_cash import save_json, sha
import run_tail_formula_morning_extrema_order as parent


def bind():
    parent.study = study; parent.common.study = study
    parent.common.EXECUTION = Path('config') / (study.STEM+'_model_protocol.json')
    parent.common.ARMS = {'control':study.CONTROL,'memory':study.EXPRESSIONS}


def freeze():
    bind(); parent.common.freeze()
    path = study.ROOT/'joint_selection_freeze.json'; r = json.loads(path.read_text())
    for item in r['models']:
        root = study.ROOT/item['arm']/item['fold']; file = root/'frozen_numeric_core.tdx'
        text = file.read_text(); assert text.count('CORE:VMREF>0 AND VP20>0 AND SC>')==1
        text = text.replace('CORE:VMREF>0 AND VP20>0 AND SC>', 'CORE:'+('EFVALID AND ' if item['arm']=='memory' else '')+'SC>')
        file.write_text(text); r['source_hashes'][str(file)] = sha(file)
        m = json.loads((root/'model_report.json').read_text())
        item['ETF_activity_split_nodes'] = sum(v>=50 for t in m['trees'] for v in t['feature']) if item['arm']=='memory' else 0
        del item['new_feature_nodes']
    assert sum(x['new_fit'] for x in r['models'])==4
    r.update(no_new_control_fits=True,native_cross_security_helper_required='YJETFQ',
             arm_display_names={'control':'原50','memory':'两ETF成交活动54'},
             legacy_volume_predicate_removed_before_final_joint=True)
    save_json(path,r); return dict(joint_sha256=sha(path),selections=r['selections'],models=r['models'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','protocols','fit','freeze','analyze','finish'])
    p.add_argument('--arm',choices=['control','memory']); p.add_argument('--fold',choices=['2024h1','2024h2','2025h1','2025h2'])
    a=p.parse_args(); bind()
    if a.stage=='prepare': result=study.prepare()
    elif a.stage=='protocols': result=parent.protocols()
    elif a.stage=='fit': result=parent.fit(a.arm,a.fold)
    elif a.stage=='freeze': result=freeze()
    else: result=getattr(parent.common,a.stage)()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
