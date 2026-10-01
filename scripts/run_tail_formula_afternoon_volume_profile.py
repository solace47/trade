"""Four volume-profile fits, with exact existing controls and shared evaluation."""
import argparse
import json
from pathlib import Path

from trade_research import tail_formula_afternoon_volume_profile as study
from trade_research.corporate_cash import save_json,sha
import run_tail_formula_morning_extrema_order as parent


def bind():
    parent.study=study;parent.common.study=study
    parent.common.EXECUTION=Path('config')/(study.STEM+'_model_protocol.json')
    parent.common.ARMS={'control':study.CONTROL,'memory':study.EXPRESSIONS}


def freeze():
    bind();parent.common.freeze();path=study.ROOT/'joint_selection_freeze.json';r=json.loads(path.read_text())
    for item in r['models']:
        root=study.ROOT/item['arm']/item['fold'];file=root/'frozen_numeric_core.tdx';text=file.read_text()
        assert text.count('CORE:VMREF>0 AND VP20>0 AND SC>')==1
        text=text.replace('CORE:VMREF>0 AND VP20>0 AND SC>','CORE:'+('AVREADY AND ' if item['arm']=='memory' else '')+'SC>')
        file.write_text(text);r['source_hashes'][str(file)]=sha(file)
        m=json.loads((root/'model_report.json').read_text())
        item['volume_profile_split_nodes']=sum(v>=50 for t in m['trees'] for v in t['feature']) if item['arm']=='memory' else 0
        del item['new_feature_nodes']
    assert sum(i['new_fit'] for i in r['models'])==4
    r.update(arm_display_names={'control':'原50','memory':'完整午后前段量分布130'},no_new_control_fits=True,
        legacy_volume_predicate_removed_before_final_joint=True,native_input_verification_sha256=sha(study.INPUTS/'native_input_verification.json'))
    save_json(path,r);return dict(joint_sha256=sha(path),selections=r['selections'],models=r['models'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['raw','prepare','protocols','fit','freeze','analyze','finish'])
    p.add_argument('--arm',choices=['control','memory']);p.add_argument('--fold',choices=['2024h1','2024h2','2025h1','2025h2']);a=p.parse_args();bind()
    if a.stage in ['raw','prepare']:result=getattr(study,a.stage)()
    elif a.stage=='protocols':result=parent.protocols()
    elif a.stage=='fit':result=parent.fit(a.arm,a.fold)
    elif a.stage=='freeze':result=freeze()
    else:result=getattr(parent.common,a.stage)()
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
