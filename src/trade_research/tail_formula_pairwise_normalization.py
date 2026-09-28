"""Fixed postprocessing of existing within-date pairwise ranking scores."""
import argparse
import importlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from . import tail_formula_boundary_evaluation as evaluation
from .corporate_cash import save_json, sha

MASTER = Path('config/tail_formula_pairwise_normalization_protocol.json')
ROOT = Path('data/research/tail_formula_pairwise_normalization')


def setup(arm):
    assert arm in ['center', 'scale']
    module = importlib.import_module('.tail_formula_score_' + arm, package=__package__)
    module.STEM = 'tail_formula_pairwise_' + arm
    module.PROTOCOL = Path('config') / (module.STEM + '_protocol.json')
    original_config = module.config
    def checked_config():
        p = original_config()
        assert p['normalization_protocol_sha256'] == sha(MASTER)
        assert p['model_family'] == 'same_date_binary_pairwise'
        for path, digest in p['references'].items():
            assert sha(Path(path)) == digest
        return p
    module.config = checked_config
    return module


def verify_center_native(module):
    # The original mean-centering verifier uses independent SQL. Add actual
    # INSUM arithmetic for every date of these new, already-fixed raw scores.
    out=module.root(module.ACTIVE_FOLD)
    r=json.loads((out/'score_report.json').read_text())
    _,source,model=module.source(module.ACTIVE_FOLD)
    text=module.native_structure(model,r['threshold'])
    lines=[line for line in text.splitlines() if line.startswith(('CSN:=','CSM:='))]
    core=text.splitlines()[-1]
    assert len(lines)==2 and core.startswith('CORE:CSN>0 AND SC-CSM>') and core.endswith(';')
    expression=core.removeprefix('CORE:CSN>0 AND ').removesuffix(';')
    f=pd.read_parquet(out/'scores.parquet');maximum=0.;days=0
    for _,day in f.groupby('date',sort=True):
        x=day.score.to_numpy();valid=day.formula_input_valid.to_numpy();safe=np.where(valid,x,0)
        sums=[int(valid.sum()),float(safe.sum())]
        def insum(block,helper,index,mode):
            assert block=='沪深Ａ股' and helper=='YJSC64' and index in [1,2] and mode==0
            return sums[index-1]
        env=dict(INSUM=insum,MAX=max,SC=x)
        for line in lines:
            name,expr=line.removesuffix(';').split(':=')
            env[name]=eval(expr,{'__builtins__':{}},env)
        actual=(env['CSN']>0)&eval(expression,{'__builtins__':{}},env)&valid
        np.testing.assert_array_equal(actual,day.centered_score.gt(r['threshold']))
        delta=np.where(valid,x-env['CSM'],np.nan)
        np.testing.assert_allclose(delta,day.centered_score,rtol=0,atol=2e-12,equal_nan=True)
        maximum=max(maximum,float(np.nanmax(np.abs(delta-day.centered_score.to_numpy()))));days+=1
    assert days==484
    proof=dict(passed=True,score_report_sha256=sha(out/'score_report.json'),
        score_verification_sha256=sha(out/'score_verification.json'),
        original_model_report_sha256=sha(source/'model_report.json'),
        all_484_dates_actual_insum_values_and_strict_flags_replayed=True,max_score_difference=maximum,
        complete_native_formula_verified=False,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(out/'native_center_verification.json',proof)
    return proof


def main(arm):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['scores','verify_scores','freeze','verify','analyze'])
    parser.add_argument('--fold',choices=['2024','recent','combined','2025'],required=True)
    args=parser.parse_args();fold='2025' if args.fold=='combined' else args.fold
    module=setup(arm);module.ACTIVE_FOLD=fold
    if args.stage=='analyze':
        assert fold=='2025'
        top=ROOT/'joint_selection_freeze.json';joint=json.loads(top.read_text())
        assert joint['passed'] and joint['protocol_sha256']==sha(MASTER) and len(joint['selections'])==6
        committed=subprocess.run(['git','show','HEAD:docs/selection-formula.md'],capture_output=True,text=True,check=True).stdout
        assert sha(top) in committed
        for record in joint['selections']:
            path=Path(record['root']);v=json.loads((path/'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256']==record['selection_report_sha256']==sha(path/'selection_report.json')
            assert record['selection_verification_sha256']==sha(path/'selection_verification.json')
        result=evaluation.analyze(module.root(fold),Path('config')/(module.STEM+'_combined_protocol.json'))
    else:
        result=getattr(module,args.stage)(fold)
        if args.stage=='verify_scores' and arm=='center':
            result=verify_center_native(module)
    print(json.dumps(result,ensure_ascii=False,indent=2))
