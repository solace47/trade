"""Fixed two-fold relative trees for ordered price-volume inputs and cached parents."""
import argparse
import json
from pathlib import Path
import subprocess

from . import tail_formula_additive as base
from . import tail_formula_dense_flow as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha
from .tail_formula_offset_logit48 import verify_scores


def protocols():
    master=study.checked()
    receipts={str(study.INPUTS/file):sha(study.INPUTS/file) for file in
        ['feature_report.json','feature_verification.json','native_input_verification.json',
         'full_label_report.json','full_label_verification.json']}
    for name in ['feature','native_input']:
        v=json.loads((study.INPUTS/(name+'_verification.json')).read_text())
        assert v['passed'] and v['feature_report_sha256']==sha(study.INPUTS/'feature_report.json')
    base.FEATURES=base.SOURCE=study.INPUTS
    counts={}
    for fold,spec in master['folds'].items():
        t=base.training(start=spec['training_start'],end=spec['training_end'])
        assert t.next_date.max()<spec['evaluation_start']
        counts[fold]=dict(rows=len(t),days=t.date.nunique(),last_observation=t.next_date.max())
    records=[]
    for arm,expr in study.ARMS.items():
        for fold,spec in master['folds'].items():
            p=dict(master_protocol_sha256=sha(study.PROTOCOL),arm=arm,fold=fold,**spec,
                expected_training_rows=counts[fold]['rows'],expected_training_days=counts[fold]['days'],
                expected_last_observation=counts[fold]['last_observation'],expected_features=len(expr),
                parameters=master['parameters'],model_max_depth=3,feature_names=list(expr),
                input_receipts=receipts,threshold=.995,target='relative',
                no_training_period_selection=True,new_2026_prices_allowed=False,no_exit_rules=True)
            path=Path('config')/(study.STEM+'_'+arm+'_'+fold+'_protocol.json')
            assert not path.exists();save_json(path,p)
            # Metadata lookup must precede fitting. Any candidate requires a
            # separate exact reuse audit, never an automatic repeated fit.
            result=subprocess.run(['.venv/bin/python','scripts/find_existing_tail_formula_models.py',
                '--protocol',str(path),'--variant','relative'],capture_output=True,text=True,check=True)
            lookup=json.loads(result.stdout)
            records.append(dict(arm=arm,fold=fold,protocol_sha256=sha(path),lookup=lookup))
    report=dict(passed=True,master_protocol_sha256=sha(study.PROTOCOL),counts=counts,records=records,
        all_six_protocols_before_fitting=True,all_candidates_require_exact_reuse_proof=True,
        no_evaluation_labels_read=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(study.ROOT/'prefit_lookup_verification.json',report)
    assert not any(x['lookup']['matches'] for x in records if x['arm']=='flow'),'Audit and reuse an existing flow fit before proceeding'
    return dict(counts=counts,existing_candidates=sum(len(x['lookup']['matches']) for x in records),
        prefit_sha256=sha(study.ROOT/'prefit_lookup_verification.json'))


def setup(arm,fold):
    master=study.checked()
    base.ROOT=study.ROOT/arm/fold;base.FEATURES=base.SOURCE=study.INPUTS
    base.EXPRESSIONS=study.ARMS[arm];base.HEADER=study.HEADER
    base.PROTOCOL=Path('config')/(study.STEM+'_'+arm+'_'+fold+'_protocol.json')
    relative.PROTOCOL=base.PROTOCOL
    p=json.loads(base.PROTOCOL.read_text())
    assert p['master_protocol_sha256']==sha(study.PROTOCOL) and p['arm']==arm and p['fold']==fold
    assert all(p[k]==v for k,v in master['folds'][fold].items())
    assert p['parameters']==master['parameters'] and p['feature_names']==list(base.EXPRESSIONS)
    assert p['expected_features']==len(base.EXPRESSIONS)
    for file,digest in p['input_receipts'].items():assert sha(Path(file))==digest
    return p


def verify_model(p):
    m=json.loads((base.ROOT/'model_report.json').read_text())
    assert m['feature_names']==p['feature_names'] and m['variant']=='relative'
    assert m['rows']==p['expected_training_rows'] and m['days']==p['expected_training_days']
    assert m['last_observation']==p['expected_last_observation']<p['evaluation_start']
    assert all(m['parameters'][k]==v for k,v in p['parameters'].items())
    return relative.verify_model('relative')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['protocols','model','verify_model','scores','verify_scores'])
    p.add_argument('--arm',choices=list(study.ARMS));p.add_argument('--fold',choices=list(['2025h1','2025h2']))
    a=p.parse_args()
    if a.stage=='protocols':result=protocols()
    else:
        assert a.arm and a.fold;spec=setup(a.arm,a.fold)
        if a.stage=='model':result=relative.model('relative')
        elif a.stage=='verify_model':result=verify_model(spec)
        elif a.stage=='scores':result=base.scores()
        else:result=verify_scores(expected_expressions=base.EXPRESSIONS)
    print(json.dumps(result,ensure_ascii=False,indent=2))
