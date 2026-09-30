"""Target-aware two-fold runner; shared selection and economics remain unchanged."""
import argparse
import json
from pathlib import Path

import run_tail_formula_paired_study as shared
from find_existing_tail_formula_models import find
from trade_research import tail_formula_industry_baseline as study
from trade_research.corporate_cash import save_json,sha


def protocols():
    p=shared.model.checked();shared.base.FEATURES=shared.base.SOURCE=study.INPUTS
    shared.base.EXPRESSIONS=study.ARMS['control'];counts={};records=[]
    receipts={str(study.INPUTS/file):sha(study.INPUTS/file) for file in
        ['feature_report.json','feature_verification.json','native_input_verification.json','full_label_report.json','full_label_verification.json']}
    for fold,spec in p['folds'].items():
        t=shared.base.training(start=spec['training_start'],end=spec['training_end'])
        _,_,report,verification=study.target_paths(fold);v=json.loads(verification.read_text())
        assert v['passed'] and v['target_report_sha256']==sha(report)
        assert len(t)==v['training_rows'] and t.date.nunique()==v['training_days']
        assert t.next_date.max()<spec['evaluation_start']
        counts[fold]=dict(rows=len(t),days=t.date.nunique(),last_observation=t.next_date.max())
        for file in [report,verification]:receipts[str(file)]=sha(file)
    for arm,expressions in study.ARMS.items():
        for fold,spec in p['folds'].items():
            n=counts[fold];event=study.TRAINING_EVENT if arm=='industry' else None
            q=dict(master_protocol_sha256=sha(shared.model.PROTOCOL),arm=arm,fold=fold,**spec,
                expected_training_rows=n['rows'],expected_training_days=n['days'],expected_last_observation=n['last_observation'],
                expected_features=len(expressions),feature_names=list(expressions),parameters=p['parameters'],model_max_depth=3,
                threshold=.995,target='relative',training_event=event,input_receipts=receipts,
                no_training_period_selection=True,new_2026_prices_allowed=False,no_exit_rules=True)
            path=shared.model.fold_protocol(arm,fold);assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True)
            save_json(path,q)
            lookup=find(path,variant='relative',training_event=event)
            records.append(dict(arm=arm,fold=fold,protocol_sha256=sha(path),lookup=lookup))
    assert not any(r['lookup']['matches'] for r in records if r['arm']=='industry'), 'Audit equivalent industry targets before fitting'
    out=dict(passed=True,master_protocol_sha256=sha(shared.model.PROTOCOL),counts=counts,records=records,
        candidate_lookup_requires_explicit_industry_training_event=True,exactly_two_new_models_allowed=True,
        two_old_controls_to_reuse_after_exact_audit=True,all_four_protocols_before_fitting=True,
        no_new_group_outcomes_read=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(study.ROOT/'prefit_lookup_verification.json',out)
    return dict(prefit_sha256=sha(study.ROOT/'prefit_lookup_verification.json'),counts=counts,candidate_metadata_matches=0)


def checked_candidate_targets():
    shared.model.checked()
    for fold in ['2025h1','2025h2']:
        root=study.ROOT/'industry'/fold
        r=json.loads((root/'model_report.json').read_text());v=json.loads((root/'model_verification.json').read_text())
        _,file,report,verification=study.target_paths(fold)
        assert r['training_event']==v['training_event']==study.TRAINING_EVENT
        assert r['target_report_sha256']==sha(report) and r['target_verification_sha256']==sha(verification)
        assert json.loads(report.read_text())['targets_sha256']==sha(file)
        assert json.loads(verification.read_text())['passed'] and v['target_verification_sha256']==sha(verification)
        assert v['passed'] and v['model_report_sha256']==sha(root/'model_report.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['protocols','reuse','model','verify_model','scores','verify_scores','freeze','analyze','finish'])
    parser.add_argument('--fold',choices=['2025h1','2025h2']);args=parser.parse_args()
    shared.configure(study.STEM)
    if args.stage=='protocols':result=protocols()
    elif args.stage in ['freeze','analyze','finish']:
        checked_candidate_targets();result=getattr(shared,args.stage)()
    elif args.stage=='reuse':
        assert args.fold;result=shared.model.reuse(args.fold)
    else:
        assert args.fold;shared.model.setup('industry',args.fold)
        if args.stage in ['model','verify_model']:result=getattr(study,args.stage)()
        elif args.stage=='scores':result=shared.base.scores()
        else:result=shared.verify_scores(expected_expressions=shared.base.EXPRESSIONS)
    print(json.dumps(result,ensure_ascii=False,indent=2))
