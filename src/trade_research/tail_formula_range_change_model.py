"""Fixed 49-input prior-range-change model with an unchanged-input 48-input control."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_prior_day as adapter
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from . import tail_formula_range_change as inputs
from .corporate_cash import sha


def setup(fold):
    inputs.checked_sources()
    root=inputs.ROOT
    r=json.loads((root/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(inputs.PROTOCOL)
    assert r['features_sha256']==sha(root/'features.parquet')
    v=json.loads((root/'feature_verification.json').read_text())
    n=json.loads((root/'native_input_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256']==sha(root/'feature_report.json')
    assert v['effective_input_intersection_unchanged'] and r['newly_invalid']==0
    assert n['passed'] and n['protocol_sha256']==sha(inputs.PROTOCOL)
    assert n['feature_report_sha256']==sha(root/'feature_report.json')
    assert n['feature_verification_sha256']==sha(root/'feature_verification.json')
    combined=Path('config')/(inputs.STEM+'_combined_protocol.json')
    p=json.loads(combined.read_text());control=Path(p['control'])
    for stage in ['selection','analysis']:
        proof=json.loads((control/(stage+'_verification.json')).read_text())
        assert proof['passed'] and proof[stage+'_report_sha256']==p['control_'+stage+'_report_sha256']==sha(control/(stage+'_report.json'))
    adapter.STEM=inputs.STEM;adapter.ROOT=root
    adapter.EXPRESSIONS=inputs.EXPRESSIONS;adapter.HEADER=inputs.HEADER
    adapter.COMBINED_PROTOCOL=combined;adapter.setup(fold);base.SOURCE=labels.ROOT
    for name in ['2024','recent']:
        p=json.loads((Path('config')/(inputs.STEM+'_'+name+'_protocol.json')).read_text())
        assert p['inputs_protocol_sha256']==sha(inputs.PROTOCOL)
        assert p['feature_report_sha256']==sha(root/'feature_report.json')
        assert p['native_input_verification_sha256']==sha(root/'native_input_verification.json')
        assert p['label_report_sha256']==sha(labels.ROOT/'full_label_report.json')
        assert p['expected_features']==49


def verify_model():
    p=json.loads(base.PROTOCOL.read_text());r=json.loads((base.ROOT/'model_report.json').read_text())
    assert r['feature_names']==list(inputs.EXPRESSIONS) and len(r['feature_names'])==49
    assert r['days']==p['expected_training_days']==241
    assert all(r['parameters'][key]==value for key,value in p['parameters'].items())
    return relative.verify_model('relative')


if __name__=='__main__':
    from .tail_formula_offset_logit48 import verify_scores
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['model','verify_model','scores','verify_scores','freeze','verify','analyze'])
    parser.add_argument('--fold',choices=['2024','recent','combined'],default='2024')
    args=parser.parse_args();setup(args.fold)
    if args.stage=='analyze':
        for fold in ['2024','recent','2025']:
            root=Path('data/research')/(inputs.STEM+'_'+fold)
            v=json.loads((root/'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256']==sha(root/'selection_report.json')
        result=evaluation.analyze(linkage.COMBINED if args.fold=='combined' else base.ROOT,
            linkage.PROTOCOL if args.fold=='combined' else base.PROTOCOL)
    elif args.fold=='combined':
        assert args.stage in ['freeze','verify']
        result=linkage.combine() if args.stage=='freeze' else linkage.verify_combined()
    elif args.stage=='model':
        result=relative.model('relative')
    elif args.stage=='verify_model':
        result=verify_model()
    elif args.stage=='verify_scores':
        result=verify_scores()
    elif args.stage=='freeze':
        result=study.freeze()
    elif args.stage=='verify':
        model=json.loads((base.ROOT/'model_report.json').read_text())
        assert (base.ROOT/'frozen_numeric_core.tdx').read_text()==base.native_core(model,model['thresholds'][3]['threshold'],base.EXPRESSIONS,base.HEADER)
        result=study.verify()
    else:
        result=base.scores()
    print(json.dumps(result,ensure_ascii=False,indent=2))
