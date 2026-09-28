"""Learn whole-morning net price area; retain the original evaluation labels."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_before1000 as original
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_early_opportunity as adapter
from . import tail_formula_path_area_labels as targets
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import sha

STEM = 'tail_formula_path_area'


def setup(fold):
    adapter.STEM = STEM; adapter.targets = targets; adapter.setup(fold)
    p = json.loads((Path('config') / (STEM+'_combined_protocol.json')).read_text())
    assert p['training_target_protocol_sha256'] == sha(targets.PROTOCOL)
    assert p['training_target_window_end'] == p['evaluation_window_end'] == '09:59'
    control = Path(p['primary_control'])
    for stage in ['selection','analysis']:
        proof = json.loads((control / (stage+'_verification.json')).read_text())
        assert proof['passed'] and proof[stage+'_report_sha256'] == p['control_'+stage+'_report_sha256'] == sha(control/(stage+'_report.json'))


def verify_model():
    p = json.loads(base.PROTOCOL.read_text()); r = json.loads((base.ROOT/'model_report.json').read_text())
    assert r['days'] == p['expected_training_days'] == 241
    assert r['feature_names'] == list(base.EXPRESSIONS) and len(r['feature_names']) == 48
    assert all(r['parameters'][key] == value for key,value in p['parameters'].items())
    # The new utility must not quietly alter the original training intersection.
    start,end,_ = relative.training_scope()
    f = base.feature_inputs()[['date','code','formula_input_valid']]
    c = base.conn(); c.register('visible',f)
    expected = c.execute('''SELECT date,code,next_date FROM visible
        JOIN read_parquet(?) USING(date,code)
        WHERE formula_input_valid AND known15 AND date>=? AND next_date<? ORDER BY date,code''',
        [str(original.ROOT/'full_labels.parquet'),start,end]).df(); c.close()
    actual = relative.training('path_area')[['date','code','next_date']]
    import pandas as pd
    pd.testing.assert_frame_equal(actual,expected,check_exact=True)
    return relative.verify_model('path_area')


if __name__ == '__main__':
    from .tail_formula_offset_logit48 import verify_scores
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['model','verify_model','scores','verify_scores','freeze','verify','analyze'])
    parser.add_argument('--fold',choices=['2024','recent','combined'],default='2024')
    args=parser.parse_args(); setup(args.fold)
    if args.stage=='analyze':
        for fold in ['2024','recent','2025']:
            root=Path('data/research')/(STEM+'_'+fold)
            proof=json.loads((root/'selection_verification.json').read_text())
            assert proof['passed'] and proof['selection_report_sha256']==sha(root/'selection_report.json')
        result=evaluation.analyze(linkage.COMBINED if args.fold=='combined' else base.ROOT,
            linkage.PROTOCOL if args.fold=='combined' else base.PROTOCOL)
    elif args.fold=='combined':
        assert args.stage in ['freeze','verify']
        result=linkage.combine() if args.stage=='freeze' else linkage.verify_combined()
    elif args.stage=='model':
        result=relative.model('path_area')
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
