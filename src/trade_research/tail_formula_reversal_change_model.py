"""Reversal-frequency changes versus an identically trained 48-input control."""
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
from . import tail_formula_reversal_change as inputs
from .corporate_cash import sha


def setup(arm,fold):
    inputs.checked_sources()
    for root in [inputs.ROOT,inputs.CONTROL_INPUTS]:
        r = json.loads((root/'feature_report.json').read_text())
        v = json.loads((root/'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(root/'feature_report.json')
        assert r['protocol_sha256'] == sha(inputs.PROTOCOL) and r['features_sha256'] == sha(root/'features.parquet')
    n = json.loads((inputs.ROOT/'native_input_verification.json').read_text())
    assert n['passed'] and n['protocol_sha256'] == sha(inputs.PROTOCOL)
    assert n['feature_report_sha256'] == sha(inputs.ROOT/'feature_report.json')
    assert n['feature_verification_sha256'] == sha(inputs.ROOT/'feature_verification.json')
    assert sha(inputs.ROOT/'features.parquet') == sha(inputs.CONTROL_INPUTS/'features.parquet')
    stem = inputs.STEM+'_'+arm
    adapter.STEM = stem;adapter.ROOT = inputs.ROOT if arm=='joint' else inputs.CONTROL_INPUTS
    adapter.EXPRESSIONS = inputs.EXPRESSIONS if arm=='joint' else inputs.previous.EXPRESSIONS
    adapter.HEADER = inputs.HEADER if arm=='joint' else inputs.previous.HEADER
    adapter.COMBINED_PROTOCOL = Path('config')/(stem+'_combined_protocol.json')
    adapter.setup(fold);base.SOURCE = labels.ROOT
    for name in ['2024','recent']:
        p = json.loads((Path('config')/(stem+'_'+name+'_protocol.json')).read_text())
        assert p['inputs_protocol_sha256'] == sha(inputs.PROTOCOL)
        assert p['feature_report_sha256'] == sha(adapter.ROOT/'feature_report.json')
        assert p['native_input_verification_sha256'] == sha(inputs.ROOT/'native_input_verification.json')
        assert p['label_report_sha256'] == sha(labels.ROOT/'full_label_report.json')
        assert p['same_training_input_table_sha256'] == sha(adapter.ROOT/'features.parquet')
        assert p['expected_features'] == len(adapter.EXPRESSIONS)


def selection_gate():
    for arm in ['joint','control']:
        for fold in ['2024','recent','2025']:
            root = Path('data/research')/(inputs.STEM+'_'+arm+'_'+fold)
            v = json.loads((root/'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == sha(root/'selection_report.json')
    for fold in ['2024','recent']:
        models = [json.loads((Path('data/research')/(inputs.STEM+'_'+arm+'_'+fold)/'model_report.json').read_text()) for arm in ['joint','control']]
        for key in ['rows','days','training_start','training_end','last_observation','label_report_sha256','parameters']:
            assert models[0][key] == models[1][key]


if __name__ == '__main__':
    from .tail_formula_offset_logit48 import verify_scores
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['model','verify_model','scores','verify_scores','freeze','verify','analyze'])
    parser.add_argument('--arm',choices=['joint','control'],required=True)
    parser.add_argument('--fold',choices=['2024','recent','combined'],default='2024')
    args = parser.parse_args();setup(args.arm,args.fold)
    if args.stage=='analyze':
        selection_gate()
        result = evaluation.analyze(linkage.COMBINED if args.fold=='combined' else base.ROOT,
            linkage.PROTOCOL if args.fold=='combined' else base.PROTOCOL)
    elif args.fold=='combined':
        assert args.stage in ['freeze','verify']
        result = linkage.combine() if args.stage=='freeze' else linkage.verify_combined()
    elif args.stage=='model':
        result = relative.model('relative')
    elif args.stage=='verify_model':
        p = json.loads(base.PROTOCOL.read_text());r = json.loads((base.ROOT/'model_report.json').read_text())
        assert r['feature_names'] == list(base.EXPRESSIONS) and len(r['feature_names']) == p['expected_features']
        assert r['days'] == p['expected_training_days'] == 241
        assert all(r['parameters'][k]==v for k,v in p['parameters'].items())
        result = relative.verify_model('relative')
    elif args.stage=='verify_scores':
        result = verify_scores()
    elif args.stage=='freeze':
        result = study.freeze()
    elif args.stage=='verify':
        model = json.loads((base.ROOT/'model_report.json').read_text())
        assert (base.ROOT/'frozen_numeric_core.tdx').read_text() == base.native_core(model,model['thresholds'][3]['threshold'],base.EXPRESSIONS,base.HEADER)
        result = study.verify()
    else:
        result = base.scores()
    print(json.dumps(result,ensure_ascii=False,indent=2))
