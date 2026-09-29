"""Six fixed shallow models using only labels preceding each 2024 test half."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_relative as relative
from . import tail_formula_units_history2024 as inputs
from .tail_formula_offset_logit48 import verify_scores
from .corporate_cash import save_json, sha


def setup(arm, fold):
    master = inputs.checked()
    base.ROOT = inputs.ROOT / arm / fold
    base.FEATURES = inputs.INPUTS; base.SOURCE = inputs.HISTORY
    base.EXPRESSIONS = inputs.ARMS[arm]; base.HEADER = inputs.HEADER
    base.PROTOCOL = Path('config') / (inputs.STEM + '_' + arm + '_' + fold + '_protocol.json')
    relative.PROTOCOL = base.PROTOCOL
    p = json.loads(base.PROTOCOL.read_text())
    assert p['master_protocol_sha256'] == sha(inputs.PROTOCOL)
    assert p['arm'] == arm and p['fold'] == fold
    assert all(p[k] == v for k, v in master['folds'][fold].items())
    assert p['parameters'] == master['parameters'] and p['model_max_depth'] == 3
    assert p['feature_names'] == list(base.EXPRESSIONS)
    for file, digest in p['input_receipts'].items():
        assert sha(Path(file)) == digest
    for kind in ['feature', 'native_input']:
        proof = json.loads((inputs.INPUTS / (kind + '_verification.json')).read_text())
        assert proof['passed'] and proof['feature_report_sha256'] == sha(inputs.INPUTS / 'feature_report.json')
    return p


def verify_model(arm, p):
    m = json.loads((base.ROOT / 'model_report.json').read_text())
    assert m['feature_names'] == p['feature_names']
    assert m['rows'] == p['expected_training_rows'] and m['days'] == p['expected_training_days']
    assert m['last_observation'] == p['expected_last_observation'] < p['evaluation_start']
    assert all(m['parameters'][k] == v for k, v in p['parameters'].items())
    proof = relative.verify_model('relative')
    if arm == 'constant':
        assert all(i < 48 for tree in m['trees'] for i in tree['feature'])
        proof['nineteen_constant_columns_unused_by_all_nodes'] = True
    proof['year_2024_is_exploratory'] = True
    proof['new_group_outcomes_read'] = False
    save_json(base.ROOT / 'model_verification.json', proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores'])
    parser.add_argument('--arm', choices=list(inputs.ARMS), required=True)
    parser.add_argument('--fold', choices=['h1', 'h2'], required=True)
    a = parser.parse_args(); p = setup(a.arm, a.fold)
    if a.stage == 'model':
        result = relative.model('relative')
    elif a.stage == 'verify_model':
        result = verify_model(a.arm, p)
    elif a.stage == 'scores':
        result = base.scores()
    else:
        result = verify_scores(expected_expressions=inputs.ARMS[a.arm])
    print(json.dumps(result, ensure_ascii=False, indent=2))
