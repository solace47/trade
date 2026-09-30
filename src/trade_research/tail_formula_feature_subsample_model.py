"""Fixed feature competition in eight otherwise unchanged shallow learners."""
import argparse
from functools import partial
import json
from pathlib import Path

from sklearn.ensemble import GradientBoostingRegressor

from . import tail_formula_additive as base
from . import tail_formula_feature_subsample as inputs
from . import tail_formula_relative as relative
from .tail_formula_offset_logit48 import verify_scores
from .corporate_cash import save_json, sha


def setup(arm, fold):
    master = inputs.checked()
    base.ROOT = inputs.ROOT / arm / fold
    base.FEATURES = base.SOURCE = inputs.INPUTS
    base.EXPRESSIONS = inputs.ARMS[arm]; base.HEADER = inputs.HEADER
    base.PROTOCOL = Path('config') / (inputs.STEM + '_' + arm + '_' + fold + '_protocol.json')
    relative.PROTOCOL = base.PROTOCOL
    p = json.loads(base.PROTOCOL.read_text())
    assert p['master_protocol_sha256'] == sha(inputs.PROTOCOL)
    assert p['arm'] == arm and p['fold'] == fold
    assert all(p[k] == v for k, v in master['folds'][fold].items())
    assert p['parameters'] == master['parameters'] and p['feature_names'] == list(base.EXPRESSIONS)
    assert len(base.EXPRESSIONS) == p['expected_features'] == 48
    for file, digest in p['input_receipts'].items():
        assert sha(Path(file)) == digest
    relative.GradientBoostingRegressor = partial(GradientBoostingRegressor, max_features=6)
    return p


def verify_model(p):
    m = json.loads((base.ROOT / 'model_report.json').read_text())
    assert m['feature_names'] == p['feature_names']
    assert m['rows'] == p['expected_training_rows'] and m['days'] == p['expected_training_days']
    assert m['last_observation'] == p['expected_last_observation'] < p['evaluation_start']
    assert all(m['parameters'][k] == v for k, v in p['parameters'].items())
    proof = relative.verify_model('relative')
    proof.update(max_features=6, nominal_random_feature_search_not_strict_cap=True,
        year_2024_and_2025_are_exploratory=True, new_group_outcomes_read=False)
    save_json(base.ROOT / 'model_verification.json', proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores'])
    parser.add_argument('--arm', choices=list(inputs.ARMS), required=True)
    parser.add_argument('--fold', choices=['2024h1', '2024h2', '2025h1', '2025h2'], required=True)
    a = parser.parse_args(); p = setup(a.arm, a.fold)
    if a.stage == 'model':
        result = relative.model('relative')
    elif a.stage == 'verify_model':
        result = verify_model(p)
    elif a.stage == 'scores':
        result = base.scores()
    else:
        result = verify_scores(expected_expressions=inputs.ARMS[a.arm])
    print(json.dumps(result, ensure_ascii=False, indent=2))
