"""Fixed agreement between two models trained on disjoint calendar halves."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_float as inputs
from . import tail_formula_relative as relative
from .corporate_cash import sha

STEM = 'tail_formula_half_consensus'
PROTOCOL = Path('config') / (STEM + '_protocol.json')
PARAMETERS = dict(n_estimators=64, max_depth=3, min_samples_leaf=300,
                  learning_rate=.05, subsample=1., random_state=20260927, loss='squared_error')


def policy():
    p = json.loads(PROTOCOL.read_text())
    assert p['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
    assert p['label_report_sha256'] == sha(labels.ROOT / 'full_label_report.json')
    assert p['label_verification_sha256'] == sha(labels.ROOT / 'full_label_verification.json')
    assert p['training_quantile'] == .995 and p['window_end'] == '09:59'
    assert p['arms'] == ['older', 'newer', 'intersection']
    assert p['no_threshold_or_split_scan'] and not p['new_2026_prices_allowed']
    for prefix in ['primary', 'secondary']:
        assert p[prefix + '_control_selection_report_sha256'] == sha(
            Path(p[prefix + '_control_root']) / 'selection_report.json')
    return p


def model_root(name):
    return Path('data/research') / (STEM + '_model_' + name)


def setup(name):
    p = policy(); spec = p['models'][name]
    config = Path('config') / (STEM + '_' + name + '_protocol.json')
    q = json.loads(config.read_text())
    assert q['parent_protocol_sha256'] == sha(PROTOCOL)
    assert all(q[k] == v for k, v in spec.items())
    assert q['parameters'] == PARAMETERS and q['model_max_depth'] == 3
    base.ROOT = model_root(name); base.PROTOCOL = relative.PROTOCOL = config
    base.FEATURES = inputs.ROOT; base.EXPRESSIONS = inputs.EXPRESSIONS; base.HEADER = inputs.HEADER
    base.SOURCE = labels.ROOT
    return spec


def verify_model(name):
    spec = setup(name)
    r = json.loads((base.ROOT / 'model_report.json').read_text())
    assert r['rows'] == spec['expected_rows'] and r['days'] == spec['expected_days']
    assert r['feature_names'] == list(inputs.EXPRESSIONS) and len(r['feature_names']) == 48
    assert all(r['parameters'][k] == v for k, v in PARAMETERS.items())
    return relative.verify_model('relative')


if __name__ == '__main__':
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores'])
    p.add_argument('--model', choices=['2024h1', '2024h2', '2025h1'], required=True)
    a = p.parse_args(); setup(a.model)
    if a.stage == 'model':
        result = relative.model('relative')
    elif a.stage == 'verify_model':
        result = verify_model(a.model)
    elif a.stage == 'scores':
        result = base.scores()
    else:
        result = verify_scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))
