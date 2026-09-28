"""Add the fixed equal-weight model before reading any new Q1 candidate inputs."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_equal_weight as inputs
from . import tail_formula_relative as relative
from . import tail_formula_daily_efficiency_q1 as other
from .corporate_cash import save_json, sha

ROOT = Path('data/research/tail_formula_equal_weight_q1')
PROTOCOL = Path('config/tail_formula_equal_weight_q1_protocol.json')


def policy():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['references'].items():
        assert sha(Path(file)) == digest
    assert (p['training_start'], p['training_end']) == ('2025-01-01', '2026-01-01')
    assert (p['signal_first'], p['signal_last'], p['observation_last']) == ('2026-01-01', '2026-03-31', '2026-04-01')
    assert p['benchmark_minimum_members'] == 2000 and p['training_quantile'] == .995
    assert not p['strict_blind'] and not p['new_2026_signal_dates_after_q1_allowed']
    return p


def setup():
    policy(); base.ROOT = ROOT/'model'; base.PROTOCOL = PROTOCOL; relative.PROTOCOL = PROTOCOL
    base.FEATURES = inputs.ROOT; base.EXPRESSIONS = inputs.EXPRESSIONS; base.HEADER = inputs.HEADER
    base.SOURCE = labels.ROOT


def verify_model():
    setup(); p = policy(); m = json.loads((base.ROOT/'model_report.json').read_text())
    assert (m['rows'], m['days']) == (p['expected_training_rows'], p['expected_training_days'])
    assert m['feature_names'] == list(inputs.EXPRESSIONS)
    assert all(m['parameters'][key] == value for key, value in p['parameters'].items())
    assert m['last_observation'] < p['training_end']
    return relative.verify_model('relative')


def freeze_model():
    setup(); p = policy(); assert not (ROOT/'model_freeze_report.json').exists()
    m = json.loads((base.ROOT/'model_report.json').read_text())
    proof = json.loads((base.ROOT/'model_verification.json').read_text())
    assert proof['passed'] and proof['model_report_sha256'] == sha(base.ROOT/'model_report.json')
    assert m['protocol_sha256'] == sha(PROTOCOL) and m['last_observation'] < '2026-01-01'
    other.policy(); frozen = json.loads((other.ROOT/'model_freeze_report.json').read_text())
    for variant, metadata in frozen['models'].items():
        folder = other.ROOT/'models'/variant
        assert metadata['model_report_sha256'] == sha(folder/'model_report.json')
        assert metadata['model_verification_sha256'] == sha(folder/'model_verification.json')
        assert metadata['core_sha256'] == sha(folder/'frozen_numeric_core.tdx')
    cut = next(t for t in m['thresholds'] if t['training_quantile'] == .995)
    core = base.ROOT/'frozen_numeric_core.tdx'
    core.write_text(inputs.native_core(m, cut['threshold'], inputs.EXPRESSIONS, inputs.HEADER))
    splits = {name:sum(sum(i>=0 and m['feature_names'][i] == name for i in t['feature']) for t in m['trees'])
              for name in inputs.NEW_EXPRESSIONS}
    result = dict(protocol_sha256=sha(PROTOCOL), model_report_sha256=sha(base.ROOT/'model_report.json'),
        model_verification_sha256=sha(base.ROOT/'model_verification.json'), core_sha256=sha(core),
        helper_sha256=sha(inputs.ROOT/'YJEW20.tdx'), chosen_threshold=cut,
        rows=m['rows'], days=m['days'], last_observation=m['last_observation'], new_feature_splits=splits,
        proceed_to_q1_inputs=sum(splits.values())>0,
        reused_other_models_freeze_sha256=sha(other.ROOT/'model_freeze_report.json'),
        matched_control_not_refitted=True, all_three_models_frozen_before_q1_candidate_inputs=True,
        q1_feature_values_read_in_this_extension=False, q1_outcomes_read_in_this_extension=False,
        prior_q1_outcomes_exposed=True, strict_blind=False, no_q2_signal_prices_read=True, no_exit_rules=True)
    save_json(ROOT/'model_freeze_report.json', result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['model', 'verify_model', 'freeze_model']); args = parser.parse_args()
    if args.stage == 'model':
        setup(); result = relative.model('relative')
    else:
        result = globals()[args.stage]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
