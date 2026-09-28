"""Freeze the 52-input 2025 model before reusing exposed Q1 inputs."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_path_relative as inputs
from . import tail_formula_relative as relative
from . import tail_formula_q1_candidates as previous
from .corporate_cash import save_json, sha

ROOT = Path('data/research/tail_formula_path_relative_q1')
PROTOCOL = Path('config/tail_formula_path_relative_q1_protocol.json')


def policy():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['references'].items():
        assert sha(Path(file)) == digest
    assert (p['training_start'], p['training_end']) == ('2025-01-01', '2026-01-01')
    assert (p['signal_first'], p['signal_last'], p['observation_last']) == ('2026-01-01', '2026-03-31', '2026-04-01')
    assert p['benchmark_minimum_members'] == 2000 and p['training_quantile'] == .995
    assert not p['strict_blind'] and not p['new_q2_signal_prices_allowed']
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
    setup(); p = policy(); assert not (ROOT/'models_gate.json').exists()
    m = json.loads((base.ROOT/'model_report.json').read_text())
    proof = json.loads((base.ROOT/'model_verification.json').read_text())
    assert proof['passed'] and proof['model_report_sha256'] == sha(base.ROOT/'model_report.json')
    assert m['protocol_sha256'] == sha(PROTOCOL) and m['last_observation'] < '2026-01-01'
    _, old = previous.checked_models()
    records = dict(old['models'])
    for metadata in records.values():
        other = json.loads((Path(metadata['root'])/'model_report.json').read_text())
        assert (other['rows'], other['days'], other['last_observation']) == (m['rows'], m['days'], m['last_observation'])
        assert other['label_report_sha256'] == m['label_report_sha256']
        assert all(other['parameters'][key] == m['parameters'][key] for key in p['parameters'])
    cut = next(t for t in m['thresholds'] if t['training_quantile'] == .995)
    core = base.ROOT/'frozen_numeric_core.tdx'
    core.write_text(inputs.native_core(m, cut['threshold'], inputs.EXPRESSIONS, inputs.HEADER))
    splits = {name:sum(sum(i>=0 and m['feature_names'][i] == name for i in t['feature']) for t in m['trees'])
              for name in inputs.NEW_EXPRESSIONS}
    records['path_relative'] = dict(root=str(base.ROOT), model_report_sha256=sha(base.ROOT/'model_report.json'),
        model_verification_sha256=sha(base.ROOT/'model_verification.json'), core_sha256=sha(core), chosen_threshold=cut)
    result = dict(protocol_sha256=sha(PROTOCOL), models=records,
        active_variants=['control','path','equal_weight','path_relative'],
        reused_models_gate_sha256=sha(previous.ROOT/'models_gate.json'),
        rows=m['rows'], days=m['days'], last_observation=m['last_observation'], new_feature_splits=splits,
        proceed_to_q1_inputs=sum(splits.values())>0, original_three_models_not_refitted=True,
        all_models_frozen_before_this_q1_input_extension=True, q1_feature_values_read_in_this_extension=False,
        q1_new_group_outcomes_read=False, prior_q1_outcomes_exposed=True, strict_blind=False,
        no_q2_signal_prices_read=True, no_exit_rules=True)
    save_json(ROOT/'models_gate.json', result); return result


def checked_models():
    p = policy(); r = json.loads((ROOT/'models_gate.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['proceed_to_q1_inputs']
    assert r['reused_models_gate_sha256'] == sha(previous.ROOT/'models_gate.json')
    previous.checked_models()
    for meta in r['models'].values():
        for key, file in [('model_report_sha256','model_report.json'),
                          ('model_verification_sha256','model_verification.json'),('core_sha256','frozen_numeric_core.tdx')]:
            assert meta[key] == sha(Path(meta['root'])/file)
    return p, r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['model','verify_model','freeze_model']); args = parser.parse_args()
    if args.stage == 'model':
        setup(); result = relative.model('relative')
    else:
        result = globals()[args.stage]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
