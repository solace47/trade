"""Freeze a 2025-only path model and matched control before the exposed Q1 audit."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_daily_efficiency as path_inputs
from . import tail_formula_float as original
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

ROOT = Path('data/research/tail_formula_daily_efficiency_q1')
PROTOCOL = Path('config/tail_formula_daily_efficiency_q1_protocol.json')


def policy():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['references'].items():
        assert sha(Path(file)) == digest
    assert (p['training_start'], p['training_end']) == ('2025-01-01', '2026-01-01')
    assert (p['signal_first'], p['signal_last'], p['observation_last']) == ('2026-01-01', '2026-03-31', '2026-04-01')
    assert p['training_quantile'] == .995 and not p['new_2026_signal_dates_after_q1_allowed']
    assert not p['strict_blind'] and p['prior_2026q1_outcomes_exposed']
    return p


def setup(variant):
    assert variant in ['control', 'path']; policy()
    inputs = original if variant == 'control' else path_inputs
    base.ROOT = ROOT/'models'/variant; base.PROTOCOL = PROTOCOL; relative.PROTOCOL = PROTOCOL
    base.FEATURES = inputs.ROOT; base.EXPRESSIONS = inputs.EXPRESSIONS; base.HEADER = inputs.HEADER
    base.SOURCE = labels.ROOT


def verify_model(variant):
    setup(variant); p = policy(); r = json.loads((base.ROOT/'model_report.json').read_text())
    assert r['rows'] == p['expected_training_rows'] and r['days'] == p['expected_training_days']
    assert r['feature_names'] == list(base.EXPRESSIONS)
    assert all(r['parameters'][key] == value for key, value in p['parameters'].items())
    assert r['last_observation'] < p['training_end']
    return relative.verify_model('relative')


def freeze_models():
    p = policy(); assert not (ROOT/'model_freeze_report.json').exists()
    reports = {}; definitions = {}; new_feature_splits = {}
    for variant in ['control', 'path']:
        setup(variant); folder = base.ROOT
        m = json.loads((folder/'model_report.json').read_text())
        v = json.loads((folder/'model_verification.json').read_text())
        assert v['passed'] and v['model_report_sha256'] == sha(folder/'model_report.json')
        assert m['protocol_sha256'] == sha(PROTOCOL) and m['label_report_sha256'] == p['training_label_report_sha256']
        assert m['last_observation'] < '2026-01-01'
        cut = next(q for q in m['thresholds'] if q['training_quantile'] == .995)
        core = base.native_core(m, cut['threshold'], base.EXPRESSIONS, base.HEADER)
        (folder/'frozen_numeric_core.tdx').write_text(core)
        reports[variant] = dict(model_report_sha256=sha(folder/'model_report.json'),
            model_verification_sha256=sha(folder/'model_verification.json'),
            core_sha256=sha(folder/'frozen_numeric_core.tdx'), rows=m['rows'], days=m['days'],
            last_observation=m['last_observation'], chosen_threshold=cut)
        names = m['feature_names']
        new_feature_splits[variant] = {name:sum(sum(index>=0 and names[index] == name for index in tree['feature'])
            for tree in m['trees']) for name in ['DE05', 'DE20']}
        definitions[variant] = dict(bias=m['bias'], learning_rate=m['learning_rate'], threshold=cut['threshold'],
            trees=[dict(features=[names[i] if i>=0 else None for i in tree['feature']],
                thresholds=[int(x//1) if i>=0 else None for i,x in zip(tree['feature'], tree['threshold'])],
                left=tree['children_left'], right=tree['children_right'], value=tree['value']) for tree in m['trees']])
    equivalent = definitions['control'] == definitions['path']
    no_path_input = sum(new_feature_splits['path'].values()) == 0
    result = dict(protocol_sha256=sha(PROTOCOL), models=reports, new_feature_splits=new_feature_splits,
        exact_all_integer_input_function_equivalence=equivalent, no_path_feature_used=no_path_input,
        proceed_to_q1_inputs=not equivalent and not no_path_input,
        both_models_frozen_before_q1_feature_values=True, q1_feature_values_read_in_this_extension=False,
        q1_outcomes_read_in_this_extension=False, prior_q1_outcomes_exposed=True, strict_blind=False,
        no_q2_signal_prices_read=True, no_exit_rules=True)
    save_json(ROOT/'model_freeze_report.json', result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['model', 'verify_model', 'freeze_models'])
    parser.add_argument('--variant', choices=['control', 'path'], default='path'); args = parser.parse_args()
    if args.stage == 'freeze_models':
        result = freeze_models()
    elif args.stage == 'verify_model':
        result = verify_model(args.variant)
    else:
        setup(args.variant); result = relative.model('relative')
    print(json.dumps(result, ensure_ascii=False, indent=2))
