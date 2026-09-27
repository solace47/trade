"""Freeze the unchanged 48-input next-morning method before its 2026 Q1 inputs."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_float as original
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

ROOT = Path('data/research/tail_formula_forward_2026q1')
MODEL = ROOT / 'model'
PROTOCOL = Path('config/tail_formula_forward_2026q1_protocol.json')


def policy():
    p = json.loads(PROTOCOL.read_text())
    assert p['training_start'] == '2025-01-01' and p['training_end'] == '2026-01-01'
    assert p['signal_first'] == '2026-01-01' and p['signal_last'] == '2026-03-31'
    assert p['observation_last'] == '2026-04-01' and p['threshold_quantile'] == .995
    assert p['training_feature_report_sha256'] == sha(original.ROOT / 'feature_report.json')
    assert p['training_label_report_sha256'] == sha(base.SOURCE / 'full_label_report.json')
    return p


def setup():
    policy()
    original.setup('recent')
    base.ROOT = MODEL; base.PROTOCOL = PROTOCOL; relative.PROTOCOL = PROTOCOL


def freeze_model():
    setup()
    if (ROOT / 'model_freeze_report.json').exists():
        raise ValueError('Do not replace the forward model freeze')
    p = policy()
    proof = json.loads((MODEL / 'model_verification.json').read_text())
    assert proof['passed'] and proof['model_report_sha256'] == sha(MODEL / 'model_report.json')
    m = json.loads((MODEL / 'model_report.json').read_text())
    assert m['protocol_sha256'] == sha(PROTOCOL) and m['last_observation'] < p['signal_first']
    assert not m['new_2026_prices_read'] and m['feature_names'] == list(original.EXPRESSIONS)
    cut = m['thresholds'][3]
    assert cut['training_quantile'] == p['threshold_quantile']
    (MODEL / 'frozen_numeric_core.tdx').write_text(base.native_core(m, cut['threshold'], original.EXPRESSIONS, original.HEADER))
    report = dict(protocol_sha256=sha(PROTOCOL), model_report_sha256=sha(MODEL / 'model_report.json'),
                  model_verification_sha256=sha(MODEL / 'model_verification.json'),
                  core_sha256=sha(MODEL / 'frozen_numeric_core.tdx'), chosen_threshold=cut,
                  training_rows=m['rows'], training_days=m['days'], last_observation=m['last_observation'],
                  new_2026_prices_read=False, old_2026_alpha158_exposure=True, strict_blind=False,
                  same_method_as_original_48=True, shortlist_top_positions=5, no_exit_rules=True)
    save_json(ROOT / 'model_freeze_report.json', report)
    return report


def checked_model():
    p = policy()
    r = json.loads((ROOT / 'model_freeze_report.json').read_text())
    for key, path in [('protocol_sha256', PROTOCOL), ('model_report_sha256', MODEL / 'model_report.json'),
                      ('model_verification_sha256', MODEL / 'model_verification.json'),
                      ('core_sha256', MODEL / 'frozen_numeric_core.tdx')]:
        assert r[key] == sha(path)
    proof = json.loads((MODEL / 'model_verification.json').read_text())
    m = json.loads((MODEL / 'model_report.json').read_text())
    assert proof['passed'] and proof['model_report_sha256'] == r['model_report_sha256']
    assert r['chosen_threshold'] == m['thresholds'][3] and m['last_observation'] < p['signal_first']
    assert (MODEL / 'frozen_numeric_core.tdx').read_text() == base.native_core(m, r['chosen_threshold']['threshold'], original.EXPRESSIONS, original.HEADER)
    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['model', 'verify_model', 'freeze_model', 'check_model'])
    a = parser.parse_args()
    if a.stage == 'freeze_model':
        result = freeze_model()
    elif a.stage == 'check_model':
        result = checked_model()
    else:
        setup(); result = getattr(relative, a.stage)('relative')
    print(json.dumps(result, ensure_ascii=False, indent=2))
