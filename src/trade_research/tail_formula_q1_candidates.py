"""Coordinate frozen 2025 models before a bounded, already-exposed Q1 comparison."""
import argparse
import json
from pathlib import Path

from . import tail_formula_daily_efficiency_q1 as path_model
from . import tail_formula_equal_weight_q1 as equal_model
from .corporate_cash import save_json, sha

ROOT = Path('data/research/tail_formula_q1_candidates')
PROTOCOL = Path('config/tail_formula_q1_candidates_protocol.json')
OLD = Path('data/research/tail_formula_forward_2026q1')


def policy():
    p = json.loads(PROTOCOL.read_text())
    for mapping in ['model_protocols', 'source_reports']:
        for file, digest in p[mapping].items():
            assert sha(Path(file)) == digest
    assert (p['signal_first'], p['signal_last'], p['observation_last']) == ('2026-01-01', '2026-03-31', '2026-04-01')
    assert p['warmup_first'] == '2025-01-01' and p['minimum_members'] == 2000
    assert not p['strict_blind'] and not p['new_q2_signal_prices_allowed']
    return p


def models_gate():
    p = policy(); assert not (ROOT/'models_gate.json').exists()
    path_model.policy(); equal_model.policy()
    first = json.loads((path_model.ROOT/'model_freeze_report.json').read_text())
    second = json.loads((equal_model.ROOT/'model_freeze_report.json').read_text())
    assert second['reused_other_models_freeze_sha256'] == sha(path_model.ROOT/'model_freeze_report.json')
    definitions = {
        'control': (path_model.ROOT/'models/control', first['models']['control']),
        'path': (path_model.ROOT/'models/path', first['models']['path']),
        'equal_weight': (equal_model.ROOT/'model', second)}
    records = {}
    for name, (folder, frozen) in definitions.items():
        for key, file in [('model_report_sha256', 'model_report.json'),
                          ('model_verification_sha256', 'model_verification.json'), ('core_sha256', 'frozen_numeric_core.tdx')]:
            assert frozen[key] == sha(folder/file)
        proof = json.loads((folder/'model_verification.json').read_text())
        m = json.loads((folder/'model_report.json').read_text())
        assert proof['passed'] and proof['model_report_sha256'] == sha(folder/'model_report.json')
        assert (m['rows'], m['days'], m['last_observation']) == (577424, 242, '2025-12-31')
        records[name] = dict(root=str(folder), **{k:frozen[k] for k in ['model_report_sha256',
            'model_verification_sha256', 'core_sha256', 'chosen_threshold']})
    active = ['control']
    if first['proceed_to_q1_inputs']:
        active.append('path')
    if second['proceed_to_q1_inputs']:
        active.append('equal_weight')
    r = dict(protocol_sha256=sha(PROTOCOL), models=records, active_variants=active,
        path_freeze_report_sha256=sha(path_model.ROOT/'model_freeze_report.json'),
        equal_weight_freeze_report_sha256=sha(equal_model.ROOT/'model_freeze_report.json'),
        all_models_frozen_before_this_q1_input_extension=True, q1_new_input_values_read=False,
        q1_new_group_outcomes_read=False, prior_q1_outcomes_exposed=True, strict_blind=False, no_exit_rules=True)
    ROOT.mkdir(parents=True, exist_ok=True); save_json(ROOT/'models_gate.json', r); return r


def checked_models():
    p = policy(); r = json.loads((ROOT/'models_gate.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL)
    for key, file in [('path_freeze_report_sha256', path_model.ROOT/'model_freeze_report.json'),
                      ('equal_weight_freeze_report_sha256', equal_model.ROOT/'model_freeze_report.json')]:
        assert r[key] == sha(file)
    for item in r['models'].values():
        for key, file in [('model_report_sha256', 'model_report.json'),
                          ('model_verification_sha256', 'model_verification.json'), ('core_sha256', 'frozen_numeric_core.tdx')]:
            assert item[key] == sha(Path(item['root'])/file)
    return p, r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['models_gate', 'checked_models']); a = parser.parse_args()
    print(json.dumps(globals()[a.stage](), ensure_ascii=False, indent=2))
