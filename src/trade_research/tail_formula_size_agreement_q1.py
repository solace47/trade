"""Fixed size-context intersection on the already exposed 2026 first quarter."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_relative as relative
from . import tail_formula_size_context as extended
from . import tail_formula_size_agreement as intersection
from .corporate_cash import save_json, sha

ROOT = Path('data/research/tail_formula_size_agreement_q1')
MODEL = ROOT / 'model'
OUT = ROOT / 'inputs'
OLD = Path('data/research/tail_formula_forward_2026q1')
PROTOCOL = Path('config/tail_formula_size_agreement_q1_protocol.json')


def policy():
    p = json.loads(PROTOCOL.read_text())
    for item in p['references'].values():
        assert sha(Path(item['path'])) == item['sha256']
    assert (p['training_start'], p['training_end']) == ('2025-01-01', '2026-01-01')
    assert (p['signal_first'], p['signal_last']) == ('2026-01-01', '2026-03-31')
    assert p['threshold_quantile'] == .995 and not p['new_2026_stock_prices_allowed']
    return p


def setup():
    policy()
    base.ROOT = MODEL; base.PROTOCOL = PROTOCOL; relative.PROTOCOL = PROTOCOL
    base.FEATURES = extended.ROOT; base.EXPRESSIONS = extended.EXPRESSIONS; base.HEADER = extended.HEADER


def freeze_model():
    setup(); assert not (ROOT / 'model_freeze_report.json').exists()
    v = json.loads((MODEL / 'model_verification.json').read_text())
    m = json.loads((MODEL / 'model_report.json').read_text())
    original = json.loads((OLD / 'model/model_report.json').read_text())
    assert v['passed'] and v['model_report_sha256'] == sha(MODEL / 'model_report.json')
    assert m['last_observation'] < '2026-01-01' and m['feature_names'] == list(extended.EXPRESSIONS)
    assert (m['rows'], m['days'], m['parameters']) == (original['rows'], original['days'], original['parameters'])
    cut = m['thresholds'][3]; assert cut['training_quantile'] == .995
    path = MODEL / 'frozen_numeric_core.tdx'
    path.write_text(base.native_core(m, cut['threshold'], extended.EXPRESSIONS, extended.HEADER))
    sources = [(OLD / 'model/frozen_numeric_core.tdx').read_text(), path.read_text()]
    prefixes, bodies = intersection.prefixes_and_bodies(sources)
    second = re.sub(r'\bT(\d{2})\b', r'U\1', bodies[1]); second = re.sub(r'\bSC\b', 'SSC', second)
    oldcut = original['thresholds'][3]['threshold']
    composed = prefixes[1] + bodies[0] + second + f'CORE:(SC>{oldcut:.17g}) AND (SSC>{cut["threshold"]:.17g});\n'
    (ROOT / 'frozen_numeric_core.tdx').write_text(composed)
    r = dict(protocol_sha256=sha(PROTOCOL), model_report_sha256=sha(MODEL / 'model_report.json'),
        model_verification_sha256=sha(MODEL / 'model_verification.json'),
        original_model_report_sha256=sha(OLD / 'model/model_report.json'),
        component_core_sha256=sha(path), core_sha256=sha(ROOT / 'frozen_numeric_core.tdx'),
        score_cuts=[oldcut, cut['threshold']], rows=m['rows'], days=m['days'],
        original_fit_parameters_and_training_scope_unchanged=True, new_2026_index_prices_read=False,
        prior_q1_original_outcomes_exposed=True, strict_blind=False, no_exit_rules=True)
    save_json(ROOT / 'model_freeze_report.json', r); return r


def checked_model():
    policy(); r = json.loads((ROOT / 'model_freeze_report.json').read_text())
    for key, path in [('protocol_sha256', PROTOCOL), ('model_report_sha256', MODEL / 'model_report.json'),
                      ('model_verification_sha256', MODEL / 'model_verification.json'),
                      ('component_core_sha256', MODEL / 'frozen_numeric_core.tdx'),
                      ('core_sha256', ROOT / 'frozen_numeric_core.tdx')]:
        assert r[key] == sha(path)
    from .tail_formula_forward import checked_model as original_model
    old = original_model()
    m = json.loads((MODEL / 'model_report.json').read_text())
    v = json.loads((MODEL / 'model_verification.json').read_text())
    assert v['passed'] and v['model_report_sha256'] == r['model_report_sha256']
    assert r['original_model_report_sha256'] == old['model_report_sha256']
    assert r['score_cuts'] == [old['chosen_threshold']['threshold'], m['thresholds'][3]['threshold']]
    core = base.native_core(m, r['score_cuts'][1], extended.EXPRESSIONS, extended.HEADER)
    assert (MODEL / 'frozen_numeric_core.tdx').read_text() == core
    sources = [(OLD / 'model/frozen_numeric_core.tdx').read_text(), core]
    prefixes, bodies = intersection.prefixes_and_bodies(sources)
    second = re.sub(r'\bT(\d{2})\b', r'U\1', bodies[1]); second = re.sub(r'\bSC\b', 'SSC', second)
    clause = f'CORE:(SC>{r["score_cuts"][0]:.17g}) AND (SSC>{r["score_cuts"][1]:.17g});\n'
    assert (ROOT / 'frozen_numeric_core.tdx').read_text() == prefixes[1] + bodies[0] + second + clause
    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['model', 'verify_model', 'freeze_model', 'check_model'])
    a = parser.parse_args()
    if a.stage in ['model', 'verify_model']:
        setup(); result = getattr(relative, a.stage)('relative')
    else:
        result = freeze_model() if a.stage == 'freeze_model' else checked_model()
    print(json.dumps(result, ensure_ascii=False, indent=2))
