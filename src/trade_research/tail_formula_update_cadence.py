"""Predeclared monthly and quarterly schedules with unchanged 48-input models."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_before1000_model as original
from . import tail_formula_float as inputs
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha
from .tail_formula_offset_logit48 import verify_scores as independent_scores

STEM = 'tail_formula_update_cadence'
ROOT = Path('data/research') / STEM
MASTER = Path('config') / (STEM + '_protocol.json')


def checked():
    p = json.loads(MASTER.read_text())
    for path, digest in p['references'].items():
        assert sha(Path(path)) == digest
    assert [m['month'] for m in p['models']] == list(range(1, 13))
    assert [m['month'] for m in p['models'] if m['reuse_existing']] == [1, 7]
    return p


def description(month):
    assert month in range(1, 13)
    return checked()['models'][month - 1]


def model_protocol(month):
    return Path('config') / f'{STEM}_model_m{month:02d}_protocol.json'


def setup(month):
    desc = description(month)
    p = json.loads(model_protocol(month).read_text())
    assert p['inputs_protocol_sha256'] == sha(MASTER)
    for key in ['training_start', 'training_end']:
        assert p[key] == desc[key]
    assert p['source_root'] == desc['source'] and p['reuse_existing_model'] == desc['reuse_existing']
    original.setup('2024')
    base.ROOT = Path(desc['source'])
    base.PROTOCOL = relative.PROTOCOL = model_protocol(month)
    return p, desc


def validate_source(month):
    desc = description(month)
    root = Path(desc['source'])
    model = json.loads((root / 'model_report.json').read_text())
    model_proof = json.loads((root / 'model_verification.json').read_text())
    score = json.loads((root / 'score_report.json').read_text())
    score_proof = json.loads((root / 'score_verification.json').read_text())
    assert model_proof['passed'] and model_proof['model_report_sha256'] == sha(root / 'model_report.json')
    assert score_proof['passed'] and score_proof['score_report_sha256'] == sha(root / 'score_report.json')
    assert score['model_report_sha256'] == sha(root / 'model_report.json') and score['scores_sha256'] == sha(root / 'scores.parquet')
    assert model['training_start'] == desc['training_start'] and model['training_end'] == desc['training_end']
    assert model['last_observation'] < desc['training_end']
    assert model['feature_names'] == list(inputs.EXPRESSIONS) and len(model['trees']) == 64
    assert model['label_report_sha256'] == sha(Path('data/research/tail_formula_before1000/full_label_report.json'))
    assert model['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
    if not desc['reuse_existing']:
        assert model['protocol_sha256'] == score['protocol_sha256'] == sha(model_protocol(month))
        assert score['new_2025H2_score_groups_read'] == model['new_2025H2_score_groups_read']
        assert score_proof['new_2025H2_score_groups_read'] == model['new_2025H2_score_groups_read']
    cut = model['thresholds'][3]
    assert cut['training_quantile'] == .995
    return root, model, cut, desc


def model(month):
    p, desc = setup(month)
    assert not desc['reuse_existing'], 'Reuse January and July; do not refit them'
    return relative.model('relative')


def verify_model(month):
    p, desc = setup(month)
    assert not desc['reuse_existing']
    result = relative.verify_model('relative')
    m = json.loads((base.ROOT / 'model_report.json').read_text())
    assert all(m['parameters'][k] == value for k, value in p['parameters'].items())
    assert m['rows'] == m['trees'][0]['n_node_samples'][0]
    result['calendar_update_month'] = month
    result['all_training_observations_strictly_before_month_start'] = True
    save_json(base.ROOT / 'model_verification.json', result)
    return result


def scores(month):
    _, desc = setup(month)
    assert not desc['reuse_existing']
    result = base.scores()
    m = json.loads((base.ROOT / 'model_report.json').read_text())
    # The legacy shared scorer hardcodes this flag false; record the actual
    # training scope for late-year models without rewriting old artifacts.
    result['new_2025H2_score_groups_read'] = m['new_2025H2_score_groups_read']
    save_json(base.ROOT / 'score_report.json', result)
    return result


def verify_scores(month):
    _, desc = setup(month)
    assert not desc['reuse_existing']
    result = independent_scores()
    m = json.loads((base.ROOT / 'model_report.json').read_text())
    result['new_2025H2_score_groups_read'] = m['new_2025H2_score_groups_read']
    save_json(base.ROOT / 'score_verification.json', result)
    core = base.native_core(m, m['thresholds'][3]['threshold'], inputs.EXPRESSIONS, inputs.HEADER)
    (base.ROOT / 'frozen_numeric_core.tdx').write_text(core)
    validate_source(month)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores'])
    parser.add_argument('--month', type=int, choices=range(1, 13), required=True)
    args = parser.parse_args()
    result = globals()[args.stage](args.month)
    print(json.dumps({k: v for k, v in result.items() if k != 'trees'}, ensure_ascii=False, indent=2))
