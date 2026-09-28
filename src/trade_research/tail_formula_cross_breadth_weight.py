"""The fixed twenty-day leaf mass constraint applied to the breadth inputs."""
import argparse
import json
from pathlib import Path

from . import tail_formula_additive as base
from . import tail_formula_cross_breadth as inputs
from . import tail_formula_day_weight48 as growth
from . import tail_formula_prior_day as adapter
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_cross_breadth_weight'
COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')


def configure():
    inputs.configure()
    adapter.STEM = STEM
    adapter.COMBINED_PROTOCOL = COMBINED_PROTOCOL
    adapter.CONTROL = Path('data/research') / (STEM + '_control')
    for fold in ['2024', 'recent']:
        p = json.loads((Path('config') / (STEM + '_' + fold + '_protocol.json')).read_text())
        assert p['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
        assert p['inputs_protocol_sha256'] == sha(inputs.PROTOCOL)
        assert p['label_report_sha256'] == sha(base.SOURCE / 'full_label_report.json')
        assert p['expected_features'] == len(inputs.EXPRESSIONS) == 50


def verify_model():
    r = json.loads((base.ROOT / 'model_report.json').read_text())
    p = json.loads(base.PROTOCOL.read_text())
    assert r['feature_names'] == list(base.EXPRESSIONS) and len(r['feature_names']) == p['expected_features'] == 50
    assert r['days'] == p['expected_training_days'] == 241
    assert r['minimum_leaf_day_weight'] == p['minimum_leaf_day_weight'] == 20
    assert r['constraint_applied_during_split_search'] and not r['post_fit_pruning']
    assert all(r['parameters'][key] == value for key, value in p['parameters'].items())
    assert r['parameters']['min_weight_fraction_leaf'] == 20 / r['days']
    proof = relative.verify_model('relative')
    masses = []; dates = []
    for tree in r['trees']:
        for node, left in enumerate(tree['children_left']):
            if left < 0:
                masses.append(tree['weighted_n_node_samples'][node])
                dates.append(tree['training_days'][node])
    assert min(masses) >= 20 - 1e-8 and min(dates) >= 20
    proof.update(all_final_leaf_day_weights_checked=True, leaf_checks=len(masses),
        minimum_leaf_day_weight_observed=min(masses), minimum_leaf_dates_observed=min(dates),
        unchanged_original_relative_objective=True, identical_breadth_50_inputs=True)
    save_json(base.ROOT / 'model_verification.json', proof)
    return proof


if __name__ == '__main__':
    from . import tail_formula_context_2024 as linkage
    from . import tail_formula_recent as study
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--fold', choices=['2024', 'recent', 'combined', 'control'], default='2024')
    a = p.parse_args(); configure()
    if a.stage == 'analyze':
        assert all((Path('data/research') / (STEM + '_' + f) / 'selection_verification.json').exists()
                   for f in ['2024', 'recent', '2025', 'control'])
    if a.fold == 'control':
        assert a.stage in ['freeze', 'verify', 'analyze']
        result = (linkage.common_analysis(adapter.CONTROL, COMBINED_PROTOCOL) if a.stage == 'analyze'
                  else adapter.control(a.stage + '_control'))
    else:
        adapter.setup(a.fold)
        if a.fold == 'combined':
            assert a.stage in ['freeze', 'verify', 'analyze']
            result = (linkage.common_analysis(linkage.COMBINED, linkage.PROTOCOL) if a.stage == 'analyze'
                      else getattr(linkage, 'combine' if a.stage == 'freeze' else 'verify_combined')())
        elif a.stage == 'model':
            result = growth.model()
        elif a.stage == 'verify_model':
            result = verify_model()
        elif a.stage == 'verify_scores':
            result = verify_scores()
        elif a.stage in ['freeze', 'verify']:
            result = getattr(study, a.stage)()
        else:
            result = getattr(base, a.stage)()
    print(json.dumps(result, ensure_ascii=False, indent=2))
