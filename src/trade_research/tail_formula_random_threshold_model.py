"""Matched whole-week tree averaging with randomized candidate thresholds."""
import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeRegressor

from . import tail_formula_week_bagging as engine
from .corporate_cash import save_json, sha

STEM = 'tail_formula_random_threshold'
PROTOCOL = Path('config') / (STEM + '_protocol.json')


def setup(fold):
    p = json.loads(PROTOCOL.read_text())
    for path, digest in p['references'].items():
        assert sha(Path(path)) == digest
    assert p['parameters']['splitter'] == 'random'
    engine.STEM = STEM
    engine.PROTOCOL = PROTOCOL
    engine.PARAMETERS = p['parameters']
    engine.DecisionTreeRegressor = partial(DecisionTreeRegressor, splitter='random')
    engine.setup(fold)


def verify_model(fold):
    proof = engine.verify_model()
    root = engine.base.ROOT
    model = json.loads((root / 'model_report.json').read_text())
    old = Path('data/research') / ('tail_formula_week_bagging_' + fold)
    prior = json.loads((old / 'model_report.json').read_text())
    assert {k: v for k, v in model['parameters'].items() if k != 'splitter'} == prior['parameters']
    for field in ['feature_names', 'rows', 'days', 'training_start', 'training_end', 'last_observation',
                  'training_week_starts', 'sampling_schedule', 'aggregation_weight', 'learning_rate']:
        assert model[field] == prior[field]
    train = engine.relative.training('relative')
    x = engine.base.encode(train)
    # Every exported integer comparison, including thresholds not at midpoints,
    # must agree with the stored floating comparison before leaf evaluation.
    checks = 0
    non_half = 0
    for tree in model['trees']:
        for feature, threshold in zip(tree['feature'], tree['threshold']):
            if feature < 0:
                continue
            np.testing.assert_array_equal(x[:, feature] <= threshold, x[:, feature] <= np.floor(threshold))
            checks += 1
            non_half += int(not float(2 * threshold).is_integer())
    proof.update(identical_week_schedules_and_training_to_matched_greedy=True,
                 matched_model_report_sha256=sha(old / 'model_report.json'),
                 all_random_threshold_integer_export_branches_verified=True,
                 random_split_nodes=checks, non_half_integer_thresholds=non_half)
    save_json(root / 'model_verification.json', proof)
    return proof


def verify():
    proof = engine.study.verify()
    root = engine.base.ROOT
    m = json.loads((root / 'model_report.json').read_text())
    report = json.loads((root / 'selection_report.json').read_text())
    core = (root / 'frozen_numeric_core.tdx').read_text()
    assert core == engine.base.native_core(m, report['chosen_threshold']['threshold'],
                                          engine.base.EXPRESSIONS, engine.base.HEADER)
    proof['all_native_tree_texts_threshold_flooring_and_export_scaling_verified'] = True
    save_json(root / 'selection_verification.json', proof)
    return proof


if __name__ == '__main__':
    from .tail_formula_offset_logit48 import verify_scores
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    parser.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    args = parser.parse_args()
    setup(args.fold)
    if args.stage == 'analyze':
        joint_path = Path('data/research') / STEM / 'joint_selection_freeze.json'
        joint = json.loads(joint_path.read_text())
        assert joint['passed'] and joint['protocol_sha256'] == sha(PROTOCOL)
        root = engine.linkage.COMBINED if args.fold == 'combined' else engine.base.ROOT
        result = engine.evaluation.analyze(root, engine.linkage.PROTOCOL if args.fold == 'combined' else engine.base.PROTOCOL)
    elif args.fold == 'combined':
        assert args.stage in ['freeze', 'verify']
        result = engine.linkage.combine() if args.stage == 'freeze' else engine.linkage.verify_combined()
    elif args.stage == 'model':
        result = engine.model()
    elif args.stage == 'verify_model':
        result = verify_model(args.fold)
    elif args.stage == 'verify_scores':
        result = verify_scores()
    elif args.stage == 'verify':
        result = verify()
    elif args.stage == 'freeze':
        result = engine.study.freeze()
    else:
        result = engine.base.scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))
