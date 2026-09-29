"""Train both input-unit arms under the same shallow learner and time folds."""
import argparse
import json
import sys

from . import tail_formula_dual_units as inputs
from . import tail_formula_range_change_model as engine
from .corporate_cash import sha, save_json

ORIGINAL_VERIFY_MODEL = engine.verify_model


def verify_model():
    proof = ORIGINAL_VERIFY_MODEL()
    if inputs.STEM.endswith('_constant'):
        model = json.loads((engine.base.ROOT / 'model_report.json').read_text())
        assert not any(index >= 48 for tree in model['trees'] for index in tree['feature'])
        proof['nineteen_constant_columns_unused_by_all_nodes'] = True
        save_json(engine.base.ROOT / 'model_verification.json', proof)
    return proof


def main(arm=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--arm', choices=['constant', 'dual'], required=arm is None, default=arm)
    args, remainder = p.parse_known_args()
    if 'analyze' in remainder:
        joint = json.loads((inputs.ROOT / 'joint_selection_freeze.json').read_text())
        assert joint['passed'] and joint['protocol_sha256'] == sha(inputs.PROTOCOL) and len(joint['selections']) == 6
    inputs.STEM = 'tail_formula_dual_units_' + args.arm
    inputs.EXPRESSIONS = {**inputs.previous.EXPRESSIONS, **{n: inputs.NEW_EXPRESSIONS[n] for n in inputs.ARM_FIELDS[args.arm]}}
    engine.inputs = inputs; engine.verify_model = verify_model
    if 'verify_scores' in remainder:
        from .tail_formula_offset_logit48 import verify_scores
        fold = remainder[remainder.index('--fold') + 1] if '--fold' in remainder else '2024'
        engine.setup(fold)
        print(json.dumps(verify_scores(expected_expressions=inputs.EXPRESSIONS), ensure_ascii=False, indent=2))
        return
    sys.argv = [sys.argv[0], *remainder]
    engine.main()


if __name__ == '__main__':
    main()
