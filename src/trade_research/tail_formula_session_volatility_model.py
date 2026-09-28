"""Matched late-only and opening-plus-late volatility arms."""
import argparse
import json
import sys

from . import tail_formula_session_volatility as inputs
from . import tail_formula_range_change_model as engine
from .corporate_cash import sha


def configure(arm):
    assert arm in ['late', 'both']
    inputs.STEM = 'tail_formula_session_volatility_' + arm
    inputs.EXPRESSIONS = {**inputs.previous.EXPRESSIONS,
                          **{n: inputs.NEW_EXPRESSIONS[n] for n in (['SV01'] if arm == 'late' else ['SV01', 'SV02', 'SV03'])}}
    engine.inputs = inputs


def main(arm=None):
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--arm', choices=['late', 'both'], required=arm is None, default=arm)
    args, remainder = p.parse_known_args()
    if 'analyze' in remainder:
        r = json.loads((inputs.ROOT / 'joint_selection_freeze.json').read_text())
        assert r['passed'] and r['protocol_sha256'] == sha(inputs.PROTOCOL) and len(r['selections']) == 6
    configure(args.arm)
    if 'verify_scores' in remainder:
        from .tail_formula_offset_logit48 import verify_scores
        fold = remainder[remainder.index('--fold')+1] if '--fold' in remainder else '2024'
        engine.setup(fold)
        print(json.dumps(verify_scores(expected_expressions=inputs.EXPRESSIONS), ensure_ascii=False, indent=2))
        return
    sys.argv = [sys.argv[0], *remainder]
    engine.main()


if __name__ == '__main__':
    main()
