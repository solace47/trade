"""Matched current-strength and historical-rank arms on the frozen input table."""
import argparse
import json
import sys

from . import tail_formula_intraday_rank as inputs
from . import tail_formula_range_change_model as engine
from .corporate_cash import sha


def configure(arm):
    assert arm in ['raw', 'rank']
    inputs.STEM = 'tail_formula_intraday_rank_' + arm
    inputs.EXPRESSIONS = {**inputs.previous.EXPRESSIONS,
                          **{n: inputs.NEW_EXPRESSIONS[n] for n in (['IT01'] if arm == 'raw' else ['IT01', 'IT02'])}}
    engine.inputs = inputs


if __name__ == '__main__':
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument('--arm', choices=['raw', 'rank'], required=True)
    args, remainder = p.parse_known_args()
    if 'analyze' in remainder:
        r = json.loads((inputs.ROOT / 'joint_selection_freeze.json').read_text())
        assert r['passed'] and r['protocol_sha256'] == sha(inputs.PROTOCOL) and len(r['selections']) == 6
    configure(args.arm)
    sys.argv = [sys.argv[0], *remainder]
    engine.main()
