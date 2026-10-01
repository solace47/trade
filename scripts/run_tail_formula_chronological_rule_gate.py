"""Search on the discovery half and gate its single rule on the next half."""
import argparse
import json
from pathlib import Path

import run_tail_formula_profit_rule_search as shared

ROOT=Path('data/research/tail_formula_chronological_rule_gate')
PROTOCOL=Path('config/tail_formula_chronological_rule_gate_protocol.json')
EXECUTION=Path('config/tail_formula_chronological_rule_gate_execution.json')
shared.ROOT,shared.PROTOCOL,shared.EXECUTION=ROOT,PROTOCOL,EXECUTION
checked=shared.checked
condition_key=shared.condition_key

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['fit'])
    parser.parse_args()
    print(json.dumps(shared.fit_all(),ensure_ascii=False))
