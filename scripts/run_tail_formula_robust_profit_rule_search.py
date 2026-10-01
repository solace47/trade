"""Run the fixed robust variant through the shared, independently audited engine."""
import argparse
import json
from pathlib import Path

import run_tail_formula_profit_rule_search as shared

ROOT=Path('data/research/tail_formula_robust_profit_rule_search')
PROTOCOL=Path('config/tail_formula_robust_profit_rule_search_protocol.json')
EXECUTION=Path('config/tail_formula_robust_profit_rule_search_execution.json')
shared.ROOT,shared.PROTOCOL,shared.EXECUTION=ROOT,PROTOCOL,EXECUTION
checked=shared.checked
condition_key=shared.condition_key

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['fit'])
    parser.parse_args()
    print(json.dumps(shared.fit_all(),ensure_ascii=False))
