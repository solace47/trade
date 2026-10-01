"""Evaluate all chronological gates together, without selecting again."""
import argparse
import json
from pathlib import Path

import run_tail_formula_chronological_rule_gate as fit
import finish_tail_formula_profit_rule_search as bridge

ROOT=fit.ROOT
EXECUTION=Path('config/tail_formula_chronological_rule_gate_evaluation.json')
bridge.fit,bridge.ROOT,bridge.EXECUTION=fit,ROOT,EXECUTION
bridge.shared.fit,bridge.shared.ROOT,bridge.shared.EXECUTION=fit,ROOT,EXECUTION

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','analyze','finish'])
    stage=parser.parse_args().stage
    print(json.dumps(bridge.finish() if stage=='finish' else getattr(bridge.shared,stage)(),ensure_ascii=False))
