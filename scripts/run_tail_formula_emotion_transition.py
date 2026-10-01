"""Run the current market-sentiment source audit; fitting is not yet enabled."""
import argparse
import json
from trade_research.tail_formula_emotion_transition import source_audit

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['source-audit'])
    args = parser.parse_args()
    print(json.dumps(source_audit(), ensure_ascii=False, indent=2), flush=True)
