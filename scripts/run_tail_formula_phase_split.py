"""Entry point for the phase-split source gate; fits require a later fixed protocol."""
import json
from trade_research.tail_formula_phase_split import prepare


if __name__ == '__main__':
    print(json.dumps(prepare(), ensure_ascii=False))
