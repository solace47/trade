"""Reuse the same three-list freeze and unchanged 49-column control checks."""
import json

import freeze_tail_formula_bipower_gap as engine
from trade_research import tail_formula_event_anchor as inputs

if __name__ == '__main__':
    engine.study = inputs
    print(json.dumps(engine.freeze(), ensure_ascii=False, indent=2))
