"""Freeze both signed-cubic models and their three complete selections."""
import json

from trade_research import tail_formula_signed_cubic as inputs
import freeze_tail_formula_bipower_gap as engine

if __name__ == '__main__':
    engine.study = inputs
    print(json.dumps(engine.freeze(), ensure_ascii=False, indent=2))
