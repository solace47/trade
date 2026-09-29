"""Use the unchanged learner with fifteen literal-zero index/financial placeholders."""
import json

from . import tail_formula_price_volume_only as inputs
from . import tail_formula_range_change_model as engine
from .corporate_cash import save_json

ORIGINAL_VERIFY_MODEL = engine.verify_model


def verify_model():
    proof = ORIGINAL_VERIFY_MODEL()
    root = engine.base.ROOT
    model = json.loads((root / 'model_report.json').read_text())
    assert not any(f in range(33, 48) for tree in model['trees'] for f in tree['feature'])
    proof['fifteen_index_financial_placeholders_unused_by_all_tree_nodes'] = True
    save_json(root / 'model_verification.json', proof)
    return proof


if __name__ == '__main__':
    engine.inputs = inputs
    engine.verify_model = verify_model
    engine.main()
