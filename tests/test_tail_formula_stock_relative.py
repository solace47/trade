from trade_research import tail_formula_stock_holdout as absolute
from trade_research import tail_formula_stock_relative as relative


def test_projection_and_calibration_namespaces_leave_absolute_artifacts_unchanged():
    assert absolute.model_root('2025h1', 0) == absolute.ROOT / 'models/2025h1/group0'
    assert relative.model_root('2025h1', 0) == relative.ROOT / 'models/2025h1/group0'
    assert relative.model_root('2025h1', 0) != absolute.model_root('2025h1', 0)
    assert relative.INPUTS == absolute.INPUTS
    assert absolute.LABEL_FIELDS == ['date', 'code', 'next_date', 'known15', 'opportunity15', 'known_no_trade']
    assert relative.LABEL_FIELDS == absolute.LABEL_FIELDS + ['adverse_return15', 'sustained_return15']
    assert absolute.PROTOCOL.name == 'tail_formula_stock_holdout_input_protocol.json'
    assert relative.shared.PROTOCOL == relative.PROTOCOL
