from trade_research import tail_formula_absolute_input_calibration as absolute
from trade_research import tail_formula_input_calibration as relative


def test_absolute_artifact_routing_does_not_change_relative_paths():
    before = relative.calibration_paths('2025h1')
    root, report, verification = absolute.calibration_paths('2025h1')
    assert root == absolute.ROOT / 'models' / '2025h1'
    assert report.parent == verification.parent == root
    assert root != before[0]
    assert relative.calibration_paths('2025h1') == before
    assert absolute.INPUTS == relative.INPUTS
