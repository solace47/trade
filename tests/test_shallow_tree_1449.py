from trade_research.shallow_tree_1449 import decision_buyable


def test_decision_filter_uses_known_quote_and_full_lot():
    assert decision_buyable("sh.600001", 10., 10.)
    assert not decision_buyable("sh.600001", 11., 10.)
    assert not decision_buyable("sh.600001", 10.99, 10.)
    assert decision_buyable("sh.600001", 10.97, 10.)
    assert not decision_buyable("sh.600001", 201., 200.)
    assert not decision_buyable("sh.600001", 10.004, 10.)
    assert not decision_buyable("sh.688001", 10., 10.)
