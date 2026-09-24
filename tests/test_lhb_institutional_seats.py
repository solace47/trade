import pytest

from trade_research.lhb_institutional_seats import _seat_amount


def test_institution_amount_uses_aligned_top_five_seats() -> None:
    row = {
        "branchNameB": "某证券营业部,机构专用,机构专用,沪股通专用",
        "branchTxAmtB": "100.00,200.50,300.25,400.00",
        "branchNameS": "机构专用,另一证券营业部",
        "branchTxAmtS": "80.00,90.00",
    }
    assert _seat_amount(row, "B") == 500.75
    assert _seat_amount(row, "S") == 80.0


def test_institution_amount_rejects_misaligned_lists() -> None:
    with pytest.raises(ValueError, match="misaligned"):
        _seat_amount({"branchNameB": "机构专用,某证券营业部",
                      "branchTxAmtB": "100.00"}, "B")
