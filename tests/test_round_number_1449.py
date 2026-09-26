import pandas as pd
import pytest

from trade_research.round_number_1449 import assess_round, price_position
from trade_research.cash_ex_matching import match_candidates


@pytest.mark.parametrize("price,expected", [
    (6.010000228881836, (601, "above", 6, 1)),
    (6.19, (619, "above", 6, 19)), (5.81, (581, "below", 6, 19)),
    (5.99, (599, "below", 6, 1)), (6., (600, "outside", 0, 0)),
    (6.2, (620, "outside", 0, 0)), (5.8, (580, "outside", 0, 0)),
    (5.01, (501, "outside", 0, 0)), (50.19, (5019, "above", 50, 19)),
    (50.99, (5099, "outside", 0, 0))])
def test_integer_cent_groups_are_symmetric_and_stable_to_storage_error(price, expected):
    assert price_position(price) == expected


def test_off_tick_quote_is_not_silently_classified():
    with pytest.raises(ValueError, match="valid cent"):
        price_position(6.014)


def test_new_price_caliper_does_not_change_the_frozen_cash_default():
    common = {"date": "2024-08-20", "half": "2024H2", "daily_rank": 1, "day_return": 0.,
              "return_last29": 0., "prior20_return": 0., "amount_1449": 200000000.}
    a = pd.DataFrame([{**common, "code": "sh.600001", "price_1449": 10.}])
    b = pd.DataFrame([{**common, "code": "sh.600002", "price_1449": 7.}])
    assert len(match_candidates(a, b)[0]) == 1
    assert match_candidates(a, b, ratio_limits={"price_1449": 1.25, "amount_1449": 2.})[0].empty


def test_integer_distance_balance_is_required_even_with_large_support():
    rows = [{"half": "2024H2", "date": f"2024-08-{i+1:03d}", "day_return_difference": 0.,
             "return_last29_difference": 0., "prior20_return_difference": 0., "price_1449_ratio": 1.,
             "amount_1449_ratio": 1., "price_1449": 10.01, "control_price_1449": 9.99,
             "distance_cents": 1, "control_distance_cents": 8} for i in range(120)]
    frame = pd.DataFrame(rows)
    result = assess_round(frame, frame)[1]
    assert result["checks"]["pairs"] and result["checks"]["pair_days"]
    assert not result["checks"]["round_distance_balance"] and not result["passed"]
