"""Exact decision quotes and unconstrained execution VWAPs are distinct inputs."""

import pytest

from trade_research.hf_outcomes import _order_shares
from trade_research.quote_precision import fixed_quote_shares, quote_cents


@pytest.mark.parametrize("quote,budget,expected", [
    (.8, 20000, 25000), (1.6, 20000, 12500), (1.600000023841858, 100000, 62500),
    (6.25, 20000, 3200), (200, 20000, 100), (200.01, 20000, 0),
])
def test_cent_reconstruction_and_integer_share_floor(quote, budget, expected):
    assert fixed_quote_shares("sh.600001", quote, budget) == expected


def test_binary_float_floor_is_not_fixed_by_display_rounding():
    assert _order_shares("sz.000001", .8, 20000) == 24900
    assert _order_shares("sz.000001", 1.6, 20000) == 12400
    assert fixed_quote_shares("sz.000001", round(1.600000023841858, 2), 20000) == 12500


@pytest.mark.parametrize("quote", [0, -1, float("nan"), float("inf"), 1.2345, .00001])
def test_invalid_or_non_cent_vwap_cannot_be_silently_quantized(quote):
    with pytest.raises(ValueError):
        quote_cents(quote)


def test_star_market_starts_at_two_hundred_but_allows_single_share_increments():
    assert fixed_quote_shares("sh.688001", 100, 19900) == 0
    assert fixed_quote_shares("sh.688001", 100, 20100) == 201
    assert fixed_quote_shares("sh.600001", 100, 20100) == 200


@pytest.mark.parametrize("budget", [0, -1, float("nan"), float("inf"), 1.111])
def test_invalid_budget_is_not_rounded_into_valid_money(budget):
    with pytest.raises(ValueError):
        fixed_quote_shares("sh.600001", 1, budget)
