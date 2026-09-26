import pandas as pd
import pytest

from trade_research.cash_ex_matching import assess, match_candidates


def record(code, **kwargs):
    return {"date": "2024-08-20", "half": "2024H2", "code": code, "daily_rank": 1,
            "day_return": -.01, "return_last29": 0., "prior20_return": .04,
            "price_1449": 10., "amount_1449": 200000000., **kwargs}


def test_frozen_rank_controls_greedy_matching_and_no_control_reuse():
    attempts = pd.DataFrame([record("sh.600002", daily_rank=2), record("sh.600001", daily_rank=1)])
    controls = pd.DataFrame([record("sh.600004"), record("sh.600003")])
    pairs, missed = match_candidates(attempts, controls)
    assert pairs.code.tolist() == ["sh.600001", "sh.600002"]
    assert pairs.control_code.tolist() == ["sh.600003", "sh.600004"]
    assert pairs.distance.tolist() == [0., 0.]
    assert missed.empty


def test_same_date_exchange_and_each_caliper_are_enforced():
    attempts = pd.DataFrame([record("sh.600001")])
    invalid = [record("sz.000001"), record("sh.600003", date="2024-08-21"),
               record("sh.600004", day_return=-.004), record("sh.600005", return_last29=.0021),
               record("sh.600006", prior20_return=.091), record("sh.600007", price_1449=4.99),
               record("sh.600008", amount_1449=99999999.)]
    pairs, missed = match_candidates(attempts, pd.DataFrame(invalid))
    assert pairs.empty and missed.code.tolist() == ["sh.600001"]


def test_exhausted_control_is_reported_as_an_unmatched_attempt():
    attempts = pd.DataFrame([record("sh.600001"), record("sh.600002", daily_rank=2)])
    pairs, missed = match_candidates(attempts, pd.DataFrame([record("sh.600003")]))
    assert len(pairs) == 1 and missed.code.tolist() == ["sh.600002"]
    assert len(pairs) + len(missed) == len(attempts)


def test_treatment_cannot_be_control_and_invalid_ratio_is_rejected():
    attempts = pd.DataFrame([record("sh.600001")])
    with pytest.raises(ValueError, match="also appear"):
        match_candidates(attempts, attempts)
    with pytest.raises(ValueError, match="positive"):
        match_candidates(attempts, pd.DataFrame([record("sh.600002", amount_1449=0)]))


def test_balance_cannot_override_insufficient_support():
    attempted = pd.DataFrame([record(f"sh.{600000+i}", date=f"2024-08-{i%30+1:02d}") for i in range(59)])
    pairs = pd.DataFrame([{**record(f"sh.{600000+i}", date=f"2024-08-{i%30+1:02d}"),
        "day_return_difference": 0., "return_last29_difference": 0., "prior20_return_difference": 0.,
        "price_1449_ratio": 1., "amount_1449_ratio": 1.} for i in range(46)])
    result = assess(attempted, pairs)[1]
    assert result["paired_days"] == 30
    assert result["checks"]["coverage"]
    assert not result["checks"]["pairs"]
    assert not result["passed"]
