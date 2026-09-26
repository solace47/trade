import pandas as pd
import pytest

from trade_research.cash_ex_matching import match_candidates
from trade_research.reference_gain_inputs import classify


def record(code, **changes):
    return {"date": "2024-05-06", "code": code, "half": "2024H1", "daily_rank": 1,
        "price_1449": 10., "preclose": 10.2, "reference_seed_double": 9.5,
        "reference_seed_half": 9.45, "return_last29": 0., "day_return": 10 / 10.2 - 1,
        "prior20_return": .03, "amount_1449": 2e8, "prior20_turnover_count": 20,
        "prior20_turnover_pct": 5., "reference_gap": False, "reference_valid": True,
        "n_1449": 1, "valid_1449": True, "raw_price_1449": 10., **changes}


def test_all_scenarios_must_support_the_group_and_width_limit():
    f = pd.DataFrame([record("sh.600001"),
        record("sh.600002", reference_seed_half=10.3, reference_seed_double=10.35),
        record("sh.600003", reference_seed_half=9., reference_seed_double=9.5),
        record("sh.600004", reference_seed_half=9.75, reference_seed_double=9.85),
        record("sh.600005", n_1449=2), record("sh.600006", prior20_turnover_count=19)])
    out, gains, losses = classify(f)
    assert gains.code.tolist() == ["sh.600001"]
    assert losses.code.tolist() == ["sh.600002"]
    assert len(out) == 6


def test_rank_is_worst_scenario_gain_then_code_and_keeps_all_candidates():
    f = pd.DataFrame([record(f"sh.{600001+i}", reference_seed_double=9.5-i*.01,
                            reference_seed_half=9.48-i*.01) for i in range(7)])
    _, gains, _ = classify(f)
    assert gains.code.tolist() == list(reversed(f.code.tolist()))
    assert gains.daily_rank.tolist() == list(range(1, 8))
    ties = pd.DataFrame([record("sh.600002"), record("sh.600001")])
    assert classify(ties)[1].code.tolist() == ["sh.600001", "sh.600002"]


def test_additional_turnover_caliper_changes_choice_but_defaults_are_unchanged():
    a = pd.DataFrame([record("sh.600001")])
    c = pd.DataFrame([record("sh.600002", prior20_turnover_pct=1.),
                      record("sh.600003", prior20_turnover_pct=4.)])
    assert match_candidates(a, c)[0].control_code.tolist() == ["sh.600002"]
    pairs, missed = match_candidates(a, c, extra_ratio_limits={"prior20_turnover_pct": 2.})
    assert pairs.control_code.tolist() == ["sh.600003"] and missed.empty
    assert pairs.prior20_turnover_pct_ratio.tolist() == [1.25]
    with pytest.raises(ValueError, match="distinct"):
        match_candidates(a, c, extra_ratio_limits={"amount_1449": 2.})


def test_cent_restoration_recomputes_reference_gain_and_current_return():
    out, _, _ = classify(pd.DataFrame([record("sh.600001", price_1449=10.000001)]))
    assert out.price_1449.iloc[0] == 10.
    assert out.screen_gain_low.iloc[0] == pytest.approx(.05)
    assert out.day_return.iloc[0] == pytest.approx(10/10.2-1)
