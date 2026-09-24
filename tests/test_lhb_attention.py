"""Check exact-day, unique matching in the frozen attention comparison."""

from __future__ import annotations

import pandas as pd

from trade_research.lhb_attention import match_controls


def test_matching_never_uses_a_listed_or_other_day_control() -> None:
    rows = [
        {"date": "2024-06-04", "code": "sh.600001", "any_list": 1,
         "open_gap": .01, "return_1450": .02, "amount_1450": 100e6,
         "t_turn": 10.0, "return20_prior_adjusted": .1, "t_amount": 100e6},
        {"date": "2024-06-04", "code": "sh.600002", "any_list": None,
         "open_gap": .01, "return_1450": .02, "amount_1450": 100e6,
         "t_turn": 10.0, "return20_prior_adjusted": .1, "t_amount": 100e6},
        {"date": "2024-06-05", "code": "sh.600003", "any_list": None,
         "open_gap": .01, "return_1450": .02, "amount_1450": 100e6,
         "t_turn": 10.0, "return20_prior_adjusted": .1, "t_amount": 100e6},
    ]
    pool = pd.DataFrame(rows)
    chosen, controls, unmatched = match_controls(pool.iloc[[0]], pool)
    assert len(chosen) == len(controls) == 1
    assert controls.code.tolist() == ["sh.600002"]
    assert unmatched == 0
