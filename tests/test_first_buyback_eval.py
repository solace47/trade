import pandas as pd
import pytest

from trade_research.buyback_inputs import CONTROL
from trade_research.first_buyback_eval import _check_pairs
from trade_research.first_buyback_inputs import EVENT


def test_first_trade_cannot_enter_on_notice_day_or_lose_its_peer() -> None:
    pairs = pd.DataFrame([
        ("2024-02-02", "sh.600001", EVENT, "sh.600001", "2024-02-01", True),
        ("2024-02-02", "sh.600002", CONTROL, "sh.600001", None, True),
    ], columns=["date", "code", "candidate", "pair_code",
                "notice_date", "recent_plan_10"])
    expected = {"matched_pairs": 1, "matched_days": 1,
                "by_year": {"2024": {"pairs": 1, "days": 1},
                            "2025": {"pairs": 0, "days": 0}}}
    _check_pairs(pairs, expected)
    same_day = pairs.copy()
    same_day.loc[0, "notice_date"] = "2024-02-02"
    with pytest.raises(ValueError, match="timing"):
        _check_pairs(same_day, expected)
    inconsistent = pairs.copy()
    inconsistent.loc[1, "recent_plan_10"] = False
    with pytest.raises(ValueError, match="peer"):
        _check_pairs(inconsistent, expected)
