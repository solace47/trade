"""The two timing arms must share one T+2 exit and existing fill rules."""

import pandas as pd

from trade_research.entry_timing import _one_trade
from trade_research.hf_outcomes import Assumptions


DATES = ["2025-09-01", "2025-09-02", "2025-09-03", "2025-09-04"]
CODE = "sh.600000"


def _quote(price: float, volume: int = 100_000) -> pd.Series:
    return pd.Series({"vwap": price, "volume": volume})


def _daily() -> dict[str, pd.Series]:
    return {
        date: pd.Series({"date": date, "tradestatus": 1,
                         "isST": 0, "preclose": prior})
        for date, prior in zip(DATES, (10.0, 10.0, 9.7, 10.0), strict=True)
    }


def test_next_morning_buy_uses_same_t_plus_two_exit() -> None:
    assumptions = Assumptions(target_notional=20_000)
    exits = {DATES[2]: _quote(10.0)}
    tail = _one_trade(CODE, DATES[0], DATES[0], 2, DATES,
                      _quote(10.0), exits, _daily(), set(), assumptions)
    morning = _one_trade(CODE, DATES[0], DATES[1], 2, DATES,
                         _quote(9.7), exits, _daily(), set(), assumptions)

    assert tail["target_exit_date"] == morning["target_exit_date"] == DATES[2]
    assert tail["exit_date"] == morning["exit_date"] == DATES[2]
    assert tail["exit_status"] == morning["exit_status"] == "filled"
    assert morning["net_return"] > tail["net_return"]


def test_unfilled_target_sells_later_and_corporate_action_rejects_raw_return() -> None:
    assumptions = Assumptions(target_notional=20_000)
    exits = {DATES[2]: _quote(10.0, volume=100), DATES[3]: _quote(10.0)}
    delayed = _one_trade(CODE, DATES[0], DATES[1], 2, DATES,
                         _quote(9.7), exits, _daily(), set(), assumptions)
    assert delayed["exit_date"] == DATES[3]
    assert delayed["exit_delay_sessions"] == 1
    assert delayed["exit_status"] == "filled"

    excluded = _one_trade(CODE, DATES[0], DATES[1], 2, DATES,
                          _quote(9.7), exits, _daily(), {DATES[1]}, assumptions)
    assert excluded["exit_status"] == "corporate_action_unadjusted"
    assert excluded["net_return"] is None
