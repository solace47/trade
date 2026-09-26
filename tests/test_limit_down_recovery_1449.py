import pandas as pd
import pytest

from trade_research.limit_down_recovery_1449 import limit_state, choose, match


def test_lower_limit_rounding_and_two_tick_reopening_use_integer_quotes():
    assert limit_state(9.06999969, 9.0499998, 10.05) == (True, False, 9.05)
    assert limit_state(9.06, 9.05, 10.05) == (False, False, 9.05)
    assert limit_state(9.07, 9.06, 10.05) == (False, True, 9.05)
    assert limit_state(9.07, 9.04, 10.05) == (False, False, 9.05)
    with pytest.raises(ValueError):
        limit_state(9.065, 9.05, 10.05)


def test_candidate_selection_is_independent_of_controls_and_respects_cooldown():
    calendar = pd.bdate_range('2024-01-01', periods=7).strftime('%Y-%m-%d').tolist()
    events = pd.DataFrame({'date': calendar, 'code': 'sh.600001', 'exchange': 'sh',
        'limit_distance': .005, 'day_return': -.06, 'late_return': .01,
        'prior20_return': -.05, 'position': .1, 'amount_1449': 1e8, 'price_1449': 10.})
    chosen = choose(events, calendar)
    assert chosen.date.tolist() == [calendar[0], calendar[6]]
    original = chosen.copy(deep=True)
    assert match(chosen, events.iloc[:0], calendar).empty
    pd.testing.assert_frame_equal(chosen, original)


def test_control_matching_cannot_cross_exchange_or_reuse_a_same_day_control():
    date = '2024-01-02'
    row = {'date': date, 'exchange': 'sh', 'day_return': -.06, 'late_return': .01,
           'prior20_return': -.05, 'position': .1, 'amount_1449': 1e8, 'price_1449': 10.}
    chosen = pd.DataFrame([{**row, 'code': 'a', 'daily_rank': 1},
                           {**row, 'code': 'b', 'daily_rank': 2}])
    peers = pd.DataFrame([{**row, 'code': 'c'}, {**row, 'exchange': 'sz', 'code': 'd'}])
    actual = match(chosen, peers, [date])
    assert actual[['code', 'pair_id']].values.tolist() == [['c', 'a']]
