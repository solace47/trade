import pandas as pd
import pytest

from trade_research.sector_resilience_1449 import sector_statistics, choose, match


def test_own_return_cannot_change_own_sector_or_market_benchmark():
    data = pd.DataFrame({'date': ['2024-01-02'] * 3, 'code': ['a', 'b', 'c'],
        'industry': ['X', 'X', 'Y'], 'return_last29': [.05, .005, -.02],
        'return20_prior_adjusted': [.5, .05, -.2]})
    first = sector_statistics(data)
    assert first.loc[0, 'sector_residual_tail'] == pytest.approx(.0125)
    assert first.loc[0, 'sector_prior20'] == pytest.approx(.05)
    data.loc[0, ['return_last29', 'return20_prior_adjusted']] = [-.05, -.5]
    second = sector_statistics(data)
    for key in ['sector_tail', 'market_tail', 'sector_prior20', 'sector_residual_tail']:
        assert first.loc[0, key] == pytest.approx(second.loc[0, key])


def test_sector_membership_duplicates_fail():
    data = pd.DataFrame({'date': ['2024-01-02'] * 2, 'code': ['a', 'a']})
    with pytest.raises(ValueError, match='Overlapping'):
        sector_statistics(data)


def test_candidate_selection_keeps_unmatched_and_respects_cooldown():
    calendar = pd.bdate_range('2024-01-01', periods=7).strftime('%Y-%m-%d').tolist()
    strong = pd.DataFrame({'date': calendar, 'code': ['a'] * 7,
        'return20_prior_adjusted': [0.] * 7, 'return_to_1420': [0.] * 7,
        'return_last29': [-.005] * 7, 'sector_prior20': [0.] * 7,
        'amount_1449': [1e8] * 7, 'price_1449': [10.] * 7, 'position_1449': [.4] * 7})
    chosen = choose(strong, calendar)
    assert chosen.date.tolist() == [calendar[0], calendar[6]]
    before = chosen.copy(deep=True)
    controls = match(chosen, strong.iloc[:0], calendar)
    assert controls.empty
    pd.testing.assert_frame_equal(chosen, before)
