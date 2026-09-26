import numpy as np
import pandas as pd
import pytest

from trade_research.alpha158_eval import compare_rows, daily_candidates


def ledger(dates, values):
    return pd.DataFrame({'date': dates, 'code': [f'sz.{i:06d}' for i in range(len(dates))],
                         'arm': 'high', 'horizon': 5, 'tick_return15': values})


def test_common_date_comparison_keeps_unknown_slots_and_excludes_unshared_dates():
    left = ledger(['2024-07-01', '2024-07-01', '2024-07-02', '2024-07-03'],
                  [0., .02, .10, .04])
    right = ledger(['2024-07-01', '2024-07-02', '2024-07-02', '2024-07-04'],
                   [.03, .20, np.nan, .30])
    compared = compare_rows(left, right, 'tick_return15').set_index('date')
    assert compared.index.tolist() == ['2024-07-01', '2024-07-02']
    assert compared.loc['2024-07-01', 'difference'] == pytest.approx(.02)
    assert np.isnan(compared.loc['2024-07-02', 'difference'])
    assert compared.loc['2024-07-02', 'size_alpha158'] == 2
    assert compared.loc['2024-07-02', 'count_alpha158'] == 1


def test_repeated_signal_is_not_silently_overweighted():
    rows = ledger(['2024-07-01'], [.03])
    with pytest.raises(ValueError, match='Duplicate'):
        daily_candidates(pd.concat([rows, rows]), 'tick_return15')
