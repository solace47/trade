import hashlib

import numpy as np
import pandas as pd
import pytest

from trade_research.positive_pool_1449 import SALT, choose_pool


def test_pool_membership_uses_sign_and_ranking_does_not_use_score_magnitude():
    date = '2024-07-01'
    rows = pd.DataFrame({'date': [date] * 9, 'code': list('abcdefghi'),
                         'score': [1., 2., 3., 4., 5., 6., 7., 0., -1.]})
    expected = sorted(list('abcdefg'), key=lambda code: hashlib.sha256((SALT + date + code).encode()).hexdigest())[:5]
    chosen = choose_pool(rows, [date])
    assert chosen.code.tolist() == expected
    rows.loc[rows.score.gt(0), 'score'] = [700., 600., 500., 400., 300., 200., 100.]
    assert choose_pool(rows.sample(frac=1, random_state=17), [date]).code.tolist() == expected


def test_pool_cooldown_uses_market_sessions():
    dates = pd.bdate_range('2024-07-01', periods=7).strftime('%Y-%m-%d').tolist()
    rows = pd.DataFrame({'date': dates, 'code': 'a', 'score': .1})
    assert choose_pool(rows, dates).date.tolist() == [dates[0], dates[6]]


def test_duplicate_and_nonfinite_predictions_do_not_silently_enter_pool():
    row = pd.DataFrame({'date': ['2024-07-01'], 'code': ['a'], 'score': [np.inf]})
    with pytest.raises(ValueError, match='unique finite'):
        choose_pool(row, ['2024-07-01'])
    row['score'] = .1
    with pytest.raises(ValueError, match='unique finite'):
        choose_pool(pd.concat([row, row]), ['2024-07-01'])
