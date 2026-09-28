import numpy as np

from trade_research.tail_formula_week_bagging import multiplicities


def test_repeated_weeks_preserve_every_stock_and_date_with_same_weight():
    weeks = np.array(['2024-01-01', '2024-01-01', '2024-01-08', '2024-01-08', '2024-01-15'])
    drawn = ['2024-01-01', '2024-01-08', '2024-01-01']
    np.testing.assert_array_equal(multiplicities(weeks, drawn), [2, 2, 1, 1, 0])
