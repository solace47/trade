import numpy as np
import pandas as pd
import pytest

from trade_research.tail_formula_input_calibration import visible_score_cut


def test_unknown_future_label_remains_in_visible_calibration_population():
    f = pd.DataFrame({'date': ['2024-10-02'] * 3, 'code': ['a', 'b', 'c'],
        'formula_input_valid': [True, True, False], 'score': [0., 100., np.nan],
        'known15': [True, False, True]})
    cut, population = visible_score_cut(f, '2024-10-01', '2025-01-01')
    assert cut == 99.5 and len(population) == 3


def test_training_and_evaluation_scores_cannot_change_calibration_cut():
    f = pd.DataFrame({'date': ['2024-09-30', '2024-10-01', '2024-12-31', '2025-01-01'],
        'code': ['a'] * 4, 'formula_input_valid': [True] * 4,
        'score': [1e9, 0., 10., -1e9]})
    cut, population = visible_score_cut(f, '2024-10-01', '2025-01-01')
    assert cut == 9.95 and population.date.tolist() == ['2024-10-01', '2024-12-31']


def test_nonfinite_valid_score_is_an_error_and_not_silently_dropped():
    f = pd.DataFrame({'date': ['2024-10-02'], 'code': ['a'],
        'formula_input_valid': [True], 'score': [np.nan]})
    with pytest.raises(AssertionError):
        visible_score_cut(f, '2024-10-01', '2025-01-01')
