import numpy as np
import pytest

from trade_research.tail_formula_confirmed_success import event_interval


def test_unknown_event_interval_is_preserved_without_reading_placeholder():
    lower, upper = event_interval([1, 1, 0, 0, 0], [0, 0, 1, 0, 0], [1, 0, np.nan, np.nan, 87])
    np.testing.assert_array_equal(lower, [1, 0, 0, 0, 0])
    np.testing.assert_array_equal(upper, [1, 0, 0, 1, 1])
    with pytest.raises(AssertionError):
        event_interval([1], [1], [1])
    with pytest.raises(AssertionError):
        event_interval([1], [0], [np.nan])
