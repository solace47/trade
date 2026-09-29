import numpy as np
import pytest

from trade_research.tail_formula_atr_target import scale_target


def test_relative_reference_is_divided_after_shared_market_mean():
    reference = np.array([2., -1., .5]); atr = np.array([4., 1., 2.])
    excess = reference - reference.mean()
    result = scale_target(excess, atr)
    np.testing.assert_array_equal(result, [.375, -1.5, 0.])
    # Normalizing first would also change the common market baseline.
    before = reference/atr - np.mean(reference/atr)
    assert not np.allclose(result, before)
    assert result.mean() != 0
    np.testing.assert_array_equal(scale_target(100*excess, 100*atr), result)


@pytest.mark.parametrize('atr', [[1., 0.], [1., -1.], [1., np.nan]])
def test_no_floor_or_silent_reference_imputation(atr):
    with pytest.raises(AssertionError):
        scale_target([1., -1.], atr)
    with pytest.raises(AssertionError):
        scale_target([1., np.nan], [1., 1.])
