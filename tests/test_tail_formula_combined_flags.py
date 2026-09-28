import pandas as pd
import pytest

from trade_research.tail_formula_context_2024 import normalized_selection_flags


def test_nullable_and_numpy_boolean_storage_preserve_exact_flags():
    a = pd.DataFrame({'code': ['a', 'b'], 'selected': pd.Series([True, False], dtype='boolean')})
    b = pd.DataFrame({'code': ['a', 'b'], 'selected': [True, False]})
    pd.testing.assert_frame_equal(normalized_selection_flags(a), b, check_exact=True)
    assert str(a.selected.dtype) == 'boolean'


@pytest.mark.parametrize('values', [pd.Series([True, None], dtype='boolean'), pd.Series([1, 0])])
def test_missing_or_numeric_flags_are_rejected(values):
    with pytest.raises(AssertionError):
        normalized_selection_flags(pd.DataFrame({'selected': values}))
