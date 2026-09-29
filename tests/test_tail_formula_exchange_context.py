import pandas as pd
import pytest

from trade_research.tail_formula_exchange_context import venue_flag


def test_venue_mapping_keeps_leading_zeroes_and_all_main_prefixes():
    codes = pd.Series(['sh.600000', 'sh.601001', 'sh.603001', 'sh.605001',
                       'sz.000001', 'sz.001001', 'sz.002001', 'sz.003001'])
    assert venue_flag(codes).tolist() == [1., 1., 1., 1., 0., 0., 0., 0.]


@pytest.mark.parametrize('code', ['sh.688001', 'sz.300001', 'sz.301001', 'bj.920001',
                                  'sz.600000', 'sh.000001', 'sz.001', None])
def test_unknown_or_inconsistent_identity_is_never_silently_zero(code):
    with pytest.raises(ValueError, match='do not impute zero'):
        venue_flag(pd.Series(['sh.600000', code], dtype='object'))
