import numpy as np
import pandas as pd
import pytest

from trade_research.tail_formula_quarter_position import position, trading_minute_projection


def test_flat_history_has_cent_denominator_and_breakouts_are_not_clamped():
    first, second = position(np.array([10., 10.01, 15.]), np.array([2., 2., 2.]),
                             np.array([60000., 60000., 60000.]),
                             np.array([1000., 1000., 900.]), np.array([1000., 1000., 1100.]))
    np.testing.assert_allclose(first, [0., .05, 25.], atol=1e-12)
    np.testing.assert_allclose(second, [0., 100., 300.], atol=1e-10)


def test_only_known_zero_quantity_halt_padding_can_be_projected_out():
    raw = pd.DataFrame({'timestamp':pd.to_datetime(['2025-01-02 15:00','2025-01-03 15:00','2025-01-06 14:49']),
                        'close':[10.,10.,10.1], 'volume':[100.,0.,100.], 'turnover':[1000.,0.,1010.]})
    daily = pd.DataFrame({'date':['2025-01-02','2025-01-03'], 'close':[10.,10.], 'tradestatus':[1,0]})
    projected, removed = trading_minute_projection(raw, daily, '2025-01-06')
    assert len(projected) == 2 and removed[0]['date'] == '2025-01-03'
    traded = raw.copy(); traded.loc[1,'volume'] = 1
    with pytest.raises(AssertionError, match='Traded records'):
        trading_minute_projection(traded, daily, '2025-01-06')
    with pytest.raises(AssertionError, match='Missing active-day'):
        trading_minute_projection(raw.iloc[1:], daily, '2025-01-06')
    with pytest.raises(AssertionError, match='Unexplained extra'):
        trading_minute_projection(raw, daily.iloc[:1], '2025-01-06')
    missing = raw.copy(); missing.loc[1,'volume'] = np.nan
    with pytest.raises(AssertionError, match='Missing halt values'):
        trading_minute_projection(missing, daily, '2025-01-06')
