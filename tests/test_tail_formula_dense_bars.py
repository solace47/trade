"""High/low sequence recovery, invalid ranges, and fixed native time cutoff."""
import numpy as np
from trade_research.tail_formula_dense_bars import measure,native_value


def test_high_low_recoverable_in_order_and_future_bars_excluded():
    close = (1200+np.arange(29)**2)/100; high = close+.03; low = close-.07; atr = 2.3
    got = measure(high.reshape(1,-1),low.reshape(1,-1),np.array([close[-1]]),np.array([atr]))[0]
    np.testing.assert_allclose(close[-1]*(1+got[:29]*atr/100),high[::-1],rtol=0,atol=2e-12)
    np.testing.assert_allclose(close[-1]*(1+got[29:]*atr/100),low[::-1],rtol=0,atol=2e-12)
    np.testing.assert_allclose(got,native_value(high,low,close,close[-1],atr),rtol=0,atol=2e-11)
    np.testing.assert_array_equal(native_value(high,low,close,close[-1],atr),native_value(high,low,close,close[-1],atr,.01))


def test_crossed_minute_ranges_are_unknown():
    close = np.full(29,10.); high = close+.02; low = close-.02; high[7]=9.99
    assert np.isnan(native_value(high,low,close,10.,2.)).all()
