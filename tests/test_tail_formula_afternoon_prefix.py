import numpy as np

from trade_research.tail_formula_afternoon_prefix import measure,native_value,WINDOW_CLOCKS


def test_full_80_price_order_with_future_and_anchor_exclusion():
    path=np.full(80,10.);path[12]=10.04;path[55]=9.96
    expected=measure(path[None,:],np.array([2.]))[0]
    np.testing.assert_allclose(native_value(path,2.),expected,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(native_value(path,2.,.01),native_value(path,2.,1000000.))
    changed=path.copy();changed[12],changed[55]=changed[55],changed[12]
    assert path[0]==changed[0] and path[-1]==changed[-1] and path.min()==changed.min() and path.max()==changed.max()
    assert not np.array_equal(measure(changed[None,:],np.array([2.]))[0],expected)
    clocks=WINDOW_CLOCKS.copy();clocks[22]=clocks[23]
    assert np.isnan(native_value(path,2.,window_clocks=clocks)).all()
    bad=path.copy();bad[20]=0
    assert np.isnan(native_value(bad,2.)).all()
