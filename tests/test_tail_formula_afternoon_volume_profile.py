import numpy as np

from trade_research.tail_formula_afternoon_volume_profile import measure,native_value
from trade_research.tail_formula_afternoon_prefix import WINDOW_CLOCKS


def test_full_volume_profile_keeps_zero_positions_and_rejects_unknown_activity():
    v=np.full(80,100.);v[12]=0;v[55]=900
    expected=measure(v[None,:])[0]
    np.testing.assert_allclose(native_value(v),expected,rtol=0,atol=1e-12)
    np.testing.assert_allclose(native_value(5*v),expected,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(native_value(v,.01),native_value(v,1000000.))
    np.testing.assert_array_equal(native_value(np.zeros(80)),np.zeros(80))
    changed=v.copy();changed[12],changed[55]=changed[55],changed[12]
    assert v.sum()==changed.sum() and not np.array_equal(native_value(v),native_value(changed))
    for bad_value in [-1.,np.inf,np.nan]:
        bad=v.copy();bad[20]=bad_value
        assert np.isnan(native_value(bad)).all()
    clocks=WINDOW_CLOCKS.copy();clocks[22]=clocks[23]
    assert np.isnan(native_value(v,window_clocks=clocks)).all()
