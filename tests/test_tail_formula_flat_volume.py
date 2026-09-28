import numpy as np

from trade_research.tail_formula_flat_volume import measure,native_value


def test_equal_prices_differ_from_offsetting_up_and_down_moves():
    fixed=np.full(30,10.)
    alternating=np.array([10.+.01*(i%2) for i in range(30)])
    np.testing.assert_array_equal(measure(np.array([fixed,alternating]),np.ones((2,29)))['FV01'],[100.,0.])


def test_volume_weight_boundary_units_and_cent_equality():
    prices=np.r_[10.,np.repeat(10.01,29)]
    volume=np.r_[100.,np.ones(28)]
    expected=100*28/128
    r=measure(prices[None,:],volume[None,:]);assert r['valid'][0]
    np.testing.assert_allclose(r['FV01'],expected,rtol=0,atol=1e-12)
    for outside in [.01,1000000.]:
        np.testing.assert_allclose(native_value(prices,np.r_[999.,volume/100],outside),expected,rtol=0,atol=1e-12)
    np.testing.assert_allclose(measure((prices+1e-10)[None,:],volume[None,:])['FV01'],expected,rtol=0,atol=1e-12)
    volume[1:]=0
    assert measure(prices[None,:],volume[None,:])['FV01'][0]==0


def test_missing_noncent_and_inactive_windows_are_unknown():
    prices=np.full((4,30),10.);volume=np.ones((4,29))
    prices[0,12]=np.nan;prices[1,12]=10.005;volume[2,12]=np.nan;volume[3]=0
    r=measure(prices,volume)
    assert not r['valid'].any() and np.isnan(r['FV01']).all()
