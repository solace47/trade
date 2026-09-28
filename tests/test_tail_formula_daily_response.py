import numpy as np
from trade_research.tail_formula_daily_response import response,native_value


def test_equal_volume_and_flat_prices_have_zero_response():
    p=np.array([10+i*.01 for i in range(21)])
    np.testing.assert_allclose(response(p[None,:],np.ones((1,20)),np.ones(1)),0,atol=1e-12)
    np.testing.assert_allclose(response(np.full((1,21),10.),np.arange(1,21)[None,:],np.ones(1)),0,atol=1e-12)


def test_volume_on_positive_and_negative_changes_has_opposite_effects():
    p=np.array([10.+i%2 for i in range(21)])
    r=p[:-1]/p[1:]-1
    for up in [True,False]:
        v=np.where(r>0,100.,1.) if up else np.where(r<0,100.,1.)
        got=response(p[None,:],v[None,:],np.array([2.]))[0]
        assert (got>0)==up
        np.testing.assert_allclose([native_value(p,v,2.),native_value(p,v/100,2.)],got,rtol=0,atol=1e-11)


def test_missing_price_zero_volume_and_invalid_atr_are_unknown():
    p=np.full((3,21),10.);v=np.ones((3,20));a=np.ones(3)
    p[0,10]=np.nan;v[1]=0;a[2]=0
    assert np.isnan(response(p,v,a)).all()
