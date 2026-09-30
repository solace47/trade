"""Independent least-squares checks for the new price-path inputs."""
import numpy as np

from trade_research.tail_formula_daily_regression import WINDOWS, regression, replay_helper


def test_daily_regression_matches_lstsq():
    rng=np.random.default_rng(20260930)
    bank=rng.integers(100,30000,(11,60))
    for n in WINDOWS:
        got=regression(bank,n)
        for i,row in enumerate(bank):
            x=row[:n][::-1].astype(float)
            design=np.column_stack([np.ones(n),np.arange(n)])
            coef=np.linalg.lstsq(design,x,rcond=None)[0]
            fitted=design@coef
            expected=[100*coef[1]/x[-1],100*(1-np.square(x-fitted).sum()/np.square(x-x.mean()).sum()),
                      100*(x[-1]-fitted[-1])/x[-1]]
            np.testing.assert_allclose(got[i],expected,rtol=0,atol=2e-9)


def test_flat_and_linear_histories():
    flat=np.full((1,60),1273)
    straight=np.arange(1273,1213,-1).reshape(1,-1)
    for n in WINDOWS:
        np.testing.assert_array_equal(regression(flat,n),[[0,0,0]])
        np.testing.assert_allclose(regression(straight,n),[[100/1273,100,0]],rtol=0,atol=1e-12)


def test_native_helper_does_not_use_current_or_future_price():
    closes=(1200+np.arange(60)**2)/100
    first=replay_helper(np.r_[closes,1.])[60]
    second=replay_helper(np.r_[closes,99999.,.01,1000000.])[60]
    np.testing.assert_array_equal(first,second)
    expected=np.concatenate([regression(np.rint(closes[::-1]*100).reshape(1,-1),n)[0] for n in WINDOWS])
    np.testing.assert_allclose(first,expected,rtol=0,atol=2e-11)
