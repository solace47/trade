"""Independent chronological least squares and native cutoff checks."""
import numpy as np

from trade_research.tail_formula_tail_regression import WINDOWS,regression,replay_native


def test_tail_regression_matches_independent_lstsq():
    rng = np.random.default_rng(20260930)
    rows = rng.integers(100,30000,(11,30)); atr = np.linspace(.5,9.,len(rows))
    for n in WINDOWS:
        got = regression(rows,atr,n)
        for i,row in enumerate(rows):
            x = row[:n][::-1].astype(float)
            design = np.column_stack([np.ones(n),np.arange(n)])
            coef = np.linalg.lstsq(design,x,rcond=None)[0]; fitted = design@coef
            expected = [100*(n-1)*coef[1]/(x[-1]*atr[i]),
                100*(1-np.square(x-fitted).sum()/np.square(x-x.mean()).sum()),
                100*(x[-1]-fitted[-1])/(x[-1]*atr[i])]
            np.testing.assert_allclose(got[i],expected,rtol=0,atol=2e-9)


def test_flat_linear_and_tail_jump_have_distinct_residuals():
    flat = np.full((1,30),1273); straight = np.arange(1273,1243,-1).reshape(1,-1)
    jump = flat.copy(); jump[0,0] += 10
    for n in WINDOWS:
        np.testing.assert_array_equal(regression(flat,np.array([2.]),n),[[0,0,0]])
        np.testing.assert_allclose(regression(straight,np.array([2.]),n),[[100*(n-1)/(1273*2),100,0]],rtol=0,atol=1e-12)
        assert regression(jump,np.array([2.]),n)[0,2] > 0


def test_native_capture_excludes_future_and_earlier_prices():
    prices = (1200+np.arange(30)**2)/100
    got = replay_native(prices,2.3)
    np.testing.assert_array_equal(got,replay_native(prices,2.3,.01))
    expected = np.concatenate([regression(np.floor(prices[::-1]*100+.5).reshape(1,-1),np.array([2.3]),n)[0] for n in WINDOWS])
    np.testing.assert_allclose(got,expected,rtol=0,atol=2e-11)
