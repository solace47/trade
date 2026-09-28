import numpy as np
import pandas as pd

from trade_research.tail_formula_date_quantile import date_quantile


def test_unequal_date_sizes_do_not_change_date_weight():
    f = pd.DataFrame(dict(date=['a']+['b']*9,code=['0']+[str(i) for i in range(9)],score=[10.]+[0.]*9))
    cut,curve = date_quantile(f,.75)
    assert cut == 10 and np.quantile(f.score,.75) == 0
    np.testing.assert_allclose(curve.weight,[1,1])


def test_pooled_ties_left_inverse_and_strict_cut():
    f = pd.DataFrame(dict(date=['a','a','b','b'],code=['0','1','0','1'],score=[0.,2.,2.,4.]))
    cut,curve = date_quantile(f,.75)
    assert cut == 2 and f.score.gt(cut).sum() == 1
    np.testing.assert_allclose(curve.cumulative_weight,[.5,1.5,2.])
    assert date_quantile(f.sample(frac=1,random_state=7),.75)[0] == cut
    assert date_quantile(f,1)[0] == 4
