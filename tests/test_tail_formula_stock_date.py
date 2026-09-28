import numpy as np
import pandas as pd

from trade_research.tail_formula_stock_date import projection, least_squares_target


def frame(y):
    return pd.DataFrame(dict(date=['a','a','b','b'],code=['x','y','x','y'],opportunity15=y))


def test_remove_additive_stock_and_date_effects():
    for y in [[0,1,0,1],[0,0,1,1],[1,1,1,1]]:
        got=projection(frame(y))
        np.testing.assert_allclose(got.target,0,atol=2e-14)
        np.testing.assert_allclose(least_squares_target(got)[0],got.target,atol=2e-14)
    got=projection(frame([0,1,0,1]))
    np.testing.assert_array_equal(got.date_only_target,[-.5,.5,-.5,.5])


def test_interaction_survives_the_two_additive_projections():
    got=projection(frame([1,0,0,1]))
    np.testing.assert_allclose(got.target,[.5,-.5,-.5,.5],atol=2e-14)
    np.testing.assert_allclose(least_squares_target(got)[0],got.target,atol=2e-14)


def test_unbalanced_panel_matches_dense_weighted_least_squares_and_preserves_singletons():
    f=pd.DataFrame(dict(date=['a','a','a','b','b','c','c','c','c'],
        code=['x','y','z','x','z','x','y','z','singleton'],opportunity15=[1,0,1,0,0,0,1,0,1]))
    got=projection(f)
    x=pd.concat([pd.get_dummies(f.date,dtype=float),pd.get_dummies(f.code,dtype=float)],axis=1).to_numpy()
    w=np.sqrt(got.w.to_numpy())
    coef=np.linalg.lstsq(x*w[:,None],f.opportunity15.to_numpy()*w,rcond=None)[0]
    expected=f.opportunity15-x@coef
    np.testing.assert_allclose(got.target,expected,atol=2e-12)
    np.testing.assert_allclose(least_squares_target(got)[0],expected,atol=2e-12)
    assert len(got)==len(f) and abs(got.loc[got.code.eq('singleton'),'target'].iloc[0])<2e-14
    shuffled=projection(f.sample(frac=1,random_state=1)).sort_values(['date','code'])
    np.testing.assert_allclose(shuffled.target,got.sort_values(['date','code']).target,atol=2e-12)
