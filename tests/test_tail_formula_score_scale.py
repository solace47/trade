import numpy as np
import pandas as pd

from trade_research.tail_formula_score_scale import standardize, native_values


def test_visible_peers_include_unknowns_and_exclude_invalid():
    f = pd.DataFrame(dict(date=['a']*4+['b']*2, code=['1','2','3','4','1','2'],
        score=[1.,3.,5.,np.nan,8.,np.nan], formula_input_valid=[True,True,True,False,True,False],
        known15=[True,False,True,True,False,True]))
    got, daily = standardize(f)
    expected = [-np.sqrt(1.5),0,np.sqrt(1.5),np.nan,0,np.nan]
    np.testing.assert_allclose(got.standardized_score, expected, equal_nan=True)
    assert daily.members.tolist() == [3,1]
    pd.testing.assert_frame_equal(got[f.columns], f)
    f.known15 = ~f.known15
    pd.testing.assert_series_equal(standardize(f)[0].standardized_score, got.standardized_score)
    native = native_values(f.score.iloc[:4], f.formula_input_valid.iloc[:4])
    np.testing.assert_allclose(native['CSZ'], expected[:4], equal_nan=True)
    assert native['CSN'] == 3


def test_translation_and_positive_scale_invariance_when_floor_inactive():
    f = pd.DataFrame(dict(date=['a']*3+['b']*3, code=['1','2','3']*2,
        score=[1.,2.,4.,102.,104.,108.], formula_input_valid=True))
    got, daily = standardize(f)
    np.testing.assert_allclose(got.standardized_score.iloc[:3], got.standardized_score.iloc[3:], atol=2e-14)
    for _, d in got.groupby('date'):
        np.testing.assert_allclose(native_values(d.score, d.formula_input_valid)['CSZ'], d.standardized_score, atol=2e-12)
        assert abs(np.mean(d.standardized_score)) < 2e-14
        np.testing.assert_allclose(np.mean(d.standardized_score**2),1,atol=2e-14)
    f.loc[f.date.eq('b'),'score'] *= -7
    np.testing.assert_allclose(standardize(f)[0].standardized_score.iloc[:3], got.standardized_score.iloc[:3])


def test_constant_and_tiny_dispersion_keep_fixed_floor_and_strict_ties():
    f = pd.DataFrame(dict(date=['a']*2+['b']*3, code=['1','2','1','2','3'],
        score=[7.,7.,-1e-8,0.,1e-8], formula_input_valid=True))
    got, _ = standardize(f)
    np.testing.assert_allclose(got.standardized_score, [0,0,-.01,0,.01], atol=2e-16)
    assert not got.standardized_score.iloc[:2].gt(0).any()
    for _, d in got.groupby('date'):
        n=native_values(d.score,d.formula_input_valid)
        np.testing.assert_allclose(n['CSZ'], d.standardized_score, atol=2e-16)
        np.testing.assert_array_equal(n['CSZ']>0, d.standardized_score.gt(0))
    empty=native_values([np.nan],[False])
    assert empty['CSN']==0 and np.isnan(empty['CSZ'][0])
