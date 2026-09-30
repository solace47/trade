import numpy as np

from trade_research.tail_formula_index_morning import measure


def test_direction_and_reference_level_are_distinct_and_scale_invariant():
    first,end,reference=np.array([1000.,1100.]),np.array([1100.,1000.]),np.array([1050.,900.])
    r=measure(first,end,reference,np.array([True,True]))
    np.testing.assert_allclose(r['IM01'],[10.,-100/11],atol=1e-12)
    np.testing.assert_allclose(r['IM02'],[100/21,100/9],atol=1e-12)
    scaled=measure(first*100,end*100,reference*100,np.array([True,True]))
    np.testing.assert_array_equal(r['IM01'],scaled['IM01'])
    np.testing.assert_array_equal(r['IM02'],scaled['IM02'])


def test_unchanged_index_can_differ_from_the_stock_day_reference():
    r=measure(np.array([1000.]),np.array([1000.]),np.array([900.]),np.array([True]))
    assert r['IM01'][0]==0 and r['IM02'][0]>0
    # An older active-stock-day reference is retained, not substituted with today's opening point.
    flat=measure(np.array([1000.]),np.array([1000.]),np.array([1000.]),np.array([True]))
    assert flat['IM02'][0]==0


def test_missing_prefix_or_reference_remains_unknown():
    r=measure(np.array([1000.,1000.,0.]),np.array([1100.,np.nan,1100.]),
        np.array([900.,900.,900.]),np.array([False,True,True]))
    assert not r['index_morning_input_valid'].any()
    assert np.isnan(r['IM01']).all() and np.isnan(r['IM02']).all()
