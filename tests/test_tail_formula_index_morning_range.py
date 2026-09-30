import numpy as np

from trade_research.tail_formula_index_morning_range import measure, quote_range


def rows():
    return [dict(sequence=i,price_raw=100000) for i in range(240)]


def test_complete_morning_extrema_differ_from_identical_early_points():
    a=rows();b=rows()
    a[60]['price_raw']=110000;a[90]['price_raw']=90000
    b[60]['price_raw']=105000;b[90]['price_raw']=95000
    pa,pb=quote_range(a),quote_range(b)
    for p in [pa,pb]: assert p['ir_prefix_valid'] and p['ir_last']==1000
    result=measure(np.array([1000.,1000.]),np.array([pa['ir_high'],pb['ir_high']]),
        np.array([pa['ir_low'],pb['ir_low']]),np.array([True,True]))
    assert result['IMH'][0]<result['IMH'][1] and result['IML'][0]>result['IML'][1]
    scaled=measure(np.array([100000.,100000.]),np.array([110000.,105000.]),
        np.array([90000.,95000.]),np.array([True,True]))
    np.testing.assert_array_equal(result['IMH'],scaled['IMH'])
    np.testing.assert_array_equal(result['IML'],scaled['IML'])


def test_120th_quote_is_included_afternoon_and_closing_suffix_do_not_set_extrema():
    a=rows();a[119]['price_raw']=110000;a[120]['price_raw']=80000
    a[227]['price_raw']=105000;a[228]['price_raw']=np.nan;a[239]['price_raw']=9999999
    p=quote_range(a)
    assert p==dict(ir_prefix_valid=True,ir_high=1100.,ir_low=1000.,ir_last=1050.)
    r=measure(np.array([1050.]),np.array([1100.]),np.array([1000.]),np.array([True]))
    assert r['IMH'][0]<0 and r['IML'][0]>0


def test_missing_bad_sequence_and_nonpositive_quote_remain_unknown():
    for change in ['missing','sequence','nonpositive','nan']:
        a=rows()
        if change=='missing': a=a[:227]
        elif change=='sequence': a[60]['sequence']=61
        elif change=='nonpositive': a[60]['price_raw']=0
        else: a[60]['price_raw']=np.nan
        assert not quote_range(a)['ir_prefix_valid']
    r=measure(np.array([1000.,1000.,1000.]),np.array([1100.,900.,1100.]),
        np.array([1000.,1000.,0.]),np.array([False,True,True]))
    assert not r['index_morning_range_valid'].any()
    assert np.isnan(r['IMH']).all() and np.isnan(r['IML']).all()
