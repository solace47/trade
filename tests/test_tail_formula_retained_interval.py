import numpy as np
import pandas as pd

from trade_research import tail_formula_retained_interval as study


def window():
    h=np.full((1,29),10.02);l=np.full((1,29),10.00);v=np.full((1,29),100.)
    return h,l,v


def test_remaining_interval_ignores_filled_regions_and_equal_endpoints():
    h,l,v=window();h[:,:2]=9.97;l[:,:2]=9.95
    expected=np.array([[3.,0.]])
    np.testing.assert_array_equal(study.measure(h,l,v),expected)
    np.testing.assert_array_equal(study.native_values(h,l,v),expected)
    l[:,-1]=9.97
    np.testing.assert_array_equal(study.measure(h,l,v),[[0.,0.]])


def test_inactive_minutes_cannot_form_or_fill_interval_and_units_cancel():
    h,l,v=window();h[:,:2]=9.97;l[:,:2]=9.95
    l[:,-1]=9.95;v[:,-1]=0
    np.testing.assert_array_equal(study.measure(h,l,v),[[3.,0.]])
    np.testing.assert_array_equal(study.native_values(h,l,v/100),[[3.,0.]])
    v[:,1]=0
    np.testing.assert_array_equal(study.measure(h,l,v),[[0.,0.]])
    np.testing.assert_array_equal(study.native_values(h,l,v),[[0.,0.]])


def test_down_interval_is_mirror_and_does_not_sum_overlapping_regions():
    h,l,v=window();h[:,:2]=10.07;l[:,:2]=10.05
    np.testing.assert_array_equal(study.measure(h,l,v),[[0.,3.]])
    np.testing.assert_array_equal(study.native_values(h,l,v),[[0.,3.]])


def test_bad_or_missing_window_is_unknown_even_with_no_pattern():
    h,l,v=window()
    raw={'date':['2024-01-02'],'code':['sh.600000'],'mp_bars':[29],'mp_clocks':[29],'mp_good_bars':[29]}
    for k,values in [('h',h),('l',l),('c',(h+l)/2),('v',v)]:
        raw.update({n:values[:,i] for i,n in enumerate(study.FIELDS[k])})
    old=pd.DataFrame({'date':['2024-01-02'],'code':['sh.600000'],'formula_input_valid':[True],'A04':[10.01],'V01':[3.]})
    source=pd.DataFrame(raw)
    out=study.transform(source,old)
    assert out.interval_source_valid.iloc[0]
    assert out.RGUP.iloc[0]==out.RGDOWN.iloc[0]==0
    source.loc[0,'mp_v21']=-1
    out=study.transform(source,old)
    assert not out.interval_source_valid.iloc[0]
    assert out[['RGUP','RGDOWN']].isna().all().all()
