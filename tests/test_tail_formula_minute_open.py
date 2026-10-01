import numpy as np
from trade_research.tail_formula_minute_open import measure,native_values


def test_same_HLCV_different_opens_has_new_information_but_same_total_change():
    close=np.full((1,30),10.);volume=np.arange(1,31,dtype=float)[None,:]*100
    a=close.copy();b=close.copy();b[0,-1]=10.1
    first,_=measure(a,close,volume,[2.]);second,_=measure(b,close,volume,[2.])
    np.testing.assert_allclose(first,0)
    assert second[0,0]>0 and second[0,1]<0
    np.testing.assert_allclose(second.sum(axis=1),0,atol=1e-12)
    np.testing.assert_allclose(second,native_values(b,close,volume,np.array([2.])),atol=1e-12)


def test_inactive_open_is_ignored_no_contiguous_pairs_is_known_zero():
    close=np.full((1,30),10.);opened=np.full((1,30),np.nan)
    volume=np.tile([100.,0.],15)[None,:]
    values,pairs=measure(opened,close,volume,[2.])
    assert not pairs.any()
    np.testing.assert_array_equal(values,[[0.,0.]])
    np.testing.assert_array_equal(native_values(opened,close,volume,np.array([2.])),values)


def test_active_bad_open_stays_unknown_and_quantity_units_cancel():
    close=np.full((1,30),10.);opened=close.copy();volume=np.arange(1,31,dtype=float)[None,:]
    opened[0,-1]=10.1
    values,_=measure(opened,close,volume,[2.]);scaled,_=measure(opened,close,volume*100,[2.])
    np.testing.assert_allclose(values,scaled,atol=1e-12)
    opened[0,-1]=np.nan;bad,_=measure(opened,close,volume,[2.])
    assert np.isnan(bad).all()


def test_bad_active_open_outside_range_is_not_a_zero_signal():
    import pandas as pd
    from trade_research.tail_formula_minute_open import transform,FIELDS
    old=pd.DataFrame(dict(date=['2024-01-02'],code=['sh.600000'],formula_input_valid=[True],A04=[10.],V01=[2.],NA05=[0.]))
    columns={'date':old.date,'code':old.code,'mo_bars':[30],'mo_clocks':[30],'mo_good_bars':[30]}
    for k,value in [('o',10.),('h',10.1),('l',9.9),('c',10.),('v',100.)]:
        columns.update({n:[value] for n in FIELDS[k]})
    raw=pd.DataFrame(columns);raw['mo_o49']=10.2
    out=transform(raw,old)
    assert not out.minute_open_valid.item()
    assert np.isnan(out.MOGAP.item()) and np.isnan(out.MOBODY.item())
