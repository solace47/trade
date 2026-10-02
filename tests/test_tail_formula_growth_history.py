import numpy as np
import pandas as pd

from trade_research.tail_formula_growth_history import align


def data():
    days=pd.date_range('2024-01-01',periods=12).strftime('%Y-%m-%d').tolist()
    indices=pd.DataFrame([dict(date=day,code=code,close=float(100+step*i))
        for code,step in [('sh.000001',1),('sz.399001',2),('sz.399006',3)]
        for i,day in enumerate(days)])
    stock=pd.DataFrame({'date':[days[i] for i in [0,1,2,3,4,5,10]],'code':'sz.000001'})
    return stock,indices


def test_stock_suspension_gap_uses_six_completed_stock_sessions():
    stock,indices=data();r=align(stock,indices).iloc[-1]
    assert r.gh_date1=='2024-01-06' and r.gh_date6=='2024-01-01'
    assert r.GH_valid
    np.testing.assert_allclose(r.GH5,100*(115/100-110/100),rtol=0,atol=1e-12)


def test_current_and_later_index_values_cannot_enter_signal():
    stock,indices=data();before=align(stock,indices)
    indices.loc[indices.date.ge('2024-01-11'),'close']=1e100
    pd.testing.assert_frame_equal(before,align(stock,indices),check_exact=True)


def test_missing_or_zero_prior_index_price_is_unknown():
    stock,indices=data()
    for bad in [np.nan,0.]:
        changed=indices.copy();changed.loc[changed.code.eq('sz.399006') & changed.date.eq('2024-01-01'),'close']=bad
        assert not align(stock,changed).iloc[-1].GH_valid
