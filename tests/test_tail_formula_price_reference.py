import duckdb
import numpy as np
import pandas as pd

from audit_tail_formula_price_reference import states, sql_states


def source():
    n=50
    d=pd.DataFrame(dict(date=pd.date_range('2023-01-01',periods=n).strftime('%Y-%m-%d'),
        code=['sh.600000']*n,open=np.full(n,10.),high=np.full(n,10.1),low=np.full(n,9.9),
        close=np.full(n,10.),preclose=np.full(n,10.),volume=np.full(n,1000),
        tradestatus=np.ones(n,dtype=int),adjustflag=np.full(n,3)))
    d.loc[25:,'open']=9.;d.loc[25:,'close']=9.;d.loc[25:,'high']=9.1;d.loc[25:,'low']=8.9
    d.loc[25:,'preclose']=9.
    return d


def independently_equal(d):
    a=states(d);c=duckdb.connect();c.register('raw',d);b=sql_states(c);c.close()
    assert a.date.tolist()==b.date.tolist()
    for n in ['prior_reference_count','prior_reset_count','atr_raw','atr_reference']:
        np.testing.assert_allclose(a[n],b[n],rtol=0,atol=2e-12,equal_nan=True)
    np.testing.assert_array_equal(a.history_price_good,b.history_price_good)
    return a


def test_prior_only_and_reference_change_is_not_price_drop():
    a=independently_equal(source())
    assert a.loc[25,'prior_reset_count']==0
    assert a.loc[26,'prior_reset_count']==1
    np.testing.assert_allclose(a.loc[26,'atr_reference'],.2,atol=1e-12)
    assert a.loc[26,'atr_raw']>a.loc[26,'atr_reference']
    assert a.loc[46,'prior_reset_count']==0


def test_bad_active_row_remains_in_the_twenty_row_window():
    d=source();d.loc[24,'preclose']=np.nan
    a=independently_equal(d)
    assert a.loc[26,'prior_reference_count']==19
    assert pd.isna(a.loc[26,'prior_reset_count'])
    assert pd.isna(a.loc[26,'atr_reference'])
    assert a.loc[45,'prior_reference_count']==20


def test_suspension_has_original_active_stock_day_semantics():
    d=source();d.loc[24,'tradestatus']=0
    a=independently_equal(d).set_index('date')
    assert d.loc[24,'date'] not in a.index
    assert a.loc[d.loc[26,'date'],'prior_reset_count']==1


def test_later_prices_do_not_change_earlier_audit_states():
    d=source();a=states(d);d.loc[30:,'close']=15.;d.loc[30:,'high']=16.
    b=states(d)
    pd.testing.assert_frame_equal(a.loc[:30,['date','prior_reset_count','atr_raw','atr_reference']],
                                   b.loc[:30,['date','prior_reset_count','atr_raw','atr_reference']],check_exact=True)
