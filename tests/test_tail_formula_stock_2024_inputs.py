import numpy as np
import pandas as pd

from trade_research import tail_formula_stock_2024_inputs as study


def source():
    return pd.DataFrame([dict(date='2023-01-03',code='sh.600000',A04=10.,V01=2.,formula_input_valid=True)])


def test_missing_full_morning_remains_unknown():
    agg=pd.DataFrame([dict(date='2023-01-03',code='sh.600000',bars=120,clocks=120,good_bars=120,
                          active=120,high_cents=1100.,low_cents=900.)])
    f=study.add_extrema(source(),agg)
    assert not f.formula_input_valid.iloc[0]
    assert f[['AMHD','AMLD']].isna().all().all()


def test_extrema_use_active_bars_and_keep_bad_bar_in_quality():
    times=pd.date_range('2023-01-03 09:30',periods=121,freq='min')
    b=pd.DataFrame(dict(date='2023-01-03',code='sh.600000',timestamp=times,
                        open=10.,high=10.,low=10.,close=10.,volume=100.))
    b.loc[0,['open','high','low','close','volume']]=[90.,100.,80.,90.,0.]
    a=study.pandas_aggregates(b)
    f=study.add_extrema(source(),a)
    assert f.formula_input_valid.iloc[0]
    np.testing.assert_array_equal(f[['AMHD','AMLD']].to_numpy(),[[0.,0.]])
    b.loc[1,'high']=9.
    f=study.add_extrema(source(),study.pandas_aggregates(b))
    assert not f.formula_input_valid.iloc[0]
