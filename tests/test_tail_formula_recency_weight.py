import numpy as np
import pandas as pd

from trade_research.tail_formula_recency_weight import date_weights


def test_date_total_does_not_depend_on_stock_count():
    dates=pd.Series(['a','b','b','b','c','c'])
    w,table=date_weights(dates)
    got=pd.DataFrame({'date':dates,'w':w}).groupby('date').w.sum()
    np.testing.assert_allclose(got,table.date_weight,rtol=0,atol=1e-15)
    np.testing.assert_allclose(w.sum(),3.,rtol=0,atol=1e-15)


def test_halving_uses_training_date_order_and_preserves_all_rows():
    dates=pd.Series(pd.bdate_range('2024-01-02',periods=127).strftime('%Y-%m-%d'))
    w,table=date_weights(dates.sample(frac=1,random_state=1).reset_index(drop=True))
    assert (w>0).all() and len(w)==127
    np.testing.assert_allclose(table.date_weight.iloc[[0,63,126]]/table.date_weight.iloc[-1],[.25,.5,1.],rtol=0,atol=1e-15)
    np.testing.assert_allclose(w.sum(),127.,rtol=0,atol=3e-14)
