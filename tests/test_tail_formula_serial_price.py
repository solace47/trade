import numpy as np

from trade_research.tail_formula_serial_price import serial_moments


def score(price):
    n,a,b=serial_moments(price)
    return 100*n/np.maximum(np.sqrt(a*b),1e-12)


def test_flat_trend_and_alternating_paths():
    flat=np.full(30,10.)
    up=10*np.exp(.001*np.arange(30))
    alternating=10*np.exp(.001*(np.arange(30)%2))
    np.testing.assert_allclose(score(np.array([flat,up,alternating])),[0,100,-100],rtol=0,atol=1e-10)


def test_order_differs_with_identical_increment_set_and_net_move():
    a=np.array([.001,-.001]*14+[.001])
    b=np.array([.001]*15+[-.001]*14)
    prices=np.exp(np.c_[np.zeros(2),np.cumsum(np.array([a,b]),axis=1)])*10
    np.testing.assert_allclose(prices[0,-1],prices[1,-1],rtol=0,atol=1e-12)
    result=score(prices)
    assert result[0]<-99 and result[1]>90
