from decimal import Decimal, ROUND_HALF_UP

import numpy as np

from trade_research.tail_formula_minute_vwap_path import measure, NEW_EXPRESSIONS


def oracle(amount, volume, quote, atr):
    cents = lambda x: int((Decimal(str(x))*100+Decimal('.000001')).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    q = int((Decimal(str(quote))*100).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    return [2900*(q*int(v)-cents(a))/(q*sum(volume))/atr for v,a in zip(volume[::-1],amount[::-1])]


def vector(r):
    return np.array([r[k][0] for k in NEW_EXPRESSIONS])


def test_ordered_redistribution_at_fixed_total_prices_and_volume():
    v = np.full((1, 29),1000.); a = v*10
    q,atr,valid = np.array([10.]),np.array([2.]),np.array([True])
    assert np.all(vector(measure(v,a,q,atr,valid))==0)
    a[0,[0,1,28]] += [20.,-10.,-10.]
    r=measure(v,a,q,atr,valid)
    np.testing.assert_allclose(vector(r),oracle(a[0],v[0],q[0],atr[0]),atol=1e-12)
    assert vector(r)[0]>0 and vector(r)[28]<0 and abs(vector(r).sum())<1e-12
    scaled=measure(v*2,a*2,q,atr,valid)
    np.testing.assert_array_equal(vector(r),vector(scaled))
    a[0,[0,28]]=a[0,[28,0]]
    assert np.allclose(vector(measure(v,a,q,atr,valid)),vector(r)[::-1]) is False
    assert vector(measure(v,a,q,atr,valid))[0]<0


def test_zero_volume_and_half_cent_are_literal_contributions():
    v = np.full((1,29),1000.); a=v*10
    q,atr,valid = np.array([10.]),np.array([2.]),np.array([True])
    a[0,28] += .005; v[0,0]=a[0,0]=0
    r=measure(v,a,q,atr,valid)
    np.testing.assert_allclose(vector(r),oracle(a[0],v[0],q[0],atr[0]),atol=1e-12)
    assert vector(r)[0]<0 and vector(r)[28]==0
    a[0,0]=1
    assert not measure(v,a,q,atr,valid)['mu_input_valid'][0]


def test_bad_parent_and_missing_bar_do_not_become_zero_contributions():
    v = np.full((1,29),1000.); a=v*10
    q,atr,valid = np.array([10.]),np.array([2.]),np.array([True])
    assert np.isnan(vector(measure(v,a,q,atr,np.array([False])))).all()
    a[0,4]=np.nan
    r=measure(v,a,q,atr,valid)
    assert not r['mu_input_valid'][0] and np.isnan(vector(r)).all()
    v[:]=a[:]=0
    assert not measure(v,a,q,atr,valid)['mu_input_valid'][0]
