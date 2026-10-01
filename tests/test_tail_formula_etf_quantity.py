import unittest
import numpy as np
from trade_research import tail_formula_baseline as timing
from trade_research import tail_formula_etf_quantity as study


def fixture(days=28, mixed_lengths=False):
    times=[];dates=[];volumes=[]
    for day in range(1,days+1):
        clocks=([930] if not mixed_lengths or day%2 else [])+timing.WINDOW_CLOCKS+list(range(1449,1460))+[1500]
        times.extend(clocks);dates.extend([1240100+day]*len(clocks))
        volumes.extend([float(day) if clock in timing.WINDOW_CLOCKS else 1e6 for clock in clocks])
    return np.asarray(times),np.asarray(dates),np.asarray(volumes)


class QuantityHistoryTest(unittest.TestCase):
    def test_relative_history_and_variable_day_lengths(self):
        for mixed in [False,True]:
            t,d,v=fixture(mixed_lengths=mixed);r=study.native_series(t,d,v)
            selected=(t==1448)&(d>=1240121)
            n=d[selected]-1240100
            expected=100*10.5/(2*n-10.5)
            np.testing.assert_allclose(r['QQ'][selected],expected,rtol=0,atol=2e-11)
            self.assertTrue(np.isnan(r['QQ'][(t==1448)&(d<1240121)]).all())
            np.testing.assert_array_equal(r['QD'][selected],d[selected])
        t,d,v=fixture(days=20)
        self.assertTrue(np.isnan(study.native_series(t,d,v)['QQ']).all())

    def test_future_units_zero_bad_history_and_clock(self):
        t,d,v=fixture();base=study.native_series(t,d,v)['QQ']
        changed=v.copy();changed[(t>=1449)|(t==930)]=.01
        np.testing.assert_allclose(base,study.native_series(t,d,changed)['QQ'],rtol=0,atol=0,equal_nan=True)
        np.testing.assert_allclose(base,study.native_series(t,d,v*5)['QQ'],rtol=0,atol=2e-11,equal_nan=True)
        zero=study.native_series(t,d,np.zeros(len(v)))['QQ']
        np.testing.assert_array_equal(zero[(t==1448)&(d>=1240121)],0)
        for bad in [-1,np.nan,np.inf]:
            changed=v.copy();changed[(d==1240105)&(t==1059)]=bad
            result=study.native_series(t,d,changed)['QQ']
            self.assertTrue(np.isnan(result[(t==1448)&(d==1240125)]).all())
            self.assertTrue(np.isfinite(result[(t==1448)&(d==1240126)]).all())
        clocks=t.copy();clocks[(d==1240105)&(t==1059)]=1058
        self.assertTrue(np.isnan(study.native_series(clocks,d,v)['QQ'][(t==1448)&(d==1240125)]).all())

    def test_two_securities_stock_date_guard(self):
        d=np.array([1240101,1240102])
        result=study.stock_values([np.array([1.,2.]),d],[np.array([3.,4.]),np.array([1240101,1240101])],d)
        np.testing.assert_array_equal(result[0],[1.,3.])
        self.assertTrue(np.isnan(result[1]).all())


if __name__=='__main__':
    unittest.main()
