import unittest
import numpy as np
from trade_research import tail_formula_etf_activity as study


class ETFPrefixTest(unittest.TestCase):
    def test_windows_units_future_and_literal_adapter(self):
        v=np.arange(1,229,dtype=float)
        expected,good=study.measure(v[None,:]); self.assertTrue(good[0])
        actual,helper=study.native_value(v)
        np.testing.assert_allclose(actual,np.tile(expected[0],2),rtol=0,atol=2e-11)
        self.assertEqual(helper[2],1240101)
        np.testing.assert_array_equal(actual,study.native_value(v,outside=.01)[0])
        np.testing.assert_allclose(actual,study.native_value(v*5)[0],rtol=0,atol=2e-11)
        self.assertTrue(np.isnan(study.native_value(v,quote_date=1240102)[0]).all())
        np.testing.assert_array_equal(study.stock_value([1,2,1240101],[3,4,1240101],1240101),[1,2,3,4])
        self.assertTrue(np.isnan(study.stock_value([1,2,1240101],[3,4,1231231],1240101)).all())

    def test_zero_bad_values_missing_and_duplicate_clock(self):
        np.testing.assert_array_equal(study.native_value(np.zeros(228))[0],np.zeros(4))
        for bad in [-1,np.nan,np.inf]:
            v=np.ones(228);v[20]=bad
            self.assertTrue(np.isnan(study.native_value(v)[0]).all())
            values,good=study.measure(v[None,:]);self.assertFalse(good[0]);self.assertTrue(np.isnan(values).all())
        clocks=study.WINDOW_CLOCKS.copy();clocks[5]=clocks[4]
        self.assertTrue(np.isnan(study.native_value(np.ones(228),window_clocks=clocks)[0]).all())
        dates=np.full(228,1240101);dates[0]=1231231
        self.assertTrue(np.isnan(study.native_value(np.ones(228),window_dates=dates)[0]).all())


if __name__=='__main__':
    unittest.main()
