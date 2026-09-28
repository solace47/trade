import unittest

import numpy as np

from trade_research.tail_formula_chrono_rounds import choose_rounds, daily_errors, native_core
from trade_research.tail_formula_additive import predict


class ChronologicalRoundsTests(unittest.TestCase):
    def test_rounding_ties_and_zero_candidate(self):
        self.assertEqual(choose_rounds([.1, .10000000001, .10000000002]), 0)
        self.assertEqual(choose_rounds([.2, .123456789049, .12345678904]), 1)
        self.assertEqual(choose_rounds([.2, .12345678905, .12345678904]), 2)
        self.assertEqual(choose_rounds([.2, .1, .05]), 2)

    def test_dates_are_equal_despite_different_stock_counts(self):
        d = daily_errors(['a', 'a', 'a', 'b'], [0, 0, 0, 1], [0, 0, 0, 0])
        self.assertEqual(d.mse.tolist(), [0, 1])
        self.assertEqual(d.mse.mean(), .5)

    def test_zero_round_core_is_valid_constant_and_never_selects(self):
        model = dict(bias=.125, trees=[])
        scores = predict(np.zeros((3, 48)), model)
        np.testing.assert_array_equal(scores, [.125]*3)
        self.assertFalse((scores > .125).any())
        core = native_core(model, .125, {}, '')
        self.assertIn('SC:=0.125;\nCORE:SC>0.125;', core)
        self.assertNotIn('+;', core)


if __name__ == '__main__':
    unittest.main()
