import unittest

import numpy as np

from trade_research.tail_formula_path_excursion import native_values, ordered_excursions


class ExcursionTests(unittest.TestCase):
    def check_case(self, cents):
        self.assertEqual(len(cents), 30)
        atr = 2.5
        pairs = [(a, b) for i, a in enumerate(cents) for b in cents[i:]]
        expected = [100*max(1-b/a for a, b in pairs)/atr,
                    100*max(b/a-1 for a, b in pairs)/atr]
        prices = (np.array(cents)/100).astype('float32').astype(float)
        np.testing.assert_allclose(ordered_excursions(prices[None, :], np.array([atr])), [expected],
                                   rtol=0, atol=1e-12)
        clocks = list(range(1420, 1450))
        np.testing.assert_allclose(native_values(clocks, cents, atr), expected, rtol=0, atol=1e-12)
        perturbed = native_values([1418, 1419]+clocks+[1450, 1451],
                                 [1, 100000000]+cents+[100000000, 1], atr)
        np.testing.assert_allclose(perturbed, expected, rtol=0, atol=1e-12)
        return expected

    def test_flat_and_monotone(self):
        self.assertEqual(self.check_case([1000]*30), [0, 0])
        self.assertEqual(self.check_case(list(range(1000, 1030)))[0], 0)
        self.assertEqual(self.check_case(list(range(1030, 1000, -1)))[1], 0)

    def test_order_and_window_edges(self):
        down_first = self.check_case([2000, 1000]+[1500]*28)
        up_first = self.check_case([1000, 2000]+[1500]*28)
        self.assertGreater(down_first[0], up_first[0])
        self.assertLess(down_first[1], up_first[1])
        self.check_case([1000]*28+[2000, 500])


if __name__ == '__main__':
    unittest.main()
