import unittest

import numpy as np

from trade_research.tail_formula_max_run import longest_runs, native_run_values


class MaxRunTests(unittest.TestCase):
    def check_case(self, changes, expected, zero_volume=()):
        self.assertEqual(len(changes), 29)
        cents = np.r_[1000, 1000 + np.cumsum(changes)]
        volume = np.ones(29)
        volume[list(zero_volume)] = 0
        # Source float32 price storage must preserve the same cent directions.
        prices = (cents / 100).astype('float32').astype(float)
        np.testing.assert_array_equal(longest_runs(prices[None, :], volume[None, :]), [expected])
        self.assertEqual(native_run_values(cents, volume), list(expected))

    def test_window_edges_and_flat_reset(self):
        self.check_case([0] * 29, [0, 0])
        self.check_case([1] + [0] * 28, [1, 0])
        self.check_case([0] * 28 + [-1], [0, 1])
        self.check_case([1, 1, 0, 1, 1, 1] + [0] * 23, [3, 0])

    def test_full_runs_and_volume_break(self):
        self.check_case([1] * 29, [29, 0])
        self.check_case([-1] * 29, [0, 29])
        self.check_case([1] * 29, [14, 0], [14])
        self.check_case([-1] * 29, [0, 28], [0])
        self.check_case([1] * 29, [0, 0], range(29))

    def test_opposite_reset_and_order(self):
        self.check_case([1, -1] * 14 + [1], [1, 1])
        self.check_case([1] * 10 + [-1] * 12 + [1] * 7, [10, 12])
        self.check_case([1] * 10 + [-1] * 12 + [1] * 7, [10, 6], [15])


if __name__ == '__main__':
    unittest.main()
