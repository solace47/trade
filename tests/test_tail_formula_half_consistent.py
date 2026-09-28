import unittest

import numpy as np

from trade_research.tail_formula_half_consistent import shared_direction


class HalfDirectionTests(unittest.TestCase):
    def test_positive_and_negative_steps_cannot_exceed_weaker_half(self):
        self.assertEqual(shared_direction([.6, .1]), .1)
        self.assertEqual(shared_direction([-.6, -.1]), -.1)
        self.assertEqual(shared_direction([.2, -.1]), 0)
        self.assertEqual(shared_direction([0, .1]), 0)
        self.assertEqual(shared_direction([None, .1]), 0)

    def test_update_reduces_each_half_empirical_squared_loss(self):
        rng = np.random.RandomState(20260927)
        for _ in range(200):
            residuals = [rng.normal(rng.uniform(-1, 1), .5, 20) for _ in range(2)]
            weights = [rng.uniform(.01, 1, 20) for _ in range(2)]
            means = [np.average(r, weights=w) for r, w in zip(residuals, weights)]
            value = shared_direction(means)
            for residual, weight in zip(residuals, weights):
                before = np.average(residual ** 2, weights=weight)
                after = np.average((residual - .05 * value) ** 2, weights=weight)
                self.assertLessEqual(after, before + 1e-12)


if __name__ == '__main__':
    unittest.main()
