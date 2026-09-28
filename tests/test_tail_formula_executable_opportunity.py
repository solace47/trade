import unittest

import numpy as np

from trade_research.tail_formula_executable_opportunity import utility


class ExecutableOpportunityTests(unittest.TestCase):
    def test_only_confirmed_nonentry_becomes_zero(self):
        got = utility([True, True, False, False], [False, False, True, False], [1., 0., np.nan, np.nan])
        np.testing.assert_allclose(got, [1., 0., 0., np.nan], rtol=0, atol=0, equal_nan=True)

    def test_conflicting_states_and_unknown_known_event_rejected(self):
        with self.assertRaises(AssertionError):
            utility([True], [True], [1.])
        with self.assertRaises(AssertionError):
            utility([True], [False], [np.nan])


if __name__ == '__main__':
    unittest.main()
