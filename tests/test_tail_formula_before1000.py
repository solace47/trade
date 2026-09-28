import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from verify_tail_formula_before1000 import array_observations
from trade_research.tail_formula_boundary_evaluation import daily_summary


class BeforeTenBoundaryTests(unittest.TestCase):
    def test_triple_only_completed_at_ambiguous_last_bar_is_excluded(self):
        close = np.ones((1, 30)) * 9
        close[0, 27:] = 11
        active = np.ones((1, 30), dtype=bool)
        self.assertEqual(array_observations(close, close, active, 30)['sustained_close'][0], 11)
        self.assertEqual(array_observations(close, close, active, 29)['sustained_close'][0], 9)

    def test_three_quotes_require_positive_volume_without_bridging_gap(self):
        close = np.ones((1, 30)) * 11
        active = np.zeros((1, 30), dtype=bool)
        active[0, [25, 26, 28]] = True
        self.assertTrue(np.isnan(array_observations(close, close, active, 29)['sustained_close'][0]))
        active[0, 27] = True
        self.assertEqual(array_observations(close, close, active, 29)['sustained_close'][0], 11)

    def test_no_active_reference_has_no_fabricated_mark(self):
        close = np.ones((1, 30)) * 11
        active = np.zeros((1, 30), dtype=bool); active[0, 29] = True
        result = array_observations(close, close, active, 29)
        self.assertEqual(result['active_minutes'][0], 0)
        for name in ['price', 'max_close', 'min_low', 'sustained_close']:
            self.assertTrue(np.isnan(result[name][0]))

    def test_all_unknown_day_kept_in_full_bounds_and_missing_reference_not_loss(self):
        rows = pd.DataFrame(dict(date=['2025-01-02', '2025-01-03'], half=['2025H1'] * 2,
            code=['sh.600000'] * 2, known15=[False, True], known_no_trade=[False, False],
            opportunity15=[np.nan, 0.], one_percent15=[np.nan, 0.], any_opportunity15=[np.nan, 0.],
            mark_0959_return15=[np.nan, np.nan], adverse_return15=[np.nan, np.nan]))
        d = daily_summary(rows, 15, False)
        self.assertEqual(len(d), 2)
        self.assertTrue(np.isnan(d.rate.iloc[0]))
        self.assertEqual(d.lower.iloc[0], 0)
        self.assertEqual(d.upper.iloc[0], 1)
        self.assertEqual(d.rate.iloc[1], 0)
        self.assertTrue(d.negative_reference.isna().all())


if __name__ == '__main__':
    unittest.main()
