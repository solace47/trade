import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research.tail_formula_chrono48 import bootstrap_lower, choose, counts, summarize


def policy():
    return json.loads(Path('config/tail_formula_chrono48_2024_protocol.json').read_text())


def rows(states):
    return pd.DataFrame([
        dict(date=day, code=str(i), known15=known, known_no_trade=no_trade,
             opportunity15=opportunity, sustained_return15=space,
             adverse_return15=bad, mark_1000_return15=mark)
        for i, (day, known, no_trade, opportunity, space, bad, mark) in enumerate(states)
    ])


def test_unknown_only_day_survives_with_worst_case_risk():
    f = rows([('2024-07-01', True, False, 1., .02, -.005, .01),
              ('2024-07-02', False, False, np.nan, np.nan, np.nan, np.nan)])
    d = counts(f)
    assert len(d) == 2 and d.loc['2024-07-02', 'rows'] == 1
    assert np.isnan(d.loc['2024-07-02', 'rate'])
    assert d.loc['2024-07-02', ['lower', 'upper', 'one_lower', 'bad_upper']].tolist() == [0., 1., 0., 1.]
    d['delta'] = d.rate - .5; d['lower_delta'] = d.lower - .5
    s = summarize(d, policy())
    assert s['conditional_gap_days'] == 1 and not s['gates']['conditional_complete']
    assert s['lower'] == .5 and s['bad_upper'] == .5 and not s['eligible']


def test_no_trade_is_not_unknown_and_does_not_invent_a_mark():
    d = counts(rows([('2024-07-01', False, True, np.nan, np.nan, np.nan, np.nan)]))
    assert d.iloc[0][['rows', 'known', 'unknown', 'no_trade']].tolist() == [1, 0, 0, 1]
    assert d.iloc[0][['lower', 'upper', 'bad_upper']].tolist() == [0., 0., 0.]
    assert np.isnan(d.iloc[0].mean1000)


def test_known_opportunity_with_missing_price_marks_keeps_risk_unknown():
    d = counts(rows([('2024-07-01', True, False, 0., np.nan, np.nan, np.nan)]))
    assert d.iloc[0].known == 1 and d.iloc[0].rate == 0
    assert d.iloc[0].one_unknown == d.iloc[0].bad_unknown == 1
    assert d.iloc[0].bad_upper == 1 and np.isnan(d.iloc[0].mean1000)


def test_single_week_cannot_supply_uncertainty_support():
    d = pd.DataFrame({'lower_delta': [.1, .2]}, index=['2024-07-01', '2024-07-02'])
    assert bootstrap_lower(d, policy()) is None


def test_fixed_choice_rejects_ineligible_and_breaks_exact_ties_by_id():
    good = dict(eligible=True, lower=.65, delta=.1, known=400)
    candidates = [dict(good, id=3), dict(good, id=1), dict(good, id=0, eligible=False, lower=.99)]
    assert choose(candidates)['id'] == 1
    assert choose([candidates[2]]) is None
