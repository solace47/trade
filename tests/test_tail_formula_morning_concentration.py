"""Semantic boundaries for completed-morning volume concentration."""
import numpy as np
import pandas as pd

from trade_research.tail_formula_morning_concentration import aggregate_rows, measure


def test_uniform_one_spike_and_complete_empty_are_distinct():
    got = measure([120, 1, 0, 0], [120, 1, 0, 0], [True, True, True, False])
    np.testing.assert_allclose(got[:3], [0, 100, -1], atol=1e-12)
    assert np.isnan(got[3])


def test_known_distribution_is_invariant_to_lots_and_order():
    v = np.arange(120, dtype=float)
    expected = measure([v.sum()], [np.square(v).sum()], [True])
    for x in [v[::-1], v/100, v*10]:
        np.testing.assert_allclose(measure([x.sum()], [np.square(x).sum()], [True]), expected, atol=1e-12)


def test_shifted_day_or_duplicate_minute_cannot_pass_window_guard():
    d = pd.DataFrame(dict(date=['2024-01-02']*120, code=['sh.600000']*120,
                         timestamp=pd.date_range('2024-01-02 09:31', periods=120, freq='min'), volume=1.))
    a = aggregate_rows(d).iloc[0]
    assert a.bars == a.labels == 120 and a.good
    duplicate = d.copy(); duplicate.loc[119, 'timestamp'] = duplicate.loc[118, 'timestamp']
    assert aggregate_rows(duplicate).iloc[0].labels == 119
    shifted = d.copy(); shifted.timestamp += pd.Timedelta(days=1)
    assert not aggregate_rows(shifted).iloc[0].good


def test_bad_volume_is_unknown_even_with_positive_total():
    d = pd.DataFrame(dict(date=['2024-01-02']*120, code=['sh.600000']*120,
                         timestamp=pd.date_range('2024-01-02 09:31', periods=120, freq='min'), volume=1.))
    for bad in [-1., np.nan, np.inf]:
        x = d.copy(); x.loc[2, 'volume'] = bad
        assert not aggregate_rows(x).iloc[0].good
