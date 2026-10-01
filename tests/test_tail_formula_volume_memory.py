import numpy as np
import pandas as pd

from trade_research.tail_formula_volume_memory import finite_states, native_states, map_states


def test_finite_reference_cancels_seed_and_volume_units():
    close = np.round(10 + np.sin(np.arange(110)/7), 2)
    volume = (1000 + 23*np.arange(110)).astype(float)
    good = np.ones(110, bool)
    actual, _, ready = finite_states(close, volume, good)
    for seed in [0., 1e6]:
        native, _, valid = native_states(close, volume, good, seed)
        np.testing.assert_array_equal(ready, valid)
        np.testing.assert_allclose(actual, native, rtol=0, atol=2e-9, equal_nan=True)
    scaled, _, _ = finite_states(close, volume/100, good)
    np.testing.assert_allclose(actual, scaled, rtol=0, atol=1e-12, equal_nan=True)
    flat, _, _ = finite_states(np.full(110, 12.34), volume, good)
    np.testing.assert_allclose(flat[ready], 12.34, rtol=0, atol=1e-12)


def test_bad_active_row_is_not_skipped_and_full_context_is_required():
    close = np.full(159, 10.)
    volume = np.ones(159)
    good = np.ones(159, bool); good[80] = False
    _, _, ready = finite_states(close, volume, good)
    assert not ready[:78].any() and ready[78:80].all()
    assert not ready[80:159].any()
    _, _, short = finite_states(close[:78], volume[:78], good[:78])
    assert not short.any()


def test_current_future_state_and_suspension_gap_do_not_change_prior_input():
    keys = pd.DataFrame({'date':['2024-01-08','2024-01-09'], 'code':['sh.600001']*2})
    states = pd.DataFrame({'history_date':['2024-01-05','2024-01-08','2024-01-10'],
                           'code':['sh.600001']*3,'anchor':[10.,20.,30.],
                           'history_input_valid':[True]*3})
    actual = map_states(keys, states)
    assert actual.anchor.tolist() == [10.,20.]
    polluted = states.copy(); polluted.loc[2,'anchor'] = 1e9
    pd.testing.assert_frame_equal(actual, map_states(keys,polluted), check_exact=True)
