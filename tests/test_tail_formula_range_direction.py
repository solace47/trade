import numpy as np

from trade_research.tail_formula_range_direction import history_arrays, native_values


def test_rising_and_falling_ranges_keep_direction_and_range_share():
    for sign in [1, -1]:
        trend = sign*np.arange(22)*.1
        h, l, c = 20+trend, 19+trend, 19.5+trend
        valid, values, _ = history_arrays(h, l, c, np.repeat(3, 22))
        assert valid[-1] and not valid[:-1].any()
        np.testing.assert_allclose(values[-1], [100*sign, 10], rtol=0, atol=1e-12)
        np.testing.assert_allclose(native_values(h, l, c, np.arange(22)), values[-1], rtol=0, atol=1e-12)


def test_equal_expansion_and_flat_ranges_do_not_invent_direction():
    for step in [0, .1]:
        h, l = 20+np.arange(22)*step, 19-np.arange(22)*step
        c = np.repeat(19.5, 22)
        valid, values, _ = history_arrays(h, l, c, np.repeat(3, 22))
        assert valid[-1]
        np.testing.assert_array_equal(values[-1], [0, 0])
        np.testing.assert_array_equal(native_values(h, l, c, np.arange(22)), [0, 0])


def test_missing_day_is_not_skipped_and_current_day_is_excluded():
    h, l, c = np.repeat(20., 23), np.repeat(19., 23), np.repeat(19.5, 23)
    h[21] = np.nan
    valid, values, _ = history_arrays(h, l, c, np.repeat(3, 23))
    assert valid[21] and not valid[22]
    assert np.isnan(values[22]).all()
    h[21], l[21], c[21] = 1e6, .01, 2.
    np.testing.assert_array_equal(native_values(h[:22], l[:22], c[:22], np.arange(22)), [0, 0])
