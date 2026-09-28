import numpy as np

from trade_research.tail_formula_finer_encoding_model import encode_values


def test_same_old_bin_can_preserve_distinct_new_subdivisions():
    x = np.array([[.011, .019, .01000000001]])
    fine = encode_values(x)
    np.testing.assert_array_equal(fine, [[100011, 100019, 100010]])
    np.testing.assert_array_equal(fine // 10, [[10001, 10001, 10001]])


def test_clip_and_offset_are_unchanged_and_all_integers_fit_float32():
    x = np.array([[-1e9, -100., 0., 9899.99, 1e9]])
    fine = encode_values(x)
    coarse = np.floor(np.clip(100 * x + 10000 + .000001, 0, 999999)).astype('int32')
    np.testing.assert_array_equal(fine // 10, coarse)
    assert fine[0, 0] == 0 and fine[0, -1] == 9999990
    np.testing.assert_array_equal(fine.astype('float32').astype('int32'), fine)


def test_known_integer_cent_boundary_is_not_shifted_to_previous_coarse_bin():
    x = np.array([[.8, 1.6, -1.6, 100.01]])
    np.testing.assert_array_equal(encode_values(x), [[100800, 101600, 98400, 200010]])
