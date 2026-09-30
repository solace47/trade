from decimal import Decimal, ROUND_HALF_UP

import numpy as np

from trade_research.tail_formula_minute_vwap_range import measure


def oracle(amount, volume, quote, atr):
    amount_cents = lambda x: float((Decimal(str(x)) * 100 + Decimal('.000001')).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    means = [amount_cents(a) / v for a, v in zip(amount, volume) if v > 0]
    q = float((Decimal(str(quote)) * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    return [100 * (q / max(means) - 1) / atr, 100 * (q / min(means) - 1) / atr]


def test_fixed_total_amount_price_and_volume_new_extrema():
    v = np.full((1, 29), 1000.); a = v * 10
    q, atr, valid = np.array([10.]), np.array([2.]), np.array([True])
    r = measure(v, a, q, atr, valid)
    assert r['MVH'][0] == r['MVL'][0] == 0
    a[0, :3] += [20., -10., -10.]
    assert a.sum() == 290000
    r = measure(v, a, q, atr, valid)
    np.testing.assert_allclose([r['MVH'][0], r['MVL'][0]], oracle(a[0], v[0], q[0], atr[0]), atol=1e-12)
    assert r['MVH'][0] < 0 < r['MVL'][0]
    scaled = measure(v * 2, a * 2, q, atr, valid)
    np.testing.assert_array_equal([r['MVH'], r['MVL']], [scaled['MVH'], scaled['MVL']])


def test_zero_volume_is_absent_and_half_cent_boundary_uses_amount():
    v = np.full((1, 29), 1000.); a = v * 10
    q, atr, valid = np.array([10.]), np.array([2.]), np.array([True])
    a[0, 0] += .005; v[0, 1] = a[0, 1] = 0
    r = measure(v, a, q, atr, valid)
    np.testing.assert_allclose([r['MVH'][0], r['MVL'][0]], oracle(a[0], v[0], q[0], atr[0]), atol=1e-12)
    assert r['MVH'][0] < 0 and r['MVL'][0] == 0
    a[0, 1] = 1
    r = measure(v, a, q, atr, valid)
    assert not r['mvr_input_valid'][0] and np.isnan(r['MVH'][0])


def test_quality_rejection_and_unusable_range_stay_missing():
    v = np.zeros((1, 29)); a = v.copy()
    q, atr, valid = np.array([10.]), np.array([2.]), np.array([True])
    assert not measure(v, a, q, atr, valid)['mvr_input_valid'][0]
    v[:] = 1000; a[:] = 10000
    assert not measure(v, a, q, atr, np.array([False]))['mvr_input_valid'][0]
    assert not measure(v, a, q, np.array([0.]), valid)['mvr_input_valid'][0]
    a[0, 0] = np.nan
    r = measure(v, a, q, atr, valid)
    assert not r['mvr_input_valid'][0] and np.isnan(r['MVL'][0])
