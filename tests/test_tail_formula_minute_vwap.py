from decimal import Decimal, ROUND_HALF_UP

import numpy as np

from trade_research.tail_formula_minute_vwap import measure


def atoms():
    c = np.full((1, 29), 10.)
    v = np.full((1, 29), 1000.)
    a = c * v
    return c, v, a, c + .02, c - .02


def decimal_oracle(c, v, a):
    cents = lambda x: int((Decimal(str(x)) * 100).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    amount_cents = lambda x: int((Decimal(str(x)) * 100 + Decimal('.000001')).quantize(Decimal('1'), rounding=ROUND_HALF_UP))
    directions = [np.sign(cents(x) * int(y) - amount_cents(z)) for x, y, z in zip(c, v, a)]
    sv = np.array(directions) * v
    return [100 * sv.sum() / v.sum(),
            100 * (sv[-4:].sum() / v[-4:].sum() - sv[:-4].sum() / v[:-4].sum())]


def test_new_information_at_fixed_total_amount_prices_and_volume():
    c, v, a, h, l = atoms(); flat = measure(c, v, a, h, l)
    a[0, :3] += [20., -10., -10.]
    assert a.sum() == 290000 and np.all(c == 10) and np.all(v == 1000)
    shifted = measure(c, v, a, h, l)
    assert flat['MV01'][0] == 0 and flat['MV02'][0] == 0
    np.testing.assert_allclose([shifted['MV01'][0], shifted['MV02'][0]], [100 / 29, -4.], atol=1e-12)
    np.testing.assert_allclose([shifted['MV01'][0], shifted['MV02'][0]], decimal_oracle(c[0], v[0], a[0]), atol=1e-12)


def test_amount_half_cent_rounding_and_terminal_direction():
    c, v, a, h, l = atoms()
    a[0, [0, 1, 26, 27, 28]] += [.0049, .005, -.0049, -.0051, .005]
    r = measure(c, v, a, h, l)
    assert r['mv_input_valid'][0]
    np.testing.assert_allclose([r['MV01'][0], r['MV02'][0]], decimal_oracle(c[0], v[0], a[0]), atol=1e-12)
    np.testing.assert_allclose([r['MV01'][0], r['MV02'][0]], [-100 / 29, 4.], atol=1e-12)


def test_zero_volume_neutral_bad_amount_and_no_tail_denominator():
    c, v, a, h, l = atoms(); v[0, 0] = a[0, 0] = 0
    r = measure(c, v, a, h, l)
    assert r['mv_input_valid'][0] and r['MV01'][0] == r['MV02'][0] == 0
    a[0, 0] = 1
    assert not measure(c, v, a, h, l)['mv_input_valid'][0]
    a[0, 0] = 0; a[0, 1] = 20000
    assert not measure(c, v, a, h, l)['mv_input_valid'][0]
    c, v, a, h, l = atoms(); v[0, -4:] = a[0, -4:] = 0
    r = measure(c, v, a, h, l)
    assert not r['mv_input_valid'][0] and np.isnan(r['MV01'][0]) and np.isnan(r['MV02'][0])
