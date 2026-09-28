import numpy as np

from trade_research.tail_formula_session_volatility import measure, native_value


def test_same_endpoints_do_not_imply_same_opening_path():
    flat = np.full(30, 10.)
    choppy = 10 + .1*np.sin(np.linspace(0, 2*np.pi, 30))
    tail = np.linspace(10., 10.3, 30)
    a = measure(flat[None, :], tail[None, :], [1.])
    b = measure(choppy[None, :], tail[None, :], [1.])
    assert choppy[0] == choppy[-1] == 10.
    assert a['SV01'][0] == b['SV01'][0]
    assert a['SV02'][0] == 0 < b['SV02'][0]
    assert a['SV03'][0] == 1 > b['SV03'][0]


def test_flat_and_missing_windows_are_distinct_and_units_cancel():
    flat = np.full((1, 30), 10.)
    result = measure(flat, flat, [2.])
    assert result['valid'][0]
    assert all(result[name][0] == 0 for name in ['SV01', 'SV02', 'SV03'])
    missing = flat.copy(); missing[0, 3] = np.nan
    result = measure(missing, flat, [2.])
    assert not result['valid'][0] and np.isnan(result['SV02'][0])
    first = np.linspace(10, 11, 30)[None, :]
    second = np.linspace(10, 10.2, 30)[None, :]
    a, b = measure(first, second, [2.]), measure(first*100, second*.1, [2.])
    for name in ['SV01', 'SV02', 'SV03']:
        np.testing.assert_allclose(a[name], b[name], rtol=0, atol=2e-10)


def test_native_anchors_exclude_bridge_and_future_prices():
    morning = np.round(10+.02*np.sin(np.arange(30)), 2)
    tail = np.round(11+np.arange(30)*.01, 2)
    clocks = np.r_[np.arange(931, 960), 1000, np.arange(1420, 1450)]
    prices = np.r_[morning, tail]
    reference = measure(morning[None, :], tail[None, :], [2.])
    result = native_value(prices, clocks, 2.)
    after = native_value(np.r_[prices, 1., 100000.], np.r_[clocks, 1450, 1451], 2.)
    for name in ['SV01', 'SV02', 'SV03']:
        np.testing.assert_allclose(result[name], reference[name][0], atol=2e-10)
        assert result[name] == after[name]
    prices[clocks > 1000] = 100000.
    assert native_value(prices, clocks, 2.)['SV02'] == result['SV02']
