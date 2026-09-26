import json
import re

import numpy as np
import pandas as pd
import pytest

from trade_research.alpha158_asof import DEFINITION, Evaluator, asof_banks, features_for_symbol


def example():
    i = np.arange(75)
    dates = pd.bdate_range('2022-01-03', periods=len(i)).strftime('%Y-%m-%d')
    factor = np.where(i < 45, 1., .5)
    adjusted = 10 + .03 * i + .2 * np.sin(i)
    close = adjusted * factor
    opening = (adjusted - .07) * factor
    volume = (10000 + (i % 9) * 700) / factor
    daily = pd.DataFrame({'date': dates, 'tradestatus': 1, 'close': close, 'open': opening,
        'high': close + .1 * factor, 'low': opening - .1 * factor,
        'preclose': np.r_[adjusted[0], adjusted[:-1]] * factor, 'volume': volume,
        'amount': volume * (close + opening) / 2})
    rows = []
    for j in (25, 30, 46, 62, 74):
        row = daily.iloc[j]
        partial = row.close - .04 * factor[j]
        v = row.volume * .9
        rows.append({'date': row.date, 'price_1449': partial, 'high_1449': row.high - .02 * factor[j],
            'low_1449': row.low + .02 * factor[j], 'volume_1449': v,
            'amount_1449': v * (partial + row.open) / 2})
    return daily, pd.DataFrame(rows)


def scalar_reference(banks, length, row):
    """Separate pandas rolling implementation of all frozen expressions."""
    size = min(length, 61)
    scope = {name: pd.Series(value[row, :size][::-1]) for name, value in banks.items()}

    def regression(values, kind):
        t = np.arange(1, len(values) + 1)
        valid = np.isfinite(values)
        if valid.sum() < 2:
            return np.nan
        design = np.column_stack([np.ones(valid.sum()), t[valid]])
        coef = np.linalg.lstsq(design, values[valid], rcond=None)[0]
        if kind == 'Slope':
            return coef[1]
        if kind == 'Resi':
            return values[-1] - coef @ [1, t[-1]]
        if np.isclose(np.nanstd(values, ddof=1), 0, atol=2e-5):
            return np.nan
        prediction = design @ coef
        return 1 - np.sum((values[valid] - prediction) ** 2) / np.sum((values[valid] - values[valid].mean()) ** 2)

    def corr(a, b, n):
        result = a.rolling(n, min_periods=1).corr(b)
        flat = np.isclose(a.rolling(n, min_periods=1).std(), 0, atol=2e-5) | np.isclose(b.rolling(n, min_periods=1).std(), 0, atol=2e-5)
        return result.mask(flat)

    scope.update({'Ref': lambda x, n: x.shift(n), 'Abs': np.abs, 'Log': np.log,
        'Greater': np.maximum, 'Less': np.minimum, 'Corr': corr,
        'Rank': lambda x, n: x.rolling(n, min_periods=1).rank(pct=True),
        'Quantile': lambda x, n, q: x.rolling(n, min_periods=1).quantile(q),
        'IdxMax': lambda x, n: x.rolling(n, min_periods=1).apply(lambda y: y.argmax() + 1, raw=True),
        'IdxMin': lambda x, n: x.rolling(n, min_periods=1).apply(lambda y: y.argmin() + 1, raw=True)})
    for name, method in [('Mean', 'mean'), ('Std', 'std'), ('Sum', 'sum'), ('Max', 'max'), ('Min', 'min')]:
        scope[name] = lambda x, n, method=method: getattr(x.rolling(n, min_periods=1), method)()
    for name in ['Slope', 'Rsquare', 'Resi']:
        scope[name] = lambda x, n, name=name: x.rolling(n, min_periods=1).apply(lambda y: regression(y, name), raw=True)
    values = {}
    for item in json.loads(DEFINITION.read_text())['fields']:
        # This oracle uses only the reviewed, checked-in expression manifest.
        expression = re.sub(r'\$(\w+)', r'\1', item['expression'])
        result = eval(expression, {'__builtins__': {}}, scope)
        values[item['name']] = result.iloc[-1]
    return values


def test_every_factor_matches_independent_pandas_partial_and_full_windows():
    daily, prefix = example()
    banks, lengths = asof_banks(daily, prefix)
    actual = Evaluator(banks, lengths).frame()
    for i, length in enumerate(lengths):
        expected = scalar_reference(banks, length, i)
        np.testing.assert_allclose(actual.loc[i].to_numpy(), np.array(list(expected.values())),
                                   atol=2e-10, rtol=1e-9, equal_nan=True)


def test_current_final_and_future_daily_values_cannot_enter_current_features():
    daily, prefix = example()
    now = prefix.iloc[:1]
    expected = features_for_symbol(daily, now)
    daily.loc[daily.date.eq(now.date.iloc[0]), ['high', 'low', 'close', 'volume', 'amount']] *= 37
    daily.loc[daily.date.gt(now.date.iloc[0]), ['open', 'high', 'low', 'close', 'preclose', 'volume', 'amount']] *= 91
    pd.testing.assert_frame_equal(features_for_symbol(daily, now), expected)


def test_past_completed_close_not_previous_partial_close():
    daily, prefix = example()
    result = features_for_symbol(daily, prefix)
    expected = daily.close.iloc[25] / prefix.price_1449.iloc[1]
    assert result.ROC5.iloc[1] == pytest.approx(expected)
    assert result.ROC5.iloc[1] != pytest.approx(prefix.price_1449.iloc[0] / prefix.price_1449.iloc[1])


def test_reference_split_keeps_price_volume_units_consistent():
    daily, prefix = example()
    banks, _ = asof_banks(daily, prefix)
    i = 2  # first decision after the synthetic two-for-one split
    assert banks['close'][i, 0] == pytest.approx(prefix.price_1449.iloc[i] * 2)
    assert banks['volume'][i, 0] == pytest.approx(prefix.volume_1449.iloc[i] / 2)
    assert banks['vwap'][i, 0] * banks['volume'][i, 0] == pytest.approx(prefix.amount_1449.iloc[i])
