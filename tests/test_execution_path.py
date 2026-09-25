"""Check the fixed minute participation execution comparison."""

import pandas as pd

from trade_research.execution_path import minute_participation


def _bars(prices: list[float], volumes: list[int]) -> pd.DataFrame:
    return pd.DataFrame({
        "volume": volumes,
        "turnover": [price * volume for price, volume in zip(prices, volumes)],
    })


def test_participation_fills_earlier_minutes_at_their_own_vwap():
    filled, price = minute_participation(
        _bars([100, 101, 102, 103], [2000] * 4),
        "sh.600000", "2025-01-02", 100, 400, 5,
    )
    assert filled == 400
    assert abs(price - 100.5 * 1.0005) < 1e-10


def test_participation_respects_lots_and_price_limit():
    filled, price = minute_participation(
        _bars([109.9, 109.96, 109.9, 109.96], [1900, 3000, 1900, 3000]),
        "sh.600000", "2025-01-02", 100, 400, 5,
    )
    # 10% of 1,900 shares rounds down to one 100-share child order.
    # The 109.96 bars become unbuyable after five basis points of slippage.
    assert filled == 200
    assert abs(price - 109.9 * 1.0005) < 1e-10


def test_star_child_orders_use_conservative_200_share_lots():
    filled, _ = minute_participation(
        _bars([100] * 4, [1900, 2000, 2100, 0]),
        "sh.688001", "2025-01-02", 100, 400, 5,
    )
    assert filled == 400
