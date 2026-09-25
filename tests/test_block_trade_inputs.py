"""Protect the source timing, discount boundary, and unexposed controls."""

import pandas as pd
import pytest

from trade_research.block_trade_inputs import (
    CONTROL, EVENT, _discount_events, _match,
)
from trade_research.block_trade_source import _validate_trade


def test_exchange_rows_accept_display_rounding_but_reject_wrong_date() -> None:
    fund = {"tradedate": "2024-01-02", "stockid": "512170",
            "tradeprice": "0.39", "tradeqty": "600", "tradeamount": "231.6"}
    _validate_trade(fund, "2024-01-02", "sse")
    with pytest.raises(ValueError, match="code/date"):
        _validate_trade(fund, "2024-01-03", "sse")
    wrong = {**fund, "tradeamount": "300"}
    with pytest.raises(ValueError, match="value"):
        _validate_trade(wrong, "2024-01-02", "sse")


def test_discount_uses_only_published_trade_date_and_strict_low() -> None:
    calendar = pd.bdate_range("2024-06-03", periods=20).strftime("%Y-%m-%d").tolist()
    trades = pd.DataFrame([
        {"trade_date": "2024-06-03", "code": "sh.600001", "price": 9.0,
         "amount_wan": 700., "low": 10.},
        {"trade_date": "2024-06-03", "code": "sh.600001", "price": 9.5,
         "amount_wan": 400., "low": 10.},
        {"trade_date": "2024-06-03", "code": "sh.600002", "price": 9.9,
         "amount_wan": 3_000., "low": 10.},
    ])
    events = _discount_events(trades, calendar)
    assert events[["date", "code"]].values.tolist() == [
        ["2024-06-04", "sh.600001"]]
    assert events.qualifying_amount_wan.iloc[0] == 1_100
    assert events.discount_depth.iloc[0] > .01


def test_discount_never_moves_a_december_event_into_next_year() -> None:
    calendar = pd.bdate_range("2024-12-02", "2025-01-31").strftime(
        "%Y-%m-%d").tolist()
    trades = pd.DataFrame([
        {"trade_date": "2024-12-02", "code": "sh.600001", "price": 9.,
         "amount_wan": 1_100., "low": 10.},
        {"trade_date": "2024-12-31", "code": "sh.600002", "price": 9.,
         "amount_wan": 1_100., "low": 10.},
    ])
    events = _discount_events(trades, calendar)
    assert events.code.tolist() == ["sh.600001"]


def test_match_excludes_all_block_trades_and_keeps_same_industry() -> None:
    day = "2024-06-04"
    source = "2024-06-03"
    base = {"date": day, "trade_date": source, "board": "sh_main",
            "industry": "C制造业", "avg20_amount": 60_000_000.,
            "float_mv": 2_000_000_000., "return20_prior_adjusted": .02,
            "return_1450": .01}
    selected = pd.DataFrame([{**base, "code": "sh.600001",
                              "discount_depth": .08}])
    universe = pd.DataFrame([
        {**base, "code": "sh.600002"},
        {**base, "code": "sh.600003", "return_1450": .011},
        {**base, "code": "sh.600004", "industry": "J金融业"},
    ])
    blocked = pd.DataFrame([{"trade_date": source, "code": "sh.600002"}])
    pairs, unmatched = _match(selected, universe, blocked)
    assert unmatched == {"no_same_industry_board": 0,
                         "outside_input_limits": 0}
    assert pairs.loc[pairs.candidate.eq(EVENT), "code"].tolist() == ["sh.600001"]
    assert pairs.loc[pairs.candidate.eq(CONTROL), "code"].tolist() == ["sh.600003"]


def test_match_reports_a_signal_day_with_no_unexposed_stock() -> None:
    day = "2024-06-04"
    source = "2024-06-03"
    base = {"date": day, "trade_date": source, "board": "sh_main",
            "industry": "C制造业", "avg20_amount": 60_000_000.,
            "float_mv": 2_000_000_000., "return20_prior_adjusted": .02,
            "return_1450": .01}
    selected = pd.DataFrame([
        {**base, "code": "sh.600001"},
        {**base, "date": "2024-06-05", "trade_date": day,
         "code": "sh.600004"},
        {**base, "date": "2024-06-06", "trade_date": "2024-06-05",
         "code": "sh.600005"},
    ])
    universe = pd.DataFrame([
        {**base, "code": "sh.600003"},
        {**base, "date": "2024-06-06", "trade_date": "2024-06-05",
         "code": "sh.600006"},
    ])
    blocked = pd.DataFrame([{"trade_date": source, "code": "sh.600003"}])
    pairs, unmatched = _match(selected, universe, blocked)
    assert unmatched == {"no_same_industry_board": 2,
                         "outside_input_limits": 0}
    assert pairs.loc[pairs.candidate.eq(EVENT), "code"].tolist() == ["sh.600005"]
