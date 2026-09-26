"""Cash before the window and aggregate volume differ from isolated fills."""

import numpy as np
import pandas as pd
import pytest

from trade_research.orderbook_funding import (
    attach_reservations, capacity_audit, funding_ledger,
)


def cash_rows():
    rows = []
    for code, buy, sell, unknown, div, tax, pay, recognized in (
            ("sh.600001", "2025-01-02", "2025-01-03", False, 20, 4,
             "2025-01-06", "2025-01-03"),
            ("sh.600002", "2025-01-03", "2025-01-06", False, np.nan, np.nan, None, None),
            ("sh.600003", "2025-01-06", None, True, np.nan, np.nan, None, None)):
        rows.append({"code": code, "date": buy, "accounting_exit_date": sell,
                     "target_notional": 20000, "horizon": 1, "candidate": "absolute_model",
                     "entry_status": "filled", "unknown_after_buy": unknown,
                     "buy_cost5": 100., "reservation5": 110.,
                     "verified_proceeds5": np.nan if unknown else 100.,
                     "dividend_gross": div, "dividend_tax_accrued": tax,
                     "dividend_pay_date": pay, "dividend_tax_recognition_date": recognized})
    return pd.DataFrame(rows)


def test_sales_and_later_dividends_cannot_pay_for_same_window_buys():
    frame, result = funding_ledger(cash_rows(), ["2025-01-02", "2025-01-03", "2025-01-06"], 5)
    assert frame.cash_change_before.tolist() == [0, -100, -104]
    assert frame.cash_change_after_buys.tolist() == [-100, -200, -204]
    assert frame.cash_change_end.tolist() == [-100, -104, -84]
    assert result["minimum_initial_actual_cash"] == 204
    assert result["minimum_initial_planned_reservation"] == 214
    assert result["unresolved_open_lots"] == 1
    assert result["unresolved_purchase_cost"] == 100
    assert result["terminal_asset_value"] is None
    assert result["complete_portfolio_return"] is None


def test_same_symbol_overlap_is_retained_and_different_books_cannot_be_mixed():
    rows = cash_rows()
    rows["code"] = "sh.600001"
    _, result = funding_ledger(rows, ["2025-01-02", "2025-01-03", "2025-01-06"], 5)
    assert result["days_with_overlapping_symbol_lots"] == 2
    rows.loc[1, "horizon"] = 5
    with pytest.raises(ValueError, match="independent funding book"):
        funding_ledger(rows, ["2025-01-02", "2025-01-03", "2025-01-06"], 5)


def test_no_open_positions_have_zero_remaining_asset_value_without_inventing_return():
    _, result = funding_ledger(cash_rows().iloc[:2],
                               ["2025-01-02", "2025-01-03", "2025-01-06"], 5)
    assert result["terminal_asset_value"] == 0
    assert result["ending_cash_change"] == 16
    assert result["complete_portfolio_return"] is None


def test_individually_feasible_orders_can_exceed_shared_window_capacity():
    events = pd.DataFrame([
        {"target_notional": amount, "horizon": 5, "candidate": "absolute_model",
         "date": "2025-01-03", "code": "sh.600001", "quantity": 60, "side": side,
         "price": 10 * (1.0005 if side == "buy" else .9995)}
        for amount, side in ((20000, "buy"), (20000, "sell"), (100000, "buy"))])
    quotes = pd.DataFrame({"date": ["2025-01-03"], "code": ["sh.600001"],
                           "volume": [1000], "vwap": [10.]})
    result = capacity_audit(events, quotes).set_index("target_notional")
    assert not result.loc[20000, "capacity_passed"]
    assert result.loc[20000, "total_participation"] == .12
    assert result.loc[100000, "capacity_passed"]
    assert result.loc[100000, "total_participation"] == .06
    events.loc[0, "price"] = 10
    with pytest.raises(ValueError, match="raw prices"):
        capacity_audit(events, quotes)


def test_failed_future_fill_still_requires_decision_time_reservation():
    rows = pd.DataFrame({
        "date": ["2025-01-02"] * 2, "code": ["sh.600001", "sh.600002"],
        "target_notional": [20000] * 2, "horizon": [5] * 2,
        "shares": [2000, 0], "entry_status": ["filled", "estimated_upper_limit"],
        "buy_cost5": [20016.2, 0], "buy_cost15": [20036.3, 0],
    })
    inputs = rows[["date", "code"]].assign(price_1449=10., preclose=10., isST=0)
    result = attach_reservations(rows, inputs)
    assert result.planned_shares.tolist() == [2000, 2000]
    assert result.reservation5.tolist() == pytest.approx([22006.82] * 2)
    assert (result.reservation15 > result.reservation5).all()
