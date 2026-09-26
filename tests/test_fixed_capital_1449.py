import numpy as np
import pandas as pd
import pytest

from trade_research.fixed_capital_1449 import cash_book


def order(date, code, exit_date, *, rank=1, reserve=100., buy=100., sell=100.,
          status='filled', ex=None, pay=None, dividend=0., tax=0., unknown=False):
    return {'date': date, 'code': code, 'order_id': date + ':' + code, 'daily_rank': rank,
        'reservation15': reserve, 'cash_buy15': buy, 'cash_sell15': sell,
        'entry_status': status, 'exit_date': exit_date, 'dividOperateDate': ex, 'dividPayDate': pay,
        'catalog_dividend_gross': dividend, 'catalog_dividend_tax': tax,
        'known_return5': np.nan if unknown else 0., 'execution_source_valid': not unknown}


def test_same_window_sales_cannot_fund_decisions_but_previous_sales_can():
    days = ['2024-07-01', '2024-07-02', '2024-07-03', '2024-07-04']
    rows = pd.DataFrame([order(days[0], 'a', days[1]), order(days[1], 'b', days[2]),
                         order(days[2], 'c', days[3], reserve=110., buy=110.)])
    ledger, decisions, result = cash_book(rows, days, 15, 150.)
    assert decisions.authorized.tolist() == [True, False, True]
    assert ledger.cash_after_buys.tolist() == [50., 50., 40., 40.]
    assert result['ending_cash'] == 140.
    assert result['bought'] == result['sold'] == 2


def test_unfilled_instruction_reservation_is_not_recycled_inside_the_window():
    days = ['2024-07-01', '2024-07-02']
    first = order(days[0], 'a', days[1], reserve=150., status='volume_cap')
    second = order(days[0], 'b', days[1], rank=2)
    rows = pd.DataFrame([first, second])
    ledger, decisions, result = cash_book(rows, days, 15, 200.)
    assert decisions.authorized.tolist() == [True, False]
    assert result['entry_rejected'] == 1 and result['bought'] == 0
    assert ledger.cash_end.tolist() == [200., 200.]
    rows.loc[0, 'entry_status'] = 'filled'
    _, changed, _ = cash_book(rows, days, 15, 200.)
    pd.testing.assert_frame_equal(changed, decisions)


def test_ex_date_tax_reserve_and_payment_day_cash_are_not_spendable_early():
    days = ['2024-07-01', '2024-07-02', '2024-07-03', '2024-07-04']
    first = order(days[0], 'a', days[3], ex=days[1], pay=days[2], dividend=20., tax=4., sell=120.)
    second = order(days[1], 'b', days[2], reserve=4., buy=4., sell=4.)
    third = order(days[2], 'c', days[3], reserve=4., buy=4., sell=4.)
    ledger, decisions, result = cash_book(pd.DataFrame([first, second, third]), days, 15, 104.)
    assert decisions.authorized.tolist() == [True, False, False]
    assert ledger.tax_reserved_before.tolist() == [0., 4., 4., 4.]
    assert ledger.dividend_received.tolist() == [0., 0., 20., 0.]
    assert ledger.dividend_tax_paid.tolist() == [0., 0., 0., 4.]
    assert result['ending_cash'] == 140.


def test_sold_position_keeps_unpaid_dividend_until_payment_even_on_weekend():
    days = pd.date_range('2024-07-04', '2024-07-07').strftime('%Y-%m-%d').tolist()
    rows = pd.DataFrame([order(days[0], 'a', days[1], ex=days[1], pay=days[2], dividend=20., tax=4.)])
    ledger, _, result = cash_book(rows, days, 15, 110.)
    assert ledger.open_lots_end.tolist() == [1, 0, 0, 0]
    assert ledger.unpaid_dividend_gross.tolist() == [0., 20., 0., 0.]
    assert ledger.cash_end.tolist() == [10., 106., 126., 126.]
    assert result['conditional_profit'] == 16.


def test_unknown_exit_is_kept_in_inventory_and_has_no_complete_return():
    days = ['2024-07-01', '2024-07-02']
    rows = pd.DataFrame([order(days[0], 'a', None, sell=np.nan, unknown=True)])
    ledger, _, result = cash_book(rows, days, 15, 150.)
    assert ledger.open_lots_end.tolist() == [1, 1]
    assert result['ending_cash'] == 50.
    assert result['remaining_purchase_cost'] == 100.
    assert not result['complete_conditional_settlement']
    assert result['conditional_capital_return'] is None
    assert result['actual_unknown_bought'] == 1


def test_cash_shortage_skips_whole_order_without_resizing():
    days = ['2024-07-01', '2024-07-02']
    rows = pd.DataFrame([order(days[0], 'a', days[1], reserve=110.),
                         order(days[0], 'b', days[1], rank=2, reserve=90., buy=90., sell=90.)])
    _, decisions, result = cash_book(rows, days, 15, 100.)
    assert decisions.authorized.tolist() == [False, True]
    assert result['bought'] == 1 and result['capital_rejected'] == 1
    assert result['conditional_capital_return'] == pytest.approx(0.)
