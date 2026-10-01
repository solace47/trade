import pandas as pd
import duckdb

from trade_research.tail_formula_external_pattern import classify
from run_tail_formula_external_pattern import summarize


def test_source_avoid_priority_over_later_buy_conditions():
    d = pd.DataFrame(dict(last_close=[10.4], first_close=[10.], vwap=[10.],
        tail_first=[10.], tail_volume=[10000.], half_volume=[15000.], tail_above=[20],
        morning_high=[10.2], afternoon_high=[10.4], day_high=[10.4], day_low=[10.],
        tail_high=[10.4]))
    branch, buy = classify(d)
    assert branch.tolist() == [1] and buy.tolist() == [False]


def test_decimal_flat_prefix_is_not_above_its_equal_weighted_price():
    clocks = [x.strftime('%H%M') for a, b in [('09:30', '11:30'), ('13:01', '14:49')]
        for x in pd.date_range('2024-01-02 '+a, '2024-01-02 '+b, freq='min')]
    raw = pd.DataFrame(dict(date=['2024-01-02']*230, code=['sh.600000']*230, clock=clocks,
        open=[10.1]*230, high=[10.1]*230, low=[10.1]*230, close=[10.1]*230,
        volume=[101]*230, turnover=[1020.1]*230))
    con = duckdb.connect()
    d = summarize(raw, con)
    con.close()
    assert d.tail_above.tolist() == [0]
    assert d.branch.tolist() == [3] and d.buy.tolist() == [False]
