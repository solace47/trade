import pandas as pd
from trade_research.first_limitup_overnight import prior_pattern,select


def test_first_board_uses_exact_rounding_and_excludes_continuation():
    assert prior_pattern(11,10,10,10)==(True,False,True)
    assert prior_pattern(12.10,11,11,10)==(False,False,True)
    assert prior_pattern(10.70,10,10,10)==(False,True,True)
    assert prior_pattern(10.99,10,10,10)==(False,False,True)
    assert prior_pattern(5.25,4.77,4.77,4.77)==(True,False,True)
    assert prior_pattern(10.701,10,10,10)==(False,False,False)


def test_liquid_first_order_and_calendar_cooldown_do_not_use_next_prices():
    days=pd.bdate_range('2025-01-02',periods=8).strftime('%Y-%m-%d').tolist()
    rows=[{'date':d,'code':'a','amount_1449':200} for d in days]
    rows+=[{'date':days[0],'code':code,'amount_1449':100} for code in 'bcdefg']
    selected=select(pd.DataFrame(rows),days)
    assert selected.loc[selected.date.eq(days[0]),'code'].tolist()==list('abcde')
    assert selected.loc[selected.code.eq('a'),'date'].tolist()==[days[0],days[6]]
    pd.testing.assert_frame_equal(selected,select(pd.DataFrame(rows).iloc[::-1],days))
