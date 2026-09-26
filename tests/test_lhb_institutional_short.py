import pandas as pd
import pytest

from trade_research.lhb_institutional_short import collapse_daily, institution_cents, money_cents, select
from trade_research.hf_outcomes import Assumptions, EXIT_WINDOWS, outcomes_for_symbol
from trade_research.reference_gain_eval import account_with_windows, quality_windows


def test_exact_cents_do_not_turn_a_flat_net_into_an_institution_signal():
    row = {"branchNameB": "机构专用,机构专用", "branchTxAmtB": "0.1,0.2",
           "branchNameS": "机构专用", "branchTxAmtS": "0.30"}
    assert institution_cents(row, "B")-institution_cents(row, "S") == 0
    for value in ("-1", "NaN", "Infinity", "1.001"):
        with pytest.raises(ValueError):
            money_cents(value)


def test_two_daily_reasons_are_one_signal_only_when_source_facts_agree():
    base = {"date": "2024-01-03", "trade_date": "2024-01-02", "code": "sh.600000",
        "disclosed_turnover_cents": 10000, "institution_buy_cents": 1000, "institution_sell_cents": 300}
    rows = pd.DataFrame([{**base, "ref_type": "11"}, {**base, "ref_type": "14"}])
    events, ambiguous = collapse_daily(rows)
    assert len(events) == 1 and ambiguous.empty
    assert events.iloc[0].institution_net_cents == 700
    assert events.iloc[0].net_fraction == .07
    rows.loc[1, "institution_sell_cents"] = 200
    events, ambiguous = collapse_daily(rows)
    assert events.empty and len(ambiguous) == 1


def test_fraction_ranking_keeps_unmatched_signals_and_uses_market_day_cooldown():
    days = pd.bdate_range("2024-01-02", periods=8).strftime("%Y-%m-%d").tolist()
    rows = [{"date": d, "code": "a", "net_fraction": .1} for d in days]
    rows += [{"date": days[0], "code": c, "net_fraction": .2} for c in "bcdefg"]
    selected = select(pd.DataFrame(rows), days)
    assert selected.loc[selected.date.eq(days[0]), "code"].tolist() == list("bcdef")
    assert selected.loc[selected.code.eq("a"), "date"].tolist() == [days[1], days[7]]
    pd.testing.assert_frame_equal(selected, select(pd.DataFrame(rows).iloc[::-1], days))


def test_t1_and_t3_calendar_targets_share_entry_and_delay_a_blocked_sale():
    days = ["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"]
    signal = pd.DataFrame([{"date": days[0], "code": "sh.600000", "isST": 0,
        "reference_gap": False, "price_1449": 10., "arm": "high", "pair_id": "a"}])
    bars = []
    for i, day in enumerate(days):
        price = 9. if i == 3 else 10.+i*.01
        for label in EXIT_WINDOWS["close"]:
            bars.append({"date": day, "code": "sh.600000", "label": label,
                "timestamp": pd.Timestamp(day+" "+label[:2]+":"+label[2:]),
                "open": price, "high": price, "low": price, "close": price,
                "volume": 100000., "turnover": 100000.*price})
    minute = pd.DataFrame(bars)
    daily = pd.DataFrame([{"date": day, "tradestatus": 1, "isST": 0,
                          "preclose": 10., "close": 10.} for day in days])
    raw = outcomes_for_symbol(signal, minute, daily, days, Assumptions(target_notional=20000),
        horizons=(1, 3), sizing_price_column="price_1449")
    raw["target_notional"] = 20000
    raw["quality_clean_exit"] = raw.exit_status.eq("filled")
    rows = account_with_windows(raw, signal, quality_windows(minute, days), days).set_index("horizon")
    assert rows.loc[1, "entry_price"] == rows.loc[3, "entry_price"]
    assert rows.loc[1, "shares"] == rows.loc[3, "shares"] == 2000
    assert rows.loc[1, "exit_date"] == days[1]
    assert rows.loc[3, "target_exit_date"] == days[3]
    assert rows.loc[3, "exit_date"] == days[4] and rows.loc[3, "exit_delay_sessions"] == 1
