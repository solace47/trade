import pandas as pd

from trade_research.risk_removal_1449 import (
    causal_exclusions, decision_buyable, document_fields, first_entries, watch_schedule,
)


def test_document_operative_dates_exclude_historical_warning_dates():
    text = ('公司自2023年5月5日起被实施退市风险警示。'
            '公司自2022年7月2日开市起撤销退市风险警示。'
            '撤销退市风险警示起始日：2024年7月2日。'
            '股票简称由“*ST全新”变更为“全新好”，涨跌幅为10%。')
    assert document_fields(text) == {'effective_dates': ['2022-07-02', '2024-07-02'],
                                    'normal_names': ['全新好'], 'normal_limit': True}


def test_low_price_half_cent_changes_necessary_buyability():
    # At 1.09 a pure 15 bp shock stays below 1.10 minus half a cent;
    # the predeclared half-cent minimum reaches that boundary and fails.
    assert not decision_buyable('sz.000001', 1.09, 1.)
    assert decision_buyable('sz.000001', 1.08, 1.)
    assert not decision_buyable('sz.000001', 1.085, 1.)
    assert not decision_buyable('sh.688001', 1.08, 1.)


def test_causal_control_exclusions_do_not_use_future_notices_or_events():
    cal = pd.bdate_range('2024-05-01', periods=30).strftime('%Y-%m-%d').tolist()
    pool = pd.DataFrame({'date': [cal[2], cal[3], cal[24], cal[3], cal[4]],
                         'code': ['a', 'a', 'a', 'b', 'b']})
    events = pd.DataFrame({'code': ['a'], 'event_date': [cal[3]]})
    notices = pd.DataFrame({'code': ['b'], 'notice_date': [cal[3]], 'title': ['进入退市整理期交易的公告']})
    actual = causal_exclusions(pool, events, notices, cal)
    assert actual.recent_removal.tolist() == [False, True, False, False, False]
    assert actual.known_delisting.tolist() == [False, False, False, False, True]


def test_watch_window_requires_ten_later_dates_without_holdout():
    cal = pd.bdate_range('2025-12-01', '2025-12-31').strftime('%Y-%m-%d').tolist()
    events = pd.DataFrame({'code': ['a', 'b'], 'event_date': [cal[-12], cal[-5]]})
    watch = watch_schedule(events, cal)
    assert watch.groupby('code').size().tolist() == [5, 5]
    assert watch.groupby('code').complete_observation.sum().tolist() == [2, 0]


def test_first_feasible_entry_is_not_rescheduled_after_daily_cap():
    rows = [{'date': date, 'code': str(code), 'event_date': '2024-05-01',
             'entry_delay': delay, 'complete_observation': True}
            for delay, date in enumerate(['2024-05-06', '2024-05-07']) for code in range(6)]
    watch = pd.DataFrame(rows)
    pool = watch[['date', 'code']].copy()
    pool['amount_1449'] = 1e8
    entries = first_entries(watch, pool)
    assert len(entries) == 6
    assert entries.date.eq('2024-05-06').all()
    assert entries.loc[entries.daily_rank.le(5), 'code'].tolist() == ['0', '1', '2', '3', '4']
