from trade_research.analyst_predecessors import select_predecessors


def record(code, stamp, display="2024-04-01", stock="600001", broker="80000031", status="valid_time_and_identity"):
    return {"info_code": code, "stock_code": stock, "broker_code": broker, "half": "2024H1",
            "production_time": stamp, "display_date": display, "source_status": status}


def test_last_produced_report_wins_over_display_date_and_forecast():
    now = record("now", "2024-04-03 18:00:00")
    old = record("old", "2024-04-01 18:00:00", "2024-04-02")
    later = record("later", "2024-04-02 18:00:00", "2024-03-30")
    future = record("future", "2024-04-04 08:00:00", "2024-03-01")
    rows = [old, now, later, future]
    assert select_predecessors([now], rows)[0]["previous_info_code"] == "later"
    later["reported_prior_forecast"] = None
    assert select_predecessors([now], rows[::-1])[0]["previous_info_code"] == "later"


def test_no_2023_fallback_or_other_broker():
    now = record("now", "2024-01-03 18:00:00")
    old = record("old", "2023-12-31 18:00:00")
    other = record("other", "2024-01-02 18:00:00", broker="other")
    result = select_predecessors([now], [now, old, other])[0]
    assert result["previous_info_code"] is None
    assert result["status"] == "no_predecessor_since_2024_in_catalog"


def test_unknown_or_tied_times_do_not_fall_back_to_an_older_report():
    now = record("now", "2024-04-03 18:00:00")
    old = record("old", "2024-04-01 18:00:00")
    for extra, status in [
        (record("unknown", None, status="missing_or_invalid_production_time"), "unknown_order_due_to_source_field"),
        (record("tie_old", old["production_time"]), "ambiguous_predecessor_time_tie"),
        (record("tie_now", now["production_time"]), "ambiguous_current_time_tie"),
    ]:
        result = select_predecessors([now], [now, old, extra])[0]
        assert result["status"] == status
        assert result["previous_info_code"] is None
