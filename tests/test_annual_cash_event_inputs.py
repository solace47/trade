from trade_research.annual_cash_event_inputs import _first_later_session


def test_annual_disclosure_uses_strictly_later_session() -> None:
    days = ["2024-04-26", "2024-04-29", "2024-04-30", "2024-05-06"]
    assert _first_later_session("2024-04-26", days) == "2024-04-29"
    assert _first_later_session("2024-04-28", days) == "2024-04-29"
    assert _first_later_session("2024-04-30", days) == "2024-05-06"
