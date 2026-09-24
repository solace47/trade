import pandas as pd

from trade_research import ipo_unlock_notice_inputs


def test_notice_signal_and_t1_exit_precede_unlock(monkeypatch, tmp_path) -> None:
    calendar = ["2024-04-26", "2024-04-29", "2024-04-30",
                "2024-05-06", "2024-05-07"]
    audits = pd.DataFrame([
        {"code": "sh.600001", "notice_date": "2024-04-26",
         "unlock_date": "2024-05-06"},
        {"code": "sh.600002", "notice_date": "2024-04-26",
         "unlock_date": "2024-04-30"},
        {"code": "sh.600003", "notice_date": "2024-04-30",
         "unlock_date": "2024-05-07"},
    ])
    monkeypatch.setattr(ipo_unlock_notice_inputs, "_confirmed_audits",
                        lambda _base: (audits, {}))
    events, excluded_controls, _ = ipo_unlock_notice_inputs._notices(
        tmp_path, calendar)
    assert events[["code", "date", "unlock_date"]].to_dict("records") == [
        {"code": "sh.600001", "date": "2024-04-29",
         "unlock_date": "2024-05-06"}]
    assert set(excluded_controls) == {
        ("2024-04-29", "sh.600001"),
        ("2024-04-29", "sh.600002"),
        ("2024-05-06", "sh.600003"),
    }
