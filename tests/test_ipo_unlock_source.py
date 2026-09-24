from trade_research.ipo_unlock_source import extract, strict_title

import pandas as pd

from trade_research.ipo_unlock_inputs import _prior_float


def test_original_preissue_notice_needs_disclosed_future_date_and_shares():
    text = ("证券代码：301149；首次公开发行前已发行股份解除限售并上市流通。"
            "特别提示：本次解除限售股东户数为6户，解除限售股份的数量为"
            "21,375,266股。本次解除限售股上市流通日为2024年11月11日。")
    assert extract(text, "sz.301149", "2024-11-08") == {
        "status": "ok", "unlock_date": "2024-11-11",
        "unlock_shares": 21375266,
    }
    assert extract(text, "sz.301148", "2024-11-08")["status"] == (
        "identity_or_type_unconfirmed")
    assert extract(text, "sz.301149", "2024-11-11")["status"] == (
        "not_disclosed_before_unlock")


def test_ambiguous_original_dates_do_not_enter_event_list():
    text = ("股票代码688001；本次为首次公开发行前股东的限售股份。"
            "本次股票上市流通总数为1,200,000股。"
            "本次股票上市流通日期为2025年5月6日。"
            "本次股票上市流通日期为2025年5月7日。")
    assert extract(text, "sh.688001", "2025-04-28")["status"] == (
        "missing_or_ambiguous_unlock_date")
    assert strict_title("首次公开发行部分限售股上市流通公告")
    assert not strict_title("首次公开发行网下配售限售股上市流通核查意见")


def test_unlock_size_uses_previous_session_float_only(tmp_path):
    pd.DataFrame([
        {"date": "2024-05-10", "code": "sz.301149", "volume": 1_000_000,
         "turn": 1.0, "tradestatus": 1},
        {"date": "2024-05-13", "code": "sz.301149", "volume": 1_000_000,
         "turn": 10.0, "tradestatus": 1},
    ]).to_parquet(tmp_path / "daily.parquet", index=False)
    event = pd.DataFrame([{"date": "2024-05-13", "code": "sz.301149",
                           "unlock_shares": 20_000_000}])
    result = _prior_float(event, tmp_path,
                          ["2024-05-10", "2024-05-13"])
    assert result.trade_date.iloc[0] == "2024-05-10"
    assert round(result.unlock_prior_float_ratio.iloc[0], 6) == .2
