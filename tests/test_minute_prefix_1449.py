from pathlib import Path

import pandas as pd
import pytest

from trade_research.minute_prefix_1449 import build_year


def _day(date: str, *, missing_last: bool = False) -> pd.DataFrame:
    times = [
        *pd.date_range(f"{date} 09:30", f"{date} 11:30", freq="min"),
        *pd.date_range(f"{date} 13:01", f"{date} 14:49", freq="min"),
    ]
    if missing_last:
        times.pop()
    prices = {"14:20": 9.9, "14:35": 9.8, "14:49": 9.7}
    frame = pd.DataFrame({"timestamp": times})
    frame["close"] = frame.timestamp.dt.strftime("%H:%M").map(prices).fillna(10.0)
    frame["open"] = frame.close
    frame["high"] = frame.close
    frame["low"] = frame.close
    frame["volume"] = 100
    frame["turnover"] = frame.close * frame.volume
    frame["symbol"] = "600000"
    frame["exchange"] = "SH"
    return frame


def test_complete_prefix_excludes_later_bar_and_incomplete_day(tmp_path: Path) -> None:
    valid = _day("2024-01-02")
    later = _day("2024-01-02").iloc[:1].copy()
    later["timestamp"] = pd.Timestamp("2024-01-02 14:50")
    later[["open", "high", "low", "close"]] = 100.0
    later["turnover"] = later.close * later.volume
    incomplete = _day("2024-01-03", missing_last=True)
    source = tmp_path / "source.parquet"
    pd.concat([valid, later, incomplete], ignore_index=True).to_parquet(source)

    output = tmp_path / "prefix.parquet"
    build_year([str(source)], 2024, output, threads=1)
    result = pd.read_parquet(output)
    assert result[["date", "code"]].to_dict("records") == [
        {"date": "2024-01-02", "code": "sh.600000"}
    ]
    assert result.price_1449.iloc[0] == pytest.approx(9.7)
    assert result.high_1449.iloc[0] == pytest.approx(10.0)
    assert result.low_1449.iloc[0] == pytest.approx(9.7)
    assert result.volume_1449.iloc[0] == 23000
    assert result.amount_1449.iloc[0] == pytest.approx(valid.turnover.sum())
    assert result.return_last29.iloc[0] == pytest.approx(9.7 / 9.9 - 1)
    assert result.vwap_1446_1449.iloc[0] == pytest.approx(10.0 * 3 / 4 + 9.7 / 4)
    assert not result.quote_outside_traded_range.iloc[0]


def test_prior_year_cannot_be_a_research_period(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="2024–2025"):
        build_year([str(tmp_path / "unused.parquet")], 2023, tmp_path / "out.parquet")


def test_older_prefix_requires_explicit_training_scope(tmp_path: Path) -> None:
    source=tmp_path/'source.parquet'
    pd.concat([_day('2022-01-04'),_day('2023-01-03'),_day('2024-01-02')],ignore_index=True).to_parquet(source)
    output=tmp_path/'older.parquet'
    build_year([str(source)],2022,output,threads=1,historical_training=True)
    assert pd.read_parquet(output).date.tolist()==['2022-01-04']
    with pytest.raises(ValueError):
        build_year([str(source)],2026,output,historical_training=True)


def test_validation_cutoff_excludes_later_days_and_requires_explicit_scope(tmp_path: Path) -> None:
    source = tmp_path / 'source.parquet'
    pd.concat([_day('2025-12-31'), _day('2026-01-05'), _day('2026-07-17'),
               _day('2026-07-20')], ignore_index=True).to_parquet(source)
    output = tmp_path / 'validation.parquet'
    build_year([str(source)], 2026, output, threads=1, validation_last='2026-07-17')
    assert sorted(pd.read_parquet(output).date) == ['2026-01-05', '2026-07-17']
    with pytest.raises(ValueError):
        build_year([str(source)], 2026, output)
    with pytest.raises(ValueError):
        build_year([str(source)], 2026, output, validation_last='2027-01-01')
