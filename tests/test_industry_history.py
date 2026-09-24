from pathlib import Path

import pandas as pd
import pytest

from trade_research.industry_history import (
    build_intervals, quarter_starts, validate_industry,
)


def test_quarter_starts_are_known_by_each_quarter(tmp_path: Path) -> None:
    dates = [f"{year}-{month:02d}-02" for year in (2024, 2025)
             for month in (1, 4, 7, 10)]
    calendar = tmp_path / "calendar.parquet"
    pd.DataFrame({"calendar_date": dates, "is_trading_day": ["1"] * 8}
                 ).to_parquet(calendar, index=False)
    assert quarter_starts(calendar) == dates


def test_industry_rejects_future_update() -> None:
    frame = pd.DataFrame({
        "code": ["sh.600000"], "updateDate": ["2025-01-06"],
        "industry": ["J66货币金融服务"],
        "industryClassification": ["证监会行业分类"],
    })
    with pytest.raises(ValueError, match="future"):
        validate_industry(frame, "2025-01-02")


def test_same_day_update_waits_and_empty_class_keeps_last_known(tmp_path: Path) -> None:
    files = []
    for asof, update, industry in (
        ("2024-01-02", "2024-01-01", "A"),
        ("2024-04-01", "2024-04-01", "B"),
        ("2024-07-01", "2024-06-30", ""),
    ):
        path = tmp_path / f"{asof}.parquet"
        pd.DataFrame({
            "code": ["sh.600000"], "updateDate": [update],
            "industry": [industry],
            "industryClassification": ["证监会行业分类"],
        }).to_parquet(path, index=False)
        files.append(path)
    frame = build_intervals(files, tmp_path / "intervals.parquet")
    assert frame.effective_date.tolist() == ["2024-01-02", "2024-04-02"]
    assert frame.next_effective_date.iloc[0] == "2024-04-02"
    assert pd.isna(frame.next_effective_date.iloc[1])
