"""Exploratory signal lists use recent dates and enforce the daily cap."""

import pandas as pd

from trade_research.exploratory_signals import select


def test_low_amount_neutral_excludes_old_and_bad_stock_days(tmp_path) -> None:
    snapshots = tmp_path / "snapshots"
    issues = tmp_path / "issues"
    snapshots.mkdir()
    issues.mkdir()
    dates24 = pd.bdate_range("2024-01-02", periods=11).strftime("%Y-%m-%d")
    dates25 = pd.bdate_range("2025-01-02", periods=11).strftime("%Y-%m-%d")
    rows = []

    def add(date: str, number: int, amount: float,
            intraday_return: float = .01) -> None:
        rows.append({
            "date": date, "code": f"sh.60000{number}",
            "amount_1450": amount, "return_1450": intraday_return,
            "isST": 0, "listing_age_sessions": 100,
            "reference_gap": False, "quote_outside_traded_range": False,
        })

    for day in list(dates24[1:]) + list(dates25[1:]):
        add(day, 1, 150_000_000, .02)
    for number in range(1, 9):
        add(dates24[0], number, 30_000_000 + number * 1_000_000)
    add(dates25[0], 1, 40_000_000)
    add("2023-12-29", 1, 31_000_000)
    pd.DataFrame(rows).to_parquet(snapshots / "shard_00_part_0000.parquet")
    pd.DataFrame([{"date": dates24[0], "code": "sh.600007",
                   "kind": "ohlc_disagreement"}]).to_csv(
        issues / "shard_00.csv", index=False
    )
    pd.DataFrame([{"code": "sh.600008", "invalid_rows": 1}]).to_csv(
        issues / "stocks.csv", index=False
    )

    result = select(snapshots, issues, "low_amount_neutral",
                    tmp_path / "signals.parquet", allow_partial=True)

    assert result["date"].unique().tolist() == [dates24[0], dates25[0]]
    assert result.loc[result.date.eq(dates24[0]), "code"].tolist() == [
        f"sh.60000{number}" for number in range(1, 6)
    ]
    assert len(result) == 6
