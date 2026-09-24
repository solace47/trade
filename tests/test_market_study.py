"""The market study must enforce its quality mask and daily selection cap."""

import pandas as pd

from trade_research.market_study import _quality_symbols, _week_bootstrap, study


def test_week_resampling_preserves_within_week_dependence():
    dates = pd.Series(pd.date_range("2024-01-01", periods=5).tolist()
                      + pd.date_range("2024-01-08", periods=5).tolist())
    values = pd.Series([1.0] * 5 + [-1.0] * 5)
    assert _week_bootstrap(values, dates, 7) == [-1.0, 1.0]


def test_source_anomalies_exclude_entire_stock_history(tmp_path):
    pd.DataFrame(columns=["code", "date", "kind"]).to_csv(
        tmp_path / "shard_00.csv", index=False,
    )
    pd.DataFrame([
        {"code": "sh.600001", "invalid_rows": 3,
         "amount_over_0_01pct_days": 0},
        {"code": "sh.600002", "invalid_rows": 0,
         "amount_over_0_01pct_days": 2},
        {"code": "sh.600003", "invalid_rows": 0,
         "amount_over_0_01pct_days": 0},
    ]).to_csv(tmp_path / "stocks.csv", index=False)
    assert _quality_symbols(tmp_path)["code"].tolist() == [
        "sh.600001", "sh.600002",
    ]


def test_quality_censoring_and_top_five_are_applied_before_metrics(tmp_path):
    snapshots = tmp_path / "snapshots"
    outcomes = tmp_path / "outcomes"
    issues = tmp_path / "issues"
    for path in (snapshots, outcomes, issues):
        path.mkdir()
    codes = [f"sh.60000{n}" for n in range(6)]
    signals = pd.DataFrame([{
        "date": "2024-01-02", "code": code,
        "isST": 0, "listing_age_sessions": 100,
        "reference_gap": False, "quote_outside_traded_range": False,
        "return_1450": 0.02 + n * 0.001,
        "position_1450": 0.9, "volume_ratio_est": 1.5,
        "price_1450": 10.0, "ma20_prior_adjusted": 9.0,
        "amount_1450": 50_000_000,
    } for n, code in enumerate(codes)])
    signals.to_parquet(snapshots / "shard_00_part_0000.parquet", index=False)
    fills = pd.DataFrame([{
        "date": "2024-01-02", "code": code, "horizon": 1,
        "entry_status": "filled", "exit_status": "filled",
        "exit_date": "2024-01-04" if code == codes[0] else "2024-01-03",
        "exit_delay_sessions": 1 if code == codes[0] else 0,
        "net_return": 0.01,
    } for code in codes])
    fills.to_parquet(outcomes / "shard_00_part_0000.parquet", index=False)
    pd.DataFrame([
        {"code": codes[-1], "date": "2024-01-03", "kind": "ohlc_disagreement"},
        {"code": codes[0], "date": "2024-01-03", "kind": "missing_active_minute"},
    ]).to_csv(issues / "shard_00.csv", index=False)
    report = study(snapshots, outcomes, issues, tmp_path / "report.json", allow_partial=True)
    period = report["policies"]
    assert period["frozen_rule"]["2024_development"]["1"]["signals"] == 6
    assert period["frozen_rule"]["2024_development"]["1"]["quality_clean_completed_exits"] == 4
    assert period["frozen_rule_top5"]["2024_development"]["1"]["signals"] == 5
    assert period["frozen_rule_top5"]["2024_development"]["1"]["quality_clean_completed_exits"] == 4
    assert report["frozen_rule_top5_by_regime"]["2024_development"]["1"][
        "broad_advance"
    ]["signals"] == 5
