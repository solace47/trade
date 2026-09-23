"""Publication gate must account for unresolved positions."""

import json

import pandas as pd

from trade_research.hf_outcomes import Assumptions, _fees
from trade_research.holdout_eval import evaluate
from trade_research.holdout_eval import _thresholds


def test_unresolved_filled_entries_prevent_formula_publication() -> None:
    metrics = {
        "signals": 200, "entry_fills": 190, "clean_completed_exits": 180,
        "date_weighted_mean_net_return": .01,
        "date_weighted_week_bootstrap_95pct_interval": [.002, .018],
        "date_weighted_mean_with_10bps_slippage_each_side": .009,
    }
    result = _thresholds({"2024": metrics, "2025_2026": metrics})

    assert not result["both_periods_pass"]
    assert "clean exits below 95% of filled entries" in \
        result["periods"]["2024"]["reasons"]


def test_frozen_holdout_query_runs_on_separate_years(tmp_path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    outcome_dir = tmp_path / "outcomes"
    issues_dir = tmp_path / "issues"
    for directory in (snapshot_dir, outcome_dir, issues_dir):
        directory.mkdir()
    code = "sh.600000"
    dates = [pd.bdate_range(f"{year}-01-02", periods=11).strftime("%Y-%m-%d").tolist()
             for year in (2024, 2025)]
    snapshots = []
    outcomes = []
    terms = Assumptions()
    for group in dates:
        for number, date in enumerate(group):
            exit_date = group[min(number + 1, len(group) - 1)]
            entry_price, exit_price, shares = 10.305, 10.495, 100
            buy_value, sell_value = entry_price * shares, exit_price * shares
            net = (sell_value - _fees(sell_value, "sell", terms, exit_date)) / \
                (buy_value + _fees(buy_value, "buy", terms, date)) - 1
            snapshots.append({
                "date": date, "code": code,
                "return_1450": .03 if number == 0 else 0,
                "position_1450": .85, "volume_ratio_est": 2.0,
                "price_1450": 10.3, "amount_1450": 100_000_000,
                "ma5_prior_adjusted": 10.0, "ma20_prior_adjusted": 10.0,
                "return5_prior_adjusted": -.06,
                "return20_prior_adjusted": -.12,
                "preclose": 10.0, "isST": 0, "listing_age_sessions": 100,
                "reference_gap": False, "quote_outside_traded_range": False,
            })
            outcomes.append({
                "date": date, "code": code, "horizon": 1,
                "entry_status": "filled", "entry_price": entry_price,
                "shares": shares, "exit_status": "filled",
                "exit_date": exit_date, "exit_price": exit_price,
                "net_return": net,
            })
    pd.DataFrame(snapshots).to_parquet(snapshot_dir / "shard_00_part_0000.parquet")
    pd.DataFrame(outcomes).to_parquet(outcome_dir / "shard_00_part_0000.parquet")
    pd.DataFrame(columns=["code", "date", "kind"]).to_csv(
        issues_dir / "shard_00.csv", index=False
    )
    pd.DataFrame({"code": [code], "invalid_rows": [0]}).to_csv(
        issues_dir / "stocks.csv", index=False
    )
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps({
        "candidate": "candidate_oversold_rebound", "horizon": 1,
        "daily_capacity": 5, "ranking": "return_1450_desc_code_asc",
    }), encoding="utf-8")

    report = evaluate(snapshot_dir, outcome_dir, issues_dir, freeze,
                      tmp_path / "report.json", tmp_path / "trades.parquet",
                      allow_partial=True)

    assert report["periods"]["2024"]["clean_completed_exits"] == 1
    assert report["periods"]["2025_2026"]["clean_completed_exits"] == 1
    assert report["last_entry_dates"]["2024"] == dates[0][0]
