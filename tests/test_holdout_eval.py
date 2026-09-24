"""Publication gate must account for unresolved positions."""

import json

import pandas as pd

from trade_research.hf_outcomes import Assumptions, _fees
from trade_research.holdout_eval import evaluate
from trade_research.holdout_eval import _thresholds
from trade_research.study_periods import HOLDOUT_YEAR


def test_unresolved_filled_entries_prevent_formula_publication() -> None:
    metrics = {
        "signals": 200, "entry_fills": 190, "clean_completed_exits": 180,
        "delayed_clean_exits": 0,
        "date_weighted_mean_net_return": .01,
        "median_net_return_per_trade": .006,
        "date_weighted_week_bootstrap_95pct_interval": [.002, .018],
        "date_weighted_mean_with_10bps_slippage_each_side": .009,
        "date_weighted_edge_vs_same_day_universe": .01,
        "edge_week_bootstrap_95pct_interval": [.001, .019],
    }
    result = _thresholds({str(HOLDOUT_YEAR): metrics})

    assert not result["holdout_pass"]
    assert "clean exits below 95% of filled entries" in \
        result["periods"][str(HOLDOUT_YEAR)]["reasons"]


def test_frozen_holdout_query_reads_only_recent_holdout(tmp_path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    outcome_dir = tmp_path / "outcomes"
    issues_dir = tmp_path / "issues"
    for directory in (snapshot_dir, outcome_dir, issues_dir):
        directory.mkdir()
    code = "sh.600000"
    dates = [pd.bdate_range(f"{year}-01-02", periods=11).strftime("%Y-%m-%d").tolist()
             for year in (2025, HOLDOUT_YEAR)]
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
                "exit_date": exit_date, "exit_delay_sessions": 0,
                "exit_price": exit_price,
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

    assert set(report["periods"]) == {str(HOLDOUT_YEAR)}
    assert report["periods"][str(HOLDOUT_YEAR)]["clean_completed_exits"] == 1
    assert report["last_entry_dates"][str(HOLDOUT_YEAR)] == dates[1][0]

    # A later audit may invalidate the outcome, but cannot retroactively
    # remove the stock from the frozen 14:50 selection.
    pd.DataFrame([{"date": dates[1][0], "code": code,
                   "kind": "ohlc_disagreement"}]).to_csv(
        issues_dir / "shard_00.csv", index=False
    )
    audited = evaluate(snapshot_dir, outcome_dir, issues_dir, freeze,
                       tmp_path / "audited.json", tmp_path / "audited.parquet",
                       allow_partial=True)
    assert audited["periods"][str(HOLDOUT_YEAR)]["signals"] == 1
    assert audited["periods"][str(HOLDOUT_YEAR)]["clean_completed_exits"] == 0
    assert not pd.read_parquet(tmp_path / "audited.parquet")[
        "quality_clean_exit"
    ].iloc[0]
