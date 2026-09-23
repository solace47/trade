"""Publication gate must account for unresolved positions."""

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
