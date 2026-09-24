"""Only a complete development scan may freeze a candidate."""

import json

from trade_research.strategy_scan import CANDIDATES, HORIZONS
from trade_research.strategy_select import propose
from trade_research.study_periods import DEVELOPMENT_YEAR, VALIDATION_YEAR


def test_proposal_uses_only_eligible_full_market_candidate(tmp_path) -> None:
    reports = {
        name: {year: {str(horizon): {"clean_completed_exits": 0}
                      for horizon in HORIZONS}
               for year in (str(DEVELOPMENT_YEAR), str(VALIDATION_YEAR))}
        for name in CANDIDATES
    }
    eligible = {
        "signals": 160, "entry_fills": 155,
        "clean_completed_exits": 150,
        "delayed_clean_exits": 2,
        "date_weighted_mean_net_return": .01,
        "date_weighted_mean_with_10bps_slippage_each_side": .009,
        "median_net_return_per_trade": .006,
        "date_weighted_edge_vs_same_day_universe": .01,
        "date_weighted_week_bootstrap_95pct_interval": [.002, .018],
        "edge_week_bootstrap_95pct_interval": [.001, .019],
    }
    reports["candidate_oversold_rebound"][str(DEVELOPMENT_YEAR)]["1"] = eligible
    reports["candidate_oversold_rebound"][str(VALIDATION_YEAR)]["1"] = eligible
    source = tmp_path / "scan.json"
    source.write_text(json.dumps({"shards": 20, "reports": reports}), encoding="utf-8")

    result = propose(source, tmp_path / "proposal.json")

    assert result["proposal"] == {
        "candidate": "candidate_oversold_rebound", "horizon": 1,
        "daily_capacity": 5, "ranking": "return_1450_desc_code_asc",
    }


def test_proposal_rejects_low_fill_rate(tmp_path) -> None:
    reports = {
        name: {year: {str(horizon): {"clean_completed_exits": 0}
                      for horizon in HORIZONS}
               for year in (str(DEVELOPMENT_YEAR), str(VALIDATION_YEAR))}
        for name in CANDIDATES
    }
    profitable = {
        "signals": 200, "entry_fills": 150, "clean_completed_exits": 145,
        "delayed_clean_exits": 2,
        "date_weighted_mean_net_return": .01,
        "date_weighted_mean_with_10bps_slippage_each_side": .009,
        "median_net_return_per_trade": .006,
        "date_weighted_edge_vs_same_day_universe": .01,
        "date_weighted_week_bootstrap_95pct_interval": [.002, .018],
        "edge_week_bootstrap_95pct_interval": [.001, .019],
    }
    for year in (str(DEVELOPMENT_YEAR), str(VALIDATION_YEAR)):
        reports["candidate_oversold_rebound"][year]["1"] = profitable
    source = tmp_path / "scan.json"
    source.write_text(json.dumps({"shards": 20, "reports": reports}), encoding="utf-8")

    result = propose(source, tmp_path / "proposal.json")

    assert result["proposal"] is None
    row = next(item for item in result["candidates"]
               if item["candidate"] == "candidate_oversold_rebound"
               and item["horizon"] == 1)
    assert "2024: entry fill rate below 80%" in row["reasons"]


def test_proposal_rejects_delayed_exits(tmp_path) -> None:
    reports = {
        name: {year: {str(horizon): {"clean_completed_exits": 0}
                      for horizon in HORIZONS}
               for year in (str(DEVELOPMENT_YEAR), str(VALIDATION_YEAR))}
        for name in CANDIDATES
    }
    apparently_profitable = {
        "signals": 160, "entry_fills": 155,
        "clean_completed_exits": 150, "delayed_clean_exits": 50,
        "date_weighted_mean_net_return": .01,
        "date_weighted_mean_with_10bps_slippage_each_side": .009,
        "median_net_return_per_trade": .006,
        "date_weighted_edge_vs_same_day_universe": .01,
        "date_weighted_week_bootstrap_95pct_interval": [.002, .018],
        "edge_week_bootstrap_95pct_interval": [.001, .019],
    }
    for year in (str(DEVELOPMENT_YEAR), str(VALIDATION_YEAR)):
        reports["candidate_oversold_rebound"][year]["1"] = apparently_profitable
    source = tmp_path / "scan.json"
    source.write_text(json.dumps({"shards": 20, "reports": reports}), encoding="utf-8")

    result = propose(source, tmp_path / "proposal.json")

    assert result["proposal"] is None
    row = next(item for item in result["candidates"]
               if item["candidate"] == "candidate_oversold_rebound"
               and item["horizon"] == 1)
    assert "2024: more than 5% of clean exits are delayed" in row["reasons"]
