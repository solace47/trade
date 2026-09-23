"""Only a complete development scan may freeze a candidate."""

import json

from trade_research.strategy_scan import CANDIDATES, HORIZONS
from trade_research.strategy_select import propose


def test_proposal_uses_only_eligible_full_market_candidate(tmp_path) -> None:
    reports = {
        name: {year: {str(horizon): {"clean_completed_exits": 0}
                      for horizon in HORIZONS}
               for year in ("2022", "2023")}
        for name in CANDIDATES
    }
    eligible = {
        "clean_completed_exits": 150,
        "date_weighted_mean_net_return": .01,
        "date_weighted_mean_with_10bps_slippage_each_side": .009,
        "median_net_return_per_trade": .006,
        "date_weighted_edge_vs_same_day_universe": .01,
        "date_weighted_week_bootstrap_95pct_interval": [.002, .018],
    }
    reports["candidate_oversold_rebound"]["2022"]["1"] = eligible
    reports["candidate_oversold_rebound"]["2023"]["1"] = eligible
    source = tmp_path / "scan.json"
    source.write_text(json.dumps({"shards": 20, "reports": reports}), encoding="utf-8")

    result = propose(source, tmp_path / "proposal.json")

    assert result["proposal"] == {
        "candidate": "candidate_oversold_rebound", "horizon": 1,
        "daily_capacity": 5, "ranking": "return_1450_desc_code_asc",
    }
