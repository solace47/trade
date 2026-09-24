"""Propose one strategy using the completed 2024-2025 scan only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .strategy_scan import CANDIDATES, HORIZONS
from .study_periods import DEVELOPMENT_YEAR, VALIDATION_YEAR


def _reasons(development: dict, validation: dict) -> list[str]:
    reasons = []
    for year, metrics in ((str(DEVELOPMENT_YEAR), development),
                          (str(VALIDATION_YEAR), validation)):
        signals = metrics.get("signals", 0)
        fills = metrics.get("entry_fills", 0)
        clean = metrics.get("clean_completed_exits", 0)
        if signals and fills / signals < .8:
            reasons.append(f"{year}: entry fill rate below 80%")
        if fills and clean / fills < .95:
            reasons.append(f"{year}: clean exits below 95% of filled entries")
        if clean and metrics["delayed_clean_exits"] / clean > .05:
            reasons.append(f"{year}: more than 5% of clean exits are delayed")
        if clean < 100:
            reasons.append(f"{year}: fewer than 100 clean exits")
            continue
        if metrics["date_weighted_mean_net_return"] <= 0:
            reasons.append(f"{year}: date-weighted net return is not positive")
        if metrics["date_weighted_mean_with_10bps_slippage_each_side"] <= 0:
            reasons.append(f"{year}: 10 bps per-side stress is not positive")
        if metrics["median_net_return_per_trade"] <= 0:
            reasons.append(f"{year}: median trade is not positive")
        if metrics["date_weighted_edge_vs_same_day_universe"] <= 0:
            reasons.append(f"{year}: no edge over the same-day universe")
        if metrics["date_weighted_week_bootstrap_95pct_interval"][0] <= 0:
            reasons.append(f"{year}: weekly bootstrap interval includes zero")
    if (validation.get("clean_completed_exits", 0) >= 100
            and validation["edge_week_bootstrap_95pct_interval"][0] <= 0):
        reasons.append(f"{VALIDATION_YEAR}: same-day edge interval includes zero")
    return reasons


def propose(scan_file: Path, output: Path) -> dict:
    scan = json.loads(scan_file.read_text(encoding="utf-8"))
    if scan["shards"] != 20:
        raise ValueError("Strategy selection requires all 20 market shards")
    if set(scan["reports"]) != set(CANDIDATES):
        raise ValueError("The scan does not contain the registered candidates")
    rows = []
    for name in CANDIDATES:
        for horizon in HORIZONS:
            development = scan["reports"][name][str(DEVELOPMENT_YEAR)][str(horizon)]
            validation = scan["reports"][name][str(VALIDATION_YEAR)][str(horizon)]
            reasons = _reasons(development, validation)
            score = (min(development["date_weighted_mean_net_return"],
                         validation["date_weighted_mean_net_return"])
                     if not reasons else None)
            rows.append({
                "candidate": name, "horizon": horizon,
                "eligible": not reasons, "reasons": reasons,
                "consistency_score": score,
            })
    eligible = sorted((row for row in rows if row["eligible"]),
                      key=lambda row: (-row["consistency_score"],
                                       row["candidate"], row["horizon"]))
    winner = eligible[0] if eligible else None
    result = {
        "scope": f"Development {DEVELOPMENT_YEAR} and validation "
                 f"{VALIDATION_YEAR} only",
        "rule": "At least 100 clean exits, 80% entry fills, 95% clean "
                "exits per filled entry and no more than 5% delayed clean "
                "exits in each year; positive date-weighted "
                "net and 10 bps stress, median, same-day edge and weekly "
                f"interval in each year; {VALIDATION_YEAR} same-day edge "
                "interval above zero; maximize the "
                "weaker annual mean",
        "proposal": ({
            "candidate": winner["candidate"], "horizon": winner["horizon"],
            "daily_capacity": 5, "ranking": "return_1450_desc_code_asc",
        } if winner else None),
        "candidates": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scan", type=Path,
                        default=Path("data/research/strategy_scan.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/research/strategy_proposal.json"))
    args = parser.parse_args()
    result = propose(args.scan, args.output)
    print(json.dumps(result["proposal"], ensure_ascii=False))


if __name__ == "__main__":
    main()
