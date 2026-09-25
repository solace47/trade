"""Freeze full-market residual-variance pairs before opening outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from .late_variance_risk import ROOT, select_inputs


OUTPUT = ROOT / "late_residual_variance_risk"
OUTPUT_MEAN = ROOT / "late_residual_mean_risk"


def freeze(output_dir: Path | None = None, factor: str = "median") -> dict:
    if factor not in ("median", "clipped_mean"):
        raise ValueError("Unknown market proxy")
    if output_dir is None:
        output_dir = OUTPUT if factor == "median" else OUTPUT_MEAN
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    connection.read_parquet(str(ROOT / "intraday_features" / "*.parquet")
                            ).create_view("intraday")
    suffix = "*_variance.parquet" if factor == "median" else "*_variance_mean.parquet"
    connection.read_parquet(str(ROOT / "late_residual_variance"
                                / suffix)).project(
        "date, code, price_1450, "
        "residual_variance_last30 AS realized_variance_last30"
    ).create_view("variance")
    selected, audit = select_inputs(connection, rank_salt="resid-risk-v1")
    all_candidates = connection.execute("SELECT * FROM candidates").df()
    audit["market_proxy"] = factor
    audit["min_half_pairs"] = min(item["pairs"] for item in audit["by_half"])
    audit["outcome_gate_passed"] = (
        len(audit["by_half"]) == 4
        and audit["min_half_pairs"] >= 50
        and audit["match_fraction"] >= .35
    )
    if factor == "clipped_mean":
        audit["market_zero_share_by_year"] = {
            str(year): json.loads((ROOT / "late_residual_variance"
                                   / f"{year}_mean_audit.json").read_text(
                                       encoding="utf-8"))["market_zero_share"]
            for year in (2024, 2025)
        }
        earlier = pd.read_parquet(
            OUTPUT / "all_candidates.parquet",
            columns=["date", "code", "surprise"],
        )
        earlier_high = set(zip(
            earlier.loc[earlier.surprise.ge(1.5), "date"],
            earlier.loc[earlier.surprise.ge(1.5), "code"],
        ))
        current_high = all_candidates.loc[all_candidates.surprise.ge(1.5)]
        newer_high = set(zip(current_high.date, current_high.code))
        audit["median_high_changed_fraction"] = (
            1 - len(earlier_high & newer_high) / len(earlier_high)
        )
        audit["outcome_gate_passed"] = (
            audit["outcome_gate_passed"]
            and max(audit["market_zero_share_by_year"].values()) < .05
            and audit["median_high_changed_fraction"] >= .10
        )
    audit["input_only"] = True
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("repricing_signals.parquet", "repriced.parquet", "report.json"):
        (output_dir / stale).unlink(missing_ok=True)
    all_candidates.to_parquet(output_dir / "all_candidates.parquet",
                              index=False, compression="zstd")
    selected.to_parquet(output_dir / "selections.parquet", index=False,
                        compression="zstd")
    if audit["outcome_gate_passed"]:
        connection.register("selected", selected)
        repricing = connection.execute("""
            SELECT s.* FROM selected r JOIN snapshots s USING (date, code)
        """).df()
        if (len(repricing) != len(selected)
                or repricing.duplicated(["date", "code"]).any()
                or not repricing.date.str[:4].isin(("2024", "2025")).all()):
            raise ValueError("Residual-variance repricing inputs invalid")
        repricing.to_parquet(output_dir / "repricing_signals.parquet",
                             index=False, compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--market-proxy", choices=("median", "clipped_mean"),
                        default="median")
    args = parser.parse_args()
    print(freeze(args.output, args.market_proxy))


if __name__ == "__main__":
    main()
