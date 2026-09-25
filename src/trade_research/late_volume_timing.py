"""Freeze matched 14:50 declining stocks by the timing of late volume."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("data/research")
OUTPUT = ROOT / "late_volume_timing"
CAPACITY = 5
COOLDOWN = 5


def select_inputs(connection: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict]:
    connection.execute("""
        CREATE TEMP TABLE eligible AS
        SELECT s.date, s.code, s.return20_prior_adjusted, s.return_1450,
               s.amount_1450, s.price_1450, s.position_1450,
               i.return_last30, i.return_last15,
               i.volume_share_last15 / NULLIF(i.volume_share_last30, 0)
                   AS late_volume_ratio
        FROM snapshots s JOIN intraday i USING (date, code)
        WHERE ((s.date BETWEEN '2024-01-01' AND '2024-12-17')
            OR (s.date BETWEEN '2025-01-01' AND '2025-12-17'))
          AND s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.price_1450 >= 5
          AND s.amount_1450 BETWEEN 100000000 AND 1000000000
          AND s.return20_prior_adjusted BETWEEN -.10 AND .10
          AND s.return_1450 BETWEEN -.03 AND .03
          AND s.position_1450 BETWEEN 0 AND 1
          AND ABS(s.price_1450 - i.price_1450) <= .005
          AND i.return_last30 BETWEEN -.01 AND -.003
          AND i.return_last15 BETWEEN -.006 AND 0
          AND i.volume_share_last30 > 0
    """)
    dispersed = connection.execute("""
        SELECT * FROM eligible WHERE late_volume_ratio <= .50
        ORDER BY date, md5('late-volume-time-v1' || date || code), code
    """).df()
    concentrated = connection.execute("""
        SELECT * FROM eligible WHERE late_volume_ratio >= .65
        ORDER BY date, code
    """).df()
    calendar = connection.execute("""
        SELECT DISTINCT date FROM snapshots
        WHERE date BETWEEN '2024-01-01' AND '2025-12-31' ORDER BY date
    """).df().date.tolist()
    calendar_index = {date: index for index, date in enumerate(calendar)}
    controls = {date: group.set_index("code", drop=False)
                for date, group in concentrated.groupby("date")}
    last_used: dict[str, int] = {}
    selected = []
    attempted = 0
    stages = ("fresh", "prior", "current", "last30", "last15",
              "position", "amount", "price")
    stage_coverage = dict.fromkeys(stages, 0)
    for date, candidates in dispersed.groupby("date", sort=True):
        if date not in controls:
            continue
        index = calendar_index[date]
        available = controls[date].copy()
        daily_attempts = 0
        for row in candidates.itertuples(index=False):
            if daily_attempts >= CAPACITY:
                break
            if index - last_used.get(row.code, -1000) <= COOLDOWN:
                continue
            daily_attempts += 1
            attempted += 1
            fresh = available.loc[available.code.map(
                lambda code: index - last_used.get(code, -1000) > COOLDOWN
            )]
            gaps = {
                "prior": (fresh.return20_prior_adjusted
                          - row.return20_prior_adjusted).abs(),
                "current": (fresh.return_1450 - row.return_1450).abs(),
                "last30": (fresh.return_last30 - row.return_last30).abs(),
                "last15": (fresh.return_last15 - row.return_last15).abs(),
                "position": (fresh.position_1450 - row.position_1450).abs(),
            }
            amount_ratio = fresh.amount_1450 / row.amount_1450
            price_ratio = fresh.price_1450 / row.price_1450
            tests = (
                pd.Series(True, index=fresh.index),
                gaps["prior"].le(.03), gaps["current"].le(.005),
                gaps["last30"].le(.002), gaps["last15"].le(.002),
                gaps["position"].le(.30), amount_ratio.between(.5, 2),
                price_ratio.between(.5, 2),
            )
            matching = pd.Series(True, index=fresh.index)
            for stage, condition in zip(stages, tests):
                matching &= condition
                stage_coverage[stage] += int(matching.any())
            possible = fresh.loc[matching].copy()
            if possible.empty:
                continue
            possible["distance"] = (
                gaps["prior"].loc[possible.index] / .03
                + gaps["current"].loc[possible.index] / .005
                + gaps["last30"].loc[possible.index] / .002
                + gaps["last15"].loc[possible.index] / .002
                + gaps["position"].loc[possible.index] / .30
                + np.abs(np.log(amount_ratio.loc[possible.index])) / np.log(2)
                + np.abs(np.log(price_ratio.loc[possible.index])) / np.log(2)
            )
            match = possible.reset_index(drop=True).sort_values(
                ["distance", "code"]).iloc[0]
            treatment = row._asdict()
            treatment.update(candidate="dispersed_decline", pair_id=row.code,
                             daily_rank=daily_attempts)
            control = match.drop(labels="distance").to_dict()
            control.update(candidate="concentrated_decline_control",
                           pair_id=row.code, daily_rank=daily_attempts)
            selected.extend((treatment, control))
            available = available.drop(index=match.code)
            last_used[row.code] = index
            last_used[match.code] = index
    signals = pd.DataFrame(selected)
    if (signals.empty or signals.duplicated(["date", "code"]).any()
            or signals.groupby(["date", "pair_id"]).candidate.nunique().ne(2).any()
            or not signals.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Invalid late-volume timing pairs")
    pairs = signals.loc[signals.candidate.eq("dispersed_decline")].merge(
        signals.loc[signals.candidate.eq("concentrated_decline_control")],
        on=["date", "pair_id"], suffixes=("_dispersed", "_concentrated"),
        validate="one_to_one",
    )
    half = pairs.date.str[:4] + "H" + np.where(
        pairs.date.str[5:7].astype(int).le(6), "1", "2")
    segments = pairs.assign(half=half).groupby("half").agg(
        pairs=("pair_id", "size"), days=("date", "nunique")
    ).reset_index().to_dict("records")
    audit = {
        "dispersed_pool": len(dispersed),
        "concentrated_pool": len(concentrated),
        "capacity_candidates": attempted,
        "paired": len(pairs),
        "days": signals.date.nunique(),
        "match_fraction": len(pairs) / attempted,
        "candidate_stage_coverage": stage_coverage,
        "median_last30_gap": float((pairs.return_last30_dispersed
                                    - pairs.return_last30_concentrated).abs().median()),
        "median_last15_gap": float((pairs.return_last15_dispersed
                                    - pairs.return_last15_concentrated).abs().median()),
        "median_current_gap": float((pairs.return_1450_dispersed
                                     - pairs.return_1450_concentrated).abs().median()),
        "median_volume_ratio_gap": float((pairs.late_volume_ratio_concentrated
                                          - pairs.late_volume_ratio_dispersed).median()),
        "by_half": segments,
    }
    audit["outcome_gate_passed"] = (
        len(segments) == 4 and all(x["pairs"] >= 50 for x in segments)
        and audit["match_fraction"] >= .35
    )
    return signals, audit


def freeze(output_dir: Path = OUTPUT) -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    connection.read_parquet(str(ROOT / "intraday_features" / "*.parquet")
                            ).create_view("intraday")
    signals, audit = select_inputs(connection)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("repricing_signals.parquet", "repriced.parquet", "report.json"):
        (output_dir / stale).unlink(missing_ok=True)
    signals.to_parquet(output_dir / "selections.parquet", index=False,
                       compression="zstd")
    if audit["outcome_gate_passed"]:
        connection.register("selected", signals)
        snapshots = connection.execute("""
            SELECT s.* FROM selected x JOIN snapshots s USING (date, code)
        """).df()
        if (len(snapshots) != len(signals)
                or snapshots.duplicated(["date", "code"]).any()):
            raise ValueError("A selected stock lacks a unique repricing snapshot")
        snapshots.to_parquet(output_dir / "repricing_signals.parquet",
                             index=False, compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(freeze(args.output))


if __name__ == "__main__":
    main()
