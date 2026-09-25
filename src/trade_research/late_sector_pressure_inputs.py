"""Freeze late-decline pairs by leave-one-out industry tail pressure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("data/research")
OUTPUT = ROOT / "late_sector_pressure"
CAPACITY = 5
COOLDOWN = 5


def select_inputs(connection: duckdb.DuckDBPyConnection
                  ) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    connection.execute("""
        CREATE TEMP TABLE peers AS
        SELECT s.date, s.code, h.industry, s.return20_prior_adjusted,
               s.return_1450, s.amount_1450, s.price_1450,
               s.position_1450, i.return_last30,
               (1 + s.return_1450) / (1 + i.return_last30) - 1
                   AS return_to_1420,
               GREATEST(-.02, LEAST(.02, i.return_last30)) AS tail_clip,
               GREATEST(-.10, LEAST(.10, s.return20_prior_adjusted))
                   AS prior20_clip,
               m.market_tail_return
        FROM snapshots s JOIN intraday i USING (date, code)
        JOIN industry_history h ON h.code = s.code
          AND s.date >= h.effective_date
          AND (h.next_effective_date IS NULL
               OR s.date < h.next_effective_date)
        JOIN market_states m USING (date)
        WHERE ((s.date BETWEEN '2024-01-01' AND '2024-12-17')
            OR (s.date BETWEEN '2025-01-01' AND '2025-12-17'))
          AND s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.amount_1450 >= 30000000
          AND s.return20_prior_adjusted IS NOT NULL
          AND ABS(s.price_1450 - i.price_1450) <= .005
          AND h.industry != ''
    """)
    duplicate_peers = connection.execute("""
        SELECT COUNT(*) FROM (
            SELECT date, code FROM peers GROUP BY date, code HAVING COUNT(*) > 1
        )
    """).fetchone()[0]
    if duplicate_peers:
        raise ValueError("Overlapping point-in-time industry memberships")
    connection.execute("""
        CREATE TEMP TABLE eligible AS
        WITH scored AS (
            SELECT p.*, COUNT(*) OVER (
                       PARTITION BY date, industry) AS industry_count,
                   SUM(tail_clip) OVER (
                       PARTITION BY date, industry) AS industry_tail_sum,
                   SUM(prior20_clip) OVER (
                       PARTITION BY date, industry) AS industry_prior_sum
            FROM peers p
        )
        SELECT date, code, industry, return20_prior_adjusted,
               return_1450, amount_1450, price_1450, position_1450,
               return_last30, return_to_1420,
               (industry_tail_sum - tail_clip) / (industry_count - 1)
                 - market_tail_return AS sector_residual_tail,
               (industry_prior_sum - prior20_clip) / (industry_count - 1)
                   AS sector_prior20,
               industry_count - 1 AS peer_count
        FROM scored
        WHERE industry_count >= 11 AND price_1450 >= 5
          AND amount_1450 BETWEEN 100000000 AND 1000000000
          AND return20_prior_adjusted BETWEEN -.10 AND .10
          AND return_1450 BETWEEN -.03 AND .03
          AND return_last30 BETWEEN -.01 AND -.003
          AND position_1450 BETWEEN 0 AND 1
    """)
    strong = connection.execute("""
        SELECT * FROM eligible WHERE sector_residual_tail >= .001
        ORDER BY date, md5('sector-tail-resilience-v1' || date || code), code
    """).df()
    weak = connection.execute("""
        SELECT * FROM eligible WHERE sector_residual_tail <= -.001
        ORDER BY date, code
    """).df()
    all_candidates = pd.concat((
        strong.assign(candidate="resilient_sector_decline"),
        weak.assign(candidate="weak_sector_decline"),
    ), ignore_index=True)
    if all_candidates.duplicated(["date", "code"]).any():
        raise ValueError("Industry pressure groups overlap")
    calendar = connection.execute("""
        SELECT DISTINCT date FROM snapshots
        WHERE date BETWEEN '2024-01-01' AND '2025-12-31' ORDER BY date
    """).df().date.tolist()
    day_index = {date: index for index, date in enumerate(calendar)}
    weak_by_date = {date: group.set_index("code", drop=False)
                    for date, group in weak.groupby("date")}
    last_used: dict[str, int] = {}
    selected = []
    attempted = 0
    stages = ("fresh", "prior20", "pre_tail", "tail", "sector_prior20",
              "amount", "price", "position")
    candidate_with_control_after = {stage: 0 for stage in stages}
    for date, candidates in strong.groupby("date", sort=True):
        if date not in weak_by_date:
            continue
        index = day_index[date]
        available = weak_by_date[date].copy()
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
            prior = (fresh.return20_prior_adjusted
                     - row.return20_prior_adjusted).abs()
            early = (fresh.return_to_1420 - row.return_to_1420).abs()
            tail = (fresh.return_last30 - row.return_last30).abs()
            sector_prior = (fresh.sector_prior20 - row.sector_prior20).abs()
            amount_ratio = fresh.amount_1450 / row.amount_1450
            price_ratio = fresh.price_1450 / row.price_1450
            position = (fresh.position_1450 - row.position_1450).abs()
            checks = (
                ("fresh", pd.Series(True, index=fresh.index)),
                ("prior20", prior.le(.03)),
                ("pre_tail", early.le(.005)),
                ("tail", tail.le(.002)),
                ("sector_prior20", sector_prior.le(.03)),
                ("amount", amount_ratio.between(.5, 2)),
                ("price", price_ratio.between(.5, 2)),
                ("position", position.le(.30)),
            )
            mask = pd.Series(True, index=fresh.index)
            for stage, passed in checks:
                mask &= passed
                candidate_with_control_after[stage] += int(mask.any())
            possible = fresh.loc[mask].copy()
            if possible.empty:
                continue
            possible["distance"] = (
                prior.loc[possible.index] / .03
                + early.loc[possible.index] / .005
                + tail.loc[possible.index] / .002
                + sector_prior.loc[possible.index] / .03
                + np.abs(np.log(amount_ratio.loc[possible.index])) / np.log(2)
                + np.abs(np.log(price_ratio.loc[possible.index])) / np.log(2)
                + position.loc[possible.index] / .30
            )
            match = possible.reset_index(drop=True).sort_values(
                ["distance", "code"]).iloc[0]
            resilient = row._asdict()
            resilient.update(candidate="resilient_sector_decline",
                             pair_id=row.code, daily_rank=daily_attempts)
            control = match.drop(labels="distance").to_dict()
            control.update(candidate="weak_sector_decline",
                           pair_id=row.code, daily_rank=daily_attempts)
            selected.extend((resilient, control))
            available = available.drop(index=match.code)
            last_used[row.code] = index
            last_used[match.code] = index
    signals = pd.DataFrame(selected)
    if (signals.empty or signals.duplicated(["date", "code"]).any()
            or signals.groupby(["date", "pair_id"]).candidate.nunique().ne(2).any()
            or not signals.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Invalid sector-tail pressure input pairs")
    pairs = signals.loc[signals.candidate.eq("resilient_sector_decline")].merge(
        signals.loc[signals.candidate.eq("weak_sector_decline")],
        on=["date", "pair_id"], suffixes=("_strong", "_weak"),
        validate="one_to_one",
    )
    if pairs.industry_strong.eq(pairs.industry_weak).any():
        raise ValueError("Sector-tail states cannot differ within an industry")
    halves = pairs.assign(half=lambda x: x.date.str[:4] + "H" + np.where(
        x.date.str[5:7].astype(int).le(6), "1", "2"))
    segments = halves.groupby("half").agg(
        pairs=("pair_id", "size"), days=("date", "nunique")
    ).reset_index().to_dict("records")
    audit = {
        "resilient_pool": len(strong), "weak_pool": len(weak),
        "capacity_candidates": attempted, "paired": len(pairs),
        "days": signals.date.nunique(),
        "match_fraction": len(pairs) / attempted,
        "median_prior20_gap": float((pairs.return20_prior_adjusted_strong
                                     - pairs.return20_prior_adjusted_weak).abs().median()),
        "median_pre_tail_gap": float((pairs.return_to_1420_strong
                                      - pairs.return_to_1420_weak).abs().median()),
        "median_tail_gap": float((pairs.return_last30_strong
                                  - pairs.return_last30_weak).abs().median()),
        "median_sector_prior_gap": float((pairs.sector_prior20_strong
                                          - pairs.sector_prior20_weak).abs().median()),
        "median_current_gap": float((pairs.return_1450_strong
                                     - pairs.return_1450_weak).abs().median()),
        "median_position_gap": float((pairs.position_1450_strong
                                      - pairs.position_1450_weak).abs().median()),
        "candidate_with_control_after": candidate_with_control_after,
        "by_half": segments,
    }
    audit["outcome_gate_passed"] = (
        len(segments) == 4
        and all(x["pairs"] >= 50 and x["days"] >= 15 for x in segments)
        and audit["match_fraction"] >= .35
    )
    return signals, audit, all_candidates


def freeze(output_dir: Path = OUTPUT) -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    connection.read_parquet(str(ROOT / "intraday_features" / "*.parquet")
                            ).create_view("intraday")
    connection.read_parquet(str(ROOT / "industry_intervals.parquet")
                            ).create_view("industry_history")
    connection.read_parquet(str(ROOT / "late_market_direction" / "states.parquet")
                            ).create_view("market_states")
    signals, audit, all_candidates = select_inputs(connection)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("repricing_signals.parquet", "repriced.parquet", "report.json"):
        (output_dir / stale).unlink(missing_ok=True)
    signals.to_parquet(output_dir / "selections.parquet", index=False,
                       compression="zstd")
    all_candidates.to_parquet(output_dir / "all_candidates.parquet",
                              index=False, compression="zstd")
    if audit["outcome_gate_passed"]:
        connection.register("selected", signals)
        snapshots = connection.execute("""
            SELECT s.* FROM selected x JOIN snapshots s USING (date, code)
        """).df()
        if (len(snapshots) != len(signals)
                or snapshots.duplicated(["date", "code"]).any()):
            raise ValueError("Frozen industry-tail stock lacks a unique snapshot")
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
