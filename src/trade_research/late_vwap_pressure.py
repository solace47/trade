"""Freeze same-day pairs for a 14:50 tail VWAP-dislocation study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .late_to_open_reversal import CAPACITY, COOLDOWN, ROOT, _board


OUTPUT = ROOT / "late_vwap_pressure"
TREATMENT = "vwap_discount"
CONTROL = "near_vwap_control"


def select_inputs(connection: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict]:
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE eligible_vwap AS
        SELECT s.date, s.code, s.return20_prior_adjusted,
               s.return_1450, s.amount_1450, s.price_1450,
               s.position_1450, i.return_last30, i.return_last15,
               i.premium_to_last30_vwap,
               (1 + s.return_1450) / (1 + i.return_last30) - 1
                   AS return_to_1420
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
          AND i.return_last15 IS NOT NULL
          AND i.premium_to_last30_vwap IS NOT NULL
    """)
    extreme = connection.execute("""
        SELECT * FROM eligible_vwap
        WHERE premium_to_last30_vwap <= -.003
        ORDER BY date, md5('late-vwap-v1' || date || code), code
    """).df()
    near = connection.execute("""
        SELECT * FROM eligible_vwap
        WHERE premium_to_last30_vwap BETWEEN -.001 AND 0
        ORDER BY date, code
    """).df()
    for frame in (extreme, near):
        frame["board"] = frame.code.map(_board)
    extreme = extreme.loc[extreme.board.notna()].copy()
    near = near.loc[near.board.notna()].copy()
    calendar = connection.execute("""
        SELECT DISTINCT date FROM snapshots
        WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
        ORDER BY date
    """).df().date.tolist()
    date_index = {date: index for index, date in enumerate(calendar)}
    near_by_date = {date: group.set_index("code", drop=False)
                    for date, group in near.groupby("date")}
    last_selected: dict[str, int] = {}
    selected: list[dict] = []
    attempted = 0
    for date, ranked in extreme.groupby("date", sort=True):
        if date not in near_by_date:
            continue
        index = date_index[date]
        available = near_by_date[date].copy()
        daily_picks = 0
        for row in ranked.itertuples(index=False):
            if daily_picks >= CAPACITY:
                break
            if index - last_selected.get(row.code, -1000) <= COOLDOWN:
                continue
            attempted += 1
            daily_picks += 1
            fresh = available.loc[
                available.board.eq(row.board)
                & available.code.map(
                    lambda code: index - last_selected.get(code, -1000) > COOLDOWN
                )
            ]
            gaps = {
                "prior20": (fresh.return20_prior_adjusted
                            - row.return20_prior_adjusted).abs(),
                "pre_tail": (fresh.return_to_1420 - row.return_to_1420).abs(),
                "tail30": (fresh.return_last30 - row.return_last30).abs(),
                "tail15": (fresh.return_last15 - row.return_last15).abs(),
                "day": (fresh.return_1450 - row.return_1450).abs(),
                "position": (fresh.position_1450 - row.position_1450).abs(),
            }
            amount_ratio = fresh.amount_1450 / row.amount_1450
            price_ratio = fresh.price_1450 / row.price_1450
            limits = {"prior20": .03, "pre_tail": .005, "tail30": .002,
                      "tail15": .002, "day": .005, "position": .30}
            mask = amount_ratio.between(.5, 2) & price_ratio.between(.5, 2)
            for name, limit in limits.items():
                mask &= gaps[name].le(limit)
            matched = fresh.loc[mask].copy()
            if matched.empty:
                continue
            matched["distance"] = sum(
                gaps[name].loc[matched.index] / limit
                for name, limit in limits.items()
            ) + (
                np.abs(np.log(amount_ratio.loc[matched.index]))
                + np.abs(np.log(price_ratio.loc[matched.index]))
            ) / np.log(2)
            control = matched.reset_index(drop=True).sort_values(
                ["distance", "code"]
            ).iloc[0]
            down = row._asdict()
            down.update(candidate=TREATMENT, pair_id=row.code,
                        daily_rank=daily_picks)
            flat = control.drop(labels="distance").to_dict()
            flat.update(candidate=CONTROL, pair_id=row.code,
                        daily_rank=daily_picks)
            selected.extend((down, flat))
            available = available.drop(index=control.code)
            last_selected[row.code] = index
            last_selected[control.code] = index
    signals = pd.DataFrame(selected)
    if (signals.empty or signals.duplicated(["date", "code"]).any()
            or signals.groupby(["date", "pair_id"]).candidate.nunique().ne(2).any()
            or not signals.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Invalid VWAP-dislocation input pairs")
    pairs = signals.loc[signals.candidate.eq(TREATMENT)].merge(
        signals.loc[signals.candidate.eq(CONTROL)],
        on=["date", "pair_id"], suffixes=("_low", "_near"),
        validate="one_to_one",
    )
    halves = pairs.assign(half=lambda rows: rows.date.str[:4] + "H"
                          + np.where(rows.date.str[5:7].astype(int) <= 6,
                                     "1", "2"))
    by_half = halves.groupby("half").agg(
        pairs=("pair_id", "size"), signal_days=("date", "nunique")
    ).reset_index().to_dict("records")
    audit = {
        "extreme_pool": len(extreme), "near_pool": len(near),
        "capacity_candidates": attempted, "paired": len(pairs),
        "match_fraction": len(pairs) / attempted,
        "by_half": by_half,
        "same_board_fraction": float(pairs.board_low.eq(pairs.board_near).mean()),
        "median_vwap_gap": float((pairs.premium_to_last30_vwap_low
                                  - pairs.premium_to_last30_vwap_near).abs().median()),
        "median_tail15_gap": float((pairs.return_last15_low
                                    - pairs.return_last15_near).abs().median()),
        "median_tail30_gap": float((pairs.return_last30_low
                                    - pairs.return_last30_near).abs().median()),
        "median_pre_tail_gap": float((pairs.return_to_1420_low
                                      - pairs.return_to_1420_near).abs().median()),
        "treatment_label": TREATMENT, "control_label": CONTROL,
    }
    audit["outcome_gate_passed"] = (
        len(by_half) == 4
        and all(item["pairs"] >= 50 and item["signal_days"] >= 15
                for item in by_half)
        and audit["match_fraction"] >= .35
        and audit["same_board_fraction"] == 1.0
        and (pairs.premium_to_last30_vwap_low <= -.003).all()
        and pairs.premium_to_last30_vwap_near.between(-.001, 0).all()
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
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(freeze(args.output))


if __name__ == "__main__":
    main()
