"""Freeze 14:50 late-decline pairs for a next-morning reversal test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("data/research")
OUTPUT = ROOT / "late_to_open_reversal"
CAPACITY = 5
COOLDOWN = 5


def select_inputs(connection: duckdb.DuckDBPyConnection, *,
                  match_pre_tail: bool = False) -> tuple[pd.DataFrame, dict]:
    connection.execute("""
        CREATE TEMP TABLE eligible AS
        SELECT s.date, s.code, s.return20_prior_adjusted,
               s.return_1450, s.amount_1450, s.price_1450,
               s.position_1450, i.return_last30,
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
    """)
    rank_seed = "late-pretrend-v1" if match_pre_tail else "late-open-v1"
    declines = connection.execute("""
        SELECT * FROM eligible
        WHERE return_last30 BETWEEN -.01 AND -.003
        ORDER BY date, md5(? || date || code), code
    """, [rank_seed]).df()
    rallies = connection.execute("""
        SELECT * FROM eligible
        WHERE return_last30 BETWEEN .003 AND .01
        ORDER BY date, code
    """).df()
    calendar = connection.execute("""
        SELECT DISTINCT date FROM snapshots
        WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
        ORDER BY date
    """).df().date.tolist()
    date_index = {date: index for index, date in enumerate(calendar)}
    rallies_by_date = {date: group.set_index("code", drop=False)
                       for date, group in rallies.groupby("date")}
    last_selected: dict[str, int] = {}
    selected: list[dict] = []
    attempted = 0
    for date, ranked in declines.groupby("date", sort=True):
        if date not in rallies_by_date:
            continue
        index = date_index[date]
        available = rallies_by_date[date].copy()
        daily_picks = 0
        for row in ranked.itertuples(index=False):
            if daily_picks >= CAPACITY:
                break
            if index - last_selected.get(row.code, -1000) <= COOLDOWN:
                continue
            attempted += 1
            daily_picks += 1
            fresh = available.loc[available.code.map(
                lambda code: index - last_selected.get(code, -1000) > COOLDOWN
            )]
            prior_gap = (fresh.return20_prior_adjusted
                         - row.return20_prior_adjusted).abs()
            matched_return = "return_to_1420" if match_pre_tail else "return_1450"
            current_gap = (fresh[matched_return] - getattr(row, matched_return)).abs()
            amount_ratio = fresh.amount_1450 / row.amount_1450
            price_ratio = fresh.price_1450 / row.price_1450
            position_gap = (fresh.position_1450 - row.position_1450).abs()
            matched = fresh.loc[
                prior_gap.le(.03) & current_gap.le(.005)
                & amount_ratio.between(.5, 2) & price_ratio.between(.5, 2)
                & position_gap.le(.30)
            ].copy()
            if matched.empty:
                continue
            matched["distance"] = (
                prior_gap.loc[matched.index] / .03
                + current_gap.loc[matched.index] / .005
                + np.abs(np.log(amount_ratio.loc[matched.index])) / np.log(2)
                + np.abs(np.log(price_ratio.loc[matched.index])) / np.log(2)
                + position_gap.loc[matched.index] / .30
            )
            control = matched.reset_index(drop=True).sort_values(
                ["distance", "code"]
            ).iloc[0]
            down = row._asdict()
            down.update(candidate="late_decline", pair_id=row.code,
                        daily_rank=daily_picks)
            up = control.drop(labels="distance").to_dict()
            up.update(candidate="late_rally_control", pair_id=row.code,
                      daily_rank=daily_picks)
            selected.extend((down, up))
            available = available.drop(index=control.code)
            last_selected[row.code] = index
            last_selected[control.code] = index
    signals = pd.DataFrame(selected)
    if (signals.empty or signals.duplicated(["date", "code"]).any()
            or signals.groupby(["date", "pair_id"]).candidate.nunique().ne(2).any()
            or not signals.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Invalid late-decline input pairs")
    pairs = signals.loc[signals.candidate.eq("late_decline")].merge(
        signals.loc[signals.candidate.eq("late_rally_control")],
        on=["date", "pair_id"], suffixes=("_down", "_up"),
        validate="one_to_one",
    )
    halves = pairs.assign(half=lambda rows: rows.date.str[:4] + "H"
                          + np.where(rows.date.str[5:7].astype(int) <= 6,
                                     "1", "2"))
    by_half = halves.groupby("half").agg(
        pairs=("pair_id", "size"), days=("date", "nunique")
    ).reset_index().to_dict("records")
    audit = {
        "decline_pool": len(declines), "rally_pool": len(rallies),
        "capacity_candidates": attempted, "paired": len(pairs),
        "days": signals.date.nunique(),
        "match_fraction": len(pairs) / attempted,
        "median_prior20_gap": float((pairs.return20_prior_adjusted_down
                                     - pairs.return20_prior_adjusted_up).abs().median()),
        "median_current_gap": float((pairs.return_1450_down
                                     - pairs.return_1450_up).abs().median()),
        "median_pre_tail_gap": float((pairs.return_to_1420_down
                                      - pairs.return_to_1420_up).abs().median()),
        "median_position_gap": float((pairs.position_1450_down
                                      - pairs.position_1450_up).abs().median()),
        "by_half": by_half,
        "matched_return": "return_to_1420" if match_pre_tail else "return_1450",
        "ranking_seed": rank_seed,
    }
    audit["outcome_gate_passed"] = (
        len(by_half) == 4 and all(item["pairs"] >= 50 for item in by_half)
        and (not match_pre_tail or all(item["days"] >= 15 for item in by_half))
        and audit["match_fraction"] >= .35
    )
    return signals, audit


def freeze(output_dir: Path = OUTPUT, *, match_pre_tail: bool = False) -> dict:
    if match_pre_tail and output_dir == OUTPUT:
        raise ValueError("Pre-tail matching requires a separate output directory")
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    connection.read_parquet(str(ROOT / "intraday_features" / "*.parquet")
                            ).create_view("intraday")
    signals, audit = select_inputs(connection, match_pre_tail=match_pre_tail)
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
    parser.add_argument("--output", type=Path)
    parser.add_argument("--match-pre-tail", action="store_true")
    args = parser.parse_args()
    output = args.output or (ROOT / "late_pretrend_match"
                             if args.match_pre_tail else OUTPUT)
    print(freeze(output, match_pre_tail=args.match_pre_tail))


if __name__ == "__main__":
    main()
