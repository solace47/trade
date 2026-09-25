"""Freeze own-history late-variance surprise pairs before outcome evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("data/research")
CAPACITY = 5
COOLDOWN = 5
BASE = """
    s.isST = 0 AND s.listing_age_sessions >= 20
    AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
    AND s.price_1450 >= 5
    AND s.amount_1450 BETWEEN 100000000 AND 1000000000
    AND s.return20_prior_adjusted BETWEEN -.10 AND .10
    AND s.return_1450 BETWEEN 0 AND .03
    AND s.position_1450 >= .7
    AND i.return_last30 BETWEEN -.003 AND .003
    AND abs(s.price_1450 - i.price_1450) <= .005
    AND abs(s.price_1450 - h.price_1450) <= .005
"""


def select_inputs(connection: duckdb.DuckDBPyConnection,
                  rank_salt: str = "rv-risk-v1") -> tuple[pd.DataFrame, dict]:
    connection.execute("""
        CREATE TEMP TABLE history AS
        SELECT date, code, price_1450,
               realized_variance_last30 AS rv,
               MEDIAN(realized_variance_last30) OVER (
                   PARTITION BY code ORDER BY date
                   ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
               ) AS prior_rv,
               LAG(date, 20) OVER (
                   PARTITION BY code ORDER BY date
               ) AS oldest_prior_date
        FROM variance
        WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
    """)
    connection.execute(f"""
        CREATE TEMP TABLE candidates AS
        SELECT s.date, s.code, s.return_1450, s.position_1450,
               s.return20_prior_adjusted, s.amount_1450, s.price_1450,
               i.return_last30, i.volume_share_last30,
               h.rv, h.prior_rv, h.rv / h.prior_rv AS surprise
        FROM snapshots s JOIN intraday i USING (date, code)
        JOIN history h USING (date, code)
        WHERE ((s.date BETWEEN '2024-01-01' AND '2024-12-17')
            OR (s.date BETWEEN '2025-01-01' AND '2025-12-17'))
          AND {BASE}
          AND h.oldest_prior_date IS NOT NULL
          AND date_diff('day', CAST(h.oldest_prior_date AS DATE),
                              CAST(s.date AS DATE)) <= 45
          AND h.prior_rv > 0
    """)
    high = connection.execute("""
        SELECT * FROM candidates WHERE surprise >= 1.5
        ORDER BY date, md5(? || date || code), code
    """, [rank_salt]).df()
    low = connection.execute("""
        SELECT * FROM candidates WHERE surprise <= 1.0
        ORDER BY date, code
    """).df()
    calendar = connection.execute("""
        SELECT DISTINCT date FROM snapshots
        WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
        ORDER BY date
    """).df().date.tolist()
    date_index = {date: index for index, date in enumerate(calendar)}
    low_by_date = {date: rows.set_index("code", drop=False)
                   for date, rows in low.groupby("date")}
    last_selected: dict[str, int] = {}
    selected: list[dict] = []
    capacity_candidates = 0
    for date, ranked in high.groupby("date", sort=True):
        if date not in low_by_date:
            continue
        market_index = date_index[date]
        available = low_by_date[date].copy()
        daily_picks = 0
        for row in ranked.itertuples(index=False):
            if daily_picks >= CAPACITY:
                break
            if market_index - last_selected.get(row.code, -1000) <= COOLDOWN:
                continue
            capacity_candidates += 1
            fresh = available.loc[
                available.code.map(
                    lambda code: market_index - last_selected.get(code, -1000)
                    > COOLDOWN
                )
            ]
            prior_gap = (fresh.return20_prior_adjusted
                         - row.return20_prior_adjusted).abs()
            current_gap = (fresh.return_1450 - row.return_1450).abs()
            tail_gap = (fresh.return_last30 - row.return_last30).abs()
            amount_ratio = fresh.amount_1450 / row.amount_1450
            position_gap = (fresh.position_1450 - row.position_1450).abs()
            price_ratio = fresh.price_1450 / row.price_1450
            volume_gap = (fresh.volume_share_last30
                          - row.volume_share_last30).abs()
            matches = fresh.loc[
                prior_gap.le(.03) & current_gap.le(.005)
                & tail_gap.le(.002) & amount_ratio.between(.5, 2)
                & position_gap.le(.15) & price_ratio.between(.5, 2)
                & volume_gap.le(.07)
            ].copy()
            if matches.empty:
                continue
            matches["distance"] = (
                prior_gap.loc[matches.index] / .03
                + current_gap.loc[matches.index] / .005
                + tail_gap.loc[matches.index] / .002
                + np.abs(np.log(amount_ratio.loc[matches.index])) / np.log(2)
                + position_gap.loc[matches.index] / .15
                + np.abs(np.log(price_ratio.loc[matches.index])) / np.log(2)
                + volume_gap.loc[matches.index] / .07
            )
            matched = matches.reset_index(drop=True).sort_values(
                ["distance", "code"]
            ).iloc[0]
            high_row = row._asdict()
            high_row.update(candidate="high_variance", pair_id=row.code,
                            daily_rank=daily_picks + 1)
            low_row = matched.drop(labels="distance").to_dict()
            low_row.update(candidate="normal_variance_control",
                           pair_id=row.code, daily_rank=daily_picks + 1)
            selected.extend((high_row, low_row))
            available = available.drop(index=matched.code)
            last_selected[row.code] = market_index
            last_selected[matched.code] = market_index
            daily_picks += 1
    signals = pd.DataFrame(selected)
    if (signals.empty or signals.duplicated(["date", "code"]).any()
            or signals.groupby(["date", "pair_id"]).candidate.nunique().ne(2).any()
            or not signals.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Invalid late-variance selection")
    pairs = signals.loc[signals.candidate.eq("high_variance")].merge(
        signals.loc[signals.candidate.eq("normal_variance_control")],
        on=["date", "pair_id"], suffixes=("_high", "_low"),
        validate="one_to_one",
    )
    audit = {
        "high_pool": len(high), "low_pool": len(low),
        "capacity_candidates": capacity_candidates,
        "paired": len(pairs), "days": signals.date.nunique(),
        "match_fraction": len(pairs) / capacity_candidates,
        "median_abs_prior20_gap": float((
            pairs.return20_prior_adjusted_high
            - pairs.return20_prior_adjusted_low).abs().median()),
        "median_abs_current_gap": float((
            pairs.return_1450_high - pairs.return_1450_low).abs().median()),
        "median_abs_last30_gap": float((
            pairs.return_last30_high - pairs.return_last30_low).abs().median()),
        "by_half": signals.loc[signals.candidate.eq("high_variance")]
            .assign(half=lambda rows: rows.date.str[:4] + "H"
                    + np.where(rows.date.str[5:7].astype(int) <= 6, "1", "2"))
            .groupby("half").agg(pairs=("code", "size"),
                                 days=("date", "nunique"))
            .reset_index().to_dict("records"),
    }
    return signals, audit


def freeze(output_dir: Path = ROOT / "late_variance_risk") -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    connection.read_parquet(str(ROOT / "intraday_features" / "*.parquet")
                            ).create_view("intraday")
    connection.read_parquet(str(ROOT / "late_variance" / "*.parquet")
                            ).create_view("variance")
    signals, audit = select_inputs(connection)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_candidates = connection.execute("SELECT * FROM candidates").df()
    all_candidates.to_parquet(output_dir / "all_candidates.parquet",
                              index=False, compression="zstd")
    signals.to_parquet(output_dir / "selections.parquet", index=False,
                       compression="zstd")
    connection.register("selected", signals)
    repricing_signals = connection.execute("""
        SELECT s.* FROM selected r JOIN snapshots s USING (date, code)
    """).df()
    if (len(repricing_signals) != len(signals)
            or repricing_signals.duplicated(["date", "code"]).any()):
        raise ValueError("Repricing inputs lack one-to-one snapshot coverage")
    repricing_signals.to_parquet(
        output_dir / "repricing_signals.parquet", index=False,
        compression="zstd",
    )
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "late_variance_risk")
    args = parser.parse_args()
    print(freeze(args.output))


if __name__ == "__main__":
    main()
