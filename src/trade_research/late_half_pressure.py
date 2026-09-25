"""Freeze same-day late-half price-pressure pairs using only 14:50 inputs."""

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
    AND i.return_last30 BETWEEN 0 AND .003
    AND abs(s.price_1450 - i.price_1450) <= .005
"""


def select_inputs(connection: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict]:
    """Pair terminal pullbacks with terminal surges before reading outcomes."""
    connection.execute(f"""
        CREATE TEMP TABLE candidates AS
        SELECT s.date, s.code, s.return_1450, s.position_1450,
               s.return20_prior_adjusted, s.amount_1450,
               i.return_last30, i.return_last15,
               i.volume_share_last30
        FROM snapshots s JOIN intraday i USING (date, code)
        WHERE ((s.date BETWEEN '2024-01-01' AND '2024-12-17')
            OR (s.date BETWEEN '2025-01-01' AND '2025-12-17'))
          AND {BASE}
    """)
    down = connection.execute("""
        SELECT * FROM candidates WHERE return_last15 <= -.0015
        ORDER BY date, return_last15, code
    """).df()
    up = connection.execute("""
        SELECT * FROM candidates WHERE return_last15 >= .0015
        ORDER BY date, code
    """).df()
    calendar = connection.execute("""
        SELECT DISTINCT date FROM snapshots
        WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
        ORDER BY date
    """).df().date.tolist()
    date_index = {date: index for index, date in enumerate(calendar)}
    up_by_date = {date: rows.set_index("code", drop=False)
                  for date, rows in up.groupby("date")}
    last_selected: dict[str, int] = {}
    selected: list[dict] = []
    capacity_candidates = 0
    for date, ranked in down.groupby("date", sort=True):
        if date not in up_by_date:
            continue
        market_index = date_index[date]
        available = up_by_date[date].copy()
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
            volume_gap = (fresh.volume_share_last30
                          - row.volume_share_last30).abs()
            matches = fresh.loc[
                prior_gap.le(.03) & current_gap.le(.005)
                & tail_gap.le(.00075) & amount_ratio.between(.5, 2)
                & position_gap.le(.15) & volume_gap.le(.07)
            ].copy()
            if matches.empty:
                continue
            matches["distance"] = (
                prior_gap.loc[matches.index] / .03
                + current_gap.loc[matches.index] / .005
                + tail_gap.loc[matches.index] / .00075
                + np.abs(np.log(amount_ratio.loc[matches.index])) / np.log(2)
                + position_gap.loc[matches.index] / .15
                + volume_gap.loc[matches.index] / .07
            )
            matched = matches.reset_index(drop=True).sort_values(
                ["distance", "code"]
            ).iloc[0]
            down_row = row._asdict()
            down_row.update(candidate="late_pullback", pair_id=row.code,
                            daily_rank=daily_picks + 1)
            up_row = matched.drop(labels="distance").to_dict()
            up_row.update(candidate="late_surge_control", pair_id=row.code,
                          daily_rank=daily_picks + 1)
            selected.extend((down_row, up_row))
            available = available.drop(index=matched.code)
            last_selected[row.code] = market_index
            last_selected[matched.code] = market_index
            daily_picks += 1
    signals = pd.DataFrame(selected)
    if (signals.empty or signals.duplicated(["date", "code"]).any()
            or signals.groupby(["date", "pair_id"]).candidate.nunique().ne(2).any()
            or not signals.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Invalid late-half pressure pairs")
    pairs = signals.loc[signals.candidate.eq("late_pullback")].merge(
        signals.loc[signals.candidate.eq("late_surge_control")],
        on=["date", "pair_id"], suffixes=("_down", "_up"),
        validate="one_to_one",
    )
    audit = {
        "down_pool": len(down), "up_pool": len(up),
        "capacity_candidates": capacity_candidates,
        "paired": len(pairs), "days": signals.date.nunique(),
        "match_fraction": len(pairs) / capacity_candidates,
        "median_abs_prior20_gap": float((
            pairs.return20_prior_adjusted_down
            - pairs.return20_prior_adjusted_up).abs().median()),
        "median_abs_current_gap": float((
            pairs.return_1450_down - pairs.return_1450_up).abs().median()),
        "median_abs_last30_gap": float((
            pairs.return_last30_down - pairs.return_last30_up).abs().median()),
        "by_half": signals.loc[signals.candidate.eq("late_pullback")]
            .assign(half=lambda rows: rows.date.str[:4] + "H"
                    + np.where(rows.date.str[5:7].astype(int) <= 6, "1", "2"))
            .groupby("half").agg(pairs=("code", "size"),
                                 days=("date", "nunique"))
            .reset_index().to_dict("records"),
    }
    audit["outcome_gate_passed"] = (
        audit["match_fraction"] >= .5
        and len(audit["by_half"]) == 4
        and all(row["pairs"] >= 50 for row in audit["by_half"])
    )
    return signals, audit


def freeze(output_dir: Path = ROOT / "late_half_pressure") -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    connection.read_parquet(str(ROOT / "intraday_features" / "*.parquet")
                            ).create_view("intraday")
    signals, audit = select_inputs(connection)
    output_dir.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(output_dir / "selections.parquet", index=False,
                       compression="zstd")
    repricing_path = output_dir / "repricing_signals.parquet"
    if audit["outcome_gate_passed"]:
        connection.register("selected", signals)
        repricing_signals = connection.execute("""
            SELECT s.* FROM selected r JOIN snapshots s USING (date, code)
        """).df()
        if (len(repricing_signals) != len(signals)
                or repricing_signals.duplicated(["date", "code"]).any()):
            raise ValueError("Repricing inputs lack one-to-one snapshot coverage")
        repricing_signals.to_parquet(repricing_path, index=False,
                                     compression="zstd")
    else:
        repricing_path.unlink(missing_ok=True)
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "late_half_pressure")
    args = parser.parse_args()
    print(freeze(args.output))


if __name__ == "__main__":
    main()
