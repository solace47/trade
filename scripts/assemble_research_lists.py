"""Assemble reproducible 2024-2025 signal and membership files.

Run exploratory_signals for both random controls, low_amount_neutral and
mid_amount_prior_loser first; run strategy_scan before the exit candidates.
"""

import argparse
from pathlib import Path

import duckdb
import pandas as pd


def assemble(root: Path, year: int, kind: str) -> tuple[int, int]:
    if year not in (2024, 2025) or kind not in ("random", "exit"):
        raise ValueError("Only recent random or exit lists are supported")
    parts = []
    if kind == "random":
        for candidate, filename in (
            ("random_low_amount", "random_low_all.parquet"),
            ("random_liquid", "random_liquid_all.parquet"),
        ):
            source = pd.read_parquet(root / filename)
            frame = source.loc[source.date.str.startswith(str(year))].copy()
            parts.append((candidate, frame))
        mapping = pd.concat(
            [frame[["date", "code"]].assign(candidate=name)
             for name, frame in parts], ignore_index=True
        )
        signals = pd.concat([frame for _, frame in parts], ignore_index=True)
    else:
        strategies = pd.read_parquet(root / "strategy_scan_trades.parquet")
        strategies = strategies.loc[
            strategies.date.str.startswith(str(year))
            & strategies.horizon.eq(1)
            & strategies.candidate.isin((
                "candidate_trend_pullback", "candidate_quiet_trend"
            )), ["date", "code", "candidate"]
        ]
        pieces = [strategies]
        for candidate, filename in (
            ("low_amount_neutral", "size_signal_neutral.parquet"),
            ("mid_amount_prior_loser", "mid_loser_all.parquet"),
        ):
            source = pd.read_parquet(root / filename)
            frame = source.loc[source.date.str.startswith(str(year)),
                               ["date", "code"]].copy()
            pieces.append(frame.assign(candidate=candidate))
        mapping = pd.concat(pieces, ignore_index=True)
        keys = mapping[["date", "code"]].drop_duplicates()
        connection = duckdb.connect()
        connection.read_parquet(str(root / "market_snapshots_ci" / "*.parquet")
                                ).create_view("snapshots")
        connection.register("keys", keys)
        signals = connection.execute("""
            SELECT s.* FROM snapshots s JOIN keys USING (date, code)
        """).df()
    if mapping.duplicated(["date", "code", "candidate"]).any():
        raise ValueError("Duplicate candidate membership")
    signals = signals.drop_duplicates(["date", "code"])
    if len(signals) != len(mapping[["date", "code"]].drop_duplicates()):
        raise ValueError("A selected signal is absent from the snapshots")
    signals.to_parquet(root / f"{kind}_signals_{year}.parquet",
                       index=False, compression="zstd")
    mapping.to_parquet(root / f"{kind}_candidate_map_{year}.parquet",
                       index=False, compression="zstd")
    return len(signals), len(mapping)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("random", "exit"), required=True)
    parser.add_argument("--year", type=int, choices=(2024, 2025), required=True)
    parser.add_argument("--research-root", type=Path,
                        default=Path("data/research"))
    args = parser.parse_args()
    signals, memberships = assemble(args.research_root, args.year, args.kind)
    print({"kind": args.kind, "year": args.year,
           "unique_signals": signals, "memberships": memberships})


if __name__ == "__main__":
    main()
