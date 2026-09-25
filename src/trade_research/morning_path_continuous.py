"""Freeze a within-day morning-path contrast using only pre-14:49 inputs."""

from __future__ import annotations

import argparse
from hashlib import md5
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("data/research")
SOURCE = ROOT / "morning_burst" / "inputs.parquet"
MORNING = ROOT / "morning_burst" / "parts"
SNAPSHOTS = ROOT / "market_snapshots_ci"
OUTPUT = ROOT / "morning_path_continuous"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")
KEY = ("half", "date", "board", "day_bin", "gap_bin")


def load_inputs(source: Path = SOURCE, morning: Path = MORNING,
                snapshots: Path = SNAPSHOTS) -> pd.DataFrame:
    """Join opening reference and order fields without reading future prices."""
    base = pd.read_parquet(source)
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 4")
        connection.register("base", base)
        connection.from_parquet(str(morning / "*.parquet")).create_view("m")
        connection.from_parquet(str(snapshots / "*.parquet")).create_view("s")
        inputs = connection.execute("""
            SELECT b.date, b.code, b.board, b.max5, b.morning_pp,
                   b.day_pp, b.tail_pp, b.prior20_pp, b.price_1449,
                   b.amount_1449, b.half,
                   100 * (m.price_0930 / s.preclose - 1) AS gap_pp,
                   s.listing_age_sessions, s.isST, s.reference_gap,
                   s.quote_outside_traded_range
            FROM base b JOIN m USING (date, code)
            JOIN s USING (date, code)
            WHERE s.preclose > 0 AND m.price_0930 > 0
        """).df()
    finally:
        connection.close()
    if len(inputs) != len(base):
        raise ValueError("Opening gap join does not cover the frozen base")
    return inputs


def select(inputs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Within day, board, day-return and gap bins, compare path quartiles."""
    needed = {"date", "code", "board", "max5", "morning_pp", "day_pp",
              "tail_pp", "prior20_pp", "price_1449", "amount_1449",
              "gap_pp", "listing_age_sessions", "isST", "reference_gap",
              "quote_outside_traded_range"}
    if not needed.issubset(inputs.columns):
        raise ValueError("Missing frozen pre-decision input")
    if (inputs.empty or inputs.duplicated(["date", "code"]).any()
            or not inputs.date.between("2024-01-01", "2025-12-17").all()
            or not np.isfinite(inputs[["max5", "morning_pp", "day_pp",
                                      "tail_pp", "prior20_pp", "price_1449",
                                      "amount_1449", "gap_pp"]]
                               .to_numpy()).all()):
        raise ValueError("Invalid, duplicate or out-of-period input")
    frame = inputs.loc[:, sorted(needed)].copy()
    frame["half"] = frame.date.str[:4] + "H" + np.where(
        frame.date.str[5:7].astype(int).le(6), "1", "2")
    if set(frame.half) != set(HALVES):
        raise ValueError("Incomplete development periods")
    frame["day_bin"] = np.floor(frame.day_pp / .5).astype(int)
    frame["gap_bin"] = np.floor(frame.gap_pp / .5).astype(int)
    grouped = frame.groupby(list(KEY), observed=True).morning_pp
    strata = grouped.agg(n="size", q25=lambda x: x.quantile(.25),
                         q75=lambda x: x.quantile(.75)).reset_index()
    strata["spread_pp"] = strata.q75 - strata.q25
    eligible = strata.loc[strata.n.ge(8) & strata.spread_pp.ge(.5),
                          list(KEY) + ["n", "spread_pp"]]
    ranked = frame.merge(eligible, on=list(KEY), how="inner",
                         validate="many_to_one")
    ranked = ranked.sort_values(list(KEY) + ["morning_pp", "code"])
    ranked["rank"] = ranked.groupby(list(KEY), observed=True).cumcount()
    ranked["arm_size"] = ranked.n // 4
    ranked["arm"] = np.select(
        [ranked["rank"].lt(ranked.arm_size),
         ranked["rank"].ge(ranked.n - ranked.arm_size)],
        ["afternoon", "morning"], default="middle")
    selected = ranked.loc[ranked.arm.ne("middle")].drop(
        columns=["n", "spread_pp", "rank", "arm_size"])
    counts = selected.groupby(list(KEY) + ["arm"], observed=True).size(
    ).unstack("arm", fill_value=0)
    if (selected.empty or not counts.morning.eq(counts.afternoon).all()
            or counts.morning.lt(2).any()
            or selected.duplicated(["date", "code"]).any()):
        raise ValueError("Unbalanced path contrast")
    by_half = {}
    for half in HALVES:
        part = selected.loc[selected.half.eq(half)]
        base = frame.loc[frame.half.eq(half)]
        high = part.loc[part.arm.eq("morning")]
        low = part.loc[part.arm.eq("afternoon")]
        diffs = {name: float(high[name].mean() - low[name].mean())
                 for name in ("morning_pp", "gap_pp", "day_pp",
                              "prior20_pp", "tail_pp", "max5")}
        by_half[half] = {
            "base_stock_days": int(len(base)),
            "eligible_strata": int(len(eligible.loc[eligible.half.eq(half)])),
            "eligible_stock_days": int(len(ranked.loc[ranked.half.eq(half)])),
            "eligible_base_fraction": float(len(ranked.loc[
                ranked.half.eq(half)]) / len(base)),
            "days": int(part.date.nunique()),
            "signals_each_arm": int(len(high)),
            "input_differences_morning_minus_afternoon": diffs,
        }
    gate = all(
        item["days"] >= 80 and item["signals_each_arm"] >= 1000
        and item["eligible_base_fraction"] >= .25
        and item["input_differences_morning_minus_afternoon"]["morning_pp"] >= .75
        and abs(item["input_differences_morning_minus_afternoon"]["gap_pp"]) <= .10
        and abs(item["input_differences_morning_minus_afternoon"]["day_pp"]) <= .10
        for item in by_half.values()
    )
    report = {
        "cutoff": "14:49", "years": [2024, 2025],
        "source_stock_days": int(len(frame)), "stratum_minimum": 8,
        "minimum_morning_iqr_pp": .5, "day_bin_pp": .5,
        "opening_gap_bin_pp": .5, "arm": "bottom/top floor(n/4)",
        "by_half": by_half, "outcome_gate_passed": bool(gate),
    }
    return selected.sort_values(["date", "code"]).reset_index(drop=True), report


def freeze(output: Path = OUTPUT) -> dict:
    selected, report = select(load_inputs())
    output.mkdir(parents=True, exist_ok=True)
    for stale in ("repricing_signals.parquet", "raw_signals.parquet",
                  "repriced.parquet", "report.json"):
        (output / stale).unlink(missing_ok=True)
    selected.to_parquet(output / "selections.parquet", index=False,
                        compression="zstd")
    if report["outcome_gate_passed"]:
        selected[["date", "code", "listing_age_sessions", "isST",
                  "reference_gap", "quote_outside_traded_range"]].to_parquet(
            output / "repricing_signals.parquet", index=False,
            compression="zstd")
        sample = selected.copy()
        sample["audit_order"] = [md5(
            ("morning-path-continuous-v1" + row.date + row.code).encode()
        ).hexdigest() for row in sample.itertuples(index=False)]
        sample = sample.sort_values("audit_order").groupby(
            ["half", "arm"], sort=True).head(40)
        if len(sample) != 320:
            raise ValueError("Incomplete raw-minute verification sample")
        sample.drop(columns="audit_order").to_parquet(
            output / "raw_signals.parquet", index=False,
            compression="zstd")
    (output / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(freeze(args.output), ensure_ascii=False, indent=2))
