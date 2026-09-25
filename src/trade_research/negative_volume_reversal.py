"""Frozen 14:49 input and executed-return test for down-day volume surprise."""

from __future__ import annotations

import argparse
from hashlib import md5
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .market_study import _quality_keys, _quality_symbols
from .tail_vwap_decomposition import _net_with_slippage


ROOT = Path("data/research")
OUTPUT = ROOT / "negative_volume_reversal"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")
SEED = 20260926


def _half(dates: pd.Series) -> pd.Series:
    return dates.str[:4] + "H" + np.where(
        dates.str[5:7].astype(int) <= 6, "1", "2")


def _inputs(prefix_dir: Path, snapshot_dir: Path) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 4")
        connection.execute("SET memory_limit = '8GB'")
        connection.from_parquet(str(prefix_dir / "*" / "part_*.parquet")
                                ).create_view("prefix")
        connection.from_parquet(str(snapshot_dir / "*.parquet")
                                ).create_view("snapshots")
        return connection.execute("""
            WITH history AS (
                SELECT date, code, price_1449, price_1420, preclose,
                       volume_1449, amount_1449, return_last29,
                       quote_outside_traded_range,
                       MEDIAN(volume_1449) OVER prior20 AS volume_median20,
                       COUNT(*) OVER prior20 AS history_days,
                       MIN(CAST(date AS DATE)) OVER prior20 AS oldest_day
                FROM (
                    SELECT p.*, s.preclose
                    FROM prefix p JOIN snapshots s USING (date, code)
                    WHERE p.date BETWEEN '2024-01-01' AND '2025-12-17'
                )
                WINDOW prior20 AS (
                    PARTITION BY code ORDER BY date
                    ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                )
            )
            SELECT h.date, h.code,
                   CASE WHEN h.code LIKE 'sh.60%' OR h.code LIKE 'sz.00%'
                        THEN 'main'
                        WHEN h.code LIKE 'sz.30%' THEN 'chinext'
                        WHEN h.code LIKE 'sh.68%' THEN 'star'
                   END AS board,
                   h.price_1449, h.amount_1449, h.volume_1449,
                   h.volume_median20, h.history_days,
                   100 * (h.price_1449 / h.preclose - 1) AS day_pp,
                   100 * h.return_last29 AS tail_pp,
                   100 * s.return20_prior_adjusted AS prior20_pp,
                   h.volume_1449 / h.volume_median20 AS volume_ratio
            FROM history h JOIN snapshots s USING (date, code)
            WHERE h.history_days = 20
              AND date_diff('day', h.oldest_day, CAST(h.date AS DATE)) <= 45
              AND h.volume_median20 > 0
              AND s.tradestatus = 1 AND s.isST = 0
              AND s.listing_age_sessions >= 20 AND NOT s.reference_gap
              AND NOT h.quote_outside_traded_range
              AND h.preclose > 0 AND h.price_1449 >= 5
              AND h.amount_1449 BETWEEN 100000000 AND 1000000000
              AND s.return20_prior_adjusted BETWEEN -.10 AND .10
              AND h.return_last29 BETWEEN -.02 AND .02
              AND h.price_1449 / h.preclose - 1 BETWEEN -.03 AND -.005
              AND (h.code LIKE 'sh.60%' OR h.code LIKE 'sz.00%'
                OR h.code LIKE 'sz.30%' OR h.code LIKE 'sh.68%')
        """).df()
    finally:
        connection.close()


def _audit_sample(inputs: pd.DataFrame) -> pd.DataFrame:
    selected = inputs.loc[inputs.group.isin(("high", "normal"))].copy()
    selected["audit_order"] = [md5(
        ("negative-volume-v1" + row.date + row.code).encode()
    ).hexdigest() for row in selected.itertuples(index=False)]
    sample = selected.sort_values("audit_order").groupby(
        ["half", "group"], sort=True).head(40)
    if (len(sample) != 320 or sample.groupby(["half", "group"])
            .size().ne(40).any()):
        raise ValueError("Incomplete input-only raw-minute sample")
    return sample.drop(columns="audit_order").sort_values(
        ["half", "group", "date", "code"])


def freeze(prefix_dir: Path = ROOT / "minute_prefix_1449",
           snapshot_dir: Path = ROOT / "market_snapshots_ci",
           output_dir: Path = OUTPUT) -> dict:
    source_audit = json.loads((prefix_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if not source_audit["input_gate_passed"]:
        raise ValueError("14:49 prefix did not pass its own audit")
    inputs = _inputs(prefix_dir, snapshot_dir)
    if (inputs.empty or inputs.duplicated(["date", "code"]).any()
            or not np.isfinite(inputs[["day_pp", "tail_pp", "prior20_pp",
                                      "volume_ratio"]].to_numpy()).all()):
        raise ValueError("Missing or duplicate pre-decision inputs")
    inputs["half"] = _half(inputs.date)
    inputs["group"] = np.select(
        [inputs.volume_ratio.ge(1.5), inputs.volume_ratio.between(.8, 1.2)],
        ["high", "normal"], default="other")
    inputs["return_bin"] = np.floor((inputs.day_pp + 3) / .5).astype(int)
    if set(inputs.half) != set(HALVES):
        raise ValueError("Incomplete development periods")
    counts = inputs.groupby(
        ["half", "date", "board", "return_bin", "group"]
    ).size().unstack("group", fill_value=0)
    for name in ("high", "normal"):
        if name not in counts:
            counts[name] = 0
    comparable = counts.high.ge(5) & counts.normal.ge(5)
    by_half = {}
    for half in HALVES:
        part = counts.loc[half]
        both = comparable.loc[half]
        segment = inputs.loc[inputs.half.eq(half)]
        by_half[half] = {
            "candidates": len(segment),
            "days": int(segment.date.nunique()),
            "comparable_strata": int(both.sum()),
            "comparable_days": int(part.index.get_level_values(
                "date")[both].nunique()),
            "high_inputs": int(part.high.sum()),
            "normal_inputs": int(part.normal.sum()),
        }
    gate = all(row["comparable_strata"] >= 100
               and row["comparable_days"] >= 60
               and row["high_inputs"] >= 1000
               and row["normal_inputs"] >= 1000
               for row in by_half.values())
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs.to_parquet(output_dir / "inputs.parquet", index=False,
                      compression="zstd")
    if gate:
        _audit_sample(inputs).to_parquet(
            output_dir / "raw_signals.parquet", index=False,
            compression="zstd")
    else:
        (output_dir / "raw_signals.parquet").unlink(missing_ok=True)
    audit = {"cutoff": "14:49", "years": [2024, 2025],
             "input_count": len(inputs), "by_half": by_half,
             "outcome_gate_passed": bool(gate),
             "raw_sample_count": 320 if gate else 0}
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze",))
    args = parser.parse_args()
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
