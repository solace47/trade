"""Input-only audit of morning five-minute bursts before any outcome lookup."""

from __future__ import annotations

import argparse
from hashlib import md5
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("data/research")
OUTPUT = ROOT / "morning_burst"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")


def build_batch(minute_paths: list[str], output: Path, threads: int = 4) -> None:
    """Extract only morning inputs from raw minutes; no later bars are read."""
    if not minute_paths or threads < 1:
        raise ValueError("Minute sources and a positive thread count are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    try:
        connection.execute(f"SET threads = {threads}")
        connection.execute("SET memory_limit = '8GB'")
        connection.execute("SET preserve_insertion_order = false")
        connection.from_parquet(minute_paths).create_view("source_minutes")
        escaped = str(output).replace("'", "''")
        connection.execute(f"""
            COPY (
                WITH labeled AS (
                    SELECT lower(exchange) || '.' || symbol AS code,
                           strftime(timestamp, '%Y-%m-%d') AS date,
                           strftime(timestamp, '%H%M') AS label,
                           close, high, low, volume, turnover,
                           close > 0 AND high >= GREATEST(low, close)
                               AND low <= close AND volume >= 0
                               AND turnover >= 0 AS valid_bar
                    FROM source_minutes
                    WHERE timestamp >= TIMESTAMP '2024-01-01'
                      AND timestamp < TIMESTAMP '2026-01-01'
                      AND strftime(timestamp, '%H%M') BETWEEN '0930' AND '1130'
                ), rolling AS (
                    SELECT *,
                           LAG(close, 5) OVER win AS price_five_before,
                           LAG(volume, 5) OVER win AS volume_five_before,
                           LAG(turnover, 5) OVER win AS amount_five_before
                    FROM labeled
                    WINDOW win AS (PARTITION BY date, code ORDER BY label)
                )
                SELECT date, code,
                       MAX(close) FILTER (WHERE label = '0930') AS price_0930,
                       MAX(close) FILTER (WHERE label = '1130') AS price_1130,
                       MAX(close / price_five_before - 1) FILTER (
                           WHERE label BETWEEN '0936' AND '1130'
                             AND valid_bar AND price_five_before > 0
                             AND volume > 0 AND turnover > 0
                             AND volume_five_before > 0
                             AND amount_five_before > 0
                       ) AS max5,
                       COUNT(*) AS bar_count,
                       COUNT(DISTINCT label) AS unique_labels,
                       COUNT(*) FILTER (WHERE valid_bar) AS valid_bars
                FROM rolling GROUP BY date, code
                HAVING bar_count = 121 AND unique_labels = 121
                   AND valid_bars = 121 AND price_0930 > 0
                   AND price_1130 > 0 AND max5 IS NOT NULL
            ) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
    finally:
        connection.close()


def _inputs(morning_dir: Path, prefix_dir: Path,
            snapshot_dir: Path) -> pd.DataFrame:
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 4")
        connection.execute("SET memory_limit = '8GB'")
        connection.from_parquet(str(morning_dir / "parts" / "*.parquet")
                                ).create_view("morning")
        connection.from_parquet(str(prefix_dir / "*" / "part_*.parquet")
                                ).create_view("prefix")
        connection.from_parquet(str(snapshot_dir / "*.parquet")
                                ).create_view("snapshots")
        return connection.execute("""
            SELECT m.date, m.code,
                   CASE WHEN m.code LIKE 'sh.60%' OR m.code LIKE 'sz.00%'
                        THEN 'main'
                        WHEN m.code LIKE 'sz.30%' THEN 'chinext'
                        WHEN m.code LIKE 'sh.68%' THEN 'star'
                   END AS board,
                   m.max5, 100 * (m.price_1130 / m.price_0930 - 1)
                       AS morning_pp,
                   100 * (p.price_1449 / s.preclose - 1) AS day_pp,
                   100 * p.return_last29 AS tail_pp,
                   100 * s.return20_prior_adjusted AS prior20_pp,
                   p.price_1449, p.amount_1449
            FROM morning m
            JOIN prefix p USING (date, code)
            JOIN snapshots s USING (date, code)
            WHERE m.date BETWEEN '2024-01-01' AND '2025-12-17'
              AND s.tradestatus = 1 AND s.isST = 0
              AND s.listing_age_sessions >= 20 AND NOT s.reference_gap
              AND NOT p.quote_outside_traded_range
              AND s.preclose > 0 AND p.price_1449 >= 5
              AND p.amount_1449 BETWEEN 100000000 AND 1000000000
              AND s.return20_prior_adjusted BETWEEN -.10 AND .10
              AND p.return_last29 BETWEEN -.01 AND .01
              AND p.price_1449 / s.preclose - 1 BETWEEN 0 AND .03
              AND m.price_1130 / m.price_0930 - 1 BETWEEN 0 AND .03
              AND (m.code LIKE 'sh.60%' OR m.code LIKE 'sz.00%'
                OR m.code LIKE 'sz.30%' OR m.code LIKE 'sh.68%')
        """).df()
    finally:
        connection.close()


def audit(inputs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    if (inputs.empty or inputs.duplicated(["date", "code"]).any()
            or not np.isfinite(inputs[["max5", "morning_pp", "day_pp",
                                      "tail_pp", "prior20_pp", "price_1449",
                                      "amount_1449"]].to_numpy()).all()):
        raise ValueError("Missing, duplicate or nonfinite pre-decision inputs")
    inputs = inputs.copy()
    inputs["half"] = inputs.date.str[:4] + "H" + np.where(
        inputs.date.str[5:7].astype(int) <= 6, "1", "2")
    inputs["group"] = np.select(
        [inputs.max5.ge(.01), inputs.max5.between(0, .005)],
        ["burst", "smooth"], default="other")
    inputs["morning_bin"] = np.floor(inputs.morning_pp / .5).astype(int)
    inputs["day_bin"] = np.floor(inputs.day_pp / .5).astype(int)
    if set(inputs.half) != set(HALVES):
        raise ValueError("Incomplete development periods")
    counts = inputs.groupby(
        ["half", "date", "board", "morning_bin", "day_bin", "group"]
    ).size().unstack("group", fill_value=0)
    for name in ("burst", "smooth"):
        if name not in counts:
            counts[name] = 0
    report = {}
    for half in HALVES:
        part = counts.loc[half]
        both = part.burst.ge(5) & part.smooth.ge(5)
        report[half] = {
            "base_stock_days": int(inputs.half.eq(half).sum()),
            "comparable_strata": int(both.sum()),
            "comparable_days": int(part.index.get_level_values(
                "date")[both].nunique()),
            "burst_inputs": int(part.burst.sum()),
            "smooth_inputs": int(part.smooth.sum()),
        }
    gate = all(x["comparable_strata"] >= 50
               and x["comparable_days"] >= 30
               and x["burst_inputs"] >= 500
               and x["smooth_inputs"] >= 500
               for x in report.values())
    return inputs, {"cutoff": "14:49", "years": [2024, 2025],
                    "input_count": len(inputs), "by_half": report,
                    "outcome_gate_passed": bool(gate)}


def freeze(morning_dir: Path = OUTPUT,
           prefix_dir: Path = ROOT / "minute_prefix_1449",
           snapshot_dir: Path = ROOT / "market_snapshots_ci") -> dict:
    prefix_audit = json.loads((prefix_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if not prefix_audit["input_gate_passed"]:
        raise ValueError("14:49 prefix did not pass its own input audit")
    inputs, report = audit(_inputs(morning_dir, prefix_dir, snapshot_dir))
    morning_dir.mkdir(parents=True, exist_ok=True)
    inputs.to_parquet(morning_dir / "inputs.parquet", index=False,
                      compression="zstd")
    if report["outcome_gate_passed"]:
        selected = inputs.loc[inputs.group.isin(("burst", "smooth"))].copy()
        selected["audit_order"] = [md5(
            ("morning-burst-v1" + row.date + row.code).encode()
        ).hexdigest() for row in selected.itertuples(index=False)]
        sample = selected.sort_values("audit_order").groupby(
            ["half", "group"], sort=True).head(40)
        if len(sample) != 320:
            raise ValueError("Incomplete raw-minute sample")
        sample.drop(columns="audit_order").to_parquet(
            morning_dir / "raw_signals.parquet", index=False,
            compression="zstd")
    else:
        (morning_dir / "raw_signals.parquet").unlink(missing_ok=True)
    (morning_dir / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minute-root", type=Path,
                        default=Path("data/hf/pilot/data/stock_1m"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--prefix-dir", type=Path,
                        default=ROOT / "minute_prefix_1449")
    parser.add_argument("--snapshot-dir", type=Path,
                        default=ROOT / "market_snapshots_ci")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    paths = sorted(str(path) for exchange in ("SH", "SZ")
                   for path in (args.minute_root / exchange).glob("*.parquet"))
    if not paths:
        parser.error("No SH/SZ source minutes found")
    for offset in range(0, len(paths), args.batch_size):
        destination = args.output_dir / "parts" / (
            f"part_{offset // args.batch_size:03d}.parquet")
        build_batch(paths[offset:offset + args.batch_size],
                    destination, args.threads)
        print(f"Wrote {destination}", flush=True)
    print(json.dumps(freeze(args.output_dir, args.prefix_dir,
                            args.snapshot_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
