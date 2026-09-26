"""Frozen 14:49 noon-down/recovery comparison; see docs/input-gates.md.

The minute vendor has not documented whether timestamps mark minute starts or
ends. All three feature labels finish well before the intended decision time
under either convention. This module reads no next-day outcomes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("data/research")
OUTPUT = ROOT / "noon_recovery_1449"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")
FEATURES = ("morning_return", "day_return", "return_last29",
            "return20_prior_adjusted")
LIMITS = (.005, .005, .002, .05)


def _half(dates: pd.Series) -> pd.Series:
    return dates.str[:4] + "H" + np.where(
        dates.str[5:7].astype(int).le(6), "1", "2")


def _connect() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '8GB'")
    connection.execute("SET preserve_insertion_order = false")
    return connection


def extract_bars(minute_root: Path = Path("data/hf/pilot/data/stock_1m"),
                 output: Path = OUTPUT, batch_size: int = 256) -> dict:
    """Materialize only the three decision-time source labels, never outcomes."""
    files = sorted((*minute_root.glob("SH/*.parquet"),
                    *minute_root.glob("SZ/*.parquet")))
    if not files or batch_size < 1:
        raise ValueError("Missing minute source or invalid batch size")
    bar_dir = output / "bars"
    bar_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("top_signals.parquet", "pairs.parquet", "input_audit.json",
                  "gross_rows.parquet", "gross_report.json"):
        (output / stale).unlink(missing_ok=True)
    for stale in bar_dir.glob("part_*.parquet"):
        stale.unlink()
    connection = _connect()
    try:
        connection.execute("SET temp_directory = ?", [str(output / "duckdb_tmp")])
        for index, start in enumerate(range(0, len(files), batch_size)):
            chunk = [str(path) for path in files[start:start + batch_size]]
            connection.from_parquet(chunk).create_view("source_minutes", replace=True)
            destination = str(bar_dir / f"part_{index:03d}.parquet").replace("'", "''")
            connection.execute(f"""
                COPY (
                    WITH labeled AS (
                        SELECT strftime(timestamp, '%Y-%m-%d') AS date,
                               lower(exchange) || '.' || symbol AS code,
                               strftime(timestamp, '%H%M') AS label,
                               open, high, low, close, volume, turnover,
                               open > 0 AND high >= GREATEST(open, close, low)
                               AND low > 0 AND low <= LEAST(open, close, high)
                               AND close > 0 AND volume > 0 AND turnover > 0
                                   AS valid
                        FROM source_minutes
                        WHERE ((timestamp >= TIMESTAMP '2024-01-01'
                                AND timestamp < TIMESTAMP '2025-01-01')
                            OR (timestamp >= TIMESTAMP '2025-01-01'
                                AND timestamp < TIMESTAMP '2026-01-01'))
                          AND strftime(timestamp, '%H%M') IN
                              ('1130', '1301', '1449')
                    )
                    SELECT date, code,
                           COUNT(*) FILTER (WHERE label = '1130') AS n_1130,
                           COUNT(*) FILTER (WHERE label = '1301') AS n_1301,
                           COUNT(*) FILTER (WHERE label = '1449') AS n_1449,
                           MAX(close) FILTER (WHERE label = '1130') AS price_1130,
                           MAX(close) FILTER (WHERE label = '1301') AS price_1301,
                           MAX(close) FILTER (WHERE label = '1449') AS raw_price_1449,
                           BOOL_AND(valid) FILTER (WHERE label = '1130') AS valid_1130,
                           BOOL_AND(valid) FILTER (WHERE label = '1301') AS valid_1301,
                           BOOL_AND(valid) FILTER (WHERE label = '1449') AS valid_1449
                    FROM labeled GROUP BY date, code
                ) TO '{destination}' (FORMAT PARQUET, COMPRESSION ZSTD)
            """)
    finally:
        connection.close()
    return {"source_files": len(files), "bar_parts": index + 1,
            "years": [2024, 2025]}


def _load_inputs(output: Path, prefix_dir: Path, snapshot_dir: Path
                 ) -> tuple[pd.DataFrame, dict]:
    connection = _connect()
    try:
        connection.from_parquet(str(prefix_dir / "*" / "part_*.parquet")
                                ).create_view("prefix")
        connection.from_parquet(str(snapshot_dir / "*.parquet")
                                ).create_view("snapshots")
        connection.from_parquet(str(output / "bars" / "part_*.parquet")
                                ).create_view("bars")
        connection.execute("""
            CREATE TEMP VIEW base AS
            SELECT p.date, p.code, LEFT(p.code, 2) AS exchange,
                   p.price_1449, p.amount_1449, p.return_last29,
                   s.preclose, s.return20_prior_adjusted
            FROM prefix p JOIN snapshots s USING (date, code)
            WHERE ((p.date BETWEEN '2024-01-01' AND '2024-12-17')
                OR (p.date BETWEEN '2025-01-01' AND '2025-12-17'))
              AND (p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%')
              AND s.isST = 0 AND s.tradestatus = 1
              AND s.listing_age_sessions >= 60 AND NOT s.reference_gap
              AND NOT p.quote_outside_traded_range AND s.preclose > 0
              AND p.price_1449 >= 5
              AND p.amount_1449 BETWEEN 100000000 AND 1000000000
              AND s.return20_prior_adjusted BETWEEN -.10 AND .10
              AND p.return_last29 BETWEEN -.01 AND .01
              AND p.price_1449 / s.preclose - 1 BETWEEN -.03 AND .03
        """)
        coverage = connection.execute("""
            SELECT LEFT(a.date, 4) || 'H' ||
                   CASE WHEN SUBSTR(a.date, 6, 2) <= '06' THEN '1' ELSE '2' END
                       AS half,
                   COUNT(*) AS base_rows,
                   COUNT(*) FILTER (WHERE b.n_1130 >= 1 AND b.n_1301 >= 1
                                      AND b.n_1449 >= 1) AS source_rows,
                   COUNT(*) FILTER (WHERE b.n_1130 = 1 AND b.n_1301 = 1
                                      AND b.n_1449 = 1) AS unique_rows,
                   COUNT(*) FILTER (WHERE b.n_1449 = 1 AND b.valid_1449
                                      AND ABS(a.price_1449 - b.raw_price_1449)
                                          > .005) AS price_mismatch_rows
            FROM base a LEFT JOIN bars b USING (date, code)
            GROUP BY half ORDER BY half
        """).df()
        inputs = connection.execute("""
            SELECT a.date, a.code, a.exchange, a.price_1449, a.amount_1449,
                   a.return_last29, a.return20_prior_adjusted,
                   b.price_1130 / a.preclose - 1 AS morning_return,
                   a.price_1449 / a.preclose - 1 AS day_return,
                   b.price_1301 / b.price_1130 - 1 AS noon_jump,
                   a.price_1449 / b.price_1130 - 1 AS afternoon_recovery
            FROM base a JOIN bars b USING (date, code)
            WHERE b.n_1130 = 1 AND b.n_1301 = 1 AND b.n_1449 = 1
              AND b.valid_1130 AND b.valid_1301 AND b.valid_1449
              AND ABS(a.price_1449 - b.raw_price_1449) <= .005
        """).df()
    finally:
        connection.close()
    if (inputs.empty or inputs.duplicated(["date", "code"]).any()
            or not inputs.date.str[:4].isin(("2024", "2025")).all()
            or not np.isfinite(inputs.select_dtypes("number").to_numpy()).all()
            or set(coverage.half) != set(HALVES)):
        raise ValueError("Missing, duplicate or nonfinite decision-time inputs")
    inputs["half"] = _half(inputs.date)
    source_report = coverage.set_index("half").to_dict(orient="index")
    return inputs, source_report


def _match(inputs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    gap = inputs.noon_jump.between(-.015, -.004)
    recovered = gap & inputs.afternoon_recovery.ge(-.001)
    signals = inputs.loc[recovered].sort_values(
        ["date", "noon_jump", "code"])
    top = signals.groupby("date", sort=False).head(5).copy()
    neutral = inputs.loc[inputs.noon_jump.abs().le(.001)]
    pools = {(date, exchange): part.set_index("code", drop=False)
             for (date, exchange), part in neutral.groupby(
                 ["date", "exchange"], sort=False)}
    pairs: list[dict] = []
    for date, group in top.groupby("date", sort=True):
        used: set[str] = set()
        for rank, row in enumerate(group.itertuples(index=False), 1):
            pool = pools.get((date, row.exchange))
            if pool is None or pool.empty:
                continue
            distance_parts = [
                (pool[name] - getattr(row, name)).abs() / limit
                for name, limit in zip(FEATURES, LIMITS, strict=True)
            ]
            price_ratio = pool.price_1449 / row.price_1449
            amount_ratio = pool.amount_1449 / row.amount_1449
            allowed = (pd.concat(distance_parts, axis=1).le(1).all(axis=1)
                       & price_ratio.between(.5, 2)
                       & amount_ratio.between(.5, 2)
                       & ~pool.code.isin(used))
            if not allowed.any():
                continue
            eligible = pool.loc[allowed].copy()
            eligible["distance"] = sum(part.loc[allowed]
                                        for part in distance_parts)
            eligible["distance"] += (
                np.abs(np.log(price_ratio.loc[allowed]))
                + np.abs(np.log(amount_ratio.loc[allowed]))) / np.log(2)
            control = eligible.reset_index(drop=True).sort_values(
                ["distance", "code"]).iloc[0]
            used.add(control.code)
            pair_id = date + ":" + row.code
            for arm, item in (("signal", row._asdict()),
                              ("control", control.drop("distance").to_dict())):
                pairs.append(dict(item, arm=arm, pair_id=pair_id,
                                  daily_rank=rank))
    pair_frame = pd.DataFrame(pairs)
    if pair_frame.empty or pair_frame.duplicated(["date", "code"]).any():
        raise ValueError("No distinct same-day pairs")
    arm_counts = pair_frame.groupby(["pair_id", "arm"]).size().unstack(
        "arm", fill_value=0)
    if not (arm_counts.signal.eq(1) & arm_counts.control.eq(1)).all():
        raise ValueError("Unbalanced noon comparison")
    by_half = {}
    for half in HALVES:
        attempted = top.loc[top.half.eq(half)]
        part = pair_frame.loc[pair_frame.half.eq(half)]
        high = part.loc[part.arm.eq("signal")]
        low = part.loc[part.arm.eq("control")]
        if len(high) != len(low):
            raise ValueError("Half-year arm count mismatch")
        differences = {name: float(high[name].mean() - low[name].mean())
                       for name in ("noon_jump", *FEATURES)} if len(high) else {}
        by_half[half] = {
            "gap_stock_days": int((gap & inputs.half.eq(half)).sum()),
            "recovered_stock_days": int((recovered
                                         & inputs.half.eq(half)).sum()),
            "recovered_days": int(inputs.loc[
                recovered & inputs.half.eq(half), "date"].nunique()),
            "top_signals": len(attempted),
            "pairs": len(high), "days": int(high.date.nunique()),
            "paired_fraction": len(high) / len(attempted) if len(attempted) else 0,
            "differences": differences,
            "price_ratio": (float(high.price_1449.mean() / low.price_1449.mean())
                            if len(high) else None),
            "amount_ratio": (float(high.amount_1449.mean() / low.amount_1449.mean())
                             if len(high) else None),
        }
    return top, pair_frame, by_half


def freeze(output: Path = OUTPUT,
           prefix_dir: Path = ROOT / "minute_prefix_1449",
           snapshot_dir: Path = ROOT / "market_snapshots_ci") -> dict:
    inputs, coverage = _load_inputs(output, prefix_dir, snapshot_dir)
    top, pairs, by_half = _match(inputs)
    for half in HALVES:
        by_half[half].update(coverage[half])
        row = by_half[half]
        row["source_coverage"] = row["source_rows"] / row["base_rows"]
    gate = all(
        row["days"] >= 80 and row["pairs"] >= 100
        and row["paired_fraction"] >= .35
        and row["source_coverage"] >= .999
        and row["unique_rows"] == row["base_rows"]
        and row["price_mismatch_rows"] == 0
        and row["differences"]["noon_jump"] <= -.003
        and all(abs(row["differences"][name]) <= .001
                for name in ("morning_return", "day_return", "return_last29"))
        and abs(row["differences"]["return20_prior_adjusted"]) <= .015
        and 2 / 3 <= row["price_ratio"] <= 1.5
        and 2 / 3 <= row["amount_ratio"] <= 1.5
        for row in by_half.values())
    output.mkdir(parents=True, exist_ok=True)
    for stale in ("gross_rows.parquet", "gross_report.json"):
        (output / stale).unlink(missing_ok=True)
    top.to_parquet(output / "top_signals.parquet", index=False,
                   compression="zstd")
    pairs.to_parquet(output / "pairs.parquet", index=False,
                     compression="zstd")
    report = {"scope": list(HALVES), "cutoff": "14:49",
              "source": "11:30 and 13:01 raw minute closes",
              "eligible_stock_days": len(inputs), "by_half": by_half,
              "outcome_gate_passed": bool(gate)}
    (output / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("extract", "freeze"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--minute-root", type=Path,
                        default=Path("data/hf/pilot/data/stock_1m"))
    args = parser.parse_args()
    actions = {"extract": lambda: extract_bars(args.minute_root, args.output),
               "freeze": lambda: freeze(args.output)}
    print(json.dumps(actions[args.stage](), ensure_ascii=False, indent=2))
