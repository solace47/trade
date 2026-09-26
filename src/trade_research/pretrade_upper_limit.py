"""Audit whether a 14:49 upper-limit quote predicts an unfilled buy window.

Only entry statuses are read from the archived raw-minute outcomes. No exit
prices or returns are used to construct the screen or assess its gate.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .execution_path import LABELS, minute_participation
from .hf_outcomes import Assumptions, _fill, _order_shares


ROOT = Path("data/research")
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")


def _connect() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '8GB'")
    return connection


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def _rows(connection: duckdb.DuckDBPyConnection, query: str) -> list[dict]:
    cursor = connection.execute(query)
    names = [field[0] for field in cursor.description]
    return [dict(zip(names, values, strict=True)) for values in cursor.fetchall()]


def freeze_inputs(prefix_dir: Path = ROOT / "minute_prefix_1449",
                  snapshot_dir: Path = ROOT / "market_snapshots_ci",
                  output_dir: Path = ROOT / "pretrade_upper_limit") -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / "inputs.parquet"
    connection = _connect()
    try:
        connection.read_parquet(str(prefix_dir / "*" / "part_*.parquet")
                                ).create_view("prefix")
        connection.read_parquet(str(snapshot_dir / "*.parquet")
                                ).create_view("snapshots")
        escaped = str(destination).replace("'", "''")
        connection.execute(f"""
            COPY (
                WITH eligible AS (
                    SELECT p.date, p.code, p.price_1449, s.preclose,
                           CAST(ROUND(s.preclose * 100, 0) AS BIGINT)
                               AS preclose_cents
                    FROM prefix p JOIN snapshots s USING (date, code)
                    WHERE p.date BETWEEN '2024-01-01' AND '2025-12-31'
                      AND (p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%')
                      AND s.isST = 0 AND s.tradestatus = 1
                      AND s.listing_age_sessions >= 20
                      AND s.reference_gap = false
                      AND p.quote_outside_traded_range = false
                      AND p.amount_1449 BETWEEN 100000000 AND 1000000000
                      AND s.preclose > 0 AND p.price_1449 > 0
                ), priced AS (
                    SELECT *, CAST(FLOOR((preclose_cents * 110 + 50) / 100)
                                   AS BIGINT) AS upper_cents
                    FROM eligible
                )
                SELECT date, code, price_1449, preclose,
                       upper_cents / 100.0 AS upper_limit,
                       price_1449 >= upper_cents / 100.0 - 0.005
                           AS at_upper,
                       LEFT(date, 4) || 'H' ||
                           CASE WHEN SUBSTR(date, 6, 2) <= '06' THEN '1'
                                ELSE '2' END AS half
                FROM priced
            ) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        connection.read_parquet(str(destination)).create_view("inputs")
        counts = _rows(connection, """
            SELECT half, COUNT(*) AS rows,
                   COUNT(DISTINCT (date, code)) AS keys,
                   COUNT(DISTINCT date) AS days,
                   COUNT(*) FILTER (WHERE at_upper) AS at_upper_rows,
                   COUNT(DISTINCT date) FILTER (WHERE at_upper)
                       AS at_upper_days
            FROM inputs GROUP BY half ORDER BY half
        """)
    finally:
        connection.close()
    audit = {
        "by_half": counts,
        "input_gate_passed": ([row["half"] for row in counts] == list(HALVES)
                              and all(row["rows"] == row["keys"]
                                      and row["at_upper_rows"] >= 100
                                      and row["at_upper_days"] >= 30
                                      for row in counts)),
    }
    _write_json(output_dir / "input_audit.json", audit)
    return audit


def evaluate(outcome_dir: Path = ROOT / "market_outcomes_ci",
             output_dir: Path = ROOT / "pretrade_upper_limit") -> dict:
    input_audit = json.loads((output_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if not input_audit["input_gate_passed"]:
        raise ValueError("Input gate failed; archived buy statuses must stay unread")
    connection = _connect()
    try:
        connection.read_parquet(str(output_dir / "inputs.parquet")
                                ).create_view("inputs")
        connection.read_parquet(str(outcome_dir / "*.parquet")
                                ).create_view("outcomes")
        # Project only entry fields, to avoid scanning any future sale or PnL.
        connection.execute("""
            CREATE TEMP VIEW joined AS
            SELECT i.half, i.date, i.code, i.at_upper,
                   o.code IS NOT NULL AS matched, o.entry_status
            FROM inputs i LEFT JOIN (
                SELECT date, code, entry_status FROM outcomes WHERE horizon = 1
            ) o USING (date, code)
        """)
        association = _rows(connection, """
            SELECT COUNT(*) AS joined_rows, COUNT(*) FILTER (WHERE matched)
                       AS matched_rows,
                   COUNT(DISTINCT (date, code)) AS unique_keys
            FROM joined
        """)[0]
        by_half = _rows(connection, """
            SELECT half, at_upper, COUNT(*) AS rows,
                   COUNT(DISTINCT date) AS days,
                   COUNT(*) FILTER (WHERE entry_status = 'filled') AS filled,
                   COUNT(*) FILTER (WHERE entry_status = 'estimated_upper_limit')
                       AS upper_limit_failure,
                   COUNT(*) FILTER (WHERE entry_status IS NOT NULL
                                     AND entry_status NOT IN
                                         ('filled', 'estimated_upper_limit'))
                       AS other_failure,
                   COUNT(*) FILTER (WHERE entry_status IS NULL)
                       AS missing_status
            FROM joined GROUP BY half, at_upper ORDER BY half, at_upper
        """)
        captured = _rows(connection, """
            SELECT COUNT(*) FILTER (WHERE entry_status =
                       'estimated_upper_limit') AS all_upper_limit_failures,
                   COUNT(*) FILTER (WHERE at_upper AND entry_status =
                       'estimated_upper_limit') AS caught_upper_limit_failures
            FROM joined
        """)[0]
    finally:
        connection.close()
    input_rows = sum(row["rows"] for row in input_audit["by_half"])
    association["coverage"] = association["matched_rows"] / input_rows
    for row in by_half:
        row["fill_rate"] = row["filled"] / row["rows"]
    captured["fraction_caught"] = (
        captured["caught_upper_limit_failures"] /
        captured["all_upper_limit_failures"]
        if captured["all_upper_limit_failures"] else None)
    groups = {(row["half"], row["at_upper"]): row for row in by_half}
    gate = (association["joined_rows"] == input_rows
            and association["unique_keys"] == input_rows
            and association["coverage"] >= .999
            and all(groups[(half, True)]["fill_rate"] <= .10
                    and groups[(half, False)]["fill_rate"] >= .90
                    for half in HALVES)) if all(
                        (half, value) in groups
                        for half in HALVES for value in (True, False)) else False
    report = {"association": association, "by_half": by_half,
              "upper_limit_failure_capture": captured,
              "execution_gate_passed": bool(gate)}
    _write_json(output_dir / "entry_audit.json", report)
    return report


def freeze_raw_sample(output_dir: Path = ROOT / "pretrade_upper_limit") -> dict:
    entry_audit = json.loads((output_dir / "entry_audit.json").read_text(
        encoding="utf-8"))
    if not entry_audit["execution_gate_passed"]:
        raise ValueError("Archived entry gate failed; raw sample stays unread")
    connection = _connect()
    try:
        connection.read_parquet(str(output_dir / "inputs.parquet")
                                ).create_view("inputs")
        destination = output_dir / "raw_sample.parquet"
        escaped = str(destination).replace("'", "''")
        connection.execute(f"""
            COPY (
                WITH ranked AS (
                    SELECT *, COUNT(*) FILTER (WHERE at_upper) OVER
                               (PARTITION BY date) AS at_upper_count,
                           ROW_NUMBER() OVER (
                               PARTITION BY date, at_upper
                               ORDER BY md5(date || code), code
                           ) AS rank_within_day
                    FROM inputs
                )
                SELECT date, code, price_1449, preclose, at_upper, half
                FROM ranked
                WHERE at_upper OR rank_within_day <= at_upper_count
            ) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        sample = _rows(connection, f"""
            SELECT half, at_upper, COUNT(*) AS rows,
                   COUNT(DISTINCT (date, code)) AS keys,
                   COUNT(DISTINCT date) AS days
            FROM read_parquet('{escaped}')
            GROUP BY half, at_upper ORDER BY half, at_upper
        """)
    finally:
        connection.close()
    groups = {(row["half"], row["at_upper"]): row for row in sample}
    valid = len(groups) == 8 and all(
        groups[(half, True)]["rows"] == groups[(half, False)]["rows"]
        and groups[(half, True)]["days"] == groups[(half, False)]["days"]
        and groups[(half, True)]["rows"] == groups[(half, True)]["keys"]
        and groups[(half, False)]["rows"] == groups[(half, False)]["keys"]
        for half in HALVES)
    report = {"by_half": sample, "sample_gate_passed": valid}
    _write_json(output_dir / "raw_sample_audit.json", report)
    return report


def _raw_stock(item: tuple[str, pd.DataFrame], minute_root: Path) -> list[dict]:
    code, signals = item
    exchange, symbol = code.split(".")
    path = minute_root / exchange.upper() / f"{symbol}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    start = pd.Timestamp(signals.date.min())
    end = pd.Timestamp(signals.date.max()) + pd.Timedelta(days=1)
    minute = pd.read_parquet(path, columns=["timestamp", "volume", "turnover"],
                             filters=[("timestamp", ">=", start),
                                      ("timestamp", "<", end)])
    clock = minute.timestamp.dt.hour * 60 + minute.timestamp.dt.minute
    minute = minute.loc[clock.between(14 * 60 + 52,
                                      14 * 60 + 55)].copy()
    minute["date"] = minute.timestamp.dt.strftime("%Y-%m-%d")
    minute["label"] = minute.timestamp.dt.strftime("%H%M")
    minute = minute.loc[minute.date.isin(signals.date)]
    by_date = {date: bars.sort_values("label")
               for date, bars in minute.groupby("date")}
    rows = []
    assumptions = Assumptions(slippage_bps_each_side=5)
    for signal in signals.itertuples(index=False):
        bars = by_date.get(signal.date)
        complete = (bars is not None and bars.label.tolist() == list(LABELS)
                    and np.isfinite(bars[["volume", "turnover"]].to_numpy()).all()
                    and not bars.volume.lt(0).any()
                    and not bars.turnover.lt(0).any())
        quote = None
        if complete:
            volume = float(bars.volume.sum())
            turnover = float(bars.turnover.sum())
            quote = pd.Series({"volume": volume,
                               "vwap": turnover / volume if volume else 0.0})
        daily = pd.Series({"date": signal.date, "preclose": signal.preclose,
                           "tradestatus": 1, "isST": 0})
        for notional in (20_000, 100_000):
            shares = _order_shares(code, signal.price_1449, notional)
            if shares == 0:
                aggregate_status = "below_minimum_lot"
                path_filled = False
            elif not complete:
                aggregate_status = "no_trading_bar"
                path_filled = False
            else:
                _, aggregate_status = _fill(
                    quote, daily, code, "buy", shares, assumptions)
                taken, _ = minute_participation(
                    bars, code, signal.date, signal.preclose, shares, 5)
                path_filled = taken == shares
            rows.append({"date": signal.date, "code": code,
                         "half": signal.half, "at_upper": signal.at_upper,
                         "notional": notional, "bars_complete": bool(complete),
                         "aggregate_status": aggregate_status,
                         "aggregate_filled": aggregate_status == "filled",
                         "participation_filled": bool(path_filled)})
    return rows


def evaluate_raw(minute_root: Path = Path("data/hf/pilot/data/stock_1m"),
                 output_dir: Path = ROOT / "pretrade_upper_limit",
                 workers: int = 4) -> dict:
    sample_audit = json.loads((output_dir / "raw_sample_audit.json").read_text(
        encoding="utf-8"))
    if not sample_audit["sample_gate_passed"]:
        raise ValueError("Raw sample gate failed")
    sample = pd.read_parquet(output_dir / "raw_sample.parquet")
    grouped = list(sample.groupby("code", sort=True))
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for count, stock_rows in enumerate(pool.map(
                lambda item: _raw_stock(item, minute_root), grouped), start=1):
            rows.extend(stock_rows)
            if count % 500 == 0 or count == len(grouped):
                print(f"Read raw entry minutes for {count}/{len(grouped)} stocks",
                      flush=True)
    output = pd.DataFrame(rows)
    if (len(output) != 2 * len(sample)
            or output.duplicated(["date", "code", "notional"]).any()):
        raise ValueError("Raw entry result keys do not match the frozen sample")
    output.to_parquet(output_dir / "raw_entries.parquet", index=False,
                      compression="zstd")
    cells = []
    for (half, upper, notional), part in output.groupby(
            ["half", "at_upper", "notional"], sort=True):
        cells.append({
            "half": half, "at_upper": bool(upper), "notional": int(notional),
            "rows": len(part), "complete_rate": float(part.bars_complete.mean()),
            "aggregate_fill_rate": float(part.aggregate_filled.mean()),
            "participation_fill_rate": float(part.participation_filled.mean()),
            "aggregate_status": part.aggregate_status.value_counts().to_dict(),
        })
    passed = (len(cells) == 16 and all(
        cell["complete_rate"] >= .999
        and (cell["aggregate_fill_rate"] <= .10
             and cell["participation_fill_rate"] <= .10
             if cell["at_upper"] else
             cell["aggregate_fill_rate"] >= .90
             and cell["participation_fill_rate"] >= .90)
        for cell in cells))
    report = {"sample_stock_days": len(sample), "result_rows": len(output),
              "by_half_group_size": cells,
              "raw_execution_gate_passed": passed}
    _write_json(output_dir / "raw_entry_audit.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "evaluate", "freeze-raw",
                                          "evaluate-raw"))
    args = parser.parse_args()
    result = {"freeze": freeze_inputs,
              "evaluate": evaluate,
              "freeze-raw": freeze_raw_sample,
              "evaluate-raw": evaluate_raw}[args.stage]()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
