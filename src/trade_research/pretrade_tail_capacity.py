"""Audit a 14:50-known estimate of the next four minutes' buy capacity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .absolute_ridge import ROOT, feature_frame


OUTPUT = ROOT / "pretrade_tail_capacity"
LAST5_OUTPUT = ROOT / "pretrade_tail_capacity_last5"
LAST5_SOURCE = ROOT / "pretrade_last5"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")
NOTIONAL = 100_000
PARTICIPATION = .1


def predict_capacity(price: pd.Series, recent_volume: pd.Series,
                     minutes: int = 15) -> pd.DataFrame:
    """The same 100-share lot rule as the main-board execution model."""
    if minutes not in (5, 15):
        raise ValueError("Only the frozen five- or fifteen-minute window is valid")
    prices = price.to_numpy(dtype=float)
    observed_volume = recent_volume.to_numpy(dtype=float)
    if (not np.isfinite(observed_volume).all()
            or np.any(observed_volume < 0)
            or not np.isfinite(prices).all()
            or np.any(prices <= 0)):
        raise ValueError("Invalid pre-trade price or minute volume")
    shares = (np.floor(NOTIONAL / prices / 100) * 100).astype(int)
    forecast_volume = observed_volume * (4 / minutes)
    return pd.DataFrame({
        "planned_shares": shares,
        "predicted_feasible": (shares >= 100)
                              & (shares <= forecast_volume * PARTICIPATION),
    }, index=price.index)


def build_last5_year(year: int, minute_root: Path,
                     output_dir: Path = LAST5_SOURCE) -> Path:
    """Extract only completed 14:46–14:50 records from the raw archive."""
    if year not in (2024, 2025):
        raise ValueError("Only 2024–2025 minute bars may be used here")
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / f"{year}.parquet"
    if destination.exists():
        return destination
    temporary = output_dir / f"{year}.partial.parquet"
    temporary.unlink(missing_ok=True)
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    sources = [str(minute_root / exchange / "*.parquet")
               for exchange in ("SH", "SZ")]
    connection.read_parquet(sources).create_view("raw_minutes")
    escaped = str(temporary).replace("'", "''")
    connection.execute(f"""
        COPY (
            SELECT lower(exchange) || '.' || symbol AS code,
                   strftime(timestamp, '%Y-%m-%d') AS date,
                   SUM(volume) AS volume_last5,
                   COUNT(*) AS bar_count
            FROM raw_minutes
            WHERE timestamp >= TIMESTAMP '{year}-01-01'
              AND timestamp < TIMESTAMP '{year + 1}-01-01'
              AND EXTRACT(HOUR FROM timestamp) = 14
              AND EXTRACT(MINUTE FROM timestamp) BETWEEN 46 AND 50
            GROUP BY code, date
            HAVING COUNT(*) = 5 AND COUNT(DISTINCT timestamp) = 5
        ) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    temporary.replace(destination)
    return destination


def freeze_inputs(output_dir: Path = OUTPUT, window: int = 15) -> dict:
    if window not in (5, 15):
        raise ValueError("Only the frozen five- or fifteen-minute window is valid")
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    eligible = feature_frame(connection, main_only=True)
    connection.register("eligible_keys", eligible[["date", "code"]])
    inputs = connection.execute("""
        SELECT k.date, k.code, s.price_1450, s.amount_1450,
               s.volume_1450 * i.volume_share_last15 AS volume_last15
        FROM eligible_keys k JOIN snapshots s USING (date, code)
        JOIN intraday i USING (date, code)
    """).df()
    if (len(inputs) != len(eligible)
            or inputs.duplicated(["date", "code"]).any()):
        raise ValueError("Eligible stock-days lack unique 14:50 inputs")
    eligible_count = len(inputs)
    if window == 5:
        recent = pd.concat([
            pd.read_parquet(LAST5_SOURCE / f"{year}.parquet")
            for year in (2024, 2025)
        ], ignore_index=True)
        if recent.duplicated(["date", "code"]).any():
            raise ValueError("Duplicated last-five-minute stock-day")
        inputs = inputs.merge(
            recent[["date", "code", "volume_last5"]],
            on=["date", "code"], how="inner", validate="one_to_one")
        recent_column = "volume_last5"
    else:
        recent_column = "volume_last15"
    inputs = pd.concat([inputs, predict_capacity(
        inputs.price_1450, inputs[recent_column], window)], axis=1)
    inputs["half"] = inputs.date.str[:4] + "H" + np.where(
        inputs.date.str[5:7].astype(int) <= 6, "1", "2")
    if not inputs.half.isin(HALVES).all():
        raise ValueError("A non-development stock-day entered capacity research")
    by_half = inputs.groupby("half").agg(
        stocks=("code", "size"), days=("date", "nunique"),
        predicted_feasible_fraction=("predicted_feasible", "mean"),
    ).reset_index().to_dict("records")
    connection.register("input_keys", inputs[["date", "code"]])
    connection.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                            ).create_view("outcomes")
    archive_rows, matched = connection.execute("""
        SELECT COUNT(*), COUNT(o.code) FROM input_keys k
        LEFT JOIN outcomes o ON k.date = o.date AND k.code = o.code
                            AND o.horizon = 5
    """).fetchone()
    gate = (len(by_half) == 4 and all(
        row["days"] >= 80 and row["stocks"] >= 50_000
        and .05 <= row["predicted_feasible_fraction"] <= .95
        for row in by_half)
        and archive_rows == matched == len(inputs))
    audit = {
        "window_minutes": window,
        "eligible_before_recent_window": eligible_count,
        "excluded_missing_recent_window": eligible_count - len(inputs),
        "input_rows": len(inputs), "by_half": by_half,
        "archive_rows": archive_rows, "archive_matched": matched,
        "outcome_gate_passed": bool(gate),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs.to_parquet(output_dir / "inputs.parquet", index=False,
                      compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--window", type=int, choices=(5, 15), default=15)
    parser.add_argument("--build-last5", type=int, choices=(2024, 2025))
    parser.add_argument("--minute-root", type=Path,
                        default=Path("data/hf/pilot/data/stock_1m"))
    args = parser.parse_args()
    if args.build_last5:
        print({"last5_file": str(build_last5_year(
            args.build_last5, args.minute_root))})
        return
    output = args.output or (LAST5_OUTPUT if args.window == 5 else OUTPUT)
    print(freeze_inputs(output, args.window))


if __name__ == "__main__":
    main()
