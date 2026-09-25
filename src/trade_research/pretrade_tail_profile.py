"""Forecast 14:52–14:55 volume from earlier days' matching minute windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .absolute_ridge import ROOT


OUTPUT = ROOT / "pretrade_tail_profile"
HISTORY = OUTPUT / "history.parquet"
PRETRADE = ROOT / "pretrade_tail_capacity_last5" / "inputs.parquet"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")


def build_history(minute_root: Path, destination: Path = HISTORY) -> Path:
    """Keep valid five- and four-minute windows; older days are warmup only."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".partial.parquet")
    temporary.unlink(missing_ok=True)
    sources = [str(minute_root / exchange / "*.parquet")
               for exchange in ("SH", "SZ")]
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(sources).create_view("raw_minutes")
    escaped = str(temporary).replace("'", "''")
    connection.execute(f"""
        COPY (
            WITH bars AS (
                SELECT lower(exchange) || '.' || symbol AS code,
                       CAST(timestamp AS DATE) AS date, timestamp, volume,
                       EXTRACT(MINUTE FROM timestamp) AS minute
                FROM raw_minutes
                WHERE timestamp >= TIMESTAMP '2023-10-01'
                  AND timestamp < TIMESTAMP '2026-01-01'
                  AND EXTRACT(HOUR FROM timestamp) = 14
                  AND volume >= 0
                  AND (EXTRACT(MINUTE FROM timestamp) BETWEEN 46 AND 50
                    OR EXTRACT(MINUTE FROM timestamp) BETWEEN 52 AND 55)
            ), daily AS (
                SELECT code, date,
                       SUM(volume) FILTER (WHERE minute BETWEEN 46 AND 50)
                           AS volume_last5,
                       SUM(volume) FILTER (WHERE minute BETWEEN 52 AND 55)
                           AS volume_next4,
                       COUNT(*) FILTER (WHERE minute BETWEEN 46 AND 50)
                           AS last5_rows,
                       COUNT(DISTINCT minute)
                           FILTER (WHERE minute BETWEEN 46 AND 50)
                           AS last5_unique,
                       COUNT(*) FILTER (WHERE minute BETWEEN 52 AND 55)
                           AS next4_rows,
                       COUNT(DISTINCT minute)
                           FILTER (WHERE minute BETWEEN 52 AND 55)
                           AS next4_unique
                FROM bars GROUP BY code, date
            )
            SELECT code, date, volume_last5, volume_next4,
                   volume_next4 / volume_last5 AS ratio
            FROM daily
            WHERE last5_rows = 5 AND last5_unique = 5
              AND next4_rows = 4 AND next4_unique = 4
              AND volume_last5 > 0 AND volume_next4 >= 0
        ) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    temporary.replace(destination)
    return destination


def forecast_ratios(signals: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Use at most 20 valid earlier sessions inside the trailing 45 days."""
    if signals.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate signal stock-day")
    if history.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate historical stock-day")
    result = signals[["date", "code"]].copy()
    result["history_days"] = 0
    result["historical_ratio_p20"] = np.nan
    prior = history[["date", "code", "ratio"]].copy()
    prior["date"] = pd.to_datetime(prior.date)
    prior = prior.sort_values(["code", "date"])
    if not np.isfinite(prior.ratio.to_numpy(dtype=float)).all():
        raise ValueError("Invalid historical volume ratio")
    prior_by_code = {
        code: (frame.date.to_numpy(dtype="datetime64[ns]"),
               frame.ratio.to_numpy(dtype=float))
        for code, frame in prior.groupby("code", sort=False)
    }
    result_dates = pd.to_datetime(result.date)
    for code, target in result.groupby("code", sort=False).groups.items():
        past = prior_by_code.get(code)
        if past is None:
            continue
        dates, ratios = past
        for row in target:
            day = result_dates.loc[row].to_datetime64()
            end = np.searchsorted(dates, day, side="left")
            start = max(np.searchsorted(dates, day - np.timedelta64(45, "D"),
                                        side="left"), end - 20)
            count = end - start
            result.at[row, "history_days"] = count
            if count >= 10:
                result.at[row, "historical_ratio_p20"] = np.quantile(
                    ratios[start:end], .2)
    return result


def freeze_inputs(history_path: Path = HISTORY, output_dir: Path = OUTPUT,
                  pretrade_path: Path = PRETRADE) -> dict:
    inputs = pd.read_parquet(pretrade_path)
    history = pd.read_parquet(history_path)
    forecast = forecast_ratios(inputs, history)
    inputs = inputs.merge(forecast, on=["date", "code"], validate="one_to_one")
    inputs["predicted_volume"] = (
        inputs.volume_last5 * inputs.historical_ratio_p20)
    inputs["predicted_feasible_profile"] = (
        inputs.planned_shares.ge(100)
        & inputs.planned_shares.le(inputs.predicted_volume * .1))
    inputs["half"] = inputs.date.str[:4] + "H" + np.where(
        inputs.date.str[5:7].astype(int).le(6), "1", "2")
    if not inputs.half.isin(HALVES).all():
        raise ValueError("A non-development signal entered capacity research")
    summary = []
    for half, group in inputs.groupby("half", sort=True):
        eligible = group.loc[group.history_days.ge(10)]
        summary.append({
            "half": half, "stocks": len(group),
            "days": int(group.date.nunique()),
            "history_coverage": float(len(eligible) / len(group)),
            "predicted_feasible_fraction": (
                float(eligible.predicted_feasible_profile.mean())
                if not eligible.empty else None),
        })
    gate = (len(summary) == 4 and all(
        row["days"] >= 80 and row["stocks"] >= 50_000
        and row["history_coverage"] >= .95
        and .05 <= row["predicted_feasible_fraction"] <= .95
        for row in summary))
    audit = {"input_rows": len(inputs), "history_rows": len(history),
             "by_half": summary, "outcome_gate_passed": bool(gate)}
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs.to_parquet(output_dir / "inputs.parquet", index=False,
                      compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-history", action="store_true")
    parser.add_argument("--minute-root", type=Path,
                        default=Path("data/hf/pilot/data/stock_1m"))
    args = parser.parse_args()
    if args.build_history:
        print({"history_file": str(build_history(args.minute_root))})
    else:
        print(freeze_inputs())


if __name__ == "__main__":
    main()
