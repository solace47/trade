"""Frozen 14:49 input and gross-space check for daytime-return asymmetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .market_study import _quality_keys, _quality_symbols


ROOT = Path("data/research")
OUTPUT = ROOT / "daytime_asymmetry"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")
MIN_GROUP_STOCK_DAYS = 5_000
MIN_COMPARABLE_DAYS = 80
MIN_COVERAGE = 0.40


def _connect() -> duckdb.DuckDBPyConnection:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '8GB'")
    return connection


def _half(date: pd.Series) -> pd.Series:
    return date.str[:4] + "H" + np.where(date.str[5:7].astype(int) <= 6, "1", "2")


def _comparable(inputs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = ["date", "board", "liquidity_third"]
    counts = inputs.groupby(keys + ["group"], observed=True).size().unstack("group", fill_value=0)
    for group in ("up", "down"):
        if group not in counts:
            counts[group] = 0
    comparable = counts.loc[counts.up.ge(5) & counts.down.ge(5), ["up", "down"]]
    return counts, comparable


def freeze(prefix_dir: Path = ROOT / "minute_prefix_1449",
           snapshot_dir: Path = ROOT / "market_snapshots_ci",
           output_dir: Path = OUTPUT) -> dict:
    source_audit = json.loads((prefix_dir / "input_audit.json").read_text(encoding="utf-8"))
    if not source_audit["input_gate_passed"]:
        raise ValueError("14:49 prefix input audit did not pass")
    connection = _connect()
    try:
        connection.from_parquet(str(prefix_dir / "*" / "part_*.parquet")).create_view("prefix")
        connection.from_parquet(str(snapshot_dir / "*.parquet")).create_view("snapshots")
        inputs = connection.execute("""
            WITH base AS (
                SELECT p.date, p.code, p.price_1449, p.amount_1449,
                       p.price_1449 / s.preclose - 1 AS day_return,
                       CASE WHEN p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%'
                                 THEN 'main'
                            WHEN p.code LIKE 'sz.30%' THEN 'chinext'
                            WHEN p.code LIKE 'sh.68%' THEN 'star'
                            ELSE NULL END AS board
                FROM prefix p JOIN snapshots s USING (date, code)
                WHERE p.date BETWEEN '2024-01-01' AND '2025-12-17'
                  AND s.tradestatus = 1 AND s.isST = 0
                  AND s.listing_age_sessions >= 20
                  AND NOT s.reference_gap
                  AND NOT p.quote_outside_traded_range
                  AND s.preclose > 0 AND p.price_1449 >= 5
                  AND p.amount_1449 BETWEEN 100000000 AND 1000000000
                  AND s.return20_prior_adjusted BETWEEN -.10 AND .10
                  AND p.return_last29 BETWEEN -.01 AND .01
                  AND p.price_1449 / s.preclose - 1 BETWEEN -.03 AND .03
            ), ranked AS (
                SELECT *, NTILE(3) OVER (
                    PARTITION BY date, board ORDER BY amount_1449, code
                ) AS liquidity_third
                FROM base WHERE board IS NOT NULL
            )
            SELECT *, CASE WHEN day_return >= .005 THEN 'up'
                           WHEN day_return <= -.005 THEN 'down'
                           ELSE 'flat' END AS "group"
            FROM ranked
        """).df()
    finally:
        connection.close()
    if inputs.empty or inputs.duplicated(["date", "code"]).any():
        raise ValueError("Missing or duplicate 14:49 inputs")
    if not inputs.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("A non-development year reached the input stage")
    if not np.isfinite(inputs[["price_1449", "amount_1449", "day_return"]].to_numpy()).all():
        raise ValueError("Nonfinite pre-decision input")
    inputs["half"] = _half(inputs.date)
    _, comparable = _comparable(inputs)
    comparable_frame = comparable.reset_index()
    comparable_frame["half"] = _half(comparable_frame.date)
    selected_keys = comparable_frame[["date", "board", "liquidity_third"]]
    marked = inputs.merge(selected_keys.assign(comparable=True),
                          on=["date", "board", "liquidity_third"], how="left",
                          validate="many_to_one")
    marked["comparable"] = marked.comparable.fillna(False).astype(bool)
    by_half = {}
    for half in HALVES:
        segment = marked.loc[marked.half.eq(half)]
        by_group = {}
        for group in ("up", "down"):
            rows = segment.loc[segment["group"].eq(group)]
            matched = rows.loc[rows.comparable]
            by_group[group] = {
                "stock_days": len(rows),
                "matched_stock_days": len(matched),
                "matched_fraction": len(matched) / len(rows) if len(rows) else 0.0,
            }
        matched_signs = segment.loc[
            segment.comparable & segment["group"].isin(("up", "down"))
        ]
        by_half[half] = {
            "comparable_strata": int(comparable_frame.half.eq(half).sum()),
            "comparable_days": int(matched_signs.date.nunique()),
            "groups": by_group,
        }
    gate = (set(marked.half) == set(HALVES)
            and all(
                by_half[half]["comparable_days"] >= MIN_COMPARABLE_DAYS
                and all(
                    by_half[half]["groups"][group]["stock_days"] >= MIN_GROUP_STOCK_DAYS
                    and by_half[half]["groups"][group]["matched_fraction"] >= MIN_COVERAGE
                    for group in ("up", "down")
                )
                for half in HALVES
            ))
    output_dir.mkdir(parents=True, exist_ok=True)
    marked.to_parquet(output_dir / "inputs.parquet", index=False, compression="zstd")
    audit = {
        "cutoff": "14:49", "periods": list(HALVES),
        "input_count": len(marked), "comparable_strata": len(comparable),
        "minimum_group_stock_days": MIN_GROUP_STOCK_DAYS,
        "minimum_comparable_days": MIN_COMPARABLE_DAYS,
        "minimum_group_coverage": MIN_COVERAGE,
        "by_half": by_half, "outcome_gate_passed": bool(gate),
    }
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def _week_interval(days: pd.DataFrame, column: str) -> list[float]:
    if days.empty:
        return [float("nan"), float("nan")]
    key = pd.to_datetime(days.date).dt.strftime("%G-%V")
    weekly = days.groupby(key)[column].agg(["sum", "count"])
    values = weekly[["sum", "count"]].to_numpy()
    rng = np.random.default_rng(20260926)
    draws = rng.integers(0, len(values), size=(2_000, len(values)))
    sampled = values[draws]
    means = sampled[:, :, 0].sum(axis=1) / sampled[:, :, 1].sum(axis=1)
    return [float(x) for x in np.quantile(means, [0.025, 0.975])]


def _summarize_valid(valid: pd.DataFrame) -> dict:
    keys = ["date", "board", "liquidity_third", "group"]
    means = valid.groupby(keys)[
        ["gross_1449", "paper_overnight", "remaining_day"]
    ].mean().unstack("group")
    counts = valid.groupby(keys).size().unstack("group", fill_value=0)
    matched = counts.up.ge(5) & counts.down.ge(5)
    means = means.loc[matched]
    if means.empty:
        raise ValueError("No quality-clean comparable strata")
    strata = means.reset_index()
    strata.columns = [
        name if not group else f"{name}_{group}" for name, group in strata.columns
    ]
    daily = strata.groupby("date", as_index=False)[
        ["gross_1449_up", "gross_1449_down",
         "paper_overnight_up", "paper_overnight_down",
         "remaining_day_up", "remaining_day_down"]
    ].mean().rename(columns={
        "gross_1449_up": "gross_up", "gross_1449_down": "gross_down",
        "paper_overnight_up": "paper_up",
        "paper_overnight_down": "paper_down",
        "remaining_day_up": "remaining_up",
        "remaining_day_down": "remaining_down",
    })
    daily["gross_edge"] = daily.gross_up - daily.gross_down
    daily["half"] = _half(daily.date)
    daily["year"] = daily.date.str[:4]
    by_half = {}
    for half in HALVES:
        part = daily.loc[daily.half.eq(half)]
        by_half[half] = {
            "days": len(part),
            "gross_up_pp": float(part.gross_up.mean() * 100),
            "gross_down_pp": float(part.gross_down.mean() * 100),
            "gross_edge_pp": float(part.gross_edge.mean() * 100),
            "paper_up_pp": float(part.paper_up.mean() * 100),
            "paper_down_pp": float(part.paper_down.mean() * 100),
            "remaining_day_up_pp": float(part.remaining_up.mean() * 100),
            "remaining_day_down_pp": float(part.remaining_down.mean() * 100),
        }
    by_year = {}
    for year in ("2024", "2025"):
        part = daily.loc[daily.year.eq(year)]
        by_year[year] = {
            "days": len(part),
            "gross_up_pp": float(part.gross_up.mean() * 100),
            "gross_edge_pp": float(part.gross_edge.mean() * 100),
            "gross_edge_week_interval_pp": [x * 100 for x in _week_interval(part, "gross_edge")],
        }
    raw_gate = all(
        row["gross_up_pp"] > 0.25 and row["gross_edge_pp"] > 0.20
        for row in by_half.values()
    ) and all(row["gross_edge_week_interval_pp"][0] > 0 for row in by_year.values())
    return {
        "valid_comparable_strata": len(strata),
        "valid_comparable_days": len(daily),
        "by_half": by_half, "by_year": by_year,
        "raw_reprice_gate_passed": bool(raw_gate),
    }


def evaluate(input_dir: Path = OUTPUT,
             daily_dir: Path = Path("data/baostock/market_2020_2026/daily"),
             calendar_file: Path = Path(
                 "data/baostock/market_2020_2026/metadata/calendar.parquet"
             ),
             issues_dir: Path = ROOT / "market_issues_ci") -> dict:
    audit = json.loads((input_dir / "input_audit.json").read_text(encoding="utf-8"))
    if not audit["outcome_gate_passed"]:
        raise ValueError("Input gate failed; next-day prices remain unread")
    connection = _connect()
    try:
        connection.from_parquet(str(input_dir / "inputs.parquet")).create_view("inputs")
        connection.from_parquet(str(daily_dir / "*.parquet")).create_view("daily_source")
        connection.from_parquet(str(calendar_file)).create_view("calendar_source")
        connection.register("bad_days", _quality_keys(issues_dir))
        connection.register("bad_symbols", _quality_symbols(issues_dir))
        joined = connection.execute("""
            WITH trading AS (
                SELECT calendar_date AS date,
                       LEAD(calendar_date) OVER (ORDER BY calendar_date) AS next_date
                FROM calendar_source
                WHERE is_trading_day = '1'
                  AND calendar_date BETWEEN '2024-01-01' AND '2025-12-31'
            ), daily AS (
                SELECT date, code, open, close, preclose, tradestatus
                FROM daily_source WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
            )
            SELECT i.date, i.code, i.board, i.liquidity_third, i."group",
                   i.half, i.comparable, t.next_date,
                   d.close AS current_close, n.open AS next_open,
                   CASE WHEN n.date IS NOT NULL
                         AND d.tradestatus = 1 AND n.tradestatus = 1
                         AND d.close > 0 AND n.open > 0
                         AND ABS(n.preclose - d.close) <= .005
                         AND b.code IS NULL AND q0.code IS NULL AND q1.code IS NULL
                        THEN TRUE ELSE FALSE END AS quality_clean,
                   n.open / i.price_1449 - 1 AS gross_1449,
                   n.open / d.close - 1 AS paper_overnight,
                   d.close / i.price_1449 - 1 AS remaining_day
            FROM inputs i JOIN trading t ON i.date = t.date
            LEFT JOIN daily d ON i.date = d.date AND i.code = d.code
            LEFT JOIN daily n ON t.next_date = n.date AND i.code = n.code
            LEFT JOIN bad_symbols b ON i.code = b.code
            LEFT JOIN bad_days q0 ON i.date = q0.date AND i.code = q0.code
            LEFT JOIN bad_days q1 ON t.next_date = q1.date AND i.code = q1.code
        """).df()
    finally:
        connection.close()
    if len(joined) != audit["input_count"] or joined.duplicated(["date", "code"]).any():
        raise ValueError("Incomplete or duplicated next-day join")
    comparable = joined.loc[
        joined.comparable & joined["group"].isin(("up", "down"))
    ]
    valid = comparable.loc[comparable.quality_clean].copy()
    if not np.isfinite(
        valid[["gross_1449", "paper_overnight", "remaining_day"]].to_numpy()
    ).all():
        raise ValueError("Nonfinite quality-clean return")
    joined.to_parquet(input_dir / "gross_rows.parquet", index=False, compression="zstd")
    report = {
        "input_gate_passed": True,
        "valid_stock_days": len(valid),
        "comparable_sign_stock_days": len(comparable),
        "quality_excluded_comparable_stock_days": int((~comparable.quality_clean).sum()),
        "year_scope": "2024–2025 only; 2026 untouched",
        "price_interpretation": "Diagnostic anchors, not executable fills",
        **_summarize_valid(valid),
    }
    (input_dir / "gross_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = (freeze(output_dir=args.output_dir) if args.stage == "freeze"
              else evaluate(input_dir=args.output_dir))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
