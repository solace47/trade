"""Freeze past overnight-up/daytime-down frequency before overnight outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .market_study import _quality_keys, _week_bootstrap
from .quality_period import load_period_bad_symbols


ROOT = Path("data/research")
SOURCE = ROOT / "overnight_composition" / "all_candidates.parquet"
DAILY = Path("data/baostock/market_2020_2026/daily")
CALENDAR = Path("data/baostock/market_2020_2026/metadata/calendar.parquet")
OUTPUT = ROOT / "overnight_reversal_frequency"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")
KEY = ("half", "date", "board", "day_bin", "today_gap_bin",
       "mean_gap_bin", "prior20_sign")


def build_inputs(source: Path = SOURCE, daily_dir: Path = DAILY) -> pd.DataFrame:
    """Count a joint event in 20 completed, valid days strictly before today."""
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 4")
        connection.execute("SET memory_limit = '8GB'")
        connection.from_parquet(str(source)).create_view("base")
        connection.from_parquet(str(daily_dir / "*.parquet")).create_view("daily")
        connection.execute("""
            CREATE TEMP TABLE event_history AS
            WITH active AS (
                SELECT date, code, open, close, preclose, isST,
                       LAG(close) OVER (PARTITION BY code ORDER BY date)
                           AS prior_active_close
                FROM daily
                WHERE date BETWEEN '2023-10-01' AND '2025-12-17'
                  AND tradestatus = 1 AND open > 0 AND close > 0
                  AND preclose > 0
            ), valid AS (
                SELECT date, code, open / preclose - 1 AS overnight_gap,
                       CASE WHEN open > preclose AND close < open
                            THEN 1 ELSE 0 END AS joint_event
                FROM active
                WHERE isST = 0 AND prior_active_close IS NOT NULL
                  AND ABS(preclose - prior_active_close) <= .005
                  AND ABS(open / preclose - 1) <= .10
            )
            SELECT date AS history_end, code,
                   COUNT(*) OVER win AS valid_days,
                   MIN(date) OVER win AS history_start,
                   AVG(overnight_gap) OVER win AS mean_gap,
                   SUM(joint_event) OVER win AS event_count
            FROM valid
            WINDOW win AS (PARTITION BY code ORDER BY date
                           ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
        """)
        inputs = connection.execute("""
            SELECT b.*, h.event_count,
                   h.mean_gap AS event_history_mean_gap,
                   h.history_end AS event_history_end
            FROM base b ASOF LEFT JOIN event_history h
              ON b.code = h.code AND b.date > h.history_end
            WHERE h.valid_days = 20
              AND date_diff('day', CAST(h.history_start AS DATE),
                                  CAST(b.date AS DATE)) <= 45
              AND date_diff('day', CAST(h.history_end AS DATE),
                                  CAST(b.date AS DATE)) <= 10
        """).df()
    finally:
        connection.close()
    base_rows = pd.read_parquet(source, columns=["date", "code"])
    if (len(inputs) != len(base_rows)
            or inputs.duplicated(["date", "code"]).any()
            or not inputs.event_history_end.lt(inputs.date).all()
            or not np.allclose(inputs.event_history_mean_gap,
                               inputs.mean_overnight_gap, atol=1e-12)):
        raise ValueError("Historical event inputs differ from the frozen base")
    return inputs


def select(inputs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Within controlled same-day strata compare frequency quartiles."""
    required = {"date", "code", "board", "price_1449", "amount_1449",
                "return_1449", "open_gap_today", "mean_overnight_gap",
                "return20_prior_adjusted", "event_count", "history_end",
                "event_history_end"}
    if not required.issubset(inputs):
        raise ValueError("Missing historical or pre-decision fields")
    if (inputs.empty or inputs.duplicated(["date", "code"]).any()
            or not inputs.date.between("2024-01-01", "2025-12-17").all()
            or not inputs.event_history_end.lt(inputs.date).all()
            or not inputs.history_end.lt(inputs.date).all()
            or not inputs.event_count.between(0, 20).all()):
        raise ValueError("Invalid or future-leaking input")
    frame = inputs.loc[:, sorted(required)].copy()
    frame["half"] = frame.date.str[:4] + "H" + np.where(
        frame.date.str[5:7].astype(int).le(6), "1", "2")
    if set(frame.half) != set(HALVES):
        raise ValueError("Incomplete development periods")
    frame["day_bin"] = np.floor(100 * frame.return_1449 / .5).astype(int)
    frame["today_gap_bin"] = np.floor(
        100 * frame.open_gap_today / .5).astype(int)
    frame["mean_gap_bin"] = np.floor(
        100 * frame.mean_overnight_gap / .25).astype(int)
    frame["prior20_sign"] = frame.return20_prior_adjusted.ge(0).astype(int)
    grouped = frame.groupby(list(KEY), observed=True).event_count
    strata = grouped.agg(n="size", q25=lambda x: x.quantile(.25),
                         q75=lambda x: x.quantile(.75)).reset_index()
    eligible = strata.loc[
        strata.n.ge(8) & (strata.q75 - strata.q25).ge(2),
        list(KEY) + ["n"]]
    ranked = frame.merge(eligible, on=list(KEY), validate="many_to_one")
    ranked = ranked.sort_values(list(KEY) + ["event_count", "code"])
    ranked["rank"] = ranked.groupby(list(KEY), observed=True).cumcount()
    ranked["arm_size"] = ranked.n // 4
    ranked["arm"] = np.select(
        [ranked["rank"].lt(ranked.arm_size),
         ranked["rank"].ge(ranked.n - ranked.arm_size)],
        ["low", "high"], default="middle")
    selected = ranked.loc[ranked.arm.ne("middle")].drop(
        columns=["n", "rank", "arm_size"])
    counts = selected.groupby(list(KEY) + ["arm"], observed=True).size(
    ).unstack("arm", fill_value=0)
    if (selected.empty or selected.duplicated(["date", "code"]).any()
            or not counts.high.eq(counts.low).all()
            or counts.high.lt(2).any()):
        raise ValueError("Historical frequency contrast is unbalanced")
    by_half = {}
    for half in HALVES:
        part = selected.loc[selected.half.eq(half)]
        high = part.loc[part.arm.eq("high")]
        low = part.loc[part.arm.eq("low")]
        differences = {
            name: float(high[name].mean() - low[name].mean())
            for name in ("event_count", "mean_overnight_gap",
                         "open_gap_today", "return_1449",
                         "return20_prior_adjusted", "price_1449",
                         "amount_1449")
        }
        by_half[half] = {
            "base_stock_days": int(frame.half.eq(half).sum()),
            "eligible_strata": int(eligible.half.eq(half).sum()),
            "eligible_stock_days": int(ranked.half.eq(half).sum()),
            "days": int(part.date.nunique()),
            "signals_each_arm": int(len(high)),
            "differences_high_minus_low": differences,
            "mean_price_ratio_high_to_low": float(high.price_1449.mean()
                                                   / low.price_1449.mean()),
        }
    gate = all(
        row["days"] >= 80 and row["signals_each_arm"] >= 1000
        and row["differences_high_minus_low"]["event_count"] >= 3
        and abs(row["differences_high_minus_low"][
            "mean_overnight_gap"]) <= .0006
        and abs(row["differences_high_minus_low"][
            "open_gap_today"]) <= .001
        and abs(row["differences_high_minus_low"][
            "return_1449"]) <= .001
        and abs(row["differences_high_minus_low"][
            "return20_prior_adjusted"]) <= .015
        and row["mean_price_ratio_high_to_low"] <= 1.6
        for row in by_half.values())
    return selected.sort_values(["date", "code"]).reset_index(drop=True), {
        "cutoff": "14:49", "years": [2024, 2025],
        "base_stock_days": int(len(frame)),
        "stratum_minimum": 8, "event_count_iqr_minimum": 2,
        "event_definition": "prior open > prior preclose and prior close < prior open",
        "by_half": by_half, "outcome_gate_passed": bool(gate),
    }


def freeze(output: Path = OUTPUT) -> dict:
    selected, report = select(build_inputs())
    output.mkdir(parents=True, exist_ok=True)
    for stale in ("gross_rows.parquet", "gross_report.json",
                  "repriced.parquet", "report.json"):
        (output / stale).unlink(missing_ok=True)
    selected.to_parquet(output / "selections.parquet", index=False,
                        compression="zstd")
    (output / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def evaluate_gross(output: Path = OUTPUT,
                   daily_dir: Path = DAILY, calendar: Path = CALENDAR,
                   issues: Path = ROOT / "market_issues_ci",
                   period_quality: Path = ROOT / "quality_period_2024_2025.json") -> dict:
    audit = json.loads((output / "input_audit.json").read_text(
        encoding="utf-8"))
    if not audit["outcome_gate_passed"]:
        raise ValueError("Input gate failed; next-day prices remain closed")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 4")
        connection.from_parquet(str(output / "selections.parquet")
                                ).create_view("signals")
        connection.from_parquet(str(daily_dir / "*.parquet")
                                ).create_view("daily")
        connection.from_parquet(str(calendar)).create_view("calendar")
        connection.register("bad_days", _quality_keys(issues))
        connection.register("bad_symbols", load_period_bad_symbols(
            period_quality, "2024-01-01", "2025-12-31"))
        joined = connection.execute("""
            WITH trading AS (
                SELECT calendar_date AS date,
                       LEAD(calendar_date) OVER (ORDER BY calendar_date)
                           AS next_date
                FROM calendar WHERE is_trading_day = '1'
                  AND calendar_date BETWEEN '2024-01-01' AND '2025-12-31'
            ), prices AS (
                SELECT date, code, open, close, preclose, tradestatus
                FROM daily WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
            )
            SELECT s.*, t.next_date, p.close AS signal_close,
                   n.open AS next_open,
                   n.open / s.price_1449 - 1 AS gross_1449,
                   p.tradestatus = 1 AND n.tradestatus = 1
                   AND p.close > 0 AND n.open > 0
                   AND ABS(n.preclose - p.close) <= .005
                   AND b.code IS NULL AND q0.code IS NULL
                   AND q1.code IS NULL AS quality_clean
            FROM signals s JOIN trading t ON s.date = t.date
            LEFT JOIN prices p ON s.date = p.date AND s.code = p.code
            LEFT JOIN prices n ON t.next_date = n.date AND s.code = n.code
            LEFT JOIN bad_symbols b ON s.code = b.code
            LEFT JOIN bad_days q0 ON s.date = q0.date AND s.code = q0.code
            LEFT JOIN bad_days q1 ON t.next_date = q1.date AND s.code = q1.code
        """).df()
    finally:
        connection.close()
    signals = pd.read_parquet(output / "selections.parquet")
    if len(joined) != len(signals) or joined.duplicated(["date", "code"]).any():
        raise ValueError("Incomplete next-market-day gross join")
    joined["quality_clean"] = joined.quality_clean.fillna(False).astype(bool)
    joined.to_parquet(output / "gross_rows.parquet", index=False,
                      compression="zstd")
    valid = joined.loc[joined.quality_clean].copy()
    if not np.isfinite(valid.gross_1449.to_numpy()).all():
        raise ValueError("Nonfinite quality-clean gross return")
    counts = valid.groupby(list(KEY) + ["arm"], observed=True).size(
    ).unstack("arm", fill_value=0)
    comparable = counts.loc[counts.high.ge(2) & counts.low.ge(2)].reset_index(
    )[list(KEY)]
    valid = valid.merge(comparable, on=list(KEY), validate="many_to_one")
    strata = valid.groupby(list(KEY) + ["arm"], observed=True).gross_1449.mean(
    ).unstack("arm").reset_index()
    strata["edge"] = strata.high - strata.low
    daily = strata.groupby(["half", "date"], observed=True)[
        ["high", "low", "edge"]].mean().reset_index()
    by_half = {}
    for half in HALVES:
        part = daily.loc[daily.half.eq(half)]
        by_half[half] = {"days": int(len(part)),
                         "gross_high_pp": float(100 * part.high.mean()),
                         "gross_low_pp": float(100 * part.low.mean()),
                         "gross_edge_pp": float(100 * part.edge.mean())}
    by_year = {}
    for year in ("2024", "2025"):
        part = daily.loc[daily.date.str.startswith(year)]
        by_year[year] = {
            "days": int(len(part)),
            "edge_week_95ci_pp": [100 * number for number in
                                  _week_bootstrap(part.edge, part.date,
                                                  620 + int(year))],
        }
    gate = all(row["gross_high_pp"] > .25
               and row["gross_edge_pp"] > .10
               for row in by_half.values()) and all(
                   row["edge_week_95ci_pp"][0] > 0
                   for row in by_year.values())
    report = {"input_gate_passed": True,
              "selected_stock_days": int(len(joined)),
              "quality_clean_comparable_stock_days": int(len(valid)),
              "comparable_strata": int(len(strata)),
              "excluded_stock_days": int((~joined.quality_clean).sum()),
              "by_half": by_half, "by_year": by_year,
              "raw_reprice_gate_passed": bool(gate),
              "interpretation": "Next daily open is diagnostic, not an executable sale"}
    (output / "gross_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "gross"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    action = freeze if args.stage == "freeze" else evaluate_gross
    print(json.dumps(action(args.output), ensure_ascii=False, indent=2))
