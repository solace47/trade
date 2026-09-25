"""Test whether a frozen main-board ridge score ranks all eligible stocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .absolute_ridge import PERIODS, ROOT, feature_frame, fit, score, training_labels
from .annual_cash_direct_eval import _month_bootstrap
from .market_study import _quality_keys, _quality_symbols, _week_bootstrap
from .strategy_scan import _stressed_returns


OUTPUT = ROOT / "absolute_ridge_calibration"
HALVES = ("2024H2", "2025H1", "2025H2")


def _half(date: str) -> str:
    return date[:4] + ("H1" if int(date[5:7]) <= 6 else "H2")


def freeze_inputs(output_dir: Path = OUTPUT) -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    features = feature_frame(connection, main_only=True)
    connection.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                            ).create_view("outcomes")
    connection.register("bad_days", _quality_keys(ROOT / "market_issues_ci"))
    frames, training = [], []
    for train_end, test_first, test_last, period in PERIODS:
        train = features.loc[features.date.le(train_end)].copy()
        labels = training_labels(connection, train, test_first)
        model, audit = fit(train, labels)
        test = features.loc[features.date.between(test_first, test_last)]
        scored = score(test, model)[["date", "code", "score"]]
        frames.append(scored)
        training.append({"period": period, "train_rows": audit["train_rows"],
                         "train_exit_last": audit["train_exit_last"]})
    scored = pd.concat(frames, ignore_index=True)
    if (scored.empty or scored.duplicated(["date", "code"]).any()
            or not scored.date.str[:4].isin(("2024", "2025")).all()
            or not np.isfinite(scored.score).all()):
        raise ValueError("Invalid frozen all-market model scores")
    scored = scored.sort_values(["date", "score", "code"]).reset_index(drop=True)
    daily_size = scored.groupby("date").code.transform("size")
    rank = scored.groupby("date").cumcount()
    scored["quintile"] = (5 * rank // daily_size + 1).astype(int)
    scored["half"] = scored.date.map(_half)
    by_half = scored.groupby("half").agg(
        rows=("code", "size"), days=("date", "nunique"),
        positive_fraction=("score", lambda x: float(x.gt(0).mean())),
    ).reset_index().to_dict("records")
    min_daily = int(scored.groupby("date").size().min())
    connection.register("scored_keys", scored[["date", "code"]])
    archive_coverage = connection.execute("""
        SELECT COUNT(*) AS rows, COUNT(o.code) AS matched
        FROM scored_keys x LEFT JOIN outcomes o
          ON x.date = o.date AND x.code = o.code AND o.horizon = 5
    """).fetchone()
    quintile_count = scored.groupby(["date", "quintile"]).size().unstack()
    gate = (len(by_half) == 3
            and all(item["days"] >= 80 for item in by_half)
            and min_daily >= 500
            and quintile_count.notna().all().all()
            and archive_coverage == (len(scored), len(scored)))
    audit = {
        "rows": len(scored), "days": scored.date.nunique(),
        "min_daily_stocks": min_daily, "by_half": by_half,
        "archive_rows": archive_coverage[1],
        "training": training, "outcome_gate_passed": gate,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").unlink(missing_ok=True)
    scored.to_parquet(output_dir / "scores.parquet", index=False,
                      compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return audit


def _interval(values: pd.Series, dates: pd.Series, seed: int) -> dict:
    return {
        "week_ci": _week_bootstrap(values, dates, seed),
        "month_ci": _month_bootstrap(values, dates, seed + 1),
    }


def summarize_daily(daily: pd.DataFrame) -> dict:
    if (daily.empty or daily.duplicated(["date", "quintile"]).any()
            or daily.groupby("date").quintile.nunique().ne(5).any()):
        raise ValueError("Missing or duplicated daily score quintiles")
    wide = daily.pivot(index="date", columns="quintile", values="cash")
    stress = daily.pivot(index="date", columns="quintile", values="stress15")
    top_middle = wide[5] - wide[3]
    top_low = wide[5] - wide[1]
    top_middle_stress = stress[5] - stress[3]
    top_low_stress = stress[5] - stress[1]
    means = daily.groupby("quintile").agg(
        cash=("cash", "mean"), score=("score", "mean"),
        entry=("entry", "mean"), clean_exit=("clean_exit", "mean"),
        stress15=("stress15", "mean"),
    )
    dates = pd.Series(wide.index)
    return {
        "days": len(wide), "quintiles": {
            str(k): {column: float(value) for column, value in row.items()}
            for k, row in means.to_dict(orient="index").items()},
        "top_minus_middle": float(top_middle.mean()),
        "top_minus_low": float(top_low.mean()),
        "top_minus_middle_stress15": float(top_middle_stress.mean()),
        "top_minus_low_stress15": float(top_low_stress.mean()),
        "top_middle_intervals": _interval(top_middle, dates, 2801),
        "top_low_intervals": _interval(top_low, dates, 2803),
        "top_cash_intervals": _interval(wide[5], dates, 2805),
    }


def evaluate(output_dir: Path = OUTPUT) -> dict:
    audit = json.loads((output_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if not audit["outcome_gate_passed"]:
        raise ValueError("Frozen calibration input gate failed")
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(output_dir / "scores.parquet")
                            ).create_view("scores")
    connection.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                            ).create_view("outcomes")
    connection.register("bad_days", _quality_keys(ROOT / "market_issues_ci"))
    connection.register("bad_symbols", _quality_symbols(ROOT / "market_issues_ci"))
    rows = connection.execute("""
        SELECT x.date, x.code, x.half, x.quintile, x.score,
               o.entry_status, o.entry_price, o.shares,
               o.exit_status, o.exit_date, o.exit_delay_sessions,
               o.exit_price, o.net_return,
               o.exit_status = 'filled' AND o.exit_delay_sessions = 0
               AND NOT EXISTS (SELECT 1 FROM bad_symbols b
                               WHERE b.code = x.code)
               AND NOT EXISTS (
                   SELECT 1 FROM bad_days q
                   WHERE q.code = x.code AND q.date >= x.date
                     AND q.date <= o.exit_date
               ) AS valid
        FROM scores x JOIN outcomes o USING (date, code)
        WHERE o.horizon = 5
    """).df()
    if (len(rows) != audit["rows"]
            or rows.duplicated(["date", "code"]).any()
            or rows.loc[rows.valid, "net_return"].isna().any()):
        raise ValueError("Incomplete archived minute outcomes")
    rows["cash"] = rows.net_return.where(rows.valid, 0.0)
    rows["stress15"] = 0.0
    rows.loc[rows.valid, "stress15"] = _stressed_returns(
        rows.loc[rows.valid], 15)
    rows["entry"] = rows.entry_status.eq("filled").astype(float)
    rows["clean_exit"] = rows.valid.astype(float)
    daily = rows.groupby(["date", "half", "quintile"], sort=True).agg(
        stocks=("code", "size"), cash=("cash", "mean"),
        stress15=("stress15", "mean"), score=("score", "mean"),
        entry=("entry", "mean"), clean_exit=("clean_exit", "mean"),
    ).reset_index()
    report = {"input_audit": audit, "results": {}}
    for half in HALVES:
        report["results"][half] = summarize_daily(daily.loc[daily.half.eq(half)])
    report["results"]["2025"] = summarize_daily(
        daily.loc[daily.date.str.startswith("2025")])
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    result = evaluate(args.output) if args.evaluate else freeze_inputs(args.output)
    print({"report": str(args.output / "report.json")}
          if args.evaluate else result)


if __name__ == "__main__":
    main()
