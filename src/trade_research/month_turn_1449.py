"""Freeze a calendar timing comparison before reading new execution outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .absolute_ridge_1449 import feature_frame
from .corporate_cash import save_json, sha
from .quote_precision import fixed_quote_shares, quote_cents
from .shallow_tree_1449 import decision_buyable
from .turnover_reference import CALENDAR

ROOT = Path("data/research/month_turn_1449")
RULE_COMMIT = "ef2911a"


def calendar_schedule(calendar: list[str]) -> pd.DataFrame:
    if calendar != sorted(set(calendar)):
        raise ValueError("The trading calendar must be ordered and unique")
    index = {date: i for i, date in enumerate(calendar)}
    dates = pd.Series(calendar, dtype=str)
    result = []
    for month, group in dates.groupby(dates.str[:7], sort=True):
        if not "2024-01" <= month <= "2025-12":
            continue
        last = group.iloc[-1]
        end = index[last]
        if end + 10 >= len(calendar) or calendar[end + 10] > "2025-12-31":
            continue
        if end < 5 or calendar[end - 5][:7] != month:
            raise ValueError("A comparison date does not belong to the same month")
        for arm, date in (("high", last), ("low", calendar[end - 5])):
            result.append({"month": month, "arm": arm, "date": date,
                "half": month[:4] + ("H1" if month[5:7] <= "06" else "H2"),
                "planned_T5": calendar[index[date] + 5],
                "last_observation_day": calendar[index[date] + 10]})
    schedule = pd.DataFrame(result)
    if schedule.empty or schedule.date.duplicated().any():
        raise ValueError("Missing or overlapping calendar timing dates")
    return schedule


def freeze(output: Path = ROOT) -> dict:
    if (output / "repriced.parquet").exists():
        raise ValueError("Cannot replace the month-turn list after its returns exist")
    output.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(CALENDAR)
    if not table.calendar_date.eq("2025-12-31").any():
        raise ValueError("The calendar does not cover the complete evaluation period")
    calendar = sorted(table.loc[table.is_trading_day.eq("1")
        & table.calendar_date.between("2024-01-01", "2025-12-31"), "calendar_date"])
    schedule = calendar_schedule(calendar)
    c = duckdb.connect()
    c.execute("SET threads=4")
    features = feature_frame(c, date_ranges=tuple((d, d) for d in schedule.date))
    c.close()
    keep = [decision_buyable(r.code, r.price_1449, r.preclose) for r in features.itertuples()]
    features = features.loc[np.asarray(keep) & features.price_1449.ge(5)].copy()
    features["price_1449"] = features.price_1449.map(lambda value: quote_cents(value) / 100)
    features["price_signal"] = features.price_1449
    features = features.sort_values(["date", "code"]).reset_index(drop=True)
    selected, coverage = [], []
    for row in schedule.itertuples(index=False):
        available = features.loc[features.date.eq(row.date)]
        chosen = available.sort_values(["amount_1449", "code"], ascending=[False, True]).head(5).copy()
        chosen["arm"], chosen["month"], chosen["half"] = row.arm, row.month, row.half
        chosen["daily_rank"] = range(1, len(chosen) + 1)
        chosen["pair_id"] = row.month + ":slot:" + chosen.daily_rank.astype(str)
        chosen["decision_shares"] = [fixed_quote_shares(code, price, 20000)
            for code, price in zip(chosen.code, chosen.price_1449)]
        selected.append(chosen)
        coverage.append({"month": row.month, "arm": row.arm, "date": row.date,
            "eligible": len(available), "selected": len(chosen), "cash_slots": 5 - len(chosen)})
    signals = pd.concat(selected, ignore_index=True)
    signals["isST"], signals["reference_gap"], signals["listing_age_sessions"] = 0, False, 20
    if signals.duplicated(["date", "code"]).any() or not signals.date.isin(schedule.date).all():
        raise ValueError("Invalid month-turn signal identities")
    for name, frame in (("calendar_schedule", schedule), ("eligible_features", features), ("signals", signals)):
        frame.to_parquet(output / f"{name}.parquet", index=False, compression="zstd")
    sources = [*sorted(Path("data/research/minute_prefix_1449").glob("*/*.parquet")),
        *sorted(Path("data/research/market_snapshots_ci").glob("*.parquet"))]
    save_json(output / "feature_sources.json", {str(path): sha(path) for path in sources})
    report = {"rule_commit": RULE_COMMIT, "independent_months": schedule.month.nunique(),
        "first_month": schedule.month.min(), "last_month": schedule.month.max(),
        "signals": len(signals), "coverage": coverage,
        "by_half": signals.groupby(["half", "arm"]).agg(rows=("code", "size"),
            dates=("date", "nunique")).reset_index().to_dict("records"),
        "signals_sha256": sha(output / "signals.parquet"),
        "calendar_schedule_sha256": sha(output / "calendar_schedule.parquet"),
        "features_sha256": sha(output / "eligible_features.parquet"),
        "calendar_sha256": sha(CALENDAR), "feature_sources_sha256": sha(output / "feature_sources.json"),
        "source_docs_sha256": {str(path): sha(path) for path in sorted((output / "source_docs").glob("*.json"))},
        "new_execution_outcomes_read": False, "holdout_prices_read": False}
    save_json(output / "input_report.json", report)
    return report


if __name__ == "__main__":
    report = freeze()
    print(json.dumps({k: report[k] for k in ("independent_months", "signals", "by_half", "signals_sha256")},
        ensure_ascii=False, indent=2))
