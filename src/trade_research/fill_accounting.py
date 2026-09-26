"""Audit the zeroed score of a fixed, rejected 14:49 model; no new selection.

Known contributions retain the original signal-day denominators. Unknown
post-entry proceeds remain missing. Neither contribution is portfolio P&L.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .absolute_ridge_1449_eval import apply_period_quality
from .hf_outcomes import Assumptions, _fees, outcomes_for_symbol


ROOT = Path("data/research")
SOURCE = ROOT / "absolute_ridge_1449_target"
OUTPUT = ROOT / "fill_accounting"
CALENDAR = Path("data/baostock/market_2020_2026/metadata/calendar.parquet")
HALVES = ("2024H2", "2025H1", "2025H2")
CATEGORIES = (
    "not_bought", "clean_ontime", "clean_delayed", "exit_quality_unverified",
    "corporate_action_unverified", "no_exit_recorded",
)
KEY = ["date", "code", "target_notional", "horizon"]


def _json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def _hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def freeze(source: Path = SOURCE, output: Path = OUTPUT) -> dict:
    files = [source / name for name in (
        "selections.parquet", "repricing_signals.parquet", "repriced.parquet",
        "report.json", "input_audit.json",
    )]
    files += [ROOT / "quality_period_2024_2025.json", CALENDAR]
    issues = sorted((ROOT / "market_issues_ci").glob("shard_*.csv"))
    if len(issues) != 20:
        raise ValueError("All 20 issue shards must be frozen")
    files += issues
    selected = pd.read_parquet(source / "selections.parquet",
                               columns=["date", "code", "candidate", "pair_id"])
    if (selected.empty or selected.duplicated(["date", "code"]).any()
            or not selected.date.between("2024-07-01", "2025-12-17").all()):
        raise ValueError("Invalid frozen selection dates or keys")
    manifest = {
        "frozen_commit": "10489ac", "source_dir": str(source),
        "selected_rows": len(selected),
        "sha256": {str(path): _hash(path) for path in files},
        "signal_cutoff": "1449", "entry_exit_labels": "1452-1455",
        "holdout_read": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "manifest.json"
    if destination.exists() and json.loads(destination.read_text()) != manifest:
        raise ValueError("An existing frozen manifest cannot be replaced")
    _json(destination, manifest)
    return manifest


def verify_manifest(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if _hash(Path(name)) != expected:
            raise ValueError(f"Frozen input changed: {name}")
    return manifest


def account_rows(trades: pd.DataFrame, calendar: list[str]) -> pd.DataFrame:
    """Keep every actual modeled purchase, including unverifiable proceeds."""
    # DuckDB joins may emit a different row order. Fix summation order too.
    rows = trades.sort_values(KEY).copy().reset_index(drop=True)
    if (rows.empty or rows.duplicated(KEY).any() or not calendar
            or calendar != sorted(set(calendar))
            or calendar[0] < "2024-01-01" or calendar[-1] > "2025-12-31"
            or not rows.date.between("2024-01-01", "2025-12-31").all()):
        raise ValueError("Invalid dates, calendar or trade keys")
    index = {day: i for i, day in enumerate(calendar)}
    for field in ("date", "target_exit_date", "exit_date"):
        if not set(rows[field].dropna()).issubset(index):
            raise ValueError(f"{field} exceeds the frozen calendar")
    bought = rows.entry_status.eq("filled")
    filled = rows.exit_status.eq("filled")
    clean = filled & rows.quality_clean_exit
    corporate = rows.exit_status.eq("corporate_action_unadjusted")
    if (rows.quality_clean_exit.isna().any()
            or ((filled | corporate) & ~bought).any()
            or (bought & (~np.isfinite(rows.entry_price) | rows.entry_price.le(0)
                         | rows.shares.le(0))).any()
            or (~bought & rows.shares.ne(0)).any()
            or (filled & (~np.isfinite(rows.net_return)
                          | rows.exit_date.isna())).any()
            or (rows.quality_clean_exit & ~filled).any()):
        raise ValueError("Inconsistent buy, sell or quality fields")
    rows["category"] = np.select(
        [~bought, clean & rows.exit_delay_sessions.eq(0),
         clean & rows.exit_delay_sessions.gt(0), filled & ~clean, corporate],
        CATEGORIES[:-1], default="no_exit_recorded",
    )
    recorded = bought & rows.exit_date.notna()
    duration = rows.exit_date.map(index) - rows.date.map(index)
    expected_delay = rows.exit_date.map(index) - rows.target_exit_date.map(index)
    target_duration = rows.target_exit_date.map(index) - rows.date.map(index)
    if ((recorded & (~(filled | corporate) | rows.exit_price.le(0)
                     | ~np.isfinite(rows.exit_price) | duration.le(0)
                     | expected_delay.lt(0)
                     | expected_delay.ne(rows.exit_delay_sessions))).any()
            or (bought & (rows.target_exit_date.isna()
                          | target_duration.ne(rows.horizon))).any()
            or ((filled | corporate) & ~recorded).any()):
        raise ValueError("Inconsistent target, exit date or delay")

    rows["half"] = rows.date.str[:4] + np.where(
        rows.date.str[5:7].astype(int) <= 6, "H1", "H2")
    rows["unknown_after_buy"] = bought & ~clean
    terms = Assumptions()
    for slip in (5, 15):
        # A cost sensitivity with original fill decisions, not a new fill test.
        buy_price = rows.entry_price / 1.0005 * (1 + slip / 10000)
        sell_price = rows.exit_price / .9995 * (1 - slip / 10000)
        buy_cost = pd.Series(0.0, index=rows.index)
        proceeds = pd.Series(np.nan, index=rows.index)
        for i in rows.index[bought]:
            value = rows.at[i, "shares"] * buy_price.at[i]
            buy_cost.at[i] = value + _fees(value, "buy", terms, rows.at[i, "date"])
        for i in rows.index[filled]:
            value = rows.at[i, "shares"] * sell_price.at[i]
            proceeds.at[i] = value - _fees(value, "sell", terms,
                                          rows.at[i, "exit_date"])
        computed = proceeds / buy_cost - 1
        if slip == 5 and not np.allclose(computed[filled], rows.net_return[filled],
                                         atol=1e-12, rtol=0):
            raise ValueError("Stored net returns disagree with cash accounting")
        known = computed.where(clean)
        known.loc[~bought] = 0.0
        rows[f"buy_cost{slip}"] = buy_cost
        rows[f"verified_proceeds{slip}"] = proceeds.where(clean)
        rows[f"known_return{slip}"] = known
        rows[f"old_score{slip}"] = known.where(
            rows.category.eq("clean_ontime"), 0.0)
        rows[f"delay_contribution{slip}"] = known.where(
            rows.category.eq("clean_delayed"), 0.0)
    rows["extra_capital_sessions5"] = (
        rows.buy_cost5 * rows.exit_delay_sessions
    ).where(rows.category.eq("clean_delayed"), 0.0)
    return rows


def daily_mean(rows: pd.DataFrame, column: str) -> float:
    return float(rows.groupby("date", sort=True)[column].mean().mean())


def summarize_cell(rows: pd.DataFrame) -> dict:
    missing = rows.unknown_after_buy
    report = {
        "signals": len(rows), "signal_days": int(rows.date.nunique()),
        "entry_fills": int(rows.entry_status.eq("filled").sum()),
        "categories": {name: int(rows.category.eq(name).sum())
                       for name in CATEGORIES},
        "buy_cost5_total": float(rows.buy_cost5.sum()),
        "unknown_after_buy": int(missing.sum()),
        "unknown_buy_cost5": float(rows.loc[missing, "buy_cost5"].sum()),
        "delayed_extra_capital_sessions5": float(rows.extra_capital_sessions5.sum()),
    }
    for slip in (5, 15):
        old = daily_mean(rows, f"old_score{slip}")
        added = daily_mean(rows, f"delay_contribution{slip}")
        report[f"old_score{slip}"] = old
        report[f"delay_contribution{slip}"] = added
        report[f"known_contribution{slip}"] = old + added
        # Never let pandas drop missing trades from a return denominator.
        report[f"complete_cohort_mean{slip}"] = (
            None if missing.any() else daily_mean(rows, f"known_return{slip}"))
        report[f"zero_score_sensitive{slip}"] = bool(
            abs(added) >= .0002 or np.sign(old) != np.sign(old + added))
    return report


def _paired_mean(rows: pd.DataFrame, column: str) -> float:
    pairs = rows.loc[rows.candidate.eq("absolute_model")].merge(
        rows.loc[rows.candidate.eq("same_day_control")],
        on=["date", "pair_id"], suffixes=("_model", "_control"),
        validate="one_to_one",
    )
    if len(pairs) != rows.candidate.eq("same_day_control").sum():
        raise ValueError("A fixed control has no selected counterpart")
    return float((pairs[f"{column}_model"] - pairs[f"{column}_control"])
                 .groupby(pairs.date).mean().mean())


def evaluate(output: Path = OUTPUT) -> dict:
    manifest = verify_manifest(output)
    source = Path(manifest["source_dir"])
    audit = json.loads((source / "input_audit.json").read_text())
    if (not audit["outcome_gate_passed"] or audit["cutoff_label"] != "1449"
            or audit["decision_price_column"] != "price_1449"):
        raise ValueError("The original input gate did not pass")
    selections = pd.read_parquet(source / "selections.parquet",
                                 columns=["date", "code", "candidate", "pair_id"])
    raw = pd.read_parquet(source / "repriced.parquet",
                           filters=[("exit_window", "==", "close")])
    if (len(raw) != len(selections) * 4 or raw.duplicated(KEY).any()
            or set(raw.target_notional) != {20000, 100000}
            or set(raw.horizon) != {1, 5}
            or not raw.entry_window.eq("baseline").all()
            or not raw.entry_label.eq("1452-1455").all()
            or not raw.exit_label.eq("1452-1455").all()):
        raise ValueError("Incomplete frozen close-window grid")
    for _, part in raw.groupby(["target_notional", "horizon"]):
        if set(zip(part.date, part.code)) != set(zip(selections.date, selections.code)):
            raise ValueError("A cell changed the frozen stock-day keys")
    qualified, released = apply_period_quality(raw)
    qualified = qualified.merge(selections, on=["date", "code"],
                                  validate="many_to_one")
    calendar_table = pd.read_parquet(CALENDAR)
    calendar = sorted(calendar_table.loc[
        calendar_table.is_trading_day.eq("1")
        & calendar_table.calendar_date.between("2024-01-01", "2025-12-31"),
        "calendar_date"].tolist())
    rows = account_rows(qualified, calendar)
    if set(rows.half) != set(HALVES):
        raise ValueError("Unexpected evaluation halves")
    old = json.loads((source / "report.json").read_text())
    cells, pairs, comparisons = [], [], []
    for (notional, horizon, half), group in rows.groupby(
            ["target_notional", "horizon", "half"], sort=True):
        archived = old["results"][str(int(notional))][str(int(horizon))]["close"][half]
        model = group.loc[group.candidate.eq("absolute_model")]
        values = {
            "own_all_cash": daily_mean(model, "old_score5"),
            "own_all_cash_stress15": daily_mean(model, "old_score15"),
            "matched_edge_cash": _paired_mean(group, "old_score5"),
            "matched_edge_stress15": _paired_mean(group, "old_score15"),
        }
        for key, value in values.items():
            delta = value - archived[key]
            if abs(delta) > 1e-12:
                raise ValueError(f"Old score does not reconcile: {half}, {key}")
            comparisons.append(abs(delta))
        for arm, part in group.groupby("candidate", sort=True):
            cells.append({"notional": int(notional), "horizon": int(horizon),
                          "half": half, "arm": arm, **summarize_cell(part)})
        pairs.append({
            "notional": int(notional), "horizon": int(horizon), "half": half,
            **{f"old_edge{slip}": _paired_mean(group, f"old_score{slip}")
               for slip in (5, 15)},
            **{f"delay_edge{slip}": _paired_mean(group, f"delay_contribution{slip}")
               for slip in (5, 15)},
        })
    report = {
        "source": str(source), "rows": len(rows),
        "quality_released_rows": released,
        "old_score_comparisons": len(comparisons),
        "max_old_score_difference": max(comparisons),
        "cells": cells, "pairs": pairs,
        "continuation_required": bool(rows.category.eq("no_exit_recorded").any()),
        "qualification": "Fixed rejected list; unknown proceeds stay missing; "
                         "not portfolio P&L. 15bps retains original fill decisions.",
        "holdout_read": False,
    }
    rows.to_parquet(output / "accounted.parquet", index=False)
    rows.loc[rows.unknown_after_buy].to_parquet(output / "unknown.parquet", index=False)
    report["accounted_sha256"] = _hash(output / "accounted.parquet")
    _json(output / "report.json", report)
    return report


def continue_exits(output: Path = OUTPUT) -> dict:
    """Trace only previously unresolved fills through the frozen 2025 end."""
    manifest = verify_manifest(output)
    source = Path(manifest["source_dir"])
    report = json.loads((output / "report.json").read_text())
    if _hash(output / "accounted.parquet") != report["accounted_sha256"]:
        raise ValueError("The evaluated accounting rows changed")
    original = pd.read_parquet(output / "accounted.parquet")
    unresolved = original.loc[original.category.eq("no_exit_recorded")]
    if unresolved.empty:
        return {"required": False, "rows": 0}
    all_signals = pd.read_parquet(source / "repricing_signals.parquet")
    calendar_table = pd.read_parquet(CALENDAR)
    calendar = sorted(calendar_table.loc[
        calendar_table.is_trading_day.eq("1")
        & calendar_table.calendar_date.between("2024-01-01", "2025-12-31"),
        "calendar_date"].tolist())
    checks = ["entry_status", "entry_price", "shares", "target_exit_date",
              "exit_status", "exit_date", "exit_delay_sessions", "exit_price",
              "net_return", "corporate_action_crossed"]
    frames, raw_hashes = [], {}
    for code, group in unresolved.groupby("code", sort=True):
        exchange, symbol = code.split(".")
        minute_file = Path("data/hf/pilot/data/stock_1m") / exchange.upper() / f"{symbol}.parquet"
        daily_file = CALENDAR.parent.parent / "daily" / f"{exchange}_{symbol}.parquet"
        raw_hashes.update({str(p): _hash(p) for p in (minute_file, daily_file)})
        minute = pd.read_parquet(
            minute_file, columns=["timestamp", "volume", "turnover"],
            filters=[("timestamp", ">=", pd.Timestamp(group.date.min())),
                     ("timestamp", "<", pd.Timestamp("2026-01-01"))],
        )
        minute["label"] = minute.timestamp.dt.strftime("%H%M")
        minute = minute.loc[minute.label.isin(["1452", "1453", "1454", "1455"])].copy()
        minute["date"] = minute.timestamp.dt.strftime("%Y-%m-%d")
        minute = minute.sort_values("timestamp")
        daily = pd.read_parquet(daily_file, filters=[
            ("date", ">=", "2024-01-01"), ("date", "<=", "2025-12-31")])
        daily = pd.concat([
            daily.loc[daily.date.lt(group.date.min())].tail(10),
            daily.loc[daily.date.ge(group.date.min())],
        ]).sort_values("date")
        for (amount, horizon), part in group.groupby(["target_notional", "horizon"]):
            signals = all_signals.loc[
                all_signals.code.eq(code) & all_signals.date.isin(part.date)]
            if len(signals) != len(part):
                raise ValueError("Continuation keys lack frozen signals")
            baseline = outcomes_for_symbol(
                signals, minute, daily, calendar,
                Assumptions(target_notional=amount), horizons=(int(horizon),),
                sizing_price_column="price_1449")
            joined = baseline.merge(part, on=["date", "code"],
                                      suffixes=("_raw", "_archive"), validate="one_to_one")
            for field in checks:
                for a, b in zip(joined[f"{field}_raw"], joined[f"{field}_archive"], strict=True):
                    if pd.isna(a) and pd.isna(b):
                        continue
                    if pd.isna(a) or pd.isna(b) or (
                            not np.isclose(a, b, atol=1e-12, rtol=0)
                            if isinstance(a, (float, np.floating)) else a != b):
                        raise ValueError(f"Original raw fill changed: {code} {field}")
            extended = outcomes_for_symbol(
                signals, minute, daily, calendar,
                Assumptions(target_notional=amount,
                            maximum_exit_delay_sessions=len(calendar)),
                horizons=(int(horizon),), sizing_price_column="price_1449")
            extended["target_notional"] = amount
            extended["entry_window"] = "baseline"
            extended["exit_window"] = "close"
            extended["quality_clean_exit"] = False
            extended, _ = apply_period_quality(extended)
            extended = extended.merge(part[["date", "code", "candidate", "pair_id"]],
                                        on=["date", "code"], validate="one_to_one")
            frames.append(extended)
    traced = pd.concat(frames, ignore_index=True)
    traced = account_rows(traced, calendar)
    final = pd.concat([
        original.loc[~original.category.eq("no_exit_recorded")], traced,
    ], ignore_index=True).sort_values(KEY)
    if len(final) != len(original) or final.duplicated(KEY).any():
        raise ValueError("Continuation changed the original cohort")
    unchanged = final.merge(original, on=KEY, suffixes=("_new", "_old"),
                              validate="one_to_one")
    for name in ("old_score5", "old_score15", "buy_cost5", "buy_cost15"):
        if not np.allclose(unchanged[f"{name}_new"], unchanged[f"{name}_old"],
                           atol=1e-12, rtol=0):
            raise ValueError("Continuation changed buys or on-time scores")
    cells = [{"notional": int(amount), "horizon": int(horizon),
              "half": half, "arm": arm, **summarize_cell(part)}
             for (amount, horizon, half, arm), part in final.groupby(
                 ["target_notional", "horizon", "half", "candidate"], sort=True)]
    report = {
        "required": True, "rows": len(traced),
        "raw_baseline_field_checks": len(traced) * len(checks),
        "last_allowed_date": calendar[-1], "raw_sha256": raw_hashes,
        "traced_categories": {name: int(traced.category.eq(name).sum())
                              for name in CATEGORIES},
        "traced_max_delay_sessions": (int(traced.exit_delay_sessions.max())
                                       if traced.exit_delay_sessions.notna().any() else None),
        "remaining_unknown_rows": int(final.unknown_after_buy.sum()),
        "cells": cells,
        "pairs": [{
            "notional": int(amount), "horizon": int(horizon), "half": half,
            **{f"known_edge{slip}": (
                _paired_mean(part, f"old_score{slip}")
                + _paired_mean(part, f"delay_contribution{slip}"))
               for slip in (5, 15)},
        } for (amount, horizon, half), part in final.groupby(
            ["target_notional", "horizon", "half"], sort=True)],
        "holdout_read": False,
        "qualification": "Continuation is an accounting diagnostic, not a new "
                         "holding rule or validation of the rejected model.",
    }
    traced.to_parquet(output / "continued_exits.parquet", index=False)
    final.to_parquet(output / "extended_accounted.parquet", index=False)
    _json(output / "continuation.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "evaluate", "continue"))
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    result = (freeze(args.source, args.output) if args.stage == "freeze"
              else evaluate(args.output) if args.stage == "evaluate"
              else continue_exits(args.output))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
