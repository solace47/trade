"""One preregistered tail exit policy on an unchanged historical buy cohort."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from math import isfinite
from pathlib import Path
import shutil

import duckdb
import numpy as np
import pandas as pd

from .absolute_ridge_1449_eval import apply_period_quality
from .corporate_cash import save_json, sha
from .hf_outcomes import Assumptions, outcomes_for_symbol
from .quote_precision import quote_cents
from .reference_gain_accounting import evaluate, weekly_interval
from .reference_gain_eval import DAILY, MINUTES, account_with_windows
from .shallow_tree_eval import daily_returns
from .turnover_reference import CALENDAR

ROOT = Path("data/research/tail_barrier_exit")
SOURCE = Path("data/research/downside_ridge_1449/downside")
CONTINUED = SOURCE.parent / "continued/downside"
SIGNAL_SHA = "94391305ea93f56832db511129700158c28c1ab72c7317e4d56869350b8986ff"
RULE = "b9ebb6a"
KEY = ["date", "code"]
CHECKS = ["entry_status", "entry_price", "shares", "target_exit_date", "exit_status",
          "exit_date", "exit_delay_sessions", "exit_price", "net_return", "corporate_action_crossed"]


def calendar_days() -> list[str]:
    table = pd.read_parquet(CALENDAR)
    return sorted(table.loc[table.is_trading_day.eq("1")
        & table.calendar_date.between("2024-01-01", "2025-12-31"), "calendar_date"].tolist())


def decision_price(bar: dict | None) -> tuple[float | None, str]:
    if bar is None:
        return None, "missing_1449"
    if not isfinite(float(bar["volume"])) or bar["volume"] <= 0:
        return None, "no_positive_volume"
    values = [float(bar[k]) for k in ("open", "high", "low", "close")]
    if any(not isfinite(v) or v <= 0 for v in values):
        return None, "invalid_ohlc"
    try:
        op, hi, lo, close = [quote_cents(v) / 100 for v in values]
    except ValueError:
        return None, "off_tick_ohlc"
    if not lo <= min(op, close) <= max(op, close) <= hi:
        return None, "inconsistent_ohlc"
    return close, "valid"


def reference_states(daily: pd.DataFrame) -> dict[str, str]:
    """Today's state uses today's preclose and an earlier close, never today's close."""
    previous = None
    states = {}
    for row in daily.sort_values("date").itertuples(index=False):
        if row.tradestatus != 1:
            states[row.date] = "not_trading"
            continue
        preclose = float(row.preclose)
        if previous is None or not isfinite(previous) or previous <= 0 or not isfinite(preclose) or preclose <= 0:
            state = "reference_unknown"
        else:
            state = "reference_gap" if abs(preclose - previous) > .005 else "ordinary"
        states[row.date] = state
        previous = float(row.close)
    return states


def plan_exit(entry: dict, bars: dict[str, dict], references: dict[str, str],
              calendar: list[str]) -> tuple[dict, list[dict]]:
    start = calendar.index(entry["date"])
    result = {"date": entry["date"], "code": entry["code"], "planned_horizon": 5,
              "decision_date": calendar[start + 5], "decision_reason": "time_exit",
              "decision_price": None, "decision_price_change": None}
    trace = []
    if entry["entry_status"] != "filled":
        result["decision_reason"] = "not_bought"
        return result, trace
    for horizon, day in enumerate(calendar[start + 1:start + 5], 1):
        state = references.get(day, "reference_unknown")
        price, status = decision_price(bars.get(day))
        event = {"date": entry["date"], "code": entry["code"], "decision_date": day,
                 "elapsed_sessions": horizon, "reference_state": state,
                 "quote_status": status, "price": price, "price_change": None,
                 "action": "wait"}
        if state in ("reference_gap", "reference_unknown"):
            event["action"] = "reference_fallback"
            trace.append(event)
            result["decision_reason"] = "reference_fallback"
            return result, trace
        if state == "ordinary" and price is not None:
            change = price / entry["entry_price"] - 1
            event["price_change"] = change
            reason = "take_profit" if change >= .03 else "stop_loss" if change <= -.03 else None
            if reason:
                event["action"] = reason
                trace.append(event)
                result.update(planned_horizon=horizon, decision_date=day,
                    decision_reason=reason, decision_price=price, decision_price_change=change)
                return result, trace
        trace.append(event)
    return result, trace


def verify_inputs(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["sources"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    return manifest


def freeze(output: Path = ROOT) -> dict:
    if (output / "repriced.parquet").exists() or (output / "targets.parquet").exists():
        raise ValueError("Do not replace the fixed exit policy inputs")
    if sha(SOURCE / "signals.parquet") != SIGNAL_SHA:
        raise ValueError("The original buy cohort changed")
    files = [CALENDAR, SOURCE / "signals.parquet", SOURCE / "repriced.parquet",
             SOURCE / "execution_report.json"]
    files += [CONTINUED / n for n in ("repriced.parquet", "raw_windows.parquet",
        "window_quality.parquet", "execution_report.json", "catalog_scenario.parquet",
        "catalog_scenario_report.json")]
    manifest = {"rule_commit": RULE, "sources": {str(p): sha(p) for p in files},
        "barrier_fraction": .03, "notional": 20000, "holdout_read": False,
        "interpretation": "exploratory_exit_change_on_previously_seen_buy_cohort"}
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "manifest.json", manifest)
    shutil.copyfile(SOURCE / "signals.parquet", output / "signals.parquet")
    original = pd.read_parquet(SOURCE / "repriced.parquet")
    entries = original.loc[original.horizon.eq(5)].copy()
    calendar = calendar_days()
    indices = {day: i for i, day in enumerate(calendar)}
    old_sources = {r["code"]: r for r in json.loads((SOURCE / "execution_sources.json").read_text())}

    def one(item):
        code, trades = item
        exchange, symbol = code.split(".")
        minute_path = MINUTES / exchange.upper() / (symbol + ".parquet")
        daily_path = DAILY / (code.replace(".", "_") + ".parquet")
        fingerprints = {"code": code, "minute_sha256": sha(minute_path), "daily_sha256": sha(daily_path)}
        if fingerprints != old_sources[code]:
            raise ValueError("The existing underlying raw stock file changed")
        dates = sorted({d for row in trades.itertuples() if row.entry_status == "filled"
            for d in calendar[indices[row.date] + 1:indices[row.date] + 5]})
        if not dates:
            return [plan_exit(row, {}, {}, calendar)[0] for row in trades.to_dict("records")], [], pd.DataFrame(), fingerprints
        c = duckdb.connect()
        c.execute("SET threads=1")
        c.read_parquet(str(minute_path)).create_view("raw_minutes")
        c.register("needed", pd.DataFrame({"date": dates}))
        bars = c.execute("""SELECT timestamp,open,high,low,close,volume,turnover
            FROM raw_minutes WHERE timestamp>=?::TIMESTAMP AND timestamp<?::DATE+INTERVAL 1 DAY
              AND strftime(timestamp,'%H%M')='1449'
              AND strftime(timestamp,'%Y-%m-%d') IN (SELECT date FROM needed)
            ORDER BY timestamp""", [dates[0], dates[-1]]).df()
        c.read_parquet(str(daily_path)).create_view("raw_daily")
        daily = c.execute("""SELECT date,tradestatus,preclose,close FROM raw_daily
            WHERE date BETWEEN ? AND ? ORDER BY date""", [trades.date.min(), dates[-1]]).df()
        c.close()
        bars["date"], bars["code"] = bars.timestamp.dt.strftime("%Y-%m-%d"), code
        if bars.duplicated("date").any():
            raise ValueError("Ambiguous 14:49 decision bar")
        lookup = {r["date"]: r for r in bars.to_dict("records")}
        references = reference_states(daily)
        plans, trace = [], []
        for entry in trades.to_dict("records"):
            target, audit = plan_exit(entry, lookup, references, calendar)
            plans.append(target); trace.extend(audit)
        return plans, trace, bars, fingerprints

    groups = list(entries.groupby("code", sort=True))
    plans, trace, bars, fingerprints = [], [], [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (p, t, b, f) in enumerate(pool.map(one, groups), 1):
            plans.extend(p); trace.extend(t); bars.append(b); fingerprints.append(f)
            if i % 200 == 0 or i == len(groups):
                print(f"Read 14:49 exit decisions {i}/{len(groups)} stocks", flush=True)
    targets = pd.DataFrame(plans).sort_values(KEY).reset_index(drop=True)
    for name, frame in (("targets", targets), ("decision_trace", pd.DataFrame(trace)),
                        ("decision_bars", pd.concat(bars, ignore_index=True))):
        frame.to_parquet(output / (name + ".parquet"), index=False, compression="zstd")
    save_json(output / "decision_sources.json", fingerprints)
    signals = pd.read_parquet(output / "signals.parquet")
    detailed = targets.merge(signals[KEY + ["arm", "half"]], on=KEY, validate="one_to_one")
    report = {**manifest, "signals_sha256": SIGNAL_SHA, "targets_sha256": sha(output / "targets.parquet"),
        "rows": len(targets), "candidates": int(signals.arm.eq("high").sum()),
        "controls": int(signals.arm.eq("low").sum()),
        "reasons": detailed.groupby(["half", "arm", "decision_reason"]).size().rename("rows").reset_index().to_dict("records"),
        "trace_sha256": sha(output / "decision_trace.parquet"),
        "decision_bars_sha256": sha(output / "decision_bars.parquet")}
    save_json(output / "input_report.json", report)
    return report


def assert_same_execution(left: pd.DataFrame, right: pd.DataFrame,
                          fields: list[str] = CHECKS) -> None:
    a, b = (f.set_index(KEY)[fields].sort_index().astype(object) for f in (left, right))
    pd.testing.assert_frame_equal(a.where(a.notna(), None), b.where(b.notna(), None),
        check_dtype=False, check_exact=False, atol=1e-12, rtol=0)


def reprice(output: Path = ROOT) -> dict:
    verify_inputs(output)
    inputs = json.loads((output / "input_report.json").read_text())
    if sha(output / "targets.parquet") != inputs["targets_sha256"]:
        raise ValueError("Fixed exit decisions changed")
    signals = pd.read_parquet(output / "signals.parquet")
    if sha(output / "signals.parquet") != SIGNAL_SHA:
        raise ValueError("Fixed buys changed")
    plans = pd.read_parquet(output / "targets.parquet")
    joined = signals.merge(plans, on=KEY, validate="one_to_one")
    bars = pd.read_parquet(CONTINUED / "raw_windows.parquet")
    windows = pd.read_parquet(CONTINUED / "window_quality.parquet")
    original = pd.read_parquet(SOURCE / "repriced.parquet")
    original = original.loc[original.horizon.eq(5)]
    calendar = calendar_days()
    sources = {r["code"]: r for r in json.loads((output / "decision_sources.json").read_text())}
    results, continued, replays = [], [], []
    bar_groups = {code: group for code, group in bars.groupby("code", sort=False)}
    groups = list(joined.groupby("code", sort=True))
    for i, (code, chosen) in enumerate(groups, 1):
        path = DAILY / (code.replace(".", "_") + ".parquet")
        if sha(path) != sources[code]["daily_sha256"]:
            raise ValueError("Daily reference data changed")
        minute = bar_groups[code]
        c = duckdb.connect()
        c.read_parquet(str(path)).create_view("daily_source")
        prior = c.execute("SELECT max(date) FROM daily_source WHERE date<? AND tradestatus=1", [chosen.date.min()]).fetchone()[0]
        daily = c.execute("SELECT * FROM daily_source WHERE date BETWEEN ? AND ? ORDER BY date", [prior, minute.date.max()]).df()
        c.close()
        replay = outcomes_for_symbol(chosen, minute, daily, calendar, Assumptions(target_notional=20000),
            horizons=(5,), sizing_price_column="price_1449")
        assert_same_execution(replay, original.loc[original.code.eq(code)])
        replays.append(replay)
        for horizon, group in chosen.groupby("planned_horizon"):
            horizon = int(horizon)
            raw = outcomes_for_symbol(group, minute, daily, calendar,
                Assumptions(target_notional=20000, maximum_exit_delay_sessions=10-horizon),
                horizons=(horizon,), sizing_price_column="price_1449")
            results.append(raw)
            pending = raw.entry_status.eq("filled") & raw.exit_price.isna()
            if pending.any():
                extended = outcomes_for_symbol(group.loc[group.date.isin(raw.loc[pending, "date"])],
                    minute, daily, calendar, Assumptions(target_notional=20000, maximum_exit_delay_sessions=len(calendar)),
                    horizons=(horizon,), sizing_price_column="price_1449")
                continued.append(extended)
        if i % 200 == 0 or i == len(groups):
            print(f"Replayed buys and conditional exits {i}/{len(groups)} stocks", flush=True)
    initial = pd.concat(results, ignore_index=True).sort_values(KEY).reset_index(drop=True)
    assert_same_execution(initial, original, ["entry_price", "entry_status", "shares"])
    initial["target_notional"], initial["entry_window"], initial["exit_window"] = 20000, "baseline", "close"
    initial["quality_clean_exit"] = False
    initial, _ = apply_period_quality(initial)
    initial.to_parquet(output / "original_delay_repriced.parquet", index=False)
    pending_keys = initial.loc[initial.entry_status.eq("filled") & initial.exit_price.isna()].set_index(KEY).index
    old_pending = original.loc[original.entry_status.eq("filled") & original.exit_price.isna()].set_index(KEY).index
    if not pending_keys.isin(old_pending).all():
        raise ValueError("New unresolved holdings need additional raw exit windows")
    replacement = pd.concat(continued, ignore_index=True) if continued else initial.iloc[:0]
    raw = pd.concat([initial.loc[~initial.set_index(KEY).index.isin(pending_keys)], replacement], ignore_index=True)
    raw["target_notional"], raw["entry_window"], raw["exit_window"] = 20000, "baseline", "close"
    raw["quality_clean_exit"] = False
    raw, _ = apply_period_quality(raw)
    raw = raw.sort_values(KEY).reset_index(drop=True)
    accounted = account_with_windows(raw, signals, windows, calendar)
    original_accounted = account_with_windows(initial, signals, windows, calendar)
    original_accounted.to_parquet(output / "original_delay_accounted.parquet", index=False)
    for name, frame in (("repriced", raw), ("accounted_initial", accounted),
                        ("raw_windows", bars), ("window_quality", windows)):
        frame.to_parquet(output / (name + ".parquet"), index=False, compression="zstd")
    report = {"rule_commit": RULE, "signals_sha256": SIGNAL_SHA, "targets_sha256": inputs["targets_sha256"],
        "interpretation": "conditional_exit_replayed_through_T10_with_separate_continuation",
        "entry_statuses": raw.entry_status.value_counts().to_dict(), "exit_statuses": raw.exit_status.value_counts().to_dict(),
        "continued_rows": len(replacement), "original_T5_replayed_rows": sum(len(p) for p in replays),
        "remaining_unresolved": int((raw.entry_status.eq("filled") & raw.exit_price.isna()).sum()),
        "source_invalid_rows": int((~accounted.execution_source_valid).sum()), "holdout_read": False,
        "output_sha256": {name: sha(output / (name + ".parquet")) for name in
            ("repriced", "accounted_initial", "raw_windows", "window_quality")}}
    save_json(output / "execution_report.json", report)
    return report


def summarize(output: Path = ROOT) -> dict:
    verify_inputs(output)
    accounting = evaluate(output)
    dynamic = pd.read_parquet(output / "catalog_scenario.parquet")
    static = pd.read_parquet(CONTINUED / "catalog_scenario.parquet")
    static = static.loc[static.horizon.eq(5)].copy()
    if set(map(tuple, dynamic[KEY].to_numpy())) != set(map(tuple, static[KEY].to_numpy())):
        raise ValueError("A policy lost one of the fixed buy attempts")
    calendar = calendar_days()
    indices = {day: i for i, day in enumerate(calendar)}
    cells, annual, contrasts, differences, risks = [], [], [], [], []
    for policy, frame in (("static_T5", static), ("barrier", dynamic)):
        bought = frame.entry_status.eq("filled")
        unresolved = bought & frame.exit_price.isna()
        frame["held_sessions"] = frame.exit_date.map(indices) - frame.date.map(indices)
        high = frame.loc[frame.arm.eq("high")]
        paired = high.merge(frame.loc[frame.arm.eq("low")], on=["date", "pair_id", "half"],
            suffixes=("_high", "_low"), validate="one_to_one")
        for (half, arm), group in frame.groupby(["half", "arm"]):
            for slip in (5, 15):
                values = {kind: daily_returns(group, f"catalog_scenario_{kind}{slip}")
                    for kind in ("return", "lower", "upper")}
                cells.append({"policy": policy, "half": half, "arm": arm, "slip": slip,
                    "rows": len(group), "days": len(values["return"]),
                    **{kind: None if series.isna().any() else float(series.mean()) for kind, series in values.items()}})
        for year, group in high.groupby(high.date.str[:4]):
            for slip in (5, 15):
                daily = daily_returns(group, f"catalog_scenario_return{slip}")
                annual.append({"policy": policy, "year": year, "slip": slip, "days": len(daily),
                    "return": None if daily.isna().any() else float(daily.mean()), "weekly_interval": weekly_interval(daily)})
        for slip in (5, 15):
            paired["edge"] = paired[f"catalog_scenario_return{slip}_high"] - paired[f"catalog_scenario_return{slip}_low"]
            for period, group in list(paired.groupby("half")) + list(paired.groupby(paired.date.str[:4])):
                daily = daily_returns(group, "edge")
                contrasts.append({"policy": policy, "period": period, "slip": slip,
                    "pairs": len(group), "days": len(daily),
                    "edge": None if daily.isna().any() else float(daily.mean()), "weekly_interval": weekly_interval(daily)})
        for half, group in high.groupby("half"):
            values = group.catalog_scenario_return15
            if values.isna().any():
                quantiles, worst_tail = None, None
            else:
                quantiles = {str(q): float(values.quantile(q)) for q in (.01, .05, .5, .95)}
                worst_tail = float(values.nsmallest(int(np.ceil(.05 * len(values)))).mean())
            risks.append({"policy": policy, "half": half,
                "interpretation": "equal_trade_return_distribution_not_portfolio_drawdown",
                "attempts": len(group), "bought": int(group.entry_status.eq("filled").sum()),
                "mean_held_sessions": float(group.held_sessions.mean()),
                "median_held_sessions": float(group.held_sessions.median()),
                "maximum_held_sessions": float(group.held_sessions.max()),
                "return_quantiles15": quantiles, "worst_five_percent_mean15": worst_tail,
                "unresolved": int(unresolved.reindex(group.index).sum())})
    matched = dynamic.merge(static, on=KEY, suffixes=("_dynamic", "_static"), validate="one_to_one")
    for slip in (5, 15):
        matched["difference"] = matched[f"catalog_scenario_return{slip}_dynamic"] - matched[f"catalog_scenario_return{slip}_static"]
        for arm, part in matched.groupby("arm_dynamic"):
            for period, group in list(part.groupby("half_dynamic")) + list(part.groupby(part.date.str[:4])):
                daily = daily_returns(group, "difference")
                differences.append({"period": period, "arm": arm, "slip": slip, "rows": len(group), "days": len(daily),
                    "barrier_minus_static": None if daily.isna().any() else float(daily.mean()), "weekly_interval": weekly_interval(daily)})
    report = {"rule_commit": RULE, "holdout_read": False,
        "interpretation": "conditional_on_decision_prices_catalogue_actions_and_recorded_fills_not_portfolio_returns",
        "ohlc_sensitivity_holds_trigger_decisions_and_exit_dates_fixed": True,
        "shared_accounting_report_groups_by_actual_planned_horizon_only": True,
        "policy_comparison_aggregates_all_planned_horizons": True,
        "source_invalid_rows": accounting["source_invalid_rows"],
        "unresolved_terminal_rows": accounting["unresolved_terminal_rows"],
        "catalogue_action_rows": accounting["catalogue_action_rows"],
        "share_change_rechecked_rows": accounting["share_change_rechecked_rows"],
        "by_half": cells, "own_annual": annual, "matched_contrasts": contrasts,
        "same_buy_differences": differences, "holding_and_tail_risk": risks}
    save_json(output / "policy_report.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "reprice", "summarize"))
    args = parser.parse_args()
    result = {"freeze": freeze, "reprice": reprice, "summarize": summarize}[args.stage]()
    print(json.dumps({k: v for k, v in result.items() if k not in
        ("sources", "reasons", "by_half", "own_annual", "matched_contrasts", "same_buy_differences", "holding_and_tail_risk")},
        ensure_ascii=False, indent=2))
