"""Compare two fixed T+1 exits on the identical first-board tail buy list."""
from __future__ import annotations

import argparse
import json
from math import ceil
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .first_limitup_overnight import ROOT, RULE_COMMIT
from .price_limit_queue_audit import audit as audit_queues, conservative_returns
from .reference_gain_eval import reprice
from .reference_gain_accounting import evaluate as account, weekly_interval
from .risk_removal_eval import evaluate as tick
from .shallow_tree_continuation import continue_model


def execute(output: Path = ROOT) -> dict:
    selection = json.loads((output / "input_report.json").read_text())
    checks = json.loads((output / "independent_input_checks.json").read_text())
    expected = selection["signals_sha256"]
    if sha(output / "signals.parquet") != expected or checks["signals_sha256"] != expected:
        raise ValueError("The independently checked buy list changed")
    results = {}
    for window in ("morning", "close"):
        folder, continued = output / window, output / "continued" / window
        folder.mkdir(exist_ok=True)
        for name in ("signals.parquet", "input_report.json"):
            destination = folder / name
            if destination.exists():
                if sha(destination) != sha(output / name):
                    raise ValueError("The copied input changed")
            else:
                shutil.copyfile(output / name, destination)
        if not (folder / "execution_report.json").exists():
            reprice(folder, expected_signal_sha=expected, rule_commit=RULE_COMMIT,
                    horizons=(1,), exit_window=window)
        if not (continued / "execution_report.json").exists():
            continue_model(folder, continued)
        if not (continued / "execution_queue_report.json").exists():
            audit_queues(continued)
        if not (continued / "catalog_scenario_report.json").exists():
            account(continued)
        if not (continued / "tick_report.json").exists():
            tick(continued, primary_horizon=1)
        results[window] = json.loads((continued / "execution_report.json").read_text())
        print(window, {key: results[window][key] for key in
            ("continued_rows", "remaining_unresolved")}, flush=True)
    return results


def daily_statistics(values: pd.Series) -> dict:
    return {"signal_dates": len(values), "unknown_dates": int(values.isna().sum()),
        "daily_mean": None if values.empty or values.isna().any() else float(values.mean()),
        "weekly_interval": weekly_interval(values)}


def trade_distribution(rows: pd.DataFrame, column: str) -> dict:
    """Keep all filled orders in the denominator, including unresolved exits."""
    bought = rows.entry_status.eq("filled")
    filled = rows.loc[bought]
    values = filled[column]
    result = {"orders":len(rows), "bought":int(bought.sum()), "not_bought":int((~bought).sum()),
        "scenario_unknown_after_buy":int(values.isna().sum()),
        "original_actual_unknown_after_buy":int(filled.known_return5.isna().sum()),
        "source_invalid_bought":int((~filled.execution_source_valid).sum()),
        "delayed_exits":int(filled.exit_delay_sessions.gt(0).sum()),
        "no_recorded_exit":int(filled.exit_date.isna().sum()),
        "max_exit_delay_sessions":None if filled.empty or filled.exit_date.isna().any()
            else int(filled.exit_delay_sessions.max())}
    metrics = ["win_rate", "mean_win", "mean_loss", "payoff_ratio", "worst_five_percent_mean",
               "worst_trade_return", "trade_mean"]
    if values.empty or values.isna().any():
        result.update(dict.fromkeys(metrics))
        return result
    wins, losses = values.loc[values.gt(0)], values.loc[values.lt(0)]
    result.update({"win_rate":float(values.gt(0).mean()),
        "mean_win":None if wins.empty else float(wins.mean()),
        "mean_loss":None if losses.empty else float(losses.mean()),
        "payoff_ratio":None if wins.empty or losses.empty else float(wins.mean() / -losses.mean()),
        "worst_five_percent_mean":float(values.nsmallest(ceil(len(values)*.05)).mean()),
        "worst_trade_return":float(values.min()), "trade_mean":float(values.mean())})
    return result


def compare(output: Path = ROOT) -> dict:
    selection = json.loads((output / "input_report.json").read_text())
    tables, sources, metrics, trades, queue_reports = {}, {}, [], [], {}
    periods = [("full","2024-01-01","2025-12-31")]
    for year in ("2024", "2025"):
        periods += [(year, year+"-01-01", year+"-12-31"),
            (year+"H1",year+"-01-01",year+"-06-30"),
            (year+"H2",year+"-07-01",year+"-12-31")]
    for window in ("morning", "close"):
        folder = output / "continued" / window
        report = json.loads((folder / "tick_report.json").read_text())
        if (sha(folder / "signals.parquet") != selection["signals_sha256"] or
                sha(folder / "tick_cost_scenario.parquet") != report["tick_scenario_sha256"]):
            raise ValueError("The frozen signals or conditional ledger changed")
        rows = pd.read_parquet(folder / "tick_cost_scenario.parquet")
        queue_report=json.loads((folder / "execution_queue_report.json").read_text())
        if sha(folder / "execution_queue_audit.parquet") != queue_report["output_sha256"]:
            raise ValueError("The execution queue audit changed")
        queue_rows=pd.read_parquet(folder / "execution_queue_audit.parquet")
        queue_reports[window]=queue_report
        for bps in (5,15):
            rows[f"queue_checked_return{bps}"]=conservative_returns(rows,queue_rows,f"tick_return{bps}")
        if set(rows.horizon) != {1}:
            raise ValueError("The overnight comparison only permits T+1")
        tables[window], sources[window] = rows, sha(folder / "tick_report.json")
        high, low = rows.loc[rows.arm.eq("high")], rows.loc[rows.arm.eq("low")]
        paired = high.merge(low, on=["date","pair_id"], validate="one_to_one", suffixes=("_high","_low"))
        for bps in (5,15):
            column = f"tick_return{bps}"
            own = high[["date",column]].rename(columns={column:"value"})
            queue_checked=high[["date",f"queue_checked_return{bps}"]].rename(columns={f"queue_checked_return{bps}":"value"})
            edge = paired[["date"]].assign(value=paired[column+"_high"]-paired[column+"_low"])
            for metric, frame in (("own",own),("same_day_edge",edge),("own_with_queue_unknown_retained",queue_checked)):
                daily = frame.groupby("date").value.agg(lambda x: x.mean() if x.notna().all() else np.nan)
                for label, first, last in periods:
                    values = daily.loc[(daily.index>=first)&(daily.index<=last)]
                    metrics.append({"window":window,"bps":bps,"metric":metric,"period":label,
                                    **daily_statistics(values)})
            for label, first, last in periods:
                part = high.loc[high.date.between(first,last)]
                trades.append({"window":window,"bps":bps,"period":label,**trade_distribution(part,column)})
    key = ["date","code","horizon"]
    identical = ["arm","pair_id","shares","entry_status","entry_price","target_exit_date"]
    pd.testing.assert_frame_equal(tables["morning"].set_index(key)[identical].sort_index(),
        tables["close"].set_index(key)[identical].sort_index(),check_exact=True)
    pair = tables["morning"].merge(tables["close"],on=key,validate="one_to_one",suffixes=("_morning","_close"))
    pair = pair.loc[pair.arm_morning.eq("high")]
    differences = pair[key].copy()
    for bps in (5,15):
        differences[f"difference{bps}"] = pair[f"tick_return{bps}_morning"]-pair[f"tick_return{bps}_close"]
        daily = differences.groupby("date")[f"difference{bps}"].agg(lambda x: x.mean() if x.notna().all() else np.nan)
        for label, first, last in periods:
            metrics.append({"window":"morning_minus_close","bps":bps,"metric":"same_stock_difference",
                "period":label,**daily_statistics(daily.loc[(daily.index>=first)&(daily.index<=last)])})
    differences.to_parquet(output / "same_stock_exit_differences.parquet",index=False,compression="zstd")
    result = {"rule_commit":RULE_COMMIT,"signals_sha256":selection["signals_sha256"],
        "interpretation":"exposed_2024_2025_catalogue_and_recorded_fill_scenario_not_portfolio_return",
        "primary":"T1_0935_0938_max_15bps_or_half_cent_per_leg", "same_buy_orders_verified":True,
        "metrics":metrics,"trade_distributions":trades,"sources":sources,"queue_reports":queue_reports,
        "same_stock_differences_sha256":sha(output / "same_stock_exit_differences.parquet"),
        "new_2026_prices_read":False,"publishable_formula":False}
    save_json(output / "comparison_report.json", result)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["execute","compare"])
    arguments=parser.parse_args()
    print(json.dumps(globals()[arguments.stage](),ensure_ascii=False,indent=2))
