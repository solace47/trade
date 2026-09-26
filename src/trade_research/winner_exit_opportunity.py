"""Hindsight intraday opportunity bounds on fixed losing lists, never a trading rule."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import DAILY, MINUTES, save_json, sha
from .hf_outcomes import _board_limit_rate, _limit_price
from .quote_precision import quote_cents
from .reference_gain_accounting import distribution_cash
from .turnover_reference import CALENDAR

ROOT = Path("data/research/winner_exit_opportunity")
SOURCE = Path("data/research/winner_direction")
PROTOCOL = Path("config/winner_exit_opportunity_protocol.json")
MODELS = ("balanced", "upside_only")
STARTS = np.array([*range(9 * 60 + 35, 11 * 60 + 28), *range(13 * 60 + 1, 14 * 60 + 53)])
LABELS = np.array([f"{v//60:02d}{v%60:02d}" for v in STARTS])
WINDOW_MINUTES = STARTS[:, None] + np.arange(4)[None, :]


def fixed_inputs() -> dict:
    report = json.loads((ROOT / "input_report.json").read_text())
    for path, digest in report["sha256"].items():
        if sha(Path(path)) != digest:
            raise ValueError(f"A fixed source changed: {path}")
    if sha(ROOT / "positions.parquet") != report["positions_sha256"]:
        raise ValueError("Fixed positions changed")
    return report


def freeze() -> dict:
    if (ROOT / "input_report.json").exists():
        raise ValueError("Do not replace fixed opportunity inputs")
    ROOT.mkdir(parents=True, exist_ok=True)
    calendar = pd.read_parquet(CALENDAR)
    days = sorted(calendar.loc[calendar.is_trading_day.eq("1") & calendar.calendar_date.between("2025-01-01", "2025-12-31"), "calendar_date"])
    next_day = dict(zip(days[:-1], days[1:]))
    holdings, hashes, expected_sources = [], {str(PROTOCOL): sha(PROTOCOL), str(CALENDAR): sha(CALENDAR)}, {}
    for model in MODELS:
        directory = SOURCE / model
        folder = directory / "continued"
        positions = pd.read_parquet(folder / "tick_cost_scenario.parquet")
        tick_report = json.loads((folder / "tick_report.json").read_text())
        assert sha(folder / "tick_cost_scenario.parquet") == tick_report["tick_scenario_sha256"]
        signals = pd.read_parquet(directory / "signals.parquet")
        positions = positions.loc[positions.arm.eq("high") & positions.horizon.eq(1)].copy()
        assert len(positions) == 1165 and positions.date.map(next_day).eq(positions.target_exit_date).all()
        positions = positions.merge(signals[["date", "code", "decision_shares"]], on=["date", "code"], validate="one_to_one")
        bought = positions.entry_status.eq("filled")
        assert positions.loc[bought, "shares"].eq(positions.loc[bought, "decision_shares"]).all()
        assert positions.loc[~bought, "shares"].eq(0).all()
        queue = pd.read_parquet(folder / "execution_queue_audit.parquet")
        positions = positions.merge(queue[["date", "code", "horizon", "entry_limit_touched"]], on=["date", "code", "horizon"], validate="one_to_one")
        positions["model"] = model
        holdings.append(positions)
        for name in ("tick_cost_scenario.parquet", "tick_report.json", "execution_queue_audit.parquet", "catalog_scenario_report.json"):
            path = folder / name
            hashes[str(path)] = sha(path)
        hashes[str(directory / "signals.parquet")] = sha(directory / "signals.parquet")
        hashes[str(directory / "execution_sources.json")] = sha(directory / "execution_sources.json")
        for item in json.loads((directory / "execution_sources.json").read_text()):
            prior = expected_sources.setdefault(item["code"], item)
            assert prior["minute_sha256"] == item["minute_sha256"] and prior["daily_sha256"] == item["daily_sha256"]
    path = SOURCE / "catalog/events_reconciled.parquet"
    for model in MODELS:
        assert sha(path) == json.loads((SOURCE / model / "continued/catalog_scenario_report.json").read_text())["catalog_sha256"]
    hashes[str(path)] = sha(path)
    combined = pd.concat(holdings, ignore_index=True)
    assert not combined.duplicated(["model", "date", "code"]).any()
    combined.to_parquet(ROOT / "positions.parquet", index=False, compression="zstd")
    result = {"rule_commit": "bccf46f", "positions": len(combined), "sha256": hashes,
        "positions_sha256": sha(ROOT / "positions.parquet"), "expected_sources": expected_sources,
        "window_count_per_trade": len(STARTS), "new_2026_prices_read": False}
    save_json(ROOT / "input_report.json", result)
    return {k: v for k, v in result.items() if k not in ("sha256", "expected_sources")}


def extract() -> dict:
    if (ROOT / "raw_report.json").exists():
        raise ValueError("Do not replace original target-day bars")
    report = fixed_inputs()
    positions = pd.read_parquet(ROOT / "positions.parquet")
    wanted = positions.loc[positions.entry_status.eq("filled"), ["date", "code", "target_exit_date"]].drop_duplicates()
    folder = ROOT / "raw_parts"
    folder.mkdir(exist_ok=True)

    def one(item):
        code, rows = item
        path = folder / (code + ".parquet")
        meta_path = path.with_suffix(".json")
        daily_out = folder / (code + ".daily.parquet")
        targets = sorted(rows.target_exit_date.unique())
        all_days = sorted(set(rows.date) | set(targets))
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            assert meta["target_dates"] == targets and meta["daily_dates"] == all_days
            assert meta["input_report_sha256"] == sha(ROOT / "input_report.json")
            if sha(path) != meta["minute_output_sha256"] or sha(daily_out) != meta["daily_output_sha256"]:
                raise ValueError("Cached target-day bars changed")
            return meta
        minute_path = MINUTES / code[:2].upper() / (code[3:] + ".parquet")
        daily_path = DAILY / (code.replace(".", "_") + ".parquet")
        expected = report["expected_sources"][code]
        assert sha(minute_path) == expected["minute_sha256"] and sha(daily_path) == expected["daily_sha256"]
        c = duckdb.connect()
        c.execute("SET threads=1")
        c.register("targets", pd.DataFrame({"date": targets}))
        minute = c.execute("""SELECT timestamp,open,high,low,close,volume,turnover
            FROM read_parquet(?) WHERE timestamp>=?::TIMESTAMP AND timestamp<?::DATE+INTERVAL 1 DAY
              AND strftime(timestamp,'%Y-%m-%d') IN(SELECT date FROM targets)
              AND (strftime(timestamp,'%H%M') BETWEEN '0935' AND '1130'
                   OR strftime(timestamp,'%H%M') BETWEEN '1301' AND '1455') ORDER BY timestamp""",
            [str(minute_path), targets[0], targets[-1]]).df()
        c.register("wanted_days", pd.DataFrame({"date": all_days}))
        daily = c.execute("SELECT * FROM read_parquet(?) WHERE date IN(SELECT date FROM wanted_days) ORDER BY date", [str(daily_path)]).df()
        c.close()
        minute["code"], minute["date"] = code, minute.timestamp.dt.strftime("%Y-%m-%d")
        minute.to_parquet(path, index=False, compression="zstd")
        daily.to_parquet(daily_out, index=False, compression="zstd")
        meta = {"code": code, "target_dates": targets, "daily_dates": all_days, "rows": len(minute),
            "input_report_sha256": sha(ROOT / "input_report.json"),
            "minute_source_sha256": expected["minute_sha256"], "daily_source_sha256": expected["daily_sha256"],
            "minute_output_sha256": sha(path), "daily_output_sha256": sha(daily_out)}
        save_json(meta_path, meta)
        return meta

    inputs = list(wanted.groupby("code", sort=True))
    results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        for index, result in enumerate(executor.map(one, inputs), 1):
            results.append(result)
            if index % 50 == 0 or index == len(inputs):
                print(json.dumps({"target_day_stocks_extracted": index, "total": len(inputs)}), flush=True)
    result = {"input_report_sha256": sha(ROOT / "input_report.json"), "sources": results,
        "target_stock_days": len(wanted[["code", "target_exit_date"]].drop_duplicates()),
        "minute_rows": sum(v["rows"] for v in results), "new_2026_prices_read": False}
    save_json(ROOT / "raw_report.json", result)
    return {k: v for k, v in result.items() if k != "sources"}


def window_quotes(bars: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame({"start": LABELS, "start_minute": STARTS})
    fields = ["open", "high", "low", "close", "volume", "turnover"]
    bars = bars.copy()
    bars["minute"] = bars.timestamp.dt.hour * 60 + bars.timestamp.dt.minute
    if bars.duplicated("minute").any() or not bars.timestamp.eq(bars.timestamp.dt.floor("min")).all():
        result["source_valid"] = False
        result["queue_bounds_valid"] = False
        result[["volume", "vwap", "positive_low"]] = np.nan
        return result
    values = bars.set_index("minute")[fields].reindex(WINDOW_MINUTES.ravel()).to_numpy().reshape(len(STARTS), 4, 6)
    o, h, l, close, v, amount = [values[:, :, i] for i in range(6)]
    valid = (np.isfinite(values).all(axis=2) & (np.minimum.reduce([o, h, l, close]) > 0)
        & (h + .0001 >= np.maximum.reduce([o, close, l])) & (l - .0001 <= np.minimum(o, close))
        & (v >= 0) & (amount >= 0) & ((v == 0) == (amount == 0)))
    minute_vwap = np.divide(amount, v, out=np.zeros_like(amount), where=v > 0)
    valid &= (v == 0) | ((minute_vwap >= l - .0101) & (minute_vwap <= h + .0101))
    total_volume = v.sum(axis=1)
    result["source_valid"] = valid.all(axis=1)
    result["queue_bounds_valid"] = ((v <= 0) | ((np.abs(h - np.rint(h * 100) / 100) <= .0001)
        & (np.abs(l - np.rint(l * 100) / 100) <= .0001) & (h > 0) & (l > 0))).all(axis=1) & (v > 0).any(axis=1)
    result["volume"] = total_volume
    result["vwap"] = np.divide(amount.sum(axis=1), total_volume, out=np.full_like(total_volume, np.nan), where=total_volume > 0)
    result["positive_low"] = np.where(v > 0, l, np.inf).min(axis=1)
    return result


def first_day_entitlement(row: pd.Series, daily: pd.DataFrame, catalog: pd.DataFrame) -> dict:
    target = row.target_exit_date
    answer = {"shares_at_target": int(row.shares), "net_receivable": 0., "entitlement_status": "none"}
    if target not in daily.index or row.date not in daily.index:
        return dict(answer, entitlement_status="missing_daily")
    events = catalog.loc[catalog.code.eq(row.code) & catalog.dividOperateDate.gt(row.date)
                         & catalog.dividOperateDate.le(target)]
    if len(events) > 1:
        return dict(answer, entitlement_status="multiple_actions_unknown")
    if len(events):
        event = events.iloc[0]
        try:
            cash = distribution_cash(event, last_date="2025-12-31")
            reserve, bonus = Decimal(event.dividReserveToStockPs or "0"), Decimal(event.dividStocksPs or "0")
            if not reserve.is_finite() or not bonus.is_finite() or reserve < 0 or bonus < 0:
                raise ValueError("Invalid share distribution terms")
            qty = Decimal(int(row.shares)) * (1 + reserve)
        except (ValueError, ArithmeticError):
            return dict(answer, entitlement_status="unverified_terms")
        if not (row.date <= event.dividRegistDate < target) or bonus or qty != qty.to_integral_value():
            return dict(answer, entitlement_status="unverified_entitlement")
        if reserve and not event.dividOperateDate <= event.dividStockMarketDate <= target:
            return dict(answer, entitlement_status="shares_not_listed")
        return {"shares_at_target": int(qty), "net_receivable": float(Decimal(int(row.shares)) * cash * Decimal('.8')),
                "entitlement_status": "catalogue_scenario"}
    if abs(daily.loc[target, "preclose"] - daily.loc[row.date, "close"]) > .005:
        return dict(answer, entitlement_status="unexplained_reference_gap")
    return answer


def evaluate() -> dict:
    if (ROOT / "analysis_report.json").exists():
        raise ValueError("Do not overwrite inspected opportunity bounds")
    fixed_inputs()
    raw_report = json.loads((ROOT / "raw_report.json").read_text())
    assert raw_report["input_report_sha256"] == sha(ROOT / "input_report.json")
    raw_sources = {item["code"]: item for item in raw_report["sources"]}
    positions = pd.read_parquet(ROOT / "positions.parquet")
    catalog = pd.read_parquet(SOURCE / "catalog/events_reconciled.parquet")
    windows, records = [], []
    for code, part in positions.groupby("code", sort=True):
        if not part.entry_status.eq("filled").any():
            for _, row in part.iterrows():
                records.append(dict(position_record(row), status="not_bought"))
            continue
        path = ROOT / "raw_parts" / (code + ".parquet")
        meta = json.loads(path.with_suffix(".json").read_text())
        assert meta == raw_sources[code]
        assert sha(path) == meta["minute_output_sha256"]
        daily_path = ROOT / "raw_parts" / (code + ".daily.parquet")
        assert sha(daily_path) == meta["daily_output_sha256"]
        bars = pd.read_parquet(path)
        daily = pd.read_parquet(daily_path).set_index("date", drop=False)
        assert daily.index.is_unique
        quotes = {date: window_quotes(group) for date, group in bars.groupby("date")}
        for _, row in part.iterrows():
            record = position_record(row)
            if row.entry_status != "filled":
                records.append(dict(record, status="not_bought")); continue
            if row.target_exit_date not in daily.index:
                records.append(dict(record, status="target_daily_unknown")); continue
            if daily.loc[row.target_exit_date, "tradestatus"] != 1 or row.target_exit_date not in quotes:
                records.append(dict(record, status="no_target_trading_window")); continue
            entitlement = first_day_entitlement(row, daily, catalog)
            record.update(entitlement)
            if entitlement["entitlement_status"] not in ("none", "catalogue_scenario"):
                records.append(dict(record, status="entitlement_unknown")); continue
            price = row.entry_price / 1.0005
            buy_value = row.shares * (price + max(price * .0015, .005))
            cost = buy_value + max(5., buy_value * .0003) + buy_value * .00001
            day = daily.loc[row.target_exit_date]
            lower = _limit_price(day.preclose, _board_limit_rate(code, int(day.isST), row.target_exit_date), False)
            q = quotes[row.target_exit_date].copy()
            q["queue_unknown"] = ~q.queue_bounds_valid | (np.rint(q.positive_low * 100) <= quote_cents(lower))
            q["capacity_ok"] = entitlement["shares_at_target"] <= q.volume * .1
            q["baseline_sellable"] = (q.vwap * .9995 > lower + .005) & q.volume.gt(0)
            q["eligible"] = q.source_valid & q.capacity_ok & q.baseline_sellable & ~q.queue_unknown
            value = entitlement["shares_at_target"] * (q.vwap - np.maximum(q.vwap * .0015, .005))
            q["net_return"] = ((value - np.maximum(5., value * .0003) - value * .00051
                                 + entitlement["net_receivable"]) / cost - 1).where(q.eligible)
            q["model"], q["date"], q["code"], q["target_date"] = row.model, row.date, code, row.target_exit_date
            windows.append(q)
            valid = q.loc[q.eligible]
            record.update(valid_windows=len(valid), source_bad_windows=int((~q.source_valid).sum()),
                queue_unknown_windows=int(q.queue_unknown.sum()), buy_cost15=cost)
            if valid.empty:
                records.append(dict(record, status="no_verified_sell_window")); continue
            best, worst = valid.loc[valid.net_return.idxmax()], valid.loc[valid.net_return.idxmin()]
            record.update(status="conditional_opportunity", best_return=float(best.net_return), worst_return=float(worst.net_return),
                best_start=best.start, worst_start=worst.start,
                ever_positive=bool(best.net_return > 0), ever_one_percent=bool(best.net_return >= .01),
                ever_three_percent=bool(best.net_return >= .03))
            records.append(record)
    outcomes = pd.DataFrame(records)
    outcomes["half"] = outcomes.date.str[:4] + np.where(outcomes.date.str[5:7].le("06"), "H1", "H2")
    outcomes.to_parquet(ROOT / "opportunities.parquet", index=False, compression="zstd")
    pd.concat(windows, ignore_index=True).to_parquet(ROOT / "window_scenarios.parquet", index=False, compression="zstd")
    rows = []
    for model in MODELS:
        for period in ("2025", "2025H1", "2025H2"):
            p = outcomes.loc[outcomes.model.eq(model) & (outcomes.date.str[:4].eq(period) if len(period) == 4 else outcomes.half.eq(period))]
            priced = p.loc[p.status.eq("conditional_opportunity")]
            valid_entry = priced.entry_source_valid & ~priced.entry_queue_unknown.astype(bool)
            for scope, subset in (("all_recorded_entries", priced), ("valid_entry_source_no_touch", priced.loc[valid_entry])):
                ontime = subset.original_exit_date.eq(subset.target_date)
                loss = subset.original_return15.lt(0) & ontime
                rows.append({"model": model, "period": period, "scope": scope, "orders": len(p),
                    "bought": int(p.bought.sum()), "opportunity_rows": len(subset), "no_verified_window": int((p.bought & p.status.ne("conditional_opportunity")).sum()),
                    "original_losses": int(loss.sum()),
                    "original_delayed_or_missing_exit_rows_in_scope": int((~ontime).sum()),
                    **{name: int(subset[name].eq(True).sum()) for name in ("ever_positive", "ever_one_percent", "ever_three_percent")},
                    **{name + "_but_original_loss": int((subset[name].eq(True) & loss).sum()) for name in ("ever_positive", "ever_one_percent", "ever_three_percent")},
                    "best_return_median": float(subset.best_return.median()), "worst_return_median": float(subset.worst_return.median()),
                    "daily_mean_best_hindsight_only": float(subset.groupby("date").best_return.mean().mean()),
                    "original_source_unknown_rows_in_scope": int((~subset.original_execution_source_valid).sum())})
    result = {"interpretation": "hindsight_upper_opportunity_envelope_not_implementable_exit_returns",
        "raw_report_sha256": sha(ROOT / "raw_report.json"), "protocol_sha256": sha(PROTOCOL),
        "statuses": outcomes.groupby(["model", "status"]).size().rename("rows").reset_index().to_dict("records"),
        "summaries": rows, "outputs_sha256": {name: sha(ROOT / name) for name in ("opportunities.parquet", "window_scenarios.parquet")},
        "new_2026_prices_read": False}
    save_json(ROOT / "analysis_report.json", result)
    return result


def position_record(row: pd.Series) -> dict:
    return {"model": row.model, "date": row.date, "code": row.code, "target_date": row.target_exit_date,
        "decision_shares": int(row.decision_shares), "bought": row.entry_status == "filled",
        "original_return15": row.tick_return15, "original_exit_date": row.exit_date,
        "original_actual_unknown": bool(row.unknown_after_buy),
        "original_execution_source_valid": bool(row.execution_source_valid),
        "entry_source_valid": row.entry_window_status == "valid",
        "entry_queue_unknown": bool(pd.isna(row.entry_limit_touched) or row.entry_limit_touched),
        "valid_windows": 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "extract", "evaluate"])
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](), ensure_ascii=False, indent=2))
