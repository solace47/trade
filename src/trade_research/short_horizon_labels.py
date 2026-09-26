"""Rebuild T+1 training labels; reuse T+5 raw facts under identical risk rules."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .downside_ridge_inputs import training_scores, window_summary, ROOT as RECENT
from .fixed_return_ranges import dated_economic_return
from .hf_outcomes import Assumptions, outcomes_for_symbol
from .long_history_inputs import ROOT as HISTORY
from .quote_precision import quote_cents
from .reference_gain_eval import DAILY, MINUTES
from .risk_removal_eval import cost_price
from .turnover_reference import CALENDAR

ROOT = Path("data/research/short_horizon_target")
PROTOCOL = Path("config/short_horizon_target_protocol.json")
RULE_COMMIT = "0630933"


def features() -> pd.DataFrame:
    protocol = json.loads(PROTOCOL.read_text())
    source = Path(protocol["features_source"])
    if sha(source) != protocol["features_sha256"]:
        raise ValueError("The fixed historical feature source changed")
    frame = pd.read_parquet(source)
    frame["price_1449"] = frame.price_1449.map(lambda p: quote_cents(p)/100)
    frame["price_signal"] = frame.price_1449
    frame["isST"], frame["reference_gap"], frame["listing_age_sessions"] = 0, False, 20
    return frame


def raw(output: Path = ROOT) -> dict:
    if any((output / model / "repriced.parquet").exists() for model in ("t1_target", "t5_target")):
        raise ValueError("Training cannot change after new selected outcomes exist")
    output.mkdir(parents=True, exist_ok=True)
    signals = features().loc[lambda x: x.date.le("2024-12-17")].copy()
    if signals.duplicated(["date", "code"]).any() or not signals.date.ge("2022-01-01").all():
        raise ValueError("Invalid historical training identities")
    signal_file = output / "training_signals.parquet"
    if signal_file.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(signal_file), signals, check_exact=True)
    else:
        signals.to_parquet(signal_file, index=False, compression="zstd")
    complete = output / "raw_report.json"
    if complete.exists():
        result = json.loads(complete.read_text())
        for name, digest in result["output_sha256"].items():
            if sha(output / name) != digest:
                raise ValueError("A completed T+1 label artifact changed")
        return result
    table = pd.read_parquet(CALENDAR)
    calendar = sorted(table.loc[table.is_trading_day.eq("1") &
        table.calendar_date.between("2022-01-01", "2024-12-31"), "calendar_date"])
    index = {d: i for i, d in enumerate(calendar)}
    code_paths = [Path(__file__), Path("src/trade_research/hf_outcomes.py"),
        Path("src/trade_research/downside_ridge_inputs.py"), PROTOCOL]
    code_sha = {str(p): sha(p) for p in code_paths}
    parts = output / "parts"; parts.mkdir(exist_ok=True)
    def one(item):
        code, group = item
        exchange, symbol = code.split(".")
        minute_path, daily_path = MINUTES/exchange.upper()/(symbol+".parquet"), DAILY/(code.replace(".", "_")+".parquet")
        fingerprint = {"code": code, "horizon": 1, "notional": 20000,
            "signals_sha256": hashlib.sha256(group[["date", "code", "price_1449"]].to_json(orient="records", double_precision=15).encode()).hexdigest(),
            "minute_sha256": sha(minute_path), "daily_sha256": sha(daily_path), "code_sha256": code_sha}
        source_file = parts/(code+".json")
        trade_file, window_file = parts/(code+".trades.parquet"), parts/(code+".windows.parquet")
        if source_file.exists():
            saved = json.loads(source_file.read_text())
            if (any(saved[k] != v for k, v in fingerprint.items()) or sha(trade_file) != saved["raw_sha256"]
                    or sha(window_file) != saved["windows_sha256"]):
                raise ValueError("A training checkpoint no longer matches its frozen inputs")
            return trade_file, window_file, saved
        if any(index[d]+10 >= len(calendar) for d in group.date):
            raise ValueError("Training must fit a complete ten-session observation window")
        dates = sorted({d for signal in group.date for d in calendar[index[signal]:index[signal]+11]})
        c = duckdb.connect(); c.execute("SET threads=1")
        c.read_parquet(str(minute_path)).create_view("minutes")
        c.register("needed", pd.DataFrame({"date": dates}))
        c.execute("""CREATE TEMP TABLE selected_bars AS SELECT timestamp,open,high,low,close,volume,turnover
          FROM minutes WHERE timestamp>=?::TIMESTAMP AND timestamp<?::DATE+INTERVAL 1 DAY
          AND strftime(timestamp,'%H%M') IN ('1452','1453','1454','1455')
          AND strftime(timestamp,'%Y-%m-%d') IN(SELECT date FROM needed)""", [dates[0], dates[-1]])
        minute = c.sql("SELECT * FROM selected_bars ORDER BY timestamp").df()
        windows = pd.DataFrame({"date": dates}).merge(window_summary(c), on="date", how="left", validate="one_to_one")
        windows["window_status"], windows["code"] = windows.window_status.fillna("missing_window"), code
        c.read_parquet(str(daily_path)).create_view("daily")
        prior = c.execute("SELECT max(date) FROM daily WHERE date<? AND tradestatus=1", [dates[0]]).fetchone()[0]
        if prior is None:
            raise ValueError("No normal daily quote precedes the training signal")
        daily = c.execute("SELECT * FROM daily WHERE date BETWEEN ? AND ? ORDER BY date", [prior, dates[-1]]).df()
        c.close()
        minute["date"], minute["label"] = minute.timestamp.dt.strftime("%Y-%m-%d"), minute.timestamp.dt.strftime("%H%M")
        trades = outcomes_for_symbol(group, minute, daily, calendar, Assumptions(target_notional=20000),
            horizons=(1,), sizing_price_column="price_1449")
        trades["target_notional"], trades["entry_window"], trades["exit_window"] = 20000, "baseline", "close"
        trades.to_parquet(trade_file, index=False, compression="zstd")
        windows.to_parquet(window_file, index=False, compression="zstd")
        saved = {**fingerprint, "first": dates[0], "last": dates[-1], "rows": len(trades),
            "raw_sha256": sha(trade_file), "windows_sha256": sha(window_file)}
        save_json(source_file, saved)
        return trade_file, window_file, saved
    groups = list(signals.groupby("code", sort=True)); trade_files, window_files, sources = [], [], []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (trade_file, window_file, source) in enumerate(pool.map(one, groups), 1):
            trade_files.append(trade_file); window_files.append(window_file); sources.append(source)
            if i % 200 == 0 or i == len(groups):
                save_json(output / "raw_progress.json", {"stocks": i, "total": len(groups)})
                print(f"T+1 historical raw labels {i}/{len(groups)} stocks", flush=True)
    trades = pd.concat([pd.read_parquet(p) for p in trade_files], ignore_index=True).sort_values(["date", "code"])
    windows = pd.concat([pd.read_parquet(p) for p in window_files], ignore_index=True).sort_values(["date", "code"])
    if len(trades) != len(signals) or trades.duplicated(["date", "code"]).any():
        raise ValueError("A historical label identity was lost or duplicated")
    trades.to_parquet(output / "t1_raw.parquet", index=False, compression="zstd")
    windows.to_parquet(output / "training_windows.parquet", index=False, compression="zstd")
    save_json(output / "training_sources.json", sources)
    result = {"rule_commit": RULE_COMMIT, "rows": len(trades), "stocks": len(groups),
        "first_signal": signals.date.min(), "last_signal": signals.date.max(),
        "last_source_date": max(s["last"] for s in sources),
        "entry_statuses": trades.entry_status.value_counts().to_dict(),
        "exit_statuses": trades.exit_status.value_counts().to_dict(),
        "output_sha256": {name: sha(output/name) for name in
            ("training_signals.parquet", "t1_raw.parquet", "training_windows.parquet", "training_sources.json")},
        "protocol_sha256": sha(PROTOCOL), "new_2026_prices_read": False}
    save_json(complete, result)
    return result


def queue_flags(labels: pd.DataFrame) -> pd.Series:
    """The cached extrema cover all positive-volume minutes of the order window."""
    c = duckdb.connect(); c.execute("SET threads=4")
    c.register("labels", labels.reset_index(drop=True).reset_index(names="row_id"))
    paths = sorted([*DAILY.glob("sh_60*.parquet"), *DAILY.glob("sz_00*.parquet")])
    c.read_parquet([str(p) for p in paths]).create_view("daily")
    c.execute("""CREATE TEMP TABLE limits AS SELECT code,date,
      round(preclose::DECIMAL(18,6)*CASE WHEN isST=1 THEN 1.05 ELSE 1.10 END,2) AS upper,
      round(preclose::DECIMAL(18,6)*CASE WHEN isST=1 THEN .95 ELSE .90 END,2) AS lower
      FROM daily WHERE date BETWEEN '2022-01-01' AND '2024-12-31'""")
    result = c.sql("""SELECT l.row_id,l.entry_status='filled' AND (
      coalesce(round(l.entry_window_high,2)>=b.upper,true) OR
      coalesce(round(l.exit_window_low,2)<=s.lower,true)) AS unverified
      FROM labels l LEFT JOIN limits b ON l.date=b.date AND l.code=b.code
      LEFT JOIN limits s ON l.exit_date=s.date AND l.code=s.code ORDER BY l.row_id""").df()
    c.close()
    if len(result) != len(labels) or result.row_id.duplicated().any():
        raise ValueError("A historical queue check changed label identities")
    return pd.Series(result.unverified.to_numpy(), index=labels.index)


def align_risk_scores(labels: pd.DataFrame, queues: pd.Series) -> pd.DataFrame:
    rows = labels.copy()
    known = rows.score_origin.eq("recorded_economic_scenario")
    part = rows.loc[known]
    buy, sell = cost_price(part.entry_price/1.0005, 15, "buy"), cost_price(part.exit_price/.9995, 15, "sell")
    values = dated_economic_return(part.shares, part.sold_shares, buy, sell,
        part.dividend_gross, part.dividend_tax, part.date, part.exit_date)
    rows.loc[known, ["downside_score", "economic_scenario15"]] = np.column_stack((values, values))
    if not queues.index.equals(rows.index) or queues.isna().any():
        raise ValueError("Queue penalties must align with every label")
    rows["queue_allocation_unverified"] = queues
    rows.loc[known & queues, "downside_score"] = -1.
    rows.loc[known & queues, "score_origin"] = "adverse_limit_queue_training_penalty"
    return rows


def assemble(output: Path = ROOT) -> dict:
    if (output / "label_report.json").exists():
        raise ValueError("Completed risk labels cannot be overwritten")
    raw_report = json.loads((output / "raw_report.json").read_text())
    for name, expected in raw_report["output_sha256"].items():
        if sha(output/name) != expected:
            raise ValueError("Raw T+1 training inputs changed")
    hist_report = json.loads((HISTORY / "historical_label_report.json").read_text())
    recent_report = json.loads((RECENT / "label_report.json").read_text())
    catalog = HISTORY / "training_catalog.parquet"
    for path, expected in ((catalog, hist_report["catalog_sha256"]),
            (HISTORY / "historical_training_labels.parquet", hist_report["labels_sha256"]),
            (RECENT / "training_labels.parquet", recent_report["labels_sha256"])):
        if sha(path) != expected:
            raise ValueError("Existing five-day training facts changed")
    t1, events = training_scores(pd.read_parquet(output / "t1_raw.parquet"),
        pd.read_parquet(output / "training_windows.parquet"), catalog_path=catalog)
    t1 = t1.sort_values(["date", "code"]).reset_index(drop=True)
    events.to_parquet(output / "training_actions.parquet", index=False, compression="zstd")
    t5 = pd.concat([pd.read_parquet(HISTORY / "historical_training_labels.parquet"),
        pd.read_parquet(RECENT / "training_labels.parquet")], ignore_index=True).sort_values(["date", "code"]).reset_index(drop=True)
    pd.testing.assert_frame_equal(t1[["date", "code"]], t5[["date", "code"]], check_exact=True)
    summaries = {}
    for name, labels in (("t1", t1), ("t5", t5)):
        labels = align_risk_scores(labels, queue_flags(labels))
        if labels.downside_score.isna().any() or labels.exit_date.dropna().gt("2024-12-31").any():
            raise ValueError("A risk score is missing or exceeds its historical scope")
        labels.to_parquet(output / (name+"_labels.parquet"), index=False, compression="zstd")
        summaries[name] = {"rows": len(labels), "horizon": int(labels.horizon.unique().item()),
            "score_origins": labels.score_origin.value_counts().to_dict(),
            "queue_unverified_bought": int(labels.queue_allocation_unverified.sum()),
            "labels_sha256": sha(output / (name+"_labels.parquet"))}
    result = {"rule_commit": RULE_COMMIT, "models": summaries, "catalog_sha256": sha(catalog),
        "historical_label_report_sha256": sha(HISTORY / "historical_label_report.json"),
        "recent_label_report_sha256": sha(RECENT / "label_report.json"), "raw_report_sha256": sha(output / "raw_report.json"),
        "training_penalties_are_not_actual_returns": True, "new_2026_prices_read": False}
    save_json(output / "label_report.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["raw", "assemble"])
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](), ensure_ascii=False, indent=2))
