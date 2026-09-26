"""Bounded 2024 labels for the preregistered conservative-target comparison."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .absolute_ridge_1449_target import TRAINING_TRADES
from .corporate_cash import save_json, sha
from .fixed_return_ranges import dated_economic_return
from .market_study import _quality_keys
from .reference_gain_eval import MINUTES
from .shallow_tree_1449 import ROOT as PREVIOUS

ROOT = Path("data/research/downside_ridge_1449")
RULE_COMMIT = "123fd1d"


def window_summary(c: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Vector equivalent of validate_window, over the requested four labels."""
    return c.execute("""WITH b AS (
        SELECT *, strftime(timestamp,'%Y-%m-%d') date,
          coalesce(isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close)
            AND isfinite(volume) AND isfinite(turnover)
            AND least(open,high,low,close)>0 AND high+.0001>=greatest(open,close,low)
            AND low-.0001<=least(open,close) AND volume>=0 AND turnover>=0
            AND ((volume=0)=(turnover=0)),false) numeric_valid
        FROM selected_bars
    ) SELECT date, count(*) bars,
        CASE WHEN count(*)<>4 OR list_sort(list(strftime(timestamp,'%H%M')))<>['1452','1453','1454','1455']
            OR NOT bool_and(timestamp=date_trunc('minute',timestamp)) THEN 'incomplete_or_duplicate_window'
          WHEN NOT bool_and(numeric_valid) THEN 'invalid_numeric_bar'
          WHEN bool_or(volume>0 AND (turnover/volume<low-.0101 OR turnover/volume>high+.0101))
            THEN 'vwap_outside_bar_range' ELSE 'valid' END window_status,
        sum(volume) volume, sum(turnover) turnover,
        sum(turnover)/nullif(sum(volume),0) raw_vwap,
        min(low) FILTER (WHERE volume>0) window_low,
        max(high) FILTER (WHERE volume>0) window_high
      FROM b GROUP BY date ORDER BY date""").df()


def load_windows(needed: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    def one(item):
        code, dates = item
        exchange, symbol = code.split(".")
        path = MINUTES / exchange.upper() / (symbol + ".parquet")
        c = duckdb.connect()
        c.execute("SET threads=1")
        c.read_parquet(str(path)).create_view("minutes")
        c.register("needed", dates[["date"]])
        c.execute("""CREATE TEMP TABLE selected_bars AS SELECT timestamp,open,high,low,close,volume,turnover
            FROM minutes WHERE timestamp>=?::TIMESTAMP AND timestamp<?::DATE+INTERVAL 1 DAY
              AND strftime(timestamp,'%H%M') IN ('1452','1453','1454','1455')
              AND strftime(timestamp,'%Y-%m-%d') IN (SELECT date FROM needed)""", [dates.date.min(), dates.date.max()])
        result = window_summary(c)
        c.close()
        result = dates[["date"]].merge(result, on="date", how="left", validate="one_to_one")
        result["window_status"] = result.window_status.fillna("missing_window")
        result["code"] = code
        return result, {"code": code, "source_sha256": sha(path),
            "first": dates.date.min(), "last": dates.date.max(), "windows": len(dates)}
    frames, sources = [], []
    groups = list(needed.groupby("code", sort=True))
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i, (frame, source) in enumerate(pool.map(one, groups), 1):
            frames.append(frame); sources.append(source)
            if i % 300 == 0 or i == len(groups):
                print(f"Training window sources {i}/{len(groups)}", flush=True)
    return pd.concat(frames, ignore_index=True), sources


def training_scores(trades: pd.DataFrame, windows: pd.DataFrame, *,
                    catalog_path: Path = Path("data/research/cash_dividend_catalog/events_augmented.parquet"),
                    last_catalog_pay_date: str = "2024-12-31") -> tuple[pd.DataFrame, pd.DataFrame]:
    for side, key in (("entry", "date"), ("exit", "exit_date")):
        w = windows.rename(columns={"date": key,
            **{k: side + "_" + k for k in windows.columns if k not in ("date", "code")}})
        trades = trades.merge(w, on=[key, "code"], how="left", validate="many_to_one")
    for side in ("entry", "exit"):
        selected = trades[side + "_price"].notna() & trades[side + "_window_status"].eq("valid")
        expected = trades.loc[selected, side + "_raw_vwap"] * (1.0005 if side == "entry" else .9995)
        if not np.allclose(expected.to_numpy(float), trades.loc[selected, side + "_price"].to_numpy(float), atol=1e-10, rtol=0):
            raise ValueError("Old training execution disagrees with raw four-minute VWAP")
    c = duckdb.connect()
    c.register("trades", trades)
    c.register("issues", _quality_keys(Path("data/research/market_issues_ci")))
    trades = c.execute("""SELECT t.*, EXISTS (SELECT 1 FROM issues q WHERE q.code=t.code
        AND q.date BETWEEN t.date AND coalesce(t.exit_date,t.date)) bad_holding_day FROM trades t""").df()
    c.close()
    catalog = pd.read_parquet(catalog_path)
    events = trades.loc[trades.entry_status.eq("filled"), ["date", "code", "exit_date"]].merge(catalog, on="code")
    events = events.loc[events.dividOperateDate.gt(events.date) & events.dividOperateDate.le(events.exit_date)]
    by_key = {key: part for key, part in events.groupby(["date", "code"])}
    trades["sold_shares"], trades["dividend_gross"], trades["dividend_tax"] = trades.shares, 0., 0.
    trades["action_status"] = np.where(trades.corporate_action_crossed.fillna(True), "unmatched_reference_change", "no_action")
    for i, row in trades.iterrows():
        items = by_key.get((row.date, row.code))
        if items is None:
            continue
        trades.loc[i, "action_status"] = "unresolved_terms"
        if len(items) != 1:
            continue
        e = items.iloc[0]
        if not (row.date <= e.dividRegistDate < row.exit_date
            and row.date < e.dividOperateDate <= row.exit_date
            and e.dividOperateDate <= e.dividPayDate <= last_catalog_pay_date
            and (pd.Timestamp(row.exit_date)-pd.Timestamp(row.date)).days <= 30):
            continue
        if Decimal(e.dividStocksPs or "0"):
            continue
        reserve = Decimal(e.dividReserveToStockPs or "0")
        quantity = Decimal(int(row.shares)) * (1 + reserve)
        if quantity != quantity.to_integral_value():
            continue
        if reserve and not (e.dividOperateDate <= e.dividStockMarketDate <= row.exit_date
                            and int(quantity) <= row.exit_volume * .1):
            continue
        gross = float(Decimal(int(row.shares)) * Decimal(e.dividCashPsBeforeTax))
        trades.loc[i, ["sold_shares", "dividend_gross", "dividend_tax", "action_status"]] = int(quantity), gross, gross * .2, "catalogue_scenario"
    bought = trades.entry_status.eq("filled")
    known = bought & trades.exit_price.notna() & trades.entry_window_status.eq("valid") & trades.exit_window_status.eq("valid") & ~trades.bad_holding_day & trades.action_status.isin(["no_action", "catalogue_scenario"])
    not_bought = ~bought & trades.entry_window_status.eq("valid") & ~trades.bad_holding_day
    trades["downside_score"] = -1.
    trades.loc[not_bought, "downside_score"] = 0.
    part = trades.loc[known]
    values = dated_economic_return(part.shares, part.sold_shares, part.entry_price/1.0005*1.0015,
        part.exit_price/.9995*.9985, part.dividend_gross, part.dividend_tax, part.date, part.exit_date)
    trades["economic_scenario15"] = np.nan
    trades.loc[known, "economic_scenario15"] = values
    trades.loc[known, "downside_score"] = values
    trades["score_origin"] = np.select([known,not_bought], ["recorded_economic_scenario","known_not_bought"], default="unresolved_full_loss_training_scenario")
    return trades, events


def build(output: Path = ROOT, *, reuse_window_cache: bool = False) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name / "repriced.parquet").exists() for name in ("old_score", "downside")):
        raise ValueError("Do not replace training inputs after new model outcomes exist")
    previous = json.loads((PREVIOUS / "input_report.json").read_text())
    if sha(PREVIOUS / "features.parquet") != previous["features_sha256"] or sha(TRAINING_TRADES) != previous["training_trades_sha256"]:
        raise ValueError("The previously frozen features or training trades changed")
    if sha(PREVIOUS / "training_labels.parquet") != previous["labels_sha256"]:
        raise ValueError("The comparison model's original scores changed")
    features = pd.read_parquet(PREVIOUS / "features.parquet")
    features = features.loc[features.decision_buyable & features.price_1449.ge(5)].copy()
    train = features.loc[features.date.str.startswith("2024")]
    trades = train[["date", "code"]].merge(pd.read_parquet(TRAINING_TRADES), on=["date", "code"], validate="one_to_one")
    if len(trades) != len(train) or trades.exit_date.dropna().ge("2025-01-01").any():
        raise ValueError("Training grid is incomplete or crosses its year")
    needed = pd.concat([trades[["date", "code"]],
        trades.loc[trades.exit_date.notna(), ["exit_date", "code"]].rename(columns={"exit_date": "date"})]).drop_duplicates()
    if not needed.date.between("2024-01-01", "2024-12-31").all():
        raise ValueError("Training windows must be in 2024")
    if reuse_window_cache:
        windows = pd.read_parquet(output / "training_windows.parquet")
        sources = json.loads((output / "window_sources.json").read_text())
        pd.testing.assert_frame_equal(windows[["date", "code"]].sort_values(["date", "code"]).reset_index(drop=True),
            needed.sort_values(["date", "code"]).reset_index(drop=True), check_dtype=False)
        for source in sources:
            exchange, symbol = source["code"].split(".")
            if sha(MINUTES / exchange.upper() / (symbol + ".parquet")) != source["source_sha256"]:
                raise ValueError("A cached training-window source changed")
    else:
        windows, sources = load_windows(needed)
        windows.to_parquet(output / "training_windows.parquet", index=False)
        save_json(output / "window_sources.json", sources)
    catalog_path = Path("data/research/cash_dividend_catalog/events_augmented.parquet")
    trades, events = training_scores(trades, windows, catalog_path=catalog_path)
    events.to_parquet(output / "training_actions.parquet", index=False)
    old = pd.read_parquet(PREVIOUS / "training_labels.parquet")[["date", "code", "net_return"]].rename(columns={"net_return": "old_score"})
    trades = trades.merge(old, on=["date", "code"], validate="one_to_one").sort_values(["date", "code"])
    trades.to_parquet(output / "training_labels.parquet", index=False, compression="zstd")
    features.sort_values(["date", "code"]).to_parquet(output / "features.parquet", index=False, compression="zstd")
    folds = []
    for last, first in (("2024-06-14", "2024-07-01"), ("2024-12-17", "2025-01-01")):
        part = trades.loc[trades.date.le(last)]
        if part.target_exit_date.ge(first).any() or part.exit_date.dropna().ge(first).any():
            raise ValueError("A training label overlaps test time")
        folds.append({"training_last": last, "test_first": first, "rows": len(part),
            "score_origins": part.score_origin.value_counts().to_dict(),
            "action_statuses": part.action_status.value_counts().to_dict(),
            "delayed_economic_labels": int((part.exit_delay_sessions.gt(0) & part.score_origin.eq("recorded_economic_scenario")).sum()),
            "old_clipped_mean": float(part.old_score.clip(-.15,.15).mean()),
            "downside_clipped_mean": float(part.downside_score.clip(-.15,.15).mean())})
    result = {"rule_commit": RULE_COMMIT, "feature_rows": len(features), "training_rows": len(trades),
        "window_rows": len(windows), "window_statuses": windows.window_status.value_counts().to_dict(),
        "training_source_sha256": sha(TRAINING_TRADES), "catalog_sha256": sha(catalog_path),
        "labels_sha256": sha(output / "training_labels.parquet"), "features_sha256": sha(output / "features.parquet"),
        "folds": folds, "new_test_outcomes_read": False, "holdout_read": False,
        "unresolved_full_loss_is_training_scenario_only": True}
    save_json(output / "label_report.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(build(), ensure_ascii=False, indent=2))
