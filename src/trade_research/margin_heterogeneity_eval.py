"""Evaluate the frozen chronological margin classifier on 2024-H2/2025."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .margin_heterogeneity import (
    CONTROL, MIN_HAC_T, MIN_INTEREST_Z, TREATED, WINDOWS,
    _prepare_inputs,
)
from .market_study import _quality_keys, _quality_symbols, _week_bootstrap
from .residual_liquidity import _period


HORIZONS = (1, 5)


def write_reprice_signals(selected_path: Path, snapshot_dir: Path,
                          output: Path) -> None:
    selected = pd.read_parquet(selected_path)[["date", "code"]]
    if selected.empty or selected.duplicated(["date", "code"]).any():
        raise ValueError("Trained-margin signals are empty or duplicated")
    c = duckdb.connect()
    c.register("selected", selected)
    c.read_parquet(str(snapshot_dir / "*.parquet")).create_view("s")
    signals = c.execute("""
        SELECT r.date, r.code, s.isST, s.reference_gap,
               s.quote_outside_traded_range, s.listing_age_sessions
        FROM selected r JOIN s USING(date, code)
        ORDER BY r.date, r.code
    """).df()
    if len(signals) != len(selected):
        raise ValueError("A trained-margin raw signal lacks a 14:50 snapshot")
    output.parent.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(output, index=False, compression="zstd")


def _all_test_inputs(universe_path: Path, margin_path: Path,
                     models_path: Path) -> pd.DataFrame:
    inputs = _prepare_inputs(universe_path, margin_path)
    models = pd.read_parquet(models_path)
    frames = []
    for name, _, _, start, end, _ in WINDOWS:
        slice_ = inputs.loc[inputs.date.between(start, end)].merge(
            models.loc[models.window.eq(name)], on="code", how="left",
            validate="many_to_one",
        )
        z = (slice_.interest - slice_.interest_mean) / slice_.interest_std
        slice_["trigger"] = (
            slice_.hac_t.gt(MIN_HAC_T) & slice_.beta.gt(0)
            & z.ge(MIN_INTEREST_Z)
        )
        slice_["window"] = name
        frames.append(slice_[["date", "code", "board", "size_bucket",
                              "trigger", "window"]])
    frame = pd.concat(frames, ignore_index=True)
    if frame.duplicated(["date", "code"]).any() or not frame.date.str[:4].isin(
            ("2024", "2025")).all():
        raise ValueError("Malformed full-market classifier test inputs")
    return frame


def evaluate(selected_path: Path, universe_path: Path, margin_path: Path,
             models_path: Path, outcome_dir: Path, issues_dir: Path,
             report_path: Path, trades_path: Path) -> dict:
    selected = pd.read_parquet(selected_path)
    if set(selected.candidate) != {TREATED, CONTROL}:
        raise ValueError("Missing trained-margin group or control")
    if not selected.date.gt(selected.trade_date).all():
        raise ValueError("Trained-margin test uses same-day financing balances")
    pairs = selected.groupby(["date", "pair_code"]).candidate.nunique()
    if not pairs.eq(2).all() or len(selected) != 2 * len(pairs):
        raise ValueError("Incomplete trained-margin matched pairs")
    c = duckdb.connect()
    c.execute("SET threads = 4")
    c.register("r", selected)
    c.read_parquet(str(outcome_dir / "*.parquet")).create_view("o")
    c.register("bad_days", _quality_keys(issues_dir))
    c.register("bad_symbols", _quality_symbols(issues_dir))
    quality = """
        o.exit_status = 'filled'
        AND NOT EXISTS (SELECT 1 FROM bad_symbols b WHERE b.code = r.code)
        AND NOT EXISTS (
            SELECT 1 FROM bad_days q
            WHERE q.code = r.code AND q.date >= r.date
              AND q.date <= o.exit_date
        )
    """
    trades = c.execute(f"""
        SELECT r.*, o.horizon, o.entry_status, o.entry_price, o.shares,
               o.exit_status, o.exit_date, o.exit_delay_sessions,
               o.exit_price, o.net_return,
               {quality} AS quality_clean_exit
        FROM r JOIN o USING(date, code)
        WHERE o.horizon IN (1, 5)
    """).df()
    if len(trades) != len(selected) * len(HORIZONS):
        raise ValueError("A trained-margin stock-day lacks minute outcomes")
    report = {"primary_horizon": 5, "matched": {}, "full_market": {},
              "note": "2025 exploratory, not blind; 2026 outcomes untouched"}
    for name, _, _, _, _, _ in WINDOWS:
        report["matched"][name] = {}
        for horizon in HORIZONS:
            current = trades.loc[trades.window.eq(name)
                                 & trades.horizon.eq(horizon)]
            high = current.loc[current.candidate.eq(TREATED)]
            low = current.loc[current.candidate.eq(CONTROL)]
            report["matched"][name][str(horizon)] = {}
            periods = [("full", high)]
            if name == "2025":
                periods.extend((
                    ("H1", high.loc[high.date.str[5:7].astype(int).le(6)]),
                    ("H2", high.loc[high.date.str[5:7].astype(int).gt(6)]),
                ))
            for label, rows in periods:
                summary = _period(rows, low)
                summary["same_day_matched_mean"] = summary.pop(
                    "same_day_random_mean")
                report["matched"][name][str(horizon)][label] = summary

    all_inputs = _all_test_inputs(universe_path, margin_path, models_path)
    c.unregister("r")
    c.register("r", all_inputs)
    full = c.execute(f"""
        SELECT r.window, r.date, r.board, r.size_bucket, r.trigger,
               COUNT(*) AS stock_days,
               AVG(CASE WHEN {quality} THEN o.net_return ELSE 0 END) AS cash_mean,
               SUM(CASE WHEN o.entry_status = 'filled' THEN 1 ELSE 0 END) AS entries,
               SUM(CASE WHEN {quality} THEN 1 ELSE 0 END) AS clean_exits
        FROM r JOIN o USING(date, code)
        WHERE o.horizon = 5
        GROUP BY r.window, r.date, r.board, r.size_bucket, r.trigger
    """).df()
    if int(full.stock_days.sum()) != len(all_inputs):
        raise ValueError("A full-market trained-margin stock-day lacks T+5 outcome")
    keys = ["window", "date", "board", "size_bucket"]
    means = full.pivot(index=keys, columns="trigger", values="cash_mean")
    comparable = means.dropna(subset=[True, False]).rename(
        columns={True: "trigger_cash", False: "other_cash"}
    ).copy()
    comparable["edge"] = comparable.trigger_cash - comparable.other_cash
    comparable = comparable.reset_index()
    for name, _, _, _, _, _ in WINDOWS:
        report["full_market"][name] = {}
        section = comparable.loc[comparable.window.eq(name)]
        for board in ("all", "sh_main", "sh_star", "sz_main", "sz_gem"):
            part = section if board == "all" else section.loc[section.board.eq(board)]
            daily = part.groupby("date", as_index=False)[
                ["trigger_cash", "other_cash", "edge"]
            ].mean()
            if daily.empty:
                continue
            report["full_market"][name][board] = {
                "days": len(daily), "matched_strata": len(part),
                "trigger_cash_mean": float(daily.trigger_cash.mean()),
                "other_cash_mean": float(daily.other_cash.mean()),
                "edge_mean": float(daily.edge.mean()),
                "edge_week_ci": _week_bootstrap(daily.edge, daily.date, 246),
            }
    report["full_market_stock_days"] = len(all_inputs)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    trades.to_parquet(trades_path, index=False, compression="zstd")
    return report


def summarize_repriced(selected_path: Path, repriced_path: Path,
                       stored_trades_path: Path, output: Path) -> dict:
    membership = pd.read_parquet(selected_path)[
        ["date", "code", "candidate", "window"]
    ]
    raw = pd.read_parquet(repriced_path)
    if (len(raw) != len(membership) * 4
            or set(raw.target_notional) != {20_000.0, 100_000.0}
            or set(raw.horizon) != {1, 5}
            or raw.duplicated(["date", "code", "target_notional", "horizon"]).any()):
        raise ValueError("Incomplete trained-margin raw-minute repricing")
    raw_100 = raw.loc[raw.target_notional.eq(100_000)]
    stored = pd.read_parquet(stored_trades_path)
    compare = stored.merge(raw_100, on=["date", "code", "horizon"],
                           suffixes=("_stored", "_raw"), validate="one_to_one")
    if len(compare) != len(stored):
        raise ValueError("A stored trained-margin trade lacks raw-minute match")
    for field in ("entry_status", "exit_status", "exit_date",
                  "quality_clean_exit"):
        if not compare[f"{field}_stored"].fillna("missing").eq(
                compare[f"{field}_raw"].fillna("missing")).all():
            raise ValueError(f"Stored/raw trained-margin {field} mismatch")
    for field in ("entry_price", "shares", "exit_delay_sessions",
                  "exit_price", "net_return"):
        if not np.isclose(
                compare[f"{field}_stored"].to_numpy(dtype=float),
                compare[f"{field}_raw"].to_numpy(dtype=float),
                atol=1e-12, equal_nan=True).all():
            raise ValueError(f"Stored/raw trained-margin {field} mismatch")
    joined = raw.merge(membership, on=["date", "code"], validate="many_to_one")
    if len(joined) != len(raw):
        raise ValueError("A raw trained-margin trade is outside frozen selection")
    result = {"exact_100k_rows": len(compare), "results": {}}
    for size in (20_000, 100_000):
        result["results"][str(size)] = {}
        for name, _, _, _, _, _ in WINDOWS:
            result["results"][str(size)][name] = {}
            for horizon in HORIZONS:
                sample = joined.loc[joined.target_notional.eq(size)
                                    & joined.window.eq(name)
                                    & joined.horizon.eq(horizon)]
                high = sample.loc[sample.candidate.eq(TREATED)]
                low = sample.loc[sample.candidate.eq(CONTROL)]
                summary = _period(high, low)
                summary["same_day_matched_mean"] = summary.pop(
                    "same_day_random_mean")
                result["results"][str(size)][name][str(horizon)] = summary
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected", type=Path,
                        default=Path("data/research/margin_trained_selected.parquet"))
    parser.add_argument("--universe", type=Path,
                        default=Path("data/research/margin_universe_quintiles.parquet"))
    parser.add_argument("--margin", type=Path,
                        default=Path("data/research/margin/margin_2024_2025.parquet"))
    parser.add_argument("--models", type=Path,
                        default=Path("data/research/margin_trained_models.parquet"))
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/margin_trained_report.json"))
    parser.add_argument("--trades", type=Path,
                        default=Path("data/research/margin_trained_trades.parquet"))
    parser.add_argument("--reprice-signals", type=Path,
                        default=Path("data/research/margin_trained_reprice_signals.parquet"))
    parser.add_argument("--repriced", type=Path)
    parser.add_argument("--repriced-report", type=Path,
                        default=Path("data/research/margin_trained_reprice_report.json"))
    args = parser.parse_args()
    write_reprice_signals(args.selected, args.snapshots, args.reprice_signals)
    result = evaluate(args.selected, args.universe, args.margin, args.models,
                      args.outcomes, args.issues, args.report, args.trades)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.repriced:
        raw = summarize_repriced(args.selected, args.repriced,
                                 args.trades, args.repriced_report)
        print({"raw_minute_sizes": list(raw["results"]),
               "exact_100k_rows": raw["exact_100k_rows"]})


if __name__ == "__main__":
    main()
