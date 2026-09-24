"""Reprice fixed residual-reversal signals for next-morning exits.

This follow-up was specified after inspecting close-exit results. It is
exploratory and never reads 2026 signal dates or outcomes.
"""

import json
from pathlib import Path

import duckdb
import pandas as pd

from trade_research.market_study import _week_bootstrap
from trade_research.size_sensitivity import run


ROOT = Path("data/research")
OUTPUT = ROOT / "residual_exit"
CANDIDATES = ("shock_reversal", "prior_residual_loser")
NOTIONALS = (20_000, 50_000)
WINDOWS = ("close", "morning")


def _signals(candidate: str, destination: Path) -> int:
    connection = duckdb.connect()
    connection.read_parquet(str(ROOT / "residual_liquidity_trades.parquet")
                            ).create_view("ranked")
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshot")
    connection.execute("""
        CREATE TEMP TABLE keys AS
        SELECT DISTINCT date, code FROM ranked
        WHERE candidate = ? AND horizon = 1
    """, [candidate])
    frame = connection.execute("""
        SELECT s.* FROM snapshot s JOIN keys USING (date, code)
    """).df()
    if frame.empty or frame.duplicated(["date", "code"]).any():
        raise ValueError("Expected unique nonempty signal keys")
    if not frame.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("Exit probe escaped 2024-2025")
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False, compression="zstd")
    return len(frame)


def _daily_cash(frame: pd.DataFrame, dates: list[str]) -> pd.Series:
    counts = frame.groupby("date").size().reindex(dates)
    if counts.isna().any():
        raise ValueError("A selected day lacks a priced signal")
    valid = frame.loc[frame.exit_status.eq("filled")
                      & frame.quality_clean_exit]
    return valid.groupby("date").net_return.sum().reindex(dates, fill_value=0) / counts


def _summary(frame: pd.DataFrame) -> dict:
    dates = sorted(frame.date.unique())
    clean = frame.loc[frame.exit_status.eq("filled")
                      & frame.quality_clean_exit]
    daily = _daily_cash(frame, dates)
    return {
        "signals": len(frame), "days": len(dates),
        "entry_fills": int(frame.entry_status.eq("filled").sum()),
        "clean_exits": len(clean),
        "ontime_exits": int(clean.exit_delay_sessions.eq(0).sum()),
        "cash_mean": float(daily.mean()),
        "cash_week_ci": _week_bootstrap(
            daily.reset_index(drop=True), pd.Series(dates), 251
        ),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    report = {"candidates": list(CANDIDATES),
              "order_sizes": list(NOTIONALS), "horizon": 1,
              "exit_windows": list(WINDOWS), "results": {}}
    for candidate in CANDIDATES:
        signal_file = OUTPUT / f"{candidate}_signals.parquet"
        count = _signals(candidate, signal_file)
        outcome_file = OUTPUT / f"{candidate}_outcomes.parquet"
        outcomes = run(
            signal_file,
            Path("data/hf/pilot/data/stock_1m"),
            Path("data/baostock/market_2020_2026/daily"),
            Path("data/baostock/market_2020_2026/metadata/calendar.parquet"),
            outcome_file, NOTIONALS,
            ROOT / "market_issues_ci", WINDOWS, (1,), 4,
        )
        if len(outcomes) != count * len(NOTIONALS) * len(WINDOWS):
            raise ValueError("Unexpected reprice coverage")
        report["results"][candidate] = {}
        for notional in NOTIONALS:
            size = outcomes.loc[outcomes.target_notional.eq(notional)]
            report["results"][candidate][str(notional)] = {}
            for year in ("2024", "2025"):
                annual = size.loc[size.date.str.startswith(year)]
                result = {}
                for window in WINDOWS:
                    subset = annual.loc[annual.exit_window.eq(window)]
                    result[window] = _summary(subset)
                dates = sorted(annual.date.unique())
                close = _daily_cash(annual.loc[annual.exit_window.eq("close")], dates)
                morning = _daily_cash(
                    annual.loc[annual.exit_window.eq("morning")], dates
                )
                delta = morning - close
                result["morning_minus_close"] = {
                    "cash_mean": float(delta.mean()),
                    "cash_week_ci": _week_bootstrap(
                        delta.reset_index(drop=True), pd.Series(dates), 252
                    ),
                }
                report["results"][candidate][str(notional)][year] = result
        print({"candidate": candidate, "signals": count}, flush=True)
    target = OUTPUT / "report.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print({"report": str(target)})


if __name__ == "__main__":
    main()
