"""Check three fixed T+1 decisions against a T+5 close exit.

The decision observes the next trading day's completed 14:50 bar and then
uses the already modeled 14:52-14:55 exit. It remains exploratory.
"""

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.market_study import _week_bootstrap


ROOT = Path("data/research")
POLICIES = {
    "stop_2pct": lambda value: value <= -.02,
    "take_2pct": lambda value: value >= .02,
    "stop_2_or_take_3pct": lambda value: (value <= -.02) | (value >= .03),
}


def check(year: int, connection: duckdb.DuckDBPyConnection) -> list[dict]:
    outcomes = pd.read_parquet(ROOT / f"random_reprice_{year}_all_horizons.parquet")
    membership = pd.read_parquet(ROOT / f"random_candidate_map_{year}.parquet")
    outcomes = outcomes.merge(membership, on=["date", "code"])
    outcomes = outcomes.loc[outcomes.candidate.eq("random_low_amount")
                            & outcomes.target_notional.eq(20_000)
                            & outcomes.exit_window.eq("close")
                            & outcomes.horizon.isin((1, 5))]
    early = outcomes.loc[outcomes.horizon.eq(1)].drop(columns="horizon")
    late = outcomes.loc[outcomes.horizon.eq(5)].drop(columns="horizon")
    pairs = early.merge(late, on=["date", "code"],
                        suffixes=("_early", "_late"), validate="one_to_one")
    for column in ("entry_status", "entry_price", "shares"):
        a = pairs[f"{column}_early"]
        b = pairs[f"{column}_late"]
        if not a.fillna(-1).equals(b.fillna(-1)):
            raise ValueError(f"Entry mismatch across horizons: {column}")
    connection.register("pairs", pairs[["target_exit_date_early", "code"]])
    marks = connection.execute("""
        SELECT p.target_exit_date_early, p.code, s.price_1450
        FROM pairs p LEFT JOIN snapshots s
          ON s.date = p.target_exit_date_early AND s.code = p.code
    """).df()
    connection.unregister("pairs")
    if marks.duplicated(["target_exit_date_early", "code"]).any():
        raise ValueError("Duplicate next-day signal price")
    pairs = pairs.merge(marks, on=["target_exit_date_early", "code"],
                        how="left", validate="many_to_one")
    pairs["decision_return"] = pairs.price_1450 / pairs.entry_price_early - 1
    rows = []
    for half, frame in pairs.groupby(
        pairs.date.str[5:7].astype(int).le(6).map({True: "H1", False: "H2"})
    ):
        dates = sorted(frame.date.unique())
        count = frame.groupby("date").size().reindex(dates)
        base_valid = (frame.exit_status_late.eq("filled")
                      & frame.quality_clean_exit_late)
        base = (frame.net_return_late.where(base_valid, 0).groupby(frame.date)
                .sum().reindex(dates, fill_value=0) / count)
        for policy_name, trigger_fn in POLICIES.items():
            trigger = trigger_fn(frame.decision_return).fillna(False)
            early_valid = (frame.exit_status_early.eq("filled")
                           & frame.quality_clean_exit_early)
            selected_return = np.where(trigger, frame.net_return_early,
                                       frame.net_return_late)
            selected_valid = np.where(trigger, early_valid, base_valid)
            amount = pd.Series(np.where(selected_valid, selected_return, 0),
                               index=frame.index)
            cash = amount.groupby(frame.date).sum().reindex(dates, fill_value=0) / count
            difference = cash - base
            rows.append({
                "year": year, "half": half, "policy": policy_name,
                "days": len(dates), "signals": len(frame),
                "early_decision_rate": float(trigger.mean()),
                "policy_cash_mean": float(cash.mean()),
                "fixed_five_day_cash_mean": float(base.mean()),
                "paired_improvement": float(difference.mean()),
                "improvement_week_ci": _week_bootstrap(
                    difference.reset_index(drop=True), pd.Series(dates), 111
                ),
            })
    return rows


def main() -> None:
    c = duckdb.connect()
    c.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")).create_view(
        "snapshots"
    )
    rows = check(2024, c) + check(2025, c)
    output = ROOT / "adaptive_exit_probe.json"
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
