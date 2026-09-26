"""Freeze 14:49 limit-down reopen pairs without reading future prices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("data/research")
OUTPUT = ROOT / "limit_down_reopen"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")
FEATURES = ("price_1449", "amount_1449", "late_return",
            "prior20_return", "day_return", "position")


def freeze(root: Path = ROOT, output: Path = OUTPUT) -> dict:
    source_audit = json.loads((root / "minute_prefix_1449" /
                               "input_audit.json").read_text(encoding="utf-8"))
    if not source_audit["input_gate_passed"]:
        raise ValueError("14:49 input source failed its coverage audit")
    c = duckdb.connect()
    try:
        c.execute("SET threads = 4")
        c.read_parquet(str(root / "minute_prefix_1449" / "*" /
                           "part_*.parquet")).create_view("prefix")
        c.read_parquet(str(root / "market_snapshots_ci" /
                           "*.parquet")).create_view("snapshots")
        inputs = c.execute("""
            WITH eligible AS (
                SELECT p.date, p.code, SUBSTR(p.code, 1, 2) AS exchange,
                       p.price_1449, p.low_1449, p.amount_1449,
                       p.return_last29 AS late_return,
                       s.return20_prior_adjusted AS prior20_return,
                       p.price_1449 / s.preclose - 1 AS day_return,
                       (p.price_1449 - p.low_1449)
                           / (p.high_1449 - p.low_1449) AS position,
                       s.preclose
                FROM prefix p JOIN snapshots s USING (date, code)
                WHERE p.date BETWEEN '2024-01-01' AND '2025-12-17'
                  AND (p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%')
                  AND s.tradestatus = 1 AND s.isST = 0
                  AND s.listing_age_sessions >= 20
                  AND s.reference_gap IS FALSE
                  AND p.quote_outside_traded_range IS FALSE
                  AND s.preclose > 0 AND p.price_1449 >= 5
                  AND p.amount_1449 >= 30000000
                  AND s.return20_prior_adjusted BETWEEN -.20 AND .20
                  AND p.return_last29 BETWEEN -.03 AND .03
                  AND p.price_1449 / s.preclose - 1
                      BETWEEN -.098 AND -.02
                  AND p.high_1449 > p.low_1449
            ), priced AS (
                SELECT *, ROUND(CAST(preclose AS DECIMAL(18, 2))
                                * CAST(.90 AS DECIMAL(4, 2)), 2) AS down_limit
                FROM eligible
            )
            SELECT date, code, exchange, price_1449, low_1449,
                   amount_1449, late_return, prior20_return, day_return,
                   position, down_limit,
                   (price_1449 - down_limit) / preclose AS limit_distance,
                   ABS(low_1449 - down_limit) <= .005
                       AND CAST(price_1449 AS DECIMAL(18, 2))
                           >= down_limit + CAST(.02 AS DECIMAL(4, 2))
                       AS reopened,
                   low_1449 > down_limit + .005 AS untouched
            FROM priced ORDER BY date, code
        """).df()
    finally:
        c.close()
    if inputs.empty or inputs.duplicated(["date", "code"]).any():
        raise ValueError("Missing or duplicate eligible inputs")
    if not inputs.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("Non-development year in input pool")
    if not np.isfinite(inputs[list(FEATURES) + ["low_1449"]].to_numpy()).all():
        raise ValueError("Nonfinite decision-time input")
    inputs["half"] = inputs.date.str[:4] + "H" + np.where(
        inputs.date.str[5:7].astype(int).le(6), "1", "2")
    events = inputs.loc[inputs.reopened].sort_values(
        ["date", "limit_distance", "code"], kind="stable")
    selected = events.groupby("date", sort=False).head(5)
    controls = inputs.loc[inputs.untouched]
    peers_by_day = {day: group.copy()
                    for day, group in controls.groupby("date", sort=False)}
    pairs = []
    for day, event_group in selected.groupby("date", sort=True):
        available = peers_by_day.get(day)
        if available is None:
            continue
        for event in event_group.itertuples(index=False):
            same_exchange = available.loc[available.exchange.eq(event.exchange)]
            amount_ratio = same_exchange.amount_1449 / event.amount_1449
            price_ratio = same_exchange.price_1449 / event.price_1449
            choices = same_exchange.loc[
                (same_exchange.day_return - event.day_return).abs().le(.01)
                & (same_exchange.late_return - event.late_return).abs().le(.015)
                & (same_exchange.prior20_return - event.prior20_return).abs().le(.10)
                & amount_ratio.between(.25, 4)
                & price_ratio.between(.3, 3)
                & (same_exchange.position - event.position).abs().le(.25)
            ].copy()
            if choices.empty:
                continue
            choices["distance"] = (
                (choices.day_return - event.day_return).abs() / .01
                + (choices.late_return - event.late_return).abs() / .015
                + (choices.prior20_return - event.prior20_return).abs() / .10
                + np.abs(np.log(choices.amount_1449 /
                                event.amount_1449)) / np.log(4)
                + np.abs(np.log(choices.price_1449 /
                                event.price_1449)) / np.log(3)
                + (choices.position - event.position).abs() / .25
            )
            peer = choices.sort_values(
                ["distance", "code"], kind="stable").iloc[0]
            pairs.append({
                "date": day, "half": event.half, "exchange": event.exchange,
                "event_code": event.code, "control_code": peer.code,
                "event_down_limit": float(event.down_limit),
                "event_low_1449": event.low_1449,
                "match_distance": float(peer.distance),
                **{f"event_{col}": getattr(event, col) for col in FEATURES},
                **{f"control_{col}": peer[col] for col in FEATURES},
            })
            available = available.loc[available.code.ne(peer.code)]
    matched = pd.DataFrame(pairs)
    if not matched.empty and (matched.duplicated(["date", "event_code"]).any()
                              or matched.duplicated(["date", "control_code"]).any()):
        raise ValueError("A signal or peer was reused")
    by_half = {}
    for half in HALVES:
        pool = selected.loc[selected.half.eq(half)]
        part = matched.loc[matched.half.eq(half)] if not matched.empty else matched
        count = len(part)
        residuals = {}
        if count:
            for col in ("day_return", "late_return", "prior20_return",
                        "position"):
                residuals[col + "_median_abs_gap"] = float((
                    part["event_" + col] - part["control_" + col]
                ).abs().median())
            for col in ("price_1449", "amount_1449"):
                residuals[col + "_control_over_event_median"] = float((
                    part["control_" + col] / part["event_" + col]
                ).median())
        by_half[half] = {
            "capacity_candidates": len(pool),
            "pairs": count,
            "signal_days": int(part.date.nunique()) if count else 0,
            "match_fraction": count / len(pool) if len(pool) else 0,
            "residual_balance": residuals,
        }
    def half_passes(row: dict) -> bool:
        balance = row["residual_balance"]
        return (row["pairs"] >= 30 and row["signal_days"] >= 15
                and row["match_fraction"] >= .35
                and balance["day_return_median_abs_gap"] <= .003
                and balance["late_return_median_abs_gap"] <= .005
                and balance["prior20_return_median_abs_gap"] <= .03
                and balance["position_median_abs_gap"] <= .10
                and .5 <= balance["price_1449_control_over_event_median"] <= 2
                and .5 <= balance["amount_1449_control_over_event_median"] <= 2)
    gate = all(half_passes(row) if row["pairs"] else False
               for row in by_half.values())
    output.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(output / "pairs.parquet", index=False,
                       compression="zstd")
    report = {"years": [2024, 2025], "last_signal_date": "2025-12-17",
              "eligible_stock_days": len(inputs),
              "reopened_events": len(events), "untouched_controls": len(controls),
              "capacity_candidates": len(selected), "matched_pairs": len(matched),
              "by_half": by_half, "outcome_gate_passed": bool(gate)}
    (output / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(freeze(args.root, args.output),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
