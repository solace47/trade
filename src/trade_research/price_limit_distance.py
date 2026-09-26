"""Freeze same-day near-down-limit pairs using 14:49 inputs only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("data/research")
OUTPUT = ROOT / "price_limit_distance"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")


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
                SELECT p.date, p.code, p.price_1449, p.amount_1449,
                       p.return_last29 AS late_return,
                       s.return20_prior_adjusted AS prior20_return,
                       p.price_1449 / s.preclose - 1 AS day_return,
                       (p.price_1449 - p.low_1449)
                           / (p.high_1449 - p.low_1449) AS position,
                       s.preclose,
                       CASE WHEN p.code LIKE 'sh.60%'
                                  OR p.code LIKE 'sz.00%' THEN 'main'
                            WHEN p.code LIKE 'sz.30%' THEN 'chinext'
                            ELSE 'star' END AS board,
                       CASE WHEN p.code LIKE 'sh.60%'
                                  OR p.code LIKE 'sz.00%'
                            THEN CAST(.90 AS DECIMAL(4, 2))
                            ELSE CAST(.80 AS DECIMAL(4, 2))
                       END AS down_factor
                FROM prefix p JOIN snapshots s USING (date, code)
                WHERE p.date BETWEEN '2024-01-01' AND '2025-12-17'
                  AND (p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%'
                    OR p.code LIKE 'sz.30%' OR p.code LIKE 'sh.68%')
                  AND s.tradestatus = 1 AND s.isST = 0
                  AND s.listing_age_sessions >= 20
                  AND s.reference_gap IS FALSE
                  AND p.quote_outside_traded_range IS FALSE
                  AND s.preclose > 0 AND p.price_1449 >= 5
                  AND p.amount_1449 >= 30000000
                  AND s.return20_prior_adjusted BETWEEN -.20 AND .20
                  AND p.return_last29 BETWEEN -.03 AND .03
                  AND p.price_1449 / s.preclose - 1
                      BETWEEN -.095 AND -.075
                  AND p.high_1449 > p.low_1449
            ), limited AS (
                SELECT *, ROUND(CAST(preclose AS DECIMAL(18, 2))
                                * down_factor, 2) AS down_limit
                FROM eligible
            )
            SELECT date, code, board, price_1449, amount_1449,
                   late_return, prior20_return, day_return, position,
                   (price_1449 - down_limit) / preclose AS limit_distance
            FROM limited
            WHERE board <> 'main'
               OR CAST(price_1449 AS DECIMAL(18, 2))
                    >= down_limit + CAST(.01 AS DECIMAL(4, 2))
            ORDER BY date, code
        """).df()
    finally:
        c.close()
    if inputs.empty or inputs.duplicated(["date", "code"]).any():
        raise ValueError("Missing or duplicate eligible inputs")
    if not inputs.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("Non-development year in input pool")
    columns = ["price_1449", "amount_1449", "late_return",
               "prior20_return", "day_return", "position"]
    if not np.isfinite(inputs[columns].to_numpy()).all():
        raise ValueError("Nonfinite decision-time input")
    inputs["half"] = inputs.date.str[:4] + "H" + np.where(
        inputs.date.str[5:7].astype(int).le(6), "1", "2")
    main = inputs.loc[inputs.board.eq("main")].sort_values(
        ["date", "limit_distance", "code"], kind="stable")
    selected = main.groupby("date", sort=False).head(5)
    wide = inputs.loc[inputs.board.ne("main")]
    peers_by_day = {day: part.copy()
                    for day, part in wide.groupby("date", sort=False)}
    pairs = []
    for day, narrow_group in selected.groupby("date", sort=True):
        available = peers_by_day.get(day)
        if available is None:
            continue
        for narrow in narrow_group.itertuples(index=False):
            amount_ratio = available.amount_1449 / narrow.amount_1449
            price_ratio = available.price_1449 / narrow.price_1449
            candidates = available.loc[
                (available.day_return - narrow.day_return).abs().le(.005)
                & (available.late_return - narrow.late_return).abs().le(.01)
                & (available.prior20_return - narrow.prior20_return).abs().le(.05)
                & amount_ratio.between(.25, 4)
                & price_ratio.between(.3, 3)
                & (available.position - narrow.position).abs().le(.25)
            ].copy()
            if candidates.empty:
                continue
            candidates["distance"] = (
                (candidates.day_return - narrow.day_return).abs() / .005
                + (candidates.late_return - narrow.late_return).abs() / .01
                + (candidates.prior20_return - narrow.prior20_return).abs() / .05
                + np.abs(np.log(candidates.amount_1449 /
                                narrow.amount_1449)) / np.log(4)
                + np.abs(np.log(candidates.price_1449 /
                                narrow.price_1449)) / np.log(3)
                + (candidates.position - narrow.position).abs() / .25
            )
            peer = candidates.sort_values(
                ["distance", "code"], kind="stable").iloc[0]
            pair_id = day + ":" + narrow.code
            pairs.append({"date": day, "half": narrow.half, "pair_id": pair_id,
                          "main_code": narrow.code, "wide_code": peer.code,
                          "wide_board": peer.board,
                          "distance": float(peer.distance),
                          **{f"main_{col}": getattr(narrow, col)
                             for col in columns},
                          **{f"wide_{col}": peer[col]
                             for col in columns}})
            available = available.loc[available.code.ne(peer.code)]
    matched = pd.DataFrame(pairs)
    if not matched.empty and (matched.duplicated("pair_id").any()
                              or matched.duplicated(["date", "wide_code"]).any()):
        raise ValueError("A signal or peer was reused")
    by_half = {}
    for half in HALVES:
        pool = selected.loc[selected.half.eq(half)]
        part = matched.loc[matched.half.eq(half)] if not matched.empty else matched
        count = len(part)
        by_half[half] = {
            "capacity_candidates": len(pool),
            "pairs": count,
            "signal_days": int(part.date.nunique()) if count else 0,
            "match_fraction": count / len(pool) if len(pool) else 0,
            "wide_chinext_pairs": int(part.wide_board.eq("chinext").sum())
            if count else 0,
            "wide_star_pairs": int(part.wide_board.eq("star").sum())
            if count else 0,
        }
    gate = all(row["pairs"] >= 30 and row["signal_days"] >= 15
               and row["match_fraction"] >= .35
               for row in by_half.values())
    residuals = {}
    if not matched.empty:
        for col in columns:
            if col in ("price_1449", "amount_1449"):
                residuals[col + "_wide_over_main_median"] = float(
                    (matched["wide_" + col] /
                     matched["main_" + col]).median())
            else:
                residuals[col + "_wide_minus_main_median"] = float(
                    (matched["wide_" + col] -
                     matched["main_" + col]).median())
    output.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(output / "pairs.parquet", index=False,
                       compression="zstd")
    report = {"years": [2024, 2025], "last_signal_date": "2025-12-17",
              "eligible_stock_days": len(inputs),
              "main_eligible": len(main), "wide_eligible": len(wide),
              "capacity_candidates": len(selected),
              "matched_pairs": len(matched),
              "by_half": by_half, "residual_balance": residuals,
              "outcome_gate_passed": bool(gate)}
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
