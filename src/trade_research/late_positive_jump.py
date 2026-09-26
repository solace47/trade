"""Freeze full-market 14:49 positive-minute-jump groups without outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


ROOT = Path("data/research")
OUTPUT = ROOT / "late_positive_jump"
HALVES = ("2024H1", "2024H2", "2025H1", "2025H2")
BALANCE_COLUMNS = ("day_return", "late_return", "prior20_return",
                   "position", "price_1449", "amount_1449", "root_variance")


def freeze(root: Path = ROOT, output: Path = OUTPUT) -> dict:
    audit = json.loads((root / "minute_prefix_1449" /
                        "input_audit.json").read_text(encoding="utf-8"))
    if not audit["input_gate_passed"]:
        raise ValueError("14:49 source coverage gate failed")
    output.mkdir(parents=True, exist_ok=True)
    c = duckdb.connect()
    try:
        c.execute("SET threads = 4")
        c.execute("SET memory_limit = '8GB'")
        c.execute("SET preserve_insertion_order = false")
        c.execute("SET temp_directory = ?", [str(output / "duckdb_tmp")])
        c.read_parquet([
            str(root / "late_residual_variance" / f"{year}_returns.parquet")
            for year in (2024, 2025)
        ]).create_view("returns")
        c.read_parquet(str(root / "minute_prefix_1449" / "*" /
                           "part_*.parquet")).create_view("prefix")
        c.read_parquet(str(root / "market_snapshots_ci" /
                           "*.parquet")).create_view("snapshots")
        joined = c.execute("""
            WITH base AS (
                SELECT p.date, p.code,
                       CASE WHEN p.code LIKE 'sh.60%'
                                  OR p.code LIKE 'sz.00%' THEN 'main'
                            WHEN p.code LIKE 'sz.30%' THEN 'chinext'
                            ELSE 'star' END AS board,
                       p.price_1449, p.amount_1449,
                       p.return_last29 AS late_return,
                       p.price_1449 / s.preclose - 1 AS day_return,
                       s.return20_prior_adjusted AS prior20_return,
                       (p.price_1449 - p.low_1449)
                           / (p.high_1449 - p.low_1449) AS position
                FROM prefix p JOIN snapshots s USING (date, code)
                WHERE p.date BETWEEN '2024-01-01' AND '2025-12-17'
                  AND (p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%'
                    OR p.code LIKE 'sz.30%' OR p.code LIKE 'sh.68%')
                  AND s.tradestatus = 1 AND s.isST = 0
                  AND s.listing_age_sessions >= 20
                  AND s.reference_gap IS FALSE
                  AND p.quote_outside_traded_range IS FALSE
                  AND s.preclose > 0 AND p.price_1449 >= 5
                  AND p.amount_1449 >= 100000000
                  AND s.return20_prior_adjusted BETWEEN -.20 AND .20
                  AND p.price_1449 / s.preclose - 1 BETWEEN -.03 AND .03
                  AND p.return_last29 BETWEEN .002 AND .02
                  AND p.high_1449 > p.low_1449
            ), minute AS (
                SELECT r.date, r.code, COUNT(*) AS minute_bars,
                       COUNT(DISTINCT r.label) AS minute_labels,
                       MAX(r.close) FILTER (WHERE r.label = '1449') AS final_close,
                       MAX(r.minute_return) AS max_positive_minute,
                       SQRT(SUM(POWER(r.minute_return, 2))) AS root_variance,
                       SUM(r.minute_return) AS sum_return
                FROM returns r JOIN base b USING (date, code)
                WHERE r.label BETWEEN '1421' AND '1449'
                GROUP BY r.date, r.code
            )
            SELECT b.*, m.minute_bars, m.minute_labels, m.final_close,
                   m.max_positive_minute, m.root_variance, m.sum_return
            FROM base b LEFT JOIN minute m USING (date, code)
            ORDER BY b.date, b.code
        """).df()
    finally:
        c.close()
    if joined.empty or joined.duplicated(["date", "code"]).any():
        raise ValueError("Missing or duplicate jump inputs")
    if not joined.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("Non-development year in input pool")
    joined["half"] = joined.date.str[:4] + "H" + np.where(
        joined.date.str[5:7].astype(int).le(6), "1", "2")
    valid = (joined.minute_bars.eq(29) & joined.minute_labels.eq(29)
             & (joined.final_close - joined.price_1449).abs().le(.005)
             & (np.expm1(joined.sum_return) - joined.late_return).abs().le(.0001)
             & joined.max_positive_minute.gt(0)
             & joined.root_variance.gt(0))
    inputs = joined.loc[valid].copy()
    columns = list(BALANCE_COLUMNS) + ["max_positive_minute"]
    if inputs.empty or not np.isfinite(inputs[columns].to_numpy()).all():
        raise ValueError("Missing or nonfinite valid jump inputs")
    groups = []
    board_days = 0
    for (_, _), group in inputs.groupby(["date", "board"], sort=True):
        if len(group) < 40:
            continue
        board_days += 1
        ranked = group.sort_values(["max_positive_minute", "code"],
                                   kind="stable")
        quintile = len(ranked) // 5
        low = ranked.head(quintile).copy()
        high = ranked.tail(quintile).copy()
        low["arm"] = "smooth"
        high["arm"] = "jump"
        groups.extend((low, high))
    if not groups:
        raise ValueError("No date-board group meets the minimum size")
    selections = pd.concat(groups, ignore_index=True)
    if selections.duplicated(["date", "code"]).any():
        raise ValueError("A stock entered both jump groups")
    by_half = {}
    for half in HALVES:
        source = joined.loc[joined.half.eq(half)]
        usable = inputs.loc[inputs.half.eq(half)]
        portion = selections.loc[selections.half.eq(half)]
        smooth = portion.loc[portion.arm.eq("smooth")]
        jump = portion.loc[portion.arm.eq("jump")]
        median_gap = (jump.max_positive_minute.median()
                      - smooth.max_positive_minute.median()) if len(jump) else 0
        corr = usable.max_positive_minute.corr(usable.root_variance)
        if not np.isfinite(corr):
            corr = 1.0
        balance = {}
        if len(smooth) and len(jump):
            for col in BALANCE_COLUMNS:
                if col in ("price_1449", "amount_1449", "root_variance"):
                    balance[col + "_jump_over_smooth_median"] = float(
                        jump[col].median() / smooth[col].median())
                else:
                    balance[col + "_jump_minus_smooth_median"] = float(
                        jump[col].median() - smooth[col].median())
        by_half[half] = {
            "base_stock_days": len(source),
            "valid_minute_stock_days": len(usable),
            "minute_coverage": len(usable) / len(source) if len(source) else 0,
            "board_days": int(portion[["date", "board"]].drop_duplicates().shape[0]),
            "signal_days": int(portion.date.nunique()),
            "smooth_stock_days": len(smooth),
            "jump_stock_days": len(jump),
            "max_positive_minute_median_gap": float(median_gap),
            "jump_variance_correlation": float(corr),
            "input_balance": balance,
        }
    gate = all(
        row["signal_days"] >= 100 and row["smooth_stock_days"] >= 1000
        and row["jump_stock_days"] >= 1000
        and row["minute_coverage"] >= .95
        and row["max_positive_minute_median_gap"] >= .003
        and abs(row["jump_variance_correlation"]) < .90
        for row in by_half.values()
    )
    report = {"years": [2024, 2025], "last_signal_date": "2025-12-17",
              "base_stock_days": len(joined), "valid_minute_stock_days": len(inputs),
              "eligible_board_days": board_days,
              "selected_stock_days": len(selections), "by_half": by_half,
              "outcome_gate_passed": bool(gate)}
    selections.to_parquet(output / "selections.parquet", index=False,
                          compression="zstd")
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
