"""Freeze 14:49 historical overnight-return-composition pairs before outcomes."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .late_to_open_reversal import _board


ROOT = Path("data/research")
OUTPUT = ROOT / "overnight_composition"
CAPACITY = 5
COOLDOWN = 5


def build_history(connection: duckdb.DuckDBPyConnection) -> None:
    """Prepare valid trailing gaps from ``daily_raw`` without signal-day data."""
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE gap_history AS
        WITH active AS (
            SELECT date, code, open, close, preclose, isST,
                   LAG(close) OVER (PARTITION BY code ORDER BY date)
                       AS prior_active_close
            FROM daily_raw
            WHERE date BETWEEN '2023-10-01' AND '2025-12-17'
              AND tradestatus = 1 AND open > 0 AND close > 0
              AND preclose > 0
        ), valid AS (
            SELECT date, code, open / preclose - 1 AS overnight_gap
            FROM active
            WHERE isST = 0 AND prior_active_close IS NOT NULL
              AND ABS(preclose - prior_active_close) <= .005
              AND ABS(open / preclose - 1) <= .10
        )
        SELECT date AS history_end, code,
               COUNT(*) OVER gap_window AS valid_days,
               MIN(date) OVER gap_window AS history_start,
               AVG(overnight_gap) OVER gap_window AS mean_overnight_gap
        FROM valid
        WINDOW gap_window AS (
            PARTITION BY code ORDER BY date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        )
    """)


def build_candidates(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE eligible_base AS
        SELECT p.date, p.code, p.price_1449, p.amount_1449,
               p.return_last29, p.price_1449 / s.preclose - 1
                   AS return_1449,
               (p.price_1449 - p.low_1449)
                   / NULLIF(p.high_1449 - p.low_1449, 0) AS position_1449,
               s.open_1450 / s.preclose - 1 AS open_gap_today,
               s.return20_prior_adjusted,
               CASE WHEN p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%'
                         THEN 'main'
                    WHEN p.code LIKE 'sz.30%' THEN 'chinext'
                    WHEN p.code LIKE 'sh.68%' THEN 'star'
                    ELSE NULL END AS board
        FROM prefix p JOIN snapshots s USING (date, code)
        WHERE ((p.date BETWEEN '2024-01-01' AND '2024-12-17')
            OR (p.date BETWEEN '2025-01-01' AND '2025-12-17'))
          AND s.isST = 0 AND s.listing_age_sessions >= 60
          AND NOT s.reference_gap AND NOT p.quote_outside_traded_range
          AND s.preclose > 0 AND s.open_1450 > 0
          AND p.price_1449 >= 5
          AND p.amount_1449 BETWEEN 100000000 AND 1000000000
          AND s.return20_prior_adjusted BETWEEN -.10 AND .10
          AND p.return_last29 BETWEEN -.01 AND .01
          AND (p.price_1449 - p.low_1449)
              / NULLIF(p.high_1449 - p.low_1449, 0) BETWEEN 0 AND 1
          AND p.price_1449 / s.preclose - 1 BETWEEN -.03 AND .03
    """)
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE candidates AS
        WITH joined AS (
            SELECT e.*, h.history_start, h.history_end, h.valid_days,
                   h.mean_overnight_gap
            FROM eligible_base e ASOF LEFT JOIN gap_history h
              ON e.code = h.code AND e.date > h.history_end
            WHERE h.valid_days = 20
              AND date_diff('day', CAST(h.history_start AS DATE),
                                   CAST(e.date AS DATE)) <= 45
              AND date_diff('day', CAST(h.history_end AS DATE),
                                   CAST(e.date AS DATE)) <= 10
              AND e.board IS NOT NULL
        ), ranked AS (
            SELECT *, COUNT(*) OVER (PARTITION BY date, board) AS board_pool,
                   NTILE(5) OVER (
                       PARTITION BY date, board
                       ORDER BY mean_overnight_gap, code
                   ) AS gap_quintile
            FROM joined
        )
        SELECT * FROM ranked WHERE board_pool >= 25
    """)


def select_inputs(connection: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict]:
    high = connection.execute("""
        SELECT * FROM candidates WHERE gap_quintile = 5
        ORDER BY date, md5('overnight-composition-v1' || date || code), code
    """).df()
    low = connection.execute("""
        SELECT * FROM candidates WHERE gap_quintile = 1
        ORDER BY date, code
    """).df()
    calendar = connection.execute("""
        SELECT DISTINCT date FROM prefix
        WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
        ORDER BY date
    """).df().date.tolist()
    date_index = {date: offset for offset, date in enumerate(calendar)}
    low_by_date = {date: rows.set_index("code", drop=False)
                   for date, rows in low.groupby("date", sort=False)}
    last_selected: dict[str, int] = {}
    selected: list[dict] = []
    attempted = 0
    for date, ranked in high.groupby("date", sort=True):
        index = date_index[date]
        available = low_by_date.get(date)
        daily_attempts = 0
        for row in ranked.itertuples(index=False):
            if daily_attempts >= CAPACITY:
                break
            if index - last_selected.get(row.code, -1000) <= COOLDOWN:
                continue
            daily_attempts += 1
            attempted += 1
            if available is None or available.empty:
                continue
            fresh = available.loc[
                available.board.eq(row.board)
                & available.code.map(
                    lambda code: index - last_selected.get(code, -1000) > COOLDOWN
                )
            ]
            if fresh.empty:
                continue
            prior_gap = (fresh.return20_prior_adjusted
                         - row.return20_prior_adjusted).abs()
            current_gap = (fresh.return_1449 - row.return_1449).abs()
            tail_gap = (fresh.return_last29 - row.return_last29).abs()
            open_gap = (fresh.open_gap_today - row.open_gap_today).abs()
            position_gap = (fresh.position_1449 - row.position_1449).abs()
            price_ratio = fresh.price_1449 / row.price_1449
            amount_ratio = fresh.amount_1449 / row.amount_1449
            matched = fresh.loc[
                prior_gap.le(.03) & current_gap.le(.005)
                & tail_gap.le(.003) & open_gap.le(.005)
                & position_gap.le(.30) & price_ratio.between(.5, 2)
                & amount_ratio.between(.5, 2)
            ].copy()
            if matched.empty:
                continue
            matched["distance"] = (
                prior_gap.loc[matched.index] / .03
                + current_gap.loc[matched.index] / .005
                + tail_gap.loc[matched.index] / .003
                + open_gap.loc[matched.index] / .005
                + position_gap.loc[matched.index] / .30
                + np.abs(np.log(price_ratio.loc[matched.index])) / np.log(2)
                + np.abs(np.log(amount_ratio.loc[matched.index])) / np.log(2)
            )
            control = matched.reset_index(drop=True).sort_values(
                ["distance", "code"]
            ).iloc[0]
            treatment = row._asdict()
            treatment.update(candidate="high_prior_overnight",
                             pair_id=row.code, daily_rank=daily_attempts)
            comparator = control.drop(labels="distance").to_dict()
            comparator.update(candidate="low_prior_overnight_control",
                              pair_id=row.code, daily_rank=daily_attempts)
            selected.extend((treatment, comparator))
            available = available.drop(index=control.code)
            last_selected[row.code] = index
            last_selected[control.code] = index
    signals = pd.DataFrame(selected)
    if not signals.empty and (
        signals.duplicated(["date", "code"]).any()
        or signals.groupby(["date", "pair_id"]).candidate.nunique().ne(2).any()
        or not signals.date.str[:4].isin(("2024", "2025")).all()
    ):
        raise ValueError("Invalid historical overnight pairs")
    pairs = (signals.loc[signals.candidate.eq("high_prior_overnight")]
             .merge(signals.loc[
                 signals.candidate.eq("low_prior_overnight_control")
             ], on=["date", "pair_id"], suffixes=("_high", "_low"),
                 validate="one_to_one") if not signals.empty else pd.DataFrame())
    halves = (signals.loc[signals.candidate.eq("high_prior_overnight")]
              .assign(half=lambda rows: rows.date.str[:4] + "H"
                      + np.where(rows.date.str[5:7].astype(int) <= 6, "1", "2"))
              if not signals.empty else pd.DataFrame())
    by_half = (halves.groupby("half").agg(
        pairs=("code", "size"), days=("date", "nunique")
    ).reset_index().to_dict("records") if not halves.empty else [])
    composition_gap = (pairs.mean_overnight_gap_high
                       - pairs.mean_overnight_gap_low
                       if not pairs.empty else pd.Series(dtype=float))
    audit = {
        "source_cutoff_label": "14:49",
        "history_strictly_before_signal": (
            bool(signals.history_end.lt(signals.date).all())
            if not signals.empty else True
        ),
        "eligible_candidates": int(connection.execute(
            "SELECT COUNT(*) FROM candidates").fetchone()[0]),
        "high_pool": len(high), "low_pool": len(low),
        "capacity_candidates": attempted, "paired": len(pairs),
        "match_fraction": len(pairs) / attempted if attempted else 0.0,
        "median_composition_gap": float(composition_gap.median())
            if not pairs.empty else None,
        "median_prior20_gap": float((
            pairs.return20_prior_adjusted_high
            - pairs.return20_prior_adjusted_low).abs().median())
            if not pairs.empty else None,
        "median_today_open_gap": float((pairs.open_gap_today_high
                                          - pairs.open_gap_today_low).abs().median())
            if not pairs.empty else None,
        "same_board_fraction": float(pairs.board_high.eq(pairs.board_low).mean())
            if not pairs.empty else None,
        "by_half": by_half,
    }
    audit["outcome_gate_passed"] = bool(
        len(by_half) == 4
        and all(item["pairs"] >= 50 and item["days"] >= 15 for item in by_half)
        and audit["match_fraction"] >= .35
        and audit["median_composition_gap"] is not None
        and audit["median_composition_gap"] >= .002
        and audit["history_strictly_before_signal"]
        and audit["same_board_fraction"] == 1.0
    )
    return signals, audit


def freeze(prefix_dir: Path = ROOT / "minute_prefix_1449",
           snapshot_dir: Path = ROOT / "market_snapshots_ci",
           daily_dir: Path = Path("data/baostock/market_2020_2026/daily"),
           output_dir: Path = OUTPUT) -> dict:
    prefix_audit = json.loads((prefix_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if not prefix_audit["input_gate_passed"]:
        raise ValueError("14:49 prefix input gate failed")
    connection = duckdb.connect()
    try:
        connection.execute("SET threads = 4")
        connection.execute("SET memory_limit = '8GB'")
        connection.from_parquet(str(prefix_dir / "*" / "part_*.parquet")
                                ).create_view("prefix")
        connection.from_parquet(str(snapshot_dir / "*.parquet")
                                ).create_view("snapshots")
        connection.from_parquet(str(daily_dir / "*.parquet")
                                ).create_view("daily_raw")
        build_history(connection)
        build_candidates(connection)
        selected, audit = select_inputs(connection)
        output_dir.mkdir(parents=True, exist_ok=True)
        for stale in ("repricing_signals.parquet", "repriced.parquet", "report.json"):
            (output_dir / stale).unlink(missing_ok=True)
        connection.execute("SELECT * FROM candidates").df().to_parquet(
            output_dir / "all_candidates.parquet", index=False,
            compression="zstd")
        selected.to_parquet(output_dir / "selections.parquet", index=False,
                            compression="zstd")
        if audit["outcome_gate_passed"]:
            connection.register("selected_keys", selected[["date", "code"]])
            repricing = connection.execute("""
                SELECT s.date, s.code, s.isST, s.reference_gap,
                       s.listing_age_sessions, p.quote_outside_traded_range,
                       p.price_1449
                FROM selected_keys k JOIN snapshots s USING (date, code)
                JOIN prefix p USING (date, code)
            """).df()
            if (len(repricing) != len(selected)
                    or repricing.duplicated(["date", "code"]).any()):
                raise ValueError("A selected stock lacks one repricing input")
            repricing.to_parquet(output_dir / "repricing_signals.parquet",
                                 index=False, compression="zstd")
        (output_dir / "input_audit.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return audit
    finally:
        connection.close()


if __name__ == "__main__":
    print(json.dumps(freeze(), ensure_ascii=False, indent=2))
