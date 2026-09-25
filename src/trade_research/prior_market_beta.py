"""Freeze 14:49 low-versus-high historical market-beta pairs before outcomes.

The beta window ends on the latest valid stock day strictly before the signal.
Only historical daily bars, the 14:49 minute prefix, and known daily metadata
enter the selection. The 2025 period is exploratory, and 2026 is untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .late_to_open_reversal import _board


ROOT = Path("data/research")
OUTPUT = ROOT / "prior_market_beta"
CAPACITY = 5
COOLDOWN = 5


def build_beta_history(connection: duckdb.DuckDBPyConnection) -> None:
    """Build 60-observation rolling betas from daily bars in ``daily_raw``."""
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE valid_daily AS
        SELECT date, code, pctChg / 100.0 AS stock_return
        FROM daily_raw
        WHERE date BETWEEN '2023-01-01' AND '2025-12-17'
          AND tradestatus = 1 AND isST = 0
          AND pctChg IS NOT NULL AND isfinite(pctChg)
          AND ABS(pctChg) <= 21
    """)
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE market_proxy AS
        SELECT date, MEDIAN(stock_return) AS market_return,
               COUNT(*) AS market_stocks
        FROM valid_daily GROUP BY date
    """)
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE beta_history AS
        SELECT d.date AS beta_date, d.code,
               COUNT(*) OVER beta_window AS prior_count,
               COVAR_SAMP(d.stock_return, m.market_return) OVER beta_window
                   / NULLIF(VAR_SAMP(m.market_return) OVER beta_window, 0)
                   AS market_beta,
               VAR_SAMP(m.market_return) OVER beta_window AS market_variance
        FROM valid_daily d JOIN market_proxy m USING (date)
        WINDOW beta_window AS (
            PARTITION BY d.code ORDER BY d.date
            ROWS BETWEEN 59 PRECEDING AND CURRENT ROW
        )
    """)


def build_candidates(connection: duckdb.DuckDBPyConnection) -> None:
    """ASOF joins guarantee that current-day daily returns cannot enter beta."""
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE eligible_base AS
        SELECT p.date, p.code, p.price_1449, p.amount_1449,
               p.return_last29,
               p.price_1449 / s.preclose - 1 AS return_1449,
               (p.price_1449 - p.low_1449)
                   / NULLIF(p.high_1449 - p.low_1449, 0) AS position_1449,
               s.return20_prior_adjusted, s.isST, s.reference_gap,
               s.listing_age_sessions
        FROM prefix p JOIN snapshots s USING (date, code)
        WHERE ((p.date BETWEEN '2024-01-01' AND '2024-12-17')
            OR (p.date BETWEEN '2025-01-01' AND '2025-12-17'))
          AND s.isST = 0 AND s.listing_age_sessions >= 60
          AND NOT s.reference_gap AND NOT p.quote_outside_traded_range
          AND s.preclose > 0 AND p.price_1449 >= 5
          AND p.amount_1449 BETWEEN 100000000 AND 1000000000
          AND s.return20_prior_adjusted BETWEEN -.10 AND .10
          AND p.return_last29 BETWEEN -.01 AND .01
          AND (p.price_1449 - p.low_1449)
              / NULLIF(p.high_1449 - p.low_1449, 0) BETWEEN 0 AND 1
          AND p.price_1449 / s.preclose - 1 BETWEEN -.03 AND .03
    """)
    connection.execute("""
        CREATE OR REPLACE TEMP TABLE candidates AS
        SELECT e.date, e.code, e.price_1449, e.amount_1449,
               e.return_last29, e.return_1449, e.position_1449,
               e.return20_prior_adjusted, b.beta_date,
               b.prior_count, b.market_beta, b.market_variance
        FROM eligible_base e ASOF LEFT JOIN beta_history b
          ON e.code = b.code AND e.date > b.beta_date
        WHERE b.prior_count >= 50 AND b.market_variance > 0
          AND b.market_beta IS NOT NULL AND isfinite(b.market_beta)
    """)


def select_inputs(connection: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict]:
    low = connection.execute("""
        SELECT * FROM candidates WHERE market_beta <= .75
        ORDER BY date, md5('prior-beta-v1' || date || code), code
    """).df()
    high = connection.execute("""
        SELECT * FROM candidates WHERE market_beta >= 1.25
        ORDER BY date, code
    """).df()
    low["board"] = low.code.map(_board)
    high["board"] = high.code.map(_board)
    low = low.loc[low.board.notna()].copy()
    high = high.loc[high.board.notna()].copy()
    calendar = connection.execute("""
        SELECT DISTINCT date FROM prefix
        WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
        ORDER BY date
    """).df().date.tolist()
    date_index = {date: offset for offset, date in enumerate(calendar)}
    high_by_date = {date: rows.set_index("code", drop=False)
                    for date, rows in high.groupby("date", sort=False)}
    last_selected: dict[str, int] = {}
    selected: list[dict] = []
    attempted = 0
    for date, ranked in low.groupby("date", sort=True):
        index = date_index[date]
        available = high_by_date.get(date)
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
            position_gap = (fresh.position_1449 - row.position_1449).abs()
            price_ratio = fresh.price_1449 / row.price_1449
            amount_ratio = fresh.amount_1449 / row.amount_1449
            matches = fresh.loc[
                prior_gap.le(.03) & current_gap.le(.005)
                & tail_gap.le(.003) & position_gap.le(.30)
                & price_ratio.between(.5, 2) & amount_ratio.between(.5, 2)
            ].copy()
            if matches.empty:
                continue
            matches["distance"] = (
                prior_gap.loc[matches.index] / .03
                + current_gap.loc[matches.index] / .005
                + tail_gap.loc[matches.index] / .003
                + position_gap.loc[matches.index] / .30
                + np.abs(np.log(price_ratio.loc[matches.index])) / np.log(2)
                + np.abs(np.log(amount_ratio.loc[matches.index])) / np.log(2)
            )
            control = matches.reset_index(drop=True).sort_values(
                ["distance", "code"]
            ).iloc[0]
            treatment = row._asdict()
            treatment.update(candidate="low_beta", pair_id=row.code,
                             daily_rank=daily_attempts)
            comparator = control.drop(labels="distance").to_dict()
            comparator.update(candidate="high_beta_control",
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
        raise ValueError("Invalid beta input pairs")
    pairs = (signals.loc[signals.candidate.eq("low_beta")].merge(
        signals.loc[signals.candidate.eq("high_beta_control")],
        on=["date", "pair_id"], suffixes=("_low", "_high"),
        validate="one_to_one",
    ) if not signals.empty else pd.DataFrame())
    halves = (signals.loc[signals.candidate.eq("low_beta")]
              .assign(half=lambda rows: rows.date.str[:4] + "H"
                      + np.where(rows.date.str[5:7].astype(int) <= 6, "1", "2"))
              if not signals.empty else pd.DataFrame())
    by_half = (halves.groupby("half").agg(
        pairs=("code", "size"), days=("date", "nunique")
    ).reset_index().to_dict("records") if not halves.empty else [])
    beta_gap = (pairs.market_beta_high - pairs.market_beta_low
                if not pairs.empty else pd.Series(dtype=float))
    audit = {
        "source_cutoff_label": "14:49",
        "beta_window": "previous 60 valid stock sessions; 50 minimum",
        "beta_latest_date_strictly_before_signal": (
            bool(signals.beta_date.lt(signals.date).all())
            if not signals.empty else True
        ),
        "eligible_with_beta": int(connection.execute(
            "SELECT COUNT(*) FROM candidates").fetchone()[0]),
        "low_pool": len(low), "high_pool": len(high),
        "capacity_candidates": attempted, "paired": len(pairs),
        "match_fraction": len(pairs) / attempted if attempted else 0.0,
        "median_beta_gap": float(beta_gap.median()) if not pairs.empty else None,
        "median_prior_count": float(signals.prior_count.median())
            if not signals.empty else None,
        "median_prior20_gap": float((
            pairs.return20_prior_adjusted_low
            - pairs.return20_prior_adjusted_high).abs().median())
            if not pairs.empty else None,
        "median_current_gap": float((pairs.return_1449_low
                                      - pairs.return_1449_high).abs().median())
            if not pairs.empty else None,
        "median_tail_gap": float((pairs.return_last29_low
                                   - pairs.return_last29_high).abs().median())
            if not pairs.empty else None,
        "same_board_fraction": float(pairs.board_low.eq(pairs.board_high).mean())
            if not pairs.empty else None,
        "by_half": by_half,
    }
    audit["outcome_gate_passed"] = bool(
        len(by_half) == 4
        and all(item["pairs"] >= 50 and item["days"] >= 15 for item in by_half)
        and audit["match_fraction"] >= .35
        and audit["median_beta_gap"] is not None
        and audit["median_beta_gap"] >= .35
        and audit["beta_latest_date_strictly_before_signal"]
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
        build_beta_history(connection)
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
