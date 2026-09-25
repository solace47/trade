"""Freeze 14:50 minute return autocorrelation pairs before outcome reads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .study_periods import DEVELOPMENT_YEAR, VALIDATION_YEAR


ROOT = Path("data/research")
CAPACITY = 5
COOLDOWN = 5
SALT = "minute-acf-v1"


def build_year(minute_paths: list[str], year: int, output: Path,
               threads: int = 4) -> None:
    """Read the 31 labels from 14:20 through 14:50 under the bar-end assumption."""
    if year not in (DEVELOPMENT_YEAR, VALIDATION_YEAR):
        raise ValueError("Minute autocorrelation extraction is restricted to 2024-2025")
    if not minute_paths or threads < 1:
        raise ValueError("Minute sources and a positive thread count are required")
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()
    connection.execute(f"SET threads = {threads}")
    connection.execute("SET memory_limit = '8GB'")
    temp_literal = str(output.parent / "duckdb_tmp").replace("'", "''")
    connection.execute(f"SET temp_directory = '{temp_literal}'")
    connection.from_parquet(minute_paths).create_view("source_minutes")
    path_literal = str(output).replace("'", "''")
    connection.execute(f"""
        COPY (
            WITH selected AS (
                SELECT lower(exchange) || '.' || symbol AS code,
                       strftime(timestamp, '%Y-%m-%d') AS date,
                       strftime(timestamp, '%H%M') AS label,
                       timestamp, close
                FROM source_minutes
                WHERE timestamp >= TIMESTAMP '{year}-01-01'
                  AND timestamp < TIMESTAMP '{year + 1}-01-01'
                  AND strftime(timestamp, '%H%M') BETWEEN '1420' AND '1450'
            ), lagged AS (
                SELECT *, LAG(close) OVER (
                    PARTITION BY date, code ORDER BY timestamp
                ) AS previous_close
                FROM selected
            ), returns AS (
                SELECT *, CASE WHEN close > 0 AND previous_close > 0
                               THEN LN(close / previous_close) END AS minute_return
                FROM lagged
            ), centered AS (
                SELECT *, AVG(minute_return) OVER (
                    PARTITION BY date, code
                ) AS average_return,
                LAG(minute_return) OVER (
                    PARTITION BY date, code ORDER BY timestamp
                ) AS prior_return
                FROM returns
            ), aggregated AS (
                SELECT date, code, COUNT(*) AS bars,
                       COUNT(DISTINCT label) AS distinct_labels,
                       COUNT(minute_return) AS valid_returns,
                       COUNT(*) FILTER (WHERE minute_return <> 0)
                           AS nonzero_returns,
                       MAX(close) FILTER (WHERE label = '1450') AS price_1450,
                       MAX(minute_return) FILTER (WHERE label = '1450')
                           AS last_minute_log_return,
                       SUM(POWER(minute_return, 2)) AS realized_variance_last30,
                       SUM((minute_return-average_return)
                           * (prior_return-average_return))
                           AS lag_one_covariance_sum,
                       SUM(POWER(minute_return-average_return, 2))
                           AS centered_square_sum
                FROM centered GROUP BY date, code
            )
            SELECT date, code, price_1450, last_minute_log_return,
                   realized_variance_last30, nonzero_returns,
                   lag_one_covariance_sum / centered_square_sum AS rho1
            FROM aggregated
            WHERE bars = 31 AND distinct_labels = 31
              AND valid_returns = 30 AND nonzero_returns >= 10
              AND price_1450 > 0 AND centered_square_sum > 0
        ) TO '{path_literal}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """)
    connection.close()


def _half(date: str) -> str:
    return date[:4] + ("H1" if int(date[5:7]) <= 6 else "H2")


def _rank_key(date: str, code: str) -> str:
    return hashlib.sha256(f"{SALT}|{date}|{code}".encode()).hexdigest()


def _daily_groups(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    strong, weak = [], []
    eligible_days = 0
    for date, group in frame.groupby("date", sort=True):
        if len(group) < 40:
            continue
        eligible_days += 1
        ranked = group.sort_values(["rho1", "code"])
        group_size = int(np.floor(len(ranked) * .2))
        low = ranked.head(group_size).copy()
        low["rank_key"] = low.code.map(lambda code: _rank_key(date, code))
        strong.append(low.sort_values(["rank_key", "code"]))
        weak.append(ranked.tail(group_size).copy().sort_values("code"))
    if not strong:
        empty = frame.iloc[:0].copy()
        return empty, empty, eligible_days
    return (pd.concat(strong, ignore_index=True),
            pd.concat(weak, ignore_index=True), eligible_days)


def _match(row, available: pd.DataFrame,
           reach: dict[str, int]) -> pd.Series | None:
    if available.empty:
        return None
    reach["pool"] += 1
    prior = (available.return20_prior_adjusted
             - row.return20_prior_adjusted).abs()
    current = (available.return_1450 - row.return_1450).abs()
    tail = (available.return_last30 - row.return_last30).abs()
    price = available.price_1450 / row.price_1450
    amount = available.amount_1450 / row.amount_1450
    variance = available.realized_variance_last30 / row.realized_variance_last30
    valid = pd.Series(True, index=available.index)
    for name, condition in (
        ("prior", prior.le(.03)), ("current", current.le(.005)),
        ("tail", tail.le(.0015)), ("price", price.between(.5, 2)),
        ("amount", amount.between(.5, 2)),
        ("variance", variance.between(.5, 2)),
    ):
        valid &= condition
        reach[name] += int(valid.any())
    matches = available.loc[valid].copy()
    if matches.empty:
        return None
    matches["distance"] = (
        prior.loc[matches.index] / .03
        + current.loc[matches.index] / .005
        + tail.loc[matches.index] / .0015
        + np.abs(np.log(price.loc[matches.index])) / np.log(2)
        + np.abs(np.log(amount.loc[matches.index])) / np.log(2)
        + np.abs(np.log(variance.loc[matches.index])) / np.log(2)
    )
    return matches.reset_index(drop=True).sort_values(
        ["distance", "code"]
    ).iloc[0]


def select_inputs(connection: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict]:
    """Make same-day pairs using pre-14:50 inputs, with five-session cooling."""
    frame = connection.execute("""
        SELECT s.date, s.code, s.price_1450, s.amount_1450,
               s.return20_prior_adjusted, s.return_1450,
               i.return_last30, a.last_minute_log_return,
               a.realized_variance_last30, a.rho1
        FROM snapshots s JOIN intraday i USING (date, code)
        JOIN autocovariance a USING (date, code)
        JOIN variance v USING (date, code)
        WHERE ((s.date BETWEEN '2024-01-01' AND '2024-12-17')
            OR (s.date BETWEEN '2025-01-01' AND '2025-12-17'))
          AND (s.code LIKE 'sh.60%' OR s.code LIKE 'sz.00%')
          AND s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.price_1450 >= 5
          AND s.amount_1450 BETWEEN 100000000 AND 1000000000
          AND s.return20_prior_adjusted BETWEEN -.10 AND .10
          AND s.return_1450 BETWEEN -.03 AND .03
          AND i.return_last30 BETWEEN -.008 AND -.002
          AND a.last_minute_log_return < 0
          AND a.realized_variance_last30 > 0
          AND ABS(s.price_1450-i.price_1450) <= .005
          AND ABS(s.price_1450-a.price_1450) <= .005
          AND ABS(s.price_1450-v.price_1450) <= .005
          AND ABS(a.realized_variance_last30
                  - v.realized_variance_last30) <= 1e-10
    """).df()
    if frame.duplicated(["date", "code"]).any() or not np.isfinite(
        frame[["rho1", "realized_variance_last30"]].to_numpy()
    ).all():
        raise ValueError("Invalid minute-autocorrelation input universe")
    strong, weak, eligible_days = _daily_groups(frame)
    calendar = connection.execute("""
        SELECT DISTINCT date FROM snapshots
        WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
        ORDER BY date
    """).df().date.tolist()
    date_index = {date: index for index, date in enumerate(calendar)}
    weak_by_date = {date: rows.set_index("code", drop=False)
                    for date, rows in weak.groupby("date")}
    last_used: dict[str, int] = {}
    selections: list[dict] = []
    attempts = {half: 0 for half in ("2024H1", "2024H2", "2025H1", "2025H2")}
    stage_names = ("pool", "prior", "current", "tail", "price", "amount",
                   "variance")
    stage_reach = {half: {name: 0 for name in stage_names}
                   for half in attempts}
    for date, ranked in strong.groupby("date", sort=True):
        market_index = date_index[date]
        available = weak_by_date[date].copy()
        attempted_today = 0
        for row in ranked.itertuples(index=False):
            if attempted_today >= CAPACITY:
                break
            if market_index - last_used.get(row.code, -1000) <= COOLDOWN:
                continue
            attempted_today += 1
            attempts[_half(date)] += 1
            fresh = available.loc[available.code.map(
                lambda code: market_index - last_used.get(code, -1000)
                > COOLDOWN
            )]
            matched = _match(row, fresh, stage_reach[_half(date)])
            if matched is None:
                continue
            strong_row = row._asdict()
            strong_row.pop("rank_key")
            strong_row.update(candidate="strong_negative", pair_id=row.code,
                              daily_rank=attempted_today)
            weak_row = matched.drop(labels="distance").to_dict()
            weak_row.update(candidate="weak_negative", pair_id=row.code,
                            daily_rank=attempted_today)
            selections.extend((strong_row, weak_row))
            available = available.drop(index=matched.code)
            last_used[row.code] = market_index
            last_used[matched.code] = market_index
    signals = pd.DataFrame(selections)
    if not signals.empty and (
        signals.duplicated(["date", "code"]).any()
        or signals.groupby(["date", "pair_id"]).candidate.nunique().ne(2).any()
    ):
        raise ValueError("Invalid minute-autocorrelation pairs")
    by_half = []
    for half in attempts:
        sub = signals.loc[
            signals.candidate.eq("strong_negative")
            & signals.date.map(_half).eq(half)
        ] if not signals.empty else signals
        paired = len(sub)
        controls = signals.loc[
            signals.candidate.eq("weak_negative")
            & signals.date.map(_half).eq(half)
        ] if not signals.empty else signals
        by_half.append({
            "half": half, "pairs": paired,
            "days": int(sub.date.nunique()) if paired else 0,
            "attempts": attempts[half],
            "match_fraction": paired / attempts[half] if attempts[half] else 0.0,
            "median_rho_gap": float(controls.rho1.median() - sub.rho1.median())
                if paired else None,
            "median_last_minute_gap": float(
                controls.last_minute_log_return.median()
                - sub.last_minute_log_return.median()
            ) if paired else None,
        })
    gate = all(
        row["pairs"] >= 50 and row["days"] >= 30
        and row["match_fraction"] >= .35
        and row["median_rho_gap"] >= .20
        for row in by_half
    )
    audit = {
        "input_only": True, "base_pool": len(frame),
        "eligible_days": eligible_days,
        "strong_pool": len(strong), "weak_pool": len(weak),
        "pairs": len(signals) // 2, "by_half": by_half,
        "stage_reach": stage_reach,
        "outcome_gate_passed": gate,
    }
    return signals, audit


def freeze(output_dir: Path = ROOT / "minute_autocovariance") -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    for name, path in (
        ("snapshots", "market_snapshots_ci"),
        ("intraday", "intraday_features"),
        ("autocovariance", "minute_autocovariance"),
        ("variance", "late_variance"),
    ):
        pattern = "20??.parquet" if name == "autocovariance" else "*.parquet"
        connection.read_parquet(str(ROOT / path / pattern)
                                ).create_view(name)
    signals, audit = select_inputs(connection)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in ("repricing_signals.parquet", "repriced.parquet", "report.json"):
        (output_dir / stale).unlink(missing_ok=True)
    if not signals.empty:
        signals.to_parquet(output_dir / "selections.parquet", index=False,
                           compression="zstd")
    if audit["outcome_gate_passed"]:
        connection.register("selected", signals)
        repricing = connection.execute("""
            SELECT s.* FROM selected r JOIN snapshots s USING (date, code)
        """).df()
        if len(repricing) != len(signals) or repricing.duplicated(
            ["date", "code"]
        ).any():
            raise ValueError("Repricing inputs lack one-to-one snapshot coverage")
        repricing.to_parquet(output_dir / "repricing_signals.parquet",
                             index=False, compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minute-root", type=Path,
                        default=Path("data/hf/pilot/data/stock_1m"))
    parser.add_argument("--output", type=Path,
                        default=ROOT / "minute_autocovariance")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    paths = [str(args.minute_root / exchange / "*.parquet")
             for exchange in ("SH", "SZ")]
    for year in (DEVELOPMENT_YEAR, VALIDATION_YEAR):
        output = args.output / f"{year}.parquet"
        build_year(paths, year, output, args.threads)
        print(f"Wrote {output}", flush=True)
    print(freeze(args.output), flush=True)


if __name__ == "__main__":
    main()
