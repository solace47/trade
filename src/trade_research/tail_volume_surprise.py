"""Evaluate own-stock seasonal tail-volume surprise without future inputs.

The 2024-2025 selector uses only bars completed by 14:50 and prior valid
sessions. Outcome joins happen strictly after selection is frozen to disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .annual_cash_direct_eval import _month_bootstrap
from .market_study import _quality_keys, _quality_symbols, _week_bootstrap
from .strategy_scan import _stressed_returns


ROOT = Path("data/research")
HORIZONS = (1, 5)
CAPACITY = 5
COOLDOWN = 5
BASE = """
    s.isST = 0 AND s.listing_age_sessions >= 20
    AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
    AND s.price_1450 >= 5
    AND s.amount_1450 BETWEEN 100000000 AND 1000000000
    AND s.return20_prior_adjusted BETWEEN -.10 AND .10
    AND s.return_1450 BETWEEN 0 AND .03
    AND s.position_1450 >= .7
    AND i.return_last30 BETWEEN 0 AND .003
    AND abs(s.price_1450 - i.price_1450) <= .005
"""


def _daily_cash(rows: pd.DataFrame, dates: list[str],
                column: str) -> pd.Series:
    counts = rows.groupby("date").size().reindex(dates)
    if counts.isna().any():
        raise ValueError("Missing same-day pair member")
    valid = rows.loc[rows.exit_status.eq("filled")
                     & rows.quality_clean_exit]
    return valid.groupby("date")[column].sum().reindex(
        dates, fill_value=0
    ) / counts


def select_inputs(connection: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, dict]:
    """Build lagged seasonal features and pair extreme groups without outcomes."""
    connection.execute("""
        CREATE TEMP TABLE historical AS
        SELECT date, code, price_1450, return_last30,
               volume_share_last30,
               MEDIAN(volume_share_last30) OVER (
                   PARTITION BY code ORDER BY date
                   ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
               ) AS prior_tail_share,
               LAG(date, 20) OVER (
                   PARTITION BY code ORDER BY date
               ) AS oldest_prior_date
        FROM intraday
        WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
    """)
    connection.execute(f"""
        CREATE TEMP TABLE candidates AS
        SELECT s.date, s.code, s.return_1450, s.position_1450,
               s.return20_prior_adjusted, s.amount_1450,
               i.return_last30, i.volume_share_last30,
               i.prior_tail_share,
               i.volume_share_last30 / i.prior_tail_share AS surprise
        FROM snapshots s JOIN historical i USING (date, code)
        WHERE ((s.date BETWEEN '2024-01-01' AND '2024-12-17')
            OR (s.date BETWEEN '2025-01-01' AND '2025-12-17'))
          AND {BASE}
          AND i.oldest_prior_date IS NOT NULL
          AND date_diff('day', CAST(i.oldest_prior_date AS DATE),
                              CAST(s.date AS DATE)) <= 45
          AND i.prior_tail_share >= .04
          AND i.volume_share_last30 > 0
    """)
    high = connection.execute("""
        SELECT * FROM candidates WHERE surprise >= 1.5
        ORDER BY date, surprise DESC, code
    """).df()
    low = connection.execute("""
        SELECT * FROM candidates WHERE surprise <= 1.0
        ORDER BY date, code
    """).df()
    calendars = connection.execute("""
        SELECT DISTINCT date FROM snapshots
        WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
        ORDER BY date
    """).df().date.tolist()
    date_index = {date: index for index, date in enumerate(calendars)}
    low_by_date = {date: rows.set_index("code", drop=False)
                   for date, rows in low.groupby("date")}
    last_high: dict[str, int] = {}
    last_low: dict[str, int] = {}
    selected = []
    capacity_candidates = 0
    for date, ranked in high.groupby("date", sort=True):
        if date not in low_by_date:
            continue
        market_index = date_index[date]
        available = low_by_date[date].copy()
        daily_picks = 0
        for row in ranked.itertuples(index=False):
            if daily_picks >= CAPACITY:
                break
            if market_index - last_high.get(row.code, -1000) <= COOLDOWN:
                continue
            capacity_candidates += 1
            fresh = available.loc[
                available.code.map(
                    lambda code: market_index - last_low.get(code, -1000)
                    > COOLDOWN
                )
            ]
            prior_gap = (fresh.return20_prior_adjusted
                         - row.return20_prior_adjusted).abs()
            current_gap = (fresh.return_1450 - row.return_1450).abs()
            amount_ratio = fresh.amount_1450 / row.amount_1450
            position_gap = (fresh.position_1450 - row.position_1450).abs()
            matches = fresh.loc[
                prior_gap.le(.03) & current_gap.le(.005)
                & amount_ratio.between(.5, 2) & position_gap.le(.15)
            ].copy()
            if matches.empty:
                continue
            matches["distance"] = (
                prior_gap.loc[matches.index] / .03
                + current_gap.loc[matches.index] / .005
                + np.abs(np.log(amount_ratio.loc[matches.index])) / np.log(2)
                + position_gap.loc[matches.index] / .15
            )
            matched = matches.reset_index(drop=True).sort_values(
                ["distance", "code"]
            ).iloc[0]
            high_row = row._asdict()
            high_row.update(candidate="high_surprise", pair_id=row.code,
                            daily_rank=daily_picks + 1)
            low_row = matched.drop(labels="distance").to_dict()
            low_row.update(candidate="low_surprise_control",
                           pair_id=row.code, daily_rank=daily_picks + 1)
            selected.extend((high_row, low_row))
            available = available.drop(index=matched.code)
            last_high[row.code] = market_index
            last_low[matched.code] = market_index
            daily_picks += 1
    signals = pd.DataFrame(selected)
    if (signals.empty or signals.duplicated(["candidate", "date", "code"]).any()
            or signals.groupby(["date", "pair_id"]).candidate.nunique().ne(2).any()
            or not signals.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Invalid seasonal tail-volume selection")
    pairs = signals.loc[signals.candidate.eq("high_surprise")].merge(
        signals.loc[signals.candidate.eq("low_surprise_control")],
        on=["date", "pair_id"], suffixes=("_high", "_low"),
        validate="one_to_one",
    )
    audit = {
        "high_pool": len(high), "low_pool": len(low),
        "capacity_candidates": capacity_candidates,
        "paired": len(pairs), "days": signals.date.nunique(),
        "prior20_median_abs_gap": float((
            pairs.return20_prior_adjusted_high
            - pairs.return20_prior_adjusted_low).abs().median()),
        "current_median_abs_gap": float((
            pairs.return_1450_high - pairs.return_1450_low).abs().median()),
        "by_year": signals.loc[signals.candidate.eq("high_surprise")]
            .groupby(signals.date.str[:4]).agg(
                signals=("code", "size"), days=("date", "nunique")
            ).reset_index().to_dict("records"),
    }
    return signals, audit


def _summarize(frame: pd.DataFrame, reference: pd.DataFrame) -> dict:
    dates = sorted(frame.date.unique())
    control = reference.loc[reference.date.isin(dates)].copy()
    if not dates or control.date.nunique() != len(dates):
        raise ValueError("Missing same-day low-surprise control")
    for rows in (frame, control):
        valid = rows.exit_status.eq("filled") & rows.quality_clean_exit
        rows.loc[:, "stress10"] = 0.0
        if valid.any():
            rows.loc[valid, "stress10"] = _stressed_returns(rows.loc[valid], 10)
    cash = _daily_cash(frame, dates, "net_return")
    other = _daily_cash(control, dates, "net_return")
    stress = _daily_cash(frame, dates, "stress10")
    other_stress = _daily_cash(control, dates, "stress10")
    edge = cash - other
    edge_stress = stress - other_stress
    date_series = pd.Series(dates)
    return {
        "signals": len(frame), "days": len(dates),
        "entry_fills": int(frame.entry_status.eq("filled").sum()),
        "clean_exits": int((frame.exit_status.eq("filled")
                            & frame.quality_clean_exit).sum()),
        "cash_mean": float(cash.mean()),
        "control_cash_mean": float(other.mean()),
        "edge_mean": float(edge.mean()),
        "stress10_cash_mean": float(stress.mean()),
        "stress10_edge_mean": float(edge_stress.mean()),
        "cash_week_ci": _week_bootstrap(cash, date_series, 562),
        "edge_week_ci": _week_bootstrap(edge, date_series, 563),
        "edge_month_ci": _month_bootstrap(edge, date_series, 564),
    }


def study(output_dir: Path = ROOT / "tail_volume_surprise") -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    connection.read_parquet(str(ROOT / "intraday_features" / "*.parquet")
                            ).create_view("intraday")
    signals, audit = select_inputs(connection)
    output_dir.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(output_dir / "selections.parquet", index=False,
                       compression="zstd")
    connection.register("selected", signals)
    repricing_signals = connection.execute("""
        SELECT s.* FROM selected r JOIN snapshots s USING (date, code)
    """).df()
    if (len(repricing_signals) != len(signals)
            or repricing_signals.duplicated(["date", "code"]).any()):
        raise ValueError("Repricing inputs lack one-to-one snapshot coverage")
    repricing_signals.to_parquet(
        output_dir / "repricing_signals.parquet", index=False,
        compression="zstd",
    )
    connection.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                            ).create_view("outcomes")
    connection.register("bad_days", _quality_keys(ROOT / "market_issues_ci"))
    connection.register("bad_symbols", _quality_symbols(ROOT / "market_issues_ci"))
    trades = connection.execute("""
        SELECT r.*, o.horizon, o.entry_status, o.entry_price, o.shares,
               o.exit_status, o.exit_date, o.exit_delay_sessions,
               o.exit_price, o.net_return,
               o.exit_status = 'filled'
               AND NOT EXISTS (SELECT 1 FROM bad_symbols b
                               WHERE b.code = r.code)
               AND NOT EXISTS (
                   SELECT 1 FROM bad_days q
                   WHERE q.code = r.code AND q.date >= r.date
                     AND q.date <= o.exit_date
               ) AS quality_clean_exit
        FROM selected r JOIN outcomes o USING (date, code)
        WHERE o.horizon IN (1, 5)
    """).df()
    if len(trades) != len(signals) * len(HORIZONS):
        raise ValueError("Missing archived outcome for a selected stock-day")
    trades.to_parquet(output_dir / "trades.parquet", index=False,
                      compression="zstd")
    report = {"input_audit": audit, "results": [],
              "note": "2025 is not blind; 2026 outcomes not used"}
    for year in ("2024", "2025"):
        for half, month_rule in (
            ("H1", lambda month: month <= 6),
            ("H2", lambda month: month > 6),
            ("full", lambda month: month > 0),
        ):
            for horizon in HORIZONS:
                rows = trades.loc[
                    trades.candidate.eq("high_surprise")
                    & trades.date.str.startswith(year)
                    & month_rule(trades.date.str[5:7].astype(int))
                    & trades.horizon.eq(horizon)
                ].copy()
                if rows.empty:
                    continue
                control = trades.loc[
                    trades.candidate.eq("low_surprise_control")
                    & trades.horizon.eq(horizon)
                ].copy()
                report["results"].append({"year": year, "half": half,
                                          "horizon": horizon,
                                          **_summarize(rows, control)})
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "tail_volume_surprise")
    args = parser.parse_args()
    report = study(args.output)
    print({"paired": report["input_audit"]["paired"],
           "output": str(args.output / "report.json")})


if __name__ == "__main__":
    main()
