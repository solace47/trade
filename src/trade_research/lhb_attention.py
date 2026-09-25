"""Frozen test: public trading-list attention as a 14:50 risk filter.

The before-outcomes plan remains in Git history. Publication on day t is first
usable on the next trading day. 2025 is exploratory, not a blind holdout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .exchange_public_events import load_complete_notices, trading_dates
from .market_study import _quality_keys, _quality_symbols
from .residual_liquidity import _period
from .strategy_scan import _last_safe_entry


CAPACITY = 5
MAX_DISTANCE = 6.0
HORIZONS = (1, 2, 5)
TREATED = "listed_limitup"
CONTROL = "same_day_unlisted_limitup"


def _match_distance(choices: pd.DataFrame, signal: pd.Series) -> pd.Series:
    return (
        (choices.open_gap - signal.open_gap).abs() / .02
        + (choices.return_1450 - signal.return_1450).abs() / .03
        + np.abs(np.log(choices.amount_1450 / signal.amount_1450)) / .7
        + (choices.t_turn - signal.t_turn).abs() / 10
        + (choices.return20_prior_adjusted
           - signal.return20_prior_adjusted).abs() / .20
        + np.abs(np.log(choices.t_amount / signal.t_amount)) / .7
    )


def match_controls(treated: pd.DataFrame, pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """Use one unique, same-day unlisted limit-up for each retained event."""
    choices_by_date = {day: group.sort_values("code").reset_index(drop=True)
                       for day, group in pool.loc[pool.any_list.isna()].groupby("date")}
    chosen = []
    controls = []
    unmatched = 0
    for day, signals in treated.groupby("date", sort=True):
        choices = choices_by_date.get(day, pd.DataFrame()).copy()
        for _, signal in signals.iterrows():
            if choices.empty:
                unmatched += 1
                continue
            distance = _match_distance(choices, signal)
            nearest = int(np.argmin(distance.to_numpy()))
            best = float(distance.iloc[nearest])
            if not np.isfinite(best) or best > MAX_DISTANCE:
                unmatched += 1
                continue
            reference = choices.iloc[nearest].copy()
            reference["pair_code"] = signal.code
            reference["match_distance"] = best
            reference["candidate"] = CONTROL
            controls.append(reference)
            item = signal.copy()
            item["pair_code"] = signal.code
            item["match_distance"] = best
            item["candidate"] = TREATED
            chosen.append(item)
            choices = choices.drop(choices.index[nearest]).reset_index(drop=True)
    if not chosen:
        raise ValueError("No listed limit-up found a qualifying same-day match")
    return pd.DataFrame(chosen), pd.DataFrame(controls), unmatched


def select(snapshot_dir: Path, daily_dir: Path, calendar: Path, sse_dir: Path,
           szse_export: Path, output: Path) -> tuple[pd.DataFrame, dict]:
    notices = load_complete_notices(calendar, sse_dir, szse_export)
    flags = notices.groupby(["trade_date", "code"], as_index=False).agg(
        positive_list=("positive_daily", "any"),
        any_list=("notice_type", "size"),
    )
    sessions = trading_dates(calendar, "2024-01-01", "2026-01-31")
    next_days = pd.DataFrame({"trade_date": sessions[:-1], "date": sessions[1:]})
    if not next_days.date.gt(next_days.trade_date).all():
        raise ValueError("Public listing was joined to its own or an earlier session")
    c = duckdb.connect()
    c.execute("SET threads = 4")
    c.read_parquet(str(snapshot_dir / "*.parquet")).create_view("snapshots")
    last_entry = {year: _last_safe_entry(c, f"{year}-01-01", f"{year+1}-01-01")
                  for year in (2024, 2025)}
    c.register("notices", flags)
    c.register("next_days", next_days)
    c.execute(f"""
        CREATE TEMP VIEW daily AS
        SELECT date, code, close, high, preclose, pctChg, turn, amount,
               isST, tradestatus
        FROM read_parquet('{daily_dir}/*.parquet')
        WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
    """)
    pool = c.execute(f"""
        WITH limitup AS (
            SELECT d.date AS trade_date, d.code, d.turn AS t_turn,
                   d.amount AS t_amount, n.positive_list, n.any_list
            FROM daily d LEFT JOIN notices n
              ON n.trade_date = d.date AND n.code = d.code
            WHERE d.tradestatus = 1 AND d.isST = 0 AND d.preclose > 0
              AND d.close >= d.high - .005
              AND (d.code LIKE 'sh.60%' OR d.code LIKE 'sz.00%')
              AND d.pctChg BETWEEN 9.5 AND 10.5
        )
        SELECT s.date, s.code, l.trade_date, l.t_turn, l.t_amount,
               l.positive_list, l.any_list, s.price_1450,
               s.open_1450, s.preclose, s.isST, s.reference_gap,
               s.quote_outside_traded_range, s.listing_age_sessions,
               s.return_1450, s.amount_1450, s.return20_prior_adjusted,
               s.open_1450 / s.preclose - 1 AS open_gap
        FROM limitup l JOIN next_days x USING (trade_date)
        JOIN snapshots s ON s.date = x.date AND s.code = l.code
        WHERE ((s.date BETWEEN '2024-01-01' AND '{last_entry[2024]}')
            OR (s.date BETWEEN '2025-01-01' AND '{last_entry[2025]}'))
          AND s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.preclose > 0 AND s.open_1450 > 0
          AND s.open_1450 BETWEEN s.low_1450 - .005 AND s.high_1450 + .005
          AND s.price_1450 >= 5 AND s.amount_1450 >= 30000000
          AND s.return20_prior_adjusted IS NOT NULL
          AND s.open_1450 / s.preclose - 1 BETWEEN -.05 AND .05
          AND s.return_1450 BETWEEN -.08 AND .08
    """).df()
    if pool.empty or pool.duplicated(["date", "code"]).any():
        raise ValueError("Empty or duplicated limit-up 14:50 stock-day pool")
    treated = (pool.loc[pool.positive_list.fillna(False)]
               .sort_values(["date", "t_turn", "code"],
                            ascending=[True, False, True])
               .groupby("date", sort=False).head(CAPACITY).copy())
    selected_treated, controls, unmatched = match_controls(treated, pool)
    selected = pd.concat([selected_treated, controls], ignore_index=True)
    if len(selected_treated) != len(controls):
        raise ValueError("Listed/unlisted selection is not one-to-one")
    if selected.duplicated(["candidate", "date", "code"]).any():
        raise ValueError("A stock-day was selected twice within a basket")
    if not selected.date.gt(selected.trade_date).all():
        raise ValueError("A signal uses same-day or future exchange disclosure")
    report = {
        "hypothesis": "Prior-day positive-list limit-ups lag same-day unlisted limit-ups after next-day 14:50",
        "primary_horizon": 5, "diagnostic_horizons": [1, 2],
        "last_entry": last_entry,
        "liquid_limitup_pool": len(pool),
        "eligible_positive_list": int(pool.positive_list.fillna(False).sum()),
        "capacity_selected": len(treated), "matched_pairs": len(selected_treated),
        "unmatched_capacity_selected": unmatched,
        "matched_days": int(selected_treated.date.nunique()),
        "match_distance_median": float(controls.match_distance.median()),
        "match_distance_p90": float(controls.match_distance.quantile(.9)),
        "note": "2025 exploratory, not blind; 2026 outcomes untouched",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output, index=False, compression="zstd")
    return selected, report


def evaluate(selected_path: Path, outcome_dir: Path, issues_dir: Path,
             report_path: Path, trades_path: Path) -> dict:
    selected = pd.read_parquet(selected_path)
    if set(selected.candidate) != {TREATED, CONTROL}:
        raise ValueError("Missing a listed or same-day unlisted limit-up group")
    if not selected.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("Listed-stock signal escaped 2024-2025")
    if not selected.date.gt(selected.trade_date).all():
        raise ValueError("Listed-stock signal uses future disclosure")
    pair_counts = selected.groupby(["date", "pair_code"]).candidate.nunique()
    if not pair_counts.eq(2).all() or len(selected) != 2 * len(pair_counts):
        raise ValueError("Listed/unlisted pairs are incomplete")
    c = duckdb.connect()
    c.execute("SET threads = 4")
    c.register("selected", selected)
    c.read_parquet(str(outcome_dir / "*.parquet")).create_view("o")
    c.register("bad_days", _quality_keys(issues_dir))
    c.register("bad_symbols", _quality_symbols(issues_dir))
    trades = c.execute("""
        SELECT r.*, o.horizon, o.entry_status, o.entry_price, o.shares,
               o.exit_status, o.exit_date, o.exit_delay_sessions,
               o.exit_price, o.net_return,
               o.exit_status = 'filled'
               AND NOT EXISTS (SELECT 1 FROM bad_symbols b WHERE b.code = r.code)
               AND NOT EXISTS (
                   SELECT 1 FROM bad_days q
                   WHERE q.code = r.code AND q.date >= r.date
                     AND q.date <= o.exit_date
               ) AS quality_clean_exit
        FROM selected r JOIN o USING (date, code)
        WHERE o.horizon IN (1, 2, 5)
    """).df()
    if len(trades) != len(selected) * len(HORIZONS):
        raise ValueError("A selected stock-day lacks minute-priced outcomes")
    report = {"primary_horizon": 5, "matched_pairs": len(pair_counts),
              "results": {}}
    for year in ("2024", "2025"):
        year_rows = trades.loc[trades.date.str.startswith(year)]
        report["results"][year] = {}
        for horizon in HORIZONS:
            same_horizon = year_rows.loc[year_rows.horizon.eq(horizon)]
            listed = same_horizon.loc[same_horizon.candidate.eq(TREATED)]
            unlisted = same_horizon.loc[same_horizon.candidate.eq(CONTROL)]
            report["results"][year][str(horizon)] = {}
            for period, sample in (
                ("H1", listed.loc[listed.date.str[5:7].astype(int).le(6)]),
                ("H2", listed.loc[listed.date.str[5:7].astype(int).gt(6)]),
                ("full", listed),
            ):
                if not sample.empty:
                    summary = _period(sample, unlisted)
                    summary["same_day_matched_mean"] = summary.pop("same_day_random_mean")
                    report["results"][year][str(horizon)][period] = summary
    report_path.parent.mkdir(parents=True, exist_ok=True)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    trades.to_parquet(trades_path, index=False, compression="zstd")
    return report


def summarize_repriced(selected_path: Path, repriced_path: Path,
                       output: Path) -> dict:
    """Summarize independent raw-minute pricing at two order sizes."""
    membership = pd.read_parquet(selected_path)[["date", "code", "candidate", "pair_code"]]
    repriced = pd.read_parquet(repriced_path)
    required_sizes = {20_000.0, 100_000.0}
    if set(repriced.target_notional) != required_sizes or set(repriced.horizon) != {1, 5}:
        raise ValueError("Missing a frozen repricing order size or horizon")
    if len(repriced) != len(membership) * 4:
        raise ValueError("Incomplete raw-minute repricing")
    rows = repriced.merge(membership, on=["date", "code"], validate="many_to_one")
    result = {"sizes_yuan": [20_000, 100_000], "results": {}}
    for size in sorted(required_sizes):
        result["results"][str(int(size))] = {}
        for year in ("2024", "2025"):
            result["results"][str(int(size))][year] = {}
            for horizon in (1, 5):
                sample = rows.loc[
                    rows.target_notional.eq(size) & rows.horizon.eq(horizon)
                    & rows.date.str.startswith(year)
                ]
                listed = sample.loc[sample.candidate.eq(TREATED)]
                unlisted = sample.loc[sample.candidate.eq(CONTROL)]
                summary = _period(listed, unlisted)
                summary["same_day_matched_mean"] = summary.pop("same_day_random_mean")
                result["results"][str(int(size))][year][str(horizon)] = summary
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return result


def posthoc_balance_sensitivity(selected_path: Path, trades_path: Path,
                                output: Path) -> dict:
    """Check whether the original edge survives tighter observed covariates.

    These filters were chosen after the primary outcome was viewed. They are
    diagnostics of confounding, not an independent strategy test.
    """
    selected = pd.read_parquet(selected_path)
    treated = selected.loc[selected.candidate.eq(TREATED)].set_index(["date", "pair_code"])
    control = selected.loc[selected.candidate.eq(CONTROL)].set_index(["date", "pair_code"])
    if not treated.index.equals(control.index):
        raise ValueError("Paired selections do not align")
    pair = pd.DataFrame(index=treated.index)
    pair["same_exchange"] = treated.code.str[:2].eq(control.code.str[:2])
    pair["turn_gap"] = (treated.t_turn - control.t_turn).abs()
    pair["prior20_gap"] = (
        treated.return20_prior_adjusted - control.return20_prior_adjusted
    ).abs()
    pair["distance"] = treated.match_distance
    masks = {
        "original": pd.Series(True, index=pair.index),
        "same_exchange": pair.same_exchange,
        "close_turn_and_prior20": pair.turn_gap.le(5) & pair.prior20_gap.le(.10),
        "same_exchange_close_calipers": (
            pair.same_exchange & pair.turn_gap.le(5) & pair.prior20_gap.le(.10)
            & pair.distance.le(4)
        ),
    }
    trades = pd.read_parquet(trades_path)
    result = {"post_hoc": True, "primary_horizon": 5, "results": {}}
    for name, mask in masks.items():
        keys = pair.loc[mask].reset_index()[["date", "pair_code"]]
        subset = trades.loc[trades.horizon.eq(5)].merge(
            keys, on=["date", "pair_code"], validate="many_to_one")
        result["results"][name] = {}
        for year in ("2024", "2025"):
            annual = subset.loc[subset.date.str.startswith(year)]
            summary = _period(annual.loc[annual.candidate.eq(TREATED)],
                              annual.loc[annual.candidate.eq(CONTROL)])
            summary["same_day_matched_mean"] = summary.pop("same_day_random_mean")
            result["results"][name][year] = summary
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    return result


def write_reprice_signals(selected: pd.DataFrame, output: Path) -> None:
    """Carry entry-state fields required by raw-minute execution pricing."""
    fields = ["date", "code", "isST", "reference_gap",
              "quote_outside_traded_range", "listing_age_sessions"]
    signals = selected[fields].drop_duplicates(["date", "code"])
    if len(signals) != len(selected):
        raise ValueError("A matched stock-day occurs in more than one basket")
    output.parent.mkdir(parents=True, exist_ok=True)
    signals.to_parquet(output, index=False, compression="zstd")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--daily", type=Path,
                        default=Path("data/baostock/market_2020_2026/daily"))
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--sse", type=Path,
                        default=Path("data/research/lhb/sse_daily"))
    parser.add_argument("--szse", type=Path,
                        default=Path("data/research/lhb/szse_2024_2025.xlsx"))
    parser.add_argument("--selected", type=Path,
                        default=Path("data/research/lhb_selected.parquet"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/lhb_report.json"))
    parser.add_argument("--trades", type=Path,
                        default=Path("data/research/lhb_trades.parquet"))
    parser.add_argument("--repriced", type=Path)
    parser.add_argument("--reprice-signals", type=Path,
                        default=Path("data/research/lhb_reprice_signals.parquet"))
    parser.add_argument("--repriced-report", type=Path,
                        default=Path("data/research/lhb_reprice_report.json"))
    parser.add_argument("--balance-report", type=Path,
                        default=Path("data/research/lhb_balance_sensitivity.json"))
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    selected, selection = select(args.snapshots, args.daily, args.calendar,
                                 args.sse, args.szse, args.selected)
    write_reprice_signals(selected, args.reprice_signals)
    print(json.dumps(selection, ensure_ascii=False, indent=2))
    if not args.select_only:
        report = evaluate(args.selected, args.outcomes, args.issues,
                          args.report, args.trades)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if args.repriced:
            raw = summarize_repriced(args.selected, args.repriced,
                                     args.repriced_report)
            print({"raw_minute_sizes": raw["sizes_yuan"],
                   "raw_minute_report": str(args.repriced_report)})
        sensitivity = posthoc_balance_sensitivity(args.selected, args.trades,
                                                  args.balance_report)
        print({"post_hoc_checks": list(sensitivity["results"]),
               "balance_report": str(args.balance_report)})


if __name__ == "__main__":
    main()
