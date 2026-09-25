"""Attribute frozen late-variance risk pairs to 14:50 market volatility."""

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
OUTPUT = ROOT / "market_variance_state"
PAIRS = ROOT / "late_variance_risk" / "selections.parquet"


def _half(date: str) -> str:
    return date[:4] + ("H1" if int(date[5:7]) <= 6 else "H2")


def build_state(output_dir: Path = OUTPUT) -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(ROOT / "late_variance" / "*.parquet")
                            ).create_view("variance")
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    state = connection.execute("""
        WITH history AS (
            SELECT date, code, realized_variance_last30 AS rv,
                   MEDIAN(realized_variance_last30) OVER (
                       PARTITION BY code ORDER BY date
                       ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING
                   ) AS prior_rv,
                   LAG(date, 20) OVER (
                       PARTITION BY code ORDER BY date
                   ) AS oldest_prior_date
            FROM variance
            WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
        ), market AS (
            SELECT s.date, COUNT(*) AS stocks,
                   MEDIAN(h.rv / h.prior_rv) AS market_surprise
            FROM snapshots s JOIN history h USING (date, code)
            WHERE ((s.date BETWEEN '2024-01-01' AND '2024-12-17')
                OR (s.date BETWEEN '2025-01-01' AND '2025-12-17'))
              AND s.isST = 0 AND s.listing_age_sessions >= 20
              AND s.price_1450 >= 5
              AND s.amount_1450 BETWEEN 100000000 AND 1000000000
              AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
              AND h.prior_rv > 0 AND h.oldest_prior_date IS NOT NULL
              AND date_diff('day', CAST(h.oldest_prior_date AS DATE),
                                  CAST(s.date AS DATE)) <= 45
            GROUP BY s.date HAVING COUNT(*) >= 200
        )
        SELECT date, stocks, market_surprise,
               CASE WHEN market_surprise > 1 THEN 'elevated'
                    ELSE 'normal' END AS market_state
        FROM market ORDER BY date
    """).df()
    if (state.empty or state.duplicated("date").any()
            or not state.date.str[:4].isin(("2024", "2025")).all()
            or state.stocks.lt(200).any()):
        raise ValueError("Invalid 14:50 market variance state")
    selected = pd.read_parquet(PAIRS)
    high = selected.loc[selected.candidate.eq("high_variance"),
                        ["date", "pair_id"]]
    assigned = high.merge(state[["date", "market_state"]], on="date",
                          validate="many_to_one")
    if len(assigned) != len(high):
        raise ValueError("A frozen variance pair lacks a market state")
    assigned["period"] = assigned.date.map(_half)
    counts = assigned.groupby(["period", "market_state"]).agg(
        pairs=("pair_id", "size"), days=("date", "nunique"),
    ).reset_index()
    by_half = {row.period: {} for row in counts.itertuples(index=False)}
    for row in counts.itertuples(index=False):
        by_half[row.period][row.market_state] = {
            "pairs": int(row.pairs), "days": int(row.days),
        }
    gate = (len(by_half) == 4
            and all(set(item) == {"elevated", "normal"}
                    and all(values["pairs"] >= 40 and values["days"] >= 15
                            for values in item.values())
                    for item in by_half.values()))
    output_dir.mkdir(parents=True, exist_ok=True)
    state.to_parquet(output_dir / "states.parquet", index=False,
                     compression="zstd")
    report = {
        "market_dates": len(state), "frozen_pairs": len(high),
        "by_half": by_half, "outcome_gate_passed": gate,
        "note": "Inputs at 14:50; 2024 first valid days only warm up; 2026 unused",
    }
    (output_dir / "input_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def _intervals(daily: pd.Series, seed: int) -> dict:
    dates = pd.Series(daily.index.tolist())
    values = daily.reset_index(drop=True)
    return {"week_ci": _week_bootstrap(values, dates, seed),
            "month_ci": _month_bootstrap(values, dates, seed + 1)}


def _interaction_interval(days: pd.DataFrame, block: str,
                          seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    sample = days.copy()
    if block == "week":
        sample["block"] = pd.to_datetime(sample.date).dt.to_period(
            "W-SUN").astype(str)
    else:
        sample["block"] = sample.date.str[:7]
    grouped = sample.groupby(["block", "market_state"]
                             ).risk_difference.agg(["sum", "count"])
    grouped = grouped.unstack("market_state", fill_value=0)
    if ("sum", "elevated") not in grouped or ("sum", "normal") not in grouped:
        raise ValueError("Market-state interaction lacks a state")
    indices = rng.integers(0, len(grouped), size=(4000, len(grouped)))
    high_sum = grouped[("sum", "elevated")].to_numpy()[indices].sum(axis=1)
    low_sum = grouped[("sum", "normal")].to_numpy()[indices].sum(axis=1)
    high_count = grouped[("count", "elevated")].to_numpy()[indices].sum(axis=1)
    low_count = grouped[("count", "normal")].to_numpy()[indices].sum(axis=1)
    valid = (high_count > 0) & (low_count > 0)
    if valid.sum() < 1000:
        raise ValueError("Insufficient resamples with both market states")
    differences = (high_sum[valid] / high_count[valid]
                   - low_sum[valid] / low_count[valid])
    return [float(value) for value in np.quantile(differences, [.025, .975])]


def _paired_summary(pairs: pd.DataFrame) -> dict:
    if pairs.empty:
        raise ValueError("Missing paired market-state observations")
    daily_cash = pairs.groupby("date", sort=True).agg(
        high_cash=("cash_high", "mean"), low_cash=("cash_low", "mean"),
        low_stress15=("stress15_low", "mean"),
    )
    daily_cash["edge"] = daily_cash.low_cash - daily_cash.high_cash
    joint = pairs.loc[pairs.valid_high & pairs.valid_low].copy()
    if joint.empty:
        raise ValueError("No jointly clean variance pairs")
    joint["risk_difference"] = (
        joint.net_return_high.le(-.02).astype(float)
        - joint.net_return_low.le(-.02).astype(float)
    )
    joint["absolute_difference"] = (
        joint.net_return_high.abs() - joint.net_return_low.abs()
    )
    daily_risk = joint.groupby("date", sort=True).agg(
        risk_difference=("risk_difference", "mean"),
        absolute_difference=("absolute_difference", "mean"),
    )
    return {
        "pairs": len(pairs), "days": len(daily_cash),
        "joint_clean_pairs": len(joint), "joint_clean_days": len(daily_risk),
        "high_loss2pct_trade_rate": float(joint.net_return_high.le(-.02).mean()),
        "low_loss2pct_trade_rate": float(joint.net_return_low.le(-.02).mean()),
        "risk_high_minus_low_equal_day": float(daily_risk.risk_difference.mean()),
        "risk_intervals": _intervals(daily_risk.risk_difference, 941),
        "abs_high_minus_low_equal_day": float(
            daily_risk.absolute_difference.mean()),
        "high_cash": float(daily_cash.high_cash.mean()),
        "low_cash": float(daily_cash.low_cash.mean()),
        "low_minus_high_cash": float(daily_cash.edge.mean()),
        "low_cash_stress15": float(daily_cash.low_stress15.mean()),
        "high_entry_rate": float(pairs.entry_status_high.eq("filled").mean()),
        "low_entry_rate": float(pairs.entry_status_low.eq("filled").mean()),
    }


def _full_pool(states: pd.DataFrame) -> dict:
    connection = duckdb.connect()
    connection.register("states", states)
    connection.read_parquet(str(ROOT / "late_variance_risk"
                                / "all_candidates.parquet")
                            ).create_view("candidates")
    connection.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                            ).create_view("outcomes")
    connection.register("bad_days", _quality_keys(ROOT / "market_issues_ci"))
    connection.register("bad_symbols", _quality_symbols(ROOT / "market_issues_ci"))
    scored = connection.execute("""
        SELECT x.date, m.market_state,
               CASE WHEN x.surprise >= 1.5 THEN 'high' ELSE 'low' END
                   AS variance_group,
               CASE WHEN o.exit_status = 'filled'
                    AND NOT EXISTS (
                        SELECT 1 FROM bad_symbols b WHERE b.code = x.code
                    ) AND NOT EXISTS (
                        SELECT 1 FROM bad_days q
                        WHERE q.code = x.code AND q.date >= x.date
                          AND q.date <= o.exit_date
                    ) THEN o.net_return ELSE 0 END AS cash
        FROM candidates x JOIN outcomes o USING (date, code)
        JOIN states m USING (date)
        WHERE o.horizon = 1
          AND (x.surprise >= 1.5 OR x.surprise <= 1.0)
    """).df()
    expected = connection.execute("""
        SELECT COUNT(*) FROM candidates
        WHERE surprise >= 1.5 OR surprise <= 1.0
    """).fetchone()[0]
    if len(scored) != expected or scored.cash.isna().any():
        raise ValueError("Full-market diagnostic lacks archived outcomes")
    daily = scored.groupby(["date", "market_state", "variance_group"]
                           ).cash.mean().unstack().dropna(subset=["high", "low"])
    daily["low_minus_high"] = daily.low - daily.high
    daily = daily.reset_index()
    daily["period"] = daily.date.map(_half)
    return {
        period: {
            state: {"days": len(section),
                    "low_minus_high_cash": float(section.low_minus_high.mean())}
            for state, section in by_period.groupby("market_state")
        }
        for period, by_period in daily.groupby("period")
    }


def evaluate(output_dir: Path = OUTPUT) -> dict:
    inputs = json.loads((output_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if not inputs["outcome_gate_passed"]:
        raise ValueError("Market-state input coverage gate failed")
    states = pd.read_parquet(output_dir / "states.parquet")
    selected = pd.read_parquet(PAIRS)
    repriced = pd.read_parquet(ROOT / "late_variance_risk" / "repriced.parquet")
    if len(repriced) != len(selected) * 4:
        raise ValueError("Missing raw-minute outcomes for frozen variance pairs")
    labelled = repriced.merge(
        selected[["date", "code", "candidate", "pair_id"]],
        on=["date", "code"], validate="many_to_one",
    ).merge(states[["date", "market_state"]], on="date",
            validate="many_to_one")
    if len(labelled) != len(repriced) or labelled.candidate.isna().any():
        raise ValueError("Frozen pair cannot be assigned to a market state")
    labelled["valid"] = (labelled.exit_status.eq("filled")
                         & labelled.quality_clean_exit)
    if not np.allclose(_stressed_returns(labelled.loc[labelled.valid], 5),
                       labelled.loc[labelled.valid, "net_return"],
                       atol=1e-12):
        raise ValueError("Frozen minute returns disagree with the fee model")
    labelled["cash"] = labelled.net_return.where(labelled.valid, 0)
    labelled["stress15"] = 0.0
    labelled.loc[labelled.valid, "stress15"] = _stressed_returns(
        labelled.loc[labelled.valid], 15,
    )
    report = {"input_audit": inputs, "full_pool_100k_t1": _full_pool(states),
              "results": {}, "note": "Post-hoc attribution; 2025 not blind; 2026 unused"}
    for amount in (20000, 100000):
        report["results"][str(amount)] = {}
        for horizon in (1, 5):
            rows = labelled.loc[
                labelled.target_notional.eq(amount)
                & labelled.horizon.eq(horizon)
            ]
            high = rows.loc[rows.candidate.eq("high_variance")]
            low = rows.loc[rows.candidate.eq("normal_variance_control")]
            pairs = high.merge(
                low, on=["date", "pair_id", "market_state"],
                suffixes=("_high", "_low"), validate="one_to_one",
            )
            if len(pairs) != len(selected) // 2:
                raise ValueError("Missing market-state pair member")
            pairs["period"] = pairs.date.map(_half)
            periods = {}
            for year in ("2024", "2025"):
                for period in (f"{year}H1", f"{year}H2", year):
                    sample = pairs.loc[
                        pairs.period.eq(period) if "H" in period else
                        pairs.date.str.startswith(year)
                    ]
                    states_report = {
                        state: _paired_summary(group)
                        for state, group in sample.groupby("market_state")
                    }
                    if set(states_report) != {"elevated", "normal"}:
                        raise ValueError("Missing a paired market state")
                    joint = sample.loc[
                        sample.valid_high & sample.valid_low
                    ].copy()
                    joint["risk_difference"] = (
                        joint.net_return_high.le(-.02).astype(float)
                        - joint.net_return_low.le(-.02).astype(float)
                    )
                    days = joint.groupby(["date", "market_state"],
                                         as_index=False).risk_difference.mean()
                    interaction = (states_report["elevated"]
                                   ["risk_high_minus_low_equal_day"]
                                   - states_report["normal"]
                                   ["risk_high_minus_low_equal_day"])
                    periods[period] = {
                        "states": states_report,
                        "elevated_minus_normal_risk_difference": interaction,
                        "interaction_week_ci": _interaction_interval(
                            days, "week", 942,
                        ),
                        "interaction_month_ci": _interaction_interval(
                            days, "month", 943,
                        ),
                    }
            report["results"][str(amount)][str(horizon)] = periods
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        report = evaluate(args.output_dir)
        print({"report": str(args.output_dir / "report.json"),
               "full_pool_periods": list(report["full_pool_100k_t1"])})
    else:
        print(build_state(args.output_dir))


if __name__ == "__main__":
    main()
