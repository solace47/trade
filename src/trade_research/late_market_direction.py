"""Stratify frozen late-decline pairs by the market's 14:50 tail direction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .annual_cash_direct_eval import _month_bootstrap
from .late_variance_risk_eval import _archive_check
from .market_study import _quality_keys, _quality_symbols, _week_bootstrap
from .strategy_scan import _stressed_returns


ROOT = Path("data/research")
SOURCE = ROOT / "late_to_open_reversal"
OUTPUT = ROOT / "late_market_direction"
TREATMENT = "late_decline"
CONTROL = "late_rally_control"


def build_inputs(output_dir: Path = OUTPUT) -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    connection.read_parquet(str(ROOT / "intraday_features" / "*.parquet")
                            ).create_view("intraday")
    states = connection.execute("""
        WITH market AS (
            SELECT i.date, COUNT(*) AS stocks,
                   AVG(GREATEST(-.02, LEAST(.02, i.return_last30)))
                       AS market_tail_return
            FROM intraday i JOIN snapshots s USING (date, code)
            WHERE i.date BETWEEN '2024-01-01' AND '2025-12-17'
              AND s.isST = 0 AND s.listing_age_sessions >= 20
              AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
              AND s.amount_1450 >= 30000000
              AND ABS(s.price_1450 - i.price_1450) <= .005
            GROUP BY i.date HAVING COUNT(*) >= 1000
        )
        SELECT date, stocks, market_tail_return,
               CASE WHEN market_tail_return >= .001 THEN 'up'
                    WHEN market_tail_return <= -.001 THEN 'down'
                    ELSE 'flat' END AS market_state
        FROM market ORDER BY date
    """).df()
    if (states.empty or states.duplicated("date").any()
            or states.stocks.lt(1000).any()
            or not states.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Invalid 14:50 market tail states")
    signals = pd.read_parquet(SOURCE / "selections.parquet",
                              columns=["date", "code", "candidate", "pair_id"])
    treatment = signals.loc[signals.candidate.eq(TREATMENT),
                            ["date", "pair_id"]]
    assigned = treatment.merge(states[["date", "market_state"]], on="date",
                               validate="many_to_one")
    if len(assigned) != len(treatment):
        raise ValueError("A frozen pair lacks a market tail state")
    assigned["half"] = assigned.date.str[:4] + "H" + np.where(
        assigned.date.str[5:7].astype(int).le(6), "1", "2")
    by_half = assigned.groupby(["half", "market_state"]).agg(
        pairs=("pair_id", "size"), days=("date", "nunique")
    ).reset_index().to_dict("records")
    counts = {(row["half"], row["market_state"]): row for row in by_half}
    gate = all(
        (half, state) in counts
        and counts[(half, state)]["pairs"] >= 30
        and counts[(half, state)]["days"] >= 15
        for half in ("2024H1", "2024H2", "2025H1", "2025H2")
        for state in ("up", "down")
    )
    audit = {"market_dates": len(states), "frozen_pairs": len(treatment),
             "by_half": by_half, "outcome_gate_passed": gate,
             "market_proxy": "clipped equal-weight return, all complete 14:50 stocks"}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "report.json").unlink(missing_ok=True)
    states.to_parquet(output_dir / "states.parquet", index=False,
                      compression="zstd")
    (output_dir / "input_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return audit


def _summary(pairs: pd.DataFrame) -> dict:
    if pairs.empty:
        raise ValueError("A requested market state has no paired trades")
    daily = pairs.groupby("date", sort=True).agg(
        treatment=("cash_treatment", "mean"),
        control=("cash_control", "mean"),
        treatment_stress=("stress_treatment", "mean"),
        control_stress=("stress_control", "mean"),
    ).reset_index()
    daily["edge"] = daily.treatment - daily.control
    daily["stress_edge"] = daily.treatment_stress - daily.control_stress
    return {
        "pairs": len(pairs), "days": len(daily),
        "treatment_cash": float(daily.treatment.mean()),
        "control_cash": float(daily.control.mean()),
        "edge_cash": float(daily.edge.mean()),
        "treatment_stress15": float(daily.treatment_stress.mean()),
        "edge_stress15": float(daily.stress_edge.mean()),
        "treatment_entry_rate": float(
            pairs.entry_status_treatment.eq("filled").mean()),
        "control_entry_rate": float(
            pairs.entry_status_control.eq("filled").mean()),
        "treatment_on_time_clean_exit_rate": float(pairs.valid_treatment.mean()),
        "control_on_time_clean_exit_rate": float(pairs.valid_control.mean()),
        "treatment_week_ci": _week_bootstrap(
            daily.treatment, daily.date, 1040),
        "treatment_month_ci": _month_bootstrap(
            daily.treatment, daily.date, 1041),
        "edge_week_ci": _week_bootstrap(daily.edge, daily.date, 1042),
        "edge_month_ci": _month_bootstrap(daily.edge, daily.date, 1043),
    }


def _state_contrast(pairs: pd.DataFrame, field: str, block: str) -> dict:
    daily = pairs.groupby(["date", "market_state"], sort=True)[field].mean(
    ).reset_index()
    daily = daily.loc[daily.market_state.isin(("up", "down"))].copy()
    if set(daily.market_state) != {"up", "down"}:
        raise ValueError("Both market states are required for contrast")
    daily["block"] = (daily.date.str[:7] if block == "month" else
                      pd.to_datetime(daily.date).dt.to_period("W-SUN").astype(str))
    sums = daily.groupby(["block", "market_state"])[field].agg(
        ["sum", "count"]).unstack("market_state", fill_value=0)
    blocks = len(sums)
    rng = np.random.default_rng(1044 if block == "month" else 1045)
    draws = rng.integers(0, blocks, size=(4000, blocks))
    up_count = sums[("count", "up")].to_numpy()[draws].sum(axis=1)
    down_count = sums[("count", "down")].to_numpy()[draws].sum(axis=1)
    valid = (up_count > 0) & (down_count > 0)
    if valid.sum() < 2000:
        raise ValueError("Too few resamples include both market states")
    sampled = (sums[("sum", "up")].to_numpy()[draws][valid].sum(axis=1)
               / up_count[valid]
               - sums[("sum", "down")].to_numpy()[draws][valid].sum(axis=1)
               / down_count[valid])
    means = daily.groupby("market_state")[field].mean()
    return {"up_minus_down": float(means["up"] - means["down"]),
            "ci": [float(v) for v in np.quantile(sampled, [.025, .975])]}


def _full_pool_diagnostic(states: pd.DataFrame) -> dict:
    """Post-outcome external-validity check, without matched-stock claims."""
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.register("states", states)
    connection.register("bad_days", _quality_keys(ROOT / "market_issues_ci"))
    connection.register("bad_symbols", _quality_symbols(ROOT / "market_issues_ci"))
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    connection.read_parquet(str(ROOT / "intraday_features" / "*.parquet")
                            ).create_view("intraday")
    connection.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                            ).create_view("archived")
    connection.execute("""
        CREATE TEMP TABLE candidates AS
        SELECT s.date, s.code,
               CASE WHEN i.return_last30 BETWEEN -.01 AND -.003
                    THEN 'decline' ELSE 'rally' END AS group_name
        FROM snapshots s JOIN intraday i USING (date, code)
        WHERE ((s.date BETWEEN '2024-01-01' AND '2024-12-17')
            OR (s.date BETWEEN '2025-01-01' AND '2025-12-17'))
          AND s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.price_1450 >= 5
          AND s.amount_1450 BETWEEN 100000000 AND 1000000000
          AND s.return20_prior_adjusted BETWEEN -.10 AND .10
          AND s.return_1450 BETWEEN -.03 AND .03
          AND s.position_1450 BETWEEN 0 AND 1
          AND ABS(s.price_1450 - i.price_1450) <= .005
          AND (i.return_last30 BETWEEN -.01 AND -.003
            OR i.return_last30 BETWEEN .003 AND .01)
    """)
    expected = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    days = connection.execute("""
        WITH scored AS (
            SELECT c.date, c.group_name,
                   CASE WHEN o.exit_status = 'filled'
                         AND o.exit_delay_sessions = 0
                         AND NOT EXISTS (
                             SELECT 1 FROM bad_symbols b WHERE b.code = c.code
                         ) AND NOT EXISTS (
                             SELECT 1 FROM bad_days q WHERE q.code = c.code
                               AND q.date >= c.date AND q.date <= o.exit_date
                         ) THEN o.net_return ELSE 0 END AS cash
            FROM candidates c JOIN archived o USING (date, code)
            WHERE o.horizon = 1
        )
        SELECT date, group_name, COUNT(*) AS stocks, AVG(cash) AS cash
        FROM scored GROUP BY date, group_name
    """).df()
    if int(days.stocks.sum()) != expected:
        raise ValueError("Full-market tail groups lack archived T+1 trades")
    wide = days.pivot(index="date", columns="group_name",
                      values=["stocks", "cash"])
    wide.columns = [f"{metric}_{group}" for metric, group in wide.columns]
    wide = wide.dropna(subset=["cash_decline", "cash_rally"]).reset_index(
    ).merge(states, on="date", validate="one_to_one")
    wide["edge"] = wide.cash_decline - wide.cash_rally
    wide["half"] = wide.date.str[:4] + "H" + np.where(
        wide.date.str[5:7].astype(int).le(6), "1", "2")
    report = {"stockdays": expected, "two_group_dates": len(wide),
              "periods": {}}
    for period in ("2024", "2024H1", "2024H2", "2025", "2025H1", "2025H2"):
        frame = wide.loc[
            wide.half.eq(period) if "H" in period else
            wide.date.str.startswith(period)]
        report["periods"][period] = {}
        for state in ("up", "down", "flat"):
            section = frame.loc[frame.market_state.eq(state)]
            if section.empty:
                raise ValueError("A full-market state has no two-group days")
            report["periods"][period][state] = {
                "days": len(section),
                "decline_stockdays": int(section.stocks_decline.sum()),
                "rally_stockdays": int(section.stocks_rally.sum()),
                "decline_cash": float(section.cash_decline.mean()),
                "rally_cash": float(section.cash_rally.mean()),
                "edge_cash": float(section.edge.mean()),
                "edge_month_ci": _month_bootstrap(
                    section.edge, section.date, 1046),
            }
    return report


def evaluate(output_dir: Path = OUTPUT) -> dict:
    audit = json.loads((output_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if not audit["outcome_gate_passed"]:
        raise ValueError("Market-state input gate failed")
    states = pd.read_parquet(output_dir / "states.parquet",
                             columns=["date", "market_state"])
    signals = pd.read_parquet(SOURCE / "selections.parquet",
                              columns=["date", "code", "candidate", "pair_id"])
    repriced = pd.read_parquet(SOURCE / "repriced.parquet")
    if (len(repriced) != len(signals) * 8
            or repriced.duplicated(["date", "code", "target_notional",
                                    "entry_window", "exit_window",
                                    "horizon"]).any()):
        raise ValueError("Frozen raw-minute repricing grid is incomplete")
    archive = _archive_check(repriced.loc[repriced.exit_window.eq("close")])
    labelled = repriced.merge(signals, on=["date", "code"],
                              validate="many_to_one")
    if len(labelled) != len(repriced):
        raise ValueError("Repriced trade lacks a frozen label")
    labelled["valid"] = (
        labelled.exit_status.eq("filled") & labelled.quality_clean_exit
        & labelled.exit_delay_sessions.eq(0)
    )
    labelled["cash"] = labelled.net_return.where(labelled.valid, 0.0)
    labelled["stress"] = 0.0
    labelled.loc[labelled.valid, "stress"] = _stressed_returns(
        labelled.loc[labelled.valid], 15)
    report = {"input_audit": audit, "archive_check_100k_close": archive,
              "full_pool_post_hoc_100k_t1_close": _full_pool_diagnostic(states),
              "results": {}}
    for amount in (20000, 100000):
        report["results"][str(amount)] = {}
        for horizon in (1, 5):
            report["results"][str(amount)][str(horizon)] = {}
            for window in ("morning", "close"):
                frame = labelled.loc[
                    labelled.target_notional.eq(amount)
                    & labelled.horizon.eq(horizon)
                    & labelled.exit_window.eq(window)]
                treatment = frame.loc[frame.candidate.eq(TREATMENT)]
                control = frame.loc[frame.candidate.eq(CONTROL)]
                pairs = treatment.merge(
                    control, on=["date", "pair_id"],
                    suffixes=("_treatment", "_control"),
                    validate="one_to_one").merge(
                        states, on="date", validate="many_to_one")
                if len(pairs) != len(signals) // 2:
                    raise ValueError("A frozen pair lacks a market state or trade")
                pairs["edge"] = pairs.cash_treatment - pairs.cash_control
                details = {}
                for year in ("2024", "2025"):
                    annual = pairs.loc[pairs.date.str.startswith(year)]
                    month = annual.date.str[5:7].astype(int)
                    for period, section in (
                        (year, annual),
                        (f"{year}H1", annual.loc[month.le(6)]),
                        (f"{year}H2", annual.loc[month.gt(6)]),
                    ):
                        details[period] = {
                            state: _summary(section.loc[
                                section.market_state.eq(state)])
                            for state in ("up", "down", "flat")
                        }
                        details[period]["up_minus_down_edge"] = {
                            block: _state_contrast(section, "edge", block)
                            for block in ("week", "month")
                        }
                        details[period]["up_minus_down_treatment"] = {
                            block: _state_contrast(
                                section, "cash_treatment", block)
                            for block in ("week", "month")
                        }
                report["results"][str(amount)][str(horizon)][window] = details
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("inputs", "evaluate"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.mode == "inputs":
        print(build_inputs(args.output))
    else:
        report = evaluate(args.output)
        print({"archive_check": report["archive_check_100k_close"],
               "report": str(args.output / "report.json")})


if __name__ == "__main__":
    main()
