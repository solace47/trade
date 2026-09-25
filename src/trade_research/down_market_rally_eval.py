"""Evaluate frozen countertrend rallies against same-day flat-tail stocks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from .annual_cash_direct_eval import _month_bootstrap
from .late_variance_risk_eval import _archive_check
from .market_study import _quality_keys, _quality_symbols, _week_bootstrap
from .strategy_scan import _stressed_returns


ROOT = Path("data/research")
OUTPUT = ROOT / "down_market_rally"
RALLY = "countertrend_rally"
FLAT = "flat_tail_control"


def _summary(pairs: pd.DataFrame) -> dict:
    if pairs.empty:
        raise ValueError("A required period lacks frozen pairs")
    daily = pairs.groupby("date", sort=True).agg(
        rally=("cash_rally", "mean"), flat=("cash_flat", "mean"),
        rally_stress=("stress_rally", "mean"),
        flat_stress=("stress_flat", "mean"),
    ).reset_index()
    daily["edge"] = daily.rally - daily.flat
    daily["stress_edge"] = daily.rally_stress - daily.flat_stress
    return {
        "pairs": int(len(pairs)), "days": int(len(daily)),
        "rally_cash": float(daily.rally.mean()),
        "flat_cash": float(daily.flat.mean()),
        "rally_minus_flat_cash": float(daily.edge.mean()),
        "rally_stress15": float(daily.rally_stress.mean()),
        "rally_minus_flat_stress15": float(daily.stress_edge.mean()),
        "rally_entry_rate": float(pairs.entry_status_rally.eq("filled").mean()),
        "flat_entry_rate": float(pairs.entry_status_flat.eq("filled").mean()),
        "rally_on_time_clean_exit_rate": float(pairs.valid_rally.mean()),
        "flat_on_time_clean_exit_rate": float(pairs.valid_flat.mean()),
        "edge_week_ci": _week_bootstrap(daily.edge, daily.date, 1101),
        "edge_month_ci": _month_bootstrap(daily.edge, daily.date, 1102),
        "rally_month_ci": _month_bootstrap(daily.rally, daily.date, 1103),
    }


def _full_pool_diagnostic() -> dict:
    """Compare all eligible stock-days; this is not a matched estimate."""
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    connection.read_parquet(str(ROOT / "intraday_features" / "*.parquet")
                            ).create_view("intraday")
    connection.read_parquet(str(ROOT / "late_market_direction" / "states.parquet")
                            ).create_view("states")
    connection.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                            ).create_view("archived")
    connection.register("bad_days", _quality_keys(ROOT / "market_issues_ci"))
    connection.register("bad_symbols", _quality_symbols(ROOT / "market_issues_ci"))
    connection.execute("""
        CREATE TEMP TABLE candidates AS
        SELECT s.date, s.code,
               CASE WHEN i.return_last30 BETWEEN .003 AND .01
                    THEN 'rally' ELSE 'flat' END AS group_name
        FROM snapshots s JOIN intraday i USING (date, code)
        JOIN states m USING (date)
        WHERE m.market_state = 'down'
          AND ((s.date BETWEEN '2024-01-01' AND '2024-12-17')
            OR (s.date BETWEEN '2025-01-01' AND '2025-12-17'))
          AND s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.price_1450 >= 5
          AND s.amount_1450 BETWEEN 100000000 AND 1000000000
          AND s.return20_prior_adjusted BETWEEN -.10 AND .10
          AND s.return_1450 BETWEEN -.03 AND .03
          AND s.position_1450 BETWEEN 0 AND 1
          AND ABS(s.price_1450 - i.price_1450) <= .005
          AND (i.return_last30 BETWEEN .003 AND .01
            OR i.return_last30 BETWEEN -.001 AND .001)
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
        raise ValueError("Full-pool groups lack archived T+1 trades")
    wide = days.pivot(index="date", columns="group_name",
                      values=["stocks", "cash"])
    wide.columns = [f"{metric}_{group}" for metric, group in wide.columns]
    wide = wide.dropna(subset=["cash_rally", "cash_flat"]).reset_index()
    wide["edge"] = wide.cash_rally - wide.cash_flat
    report = {"stockdays": int(expected), "two_group_days": len(wide),
              "periods": {}}
    for year in ("2024", "2025"):
        annual = wide.loc[wide.date.str.startswith(year)]
        month = annual.date.str[5:7].astype(int)
        for label, frame in (
            (year, annual), (f"{year}H1", annual.loc[month.le(6)]),
            (f"{year}H2", annual.loc[month.gt(6)]),
        ):
            if frame.empty:
                raise ValueError("A full-pool period lacks both groups")
            report["periods"][label] = {
                "days": len(frame),
                "rally_stockdays": int(frame.stocks_rally.sum()),
                "flat_stockdays": int(frame.stocks_flat.sum()),
                "rally_cash": float(frame.cash_rally.mean()),
                "flat_cash": float(frame.cash_flat.mean()),
                "rally_minus_flat_cash": float(frame.edge.mean()),
                "edge_month_ci": _month_bootstrap(
                    frame.edge, frame.date, 1104),
            }
    return report


def evaluate(output_dir: Path = OUTPUT) -> dict:
    audit = json.loads((output_dir / "input_audit.json").read_text(
        encoding="utf-8"))
    if not audit["outcome_gate_passed"]:
        raise ValueError("Frozen input gate failed; outcomes must stay closed")
    signals = pd.read_parquet(output_dir / "selections.parquet")
    repriced = pd.read_parquet(output_dir / "repriced.parquet")
    key = ["date", "code", "target_notional", "entry_window",
           "exit_window", "horizon"]
    expected = len(signals) * 8
    if (len(repriced) != expected or repriced.duplicated(key).any()
            or set(repriced.target_notional) != {20000, 100000}
            or set(repriced.entry_window) != {"baseline"}
            or set(repriced.exit_window) != {"morning", "close"}
            or set(repriced.horizon) != {1, 5}):
        raise ValueError("Incomplete frozen raw-minute repricing grid")
    archive = _archive_check(repriced.loc[repriced.exit_window.eq("close")])
    labelled = repriced.merge(
        signals[["date", "code", "candidate", "pair_id"]],
        on=["date", "code"], validate="many_to_one")
    if len(labelled) != expected:
        raise ValueError("A repriced trade lacks its frozen pair label")
    labelled["valid"] = (
        labelled.exit_status.eq("filled") & labelled.quality_clean_exit
        & labelled.exit_delay_sessions.eq(0)
    )
    labelled["cash"] = labelled.net_return.where(labelled.valid, 0.0)
    labelled["stress"] = 0.0
    labelled.loc[labelled.valid, "stress"] = _stressed_returns(
        labelled.loc[labelled.valid], 15)
    report = {"input_audit": audit, "archive_check_100k_close": archive,
              "cost_stress_bps_each_side": 15,
              "delayed_or_unfilled_exit_cash": 0,
              "results": {}}
    for amount in (20000, 100000):
        report["results"][str(amount)] = {}
        for horizon in (1, 5):
            report["results"][str(amount)][str(horizon)] = {}
            for window in ("morning", "close"):
                rows = labelled.loc[
                    labelled.target_notional.eq(amount)
                    & labelled.horizon.eq(horizon)
                    & labelled.exit_window.eq(window)]
                rally = rows.loc[rows.candidate.eq(RALLY)]
                flat = rows.loc[rows.candidate.eq(FLAT)]
                pairs = rally.merge(
                    flat, on=["date", "pair_id"],
                    suffixes=("_rally", "_flat"),
                    validate="one_to_one")
                if len(pairs) != len(signals) // 2:
                    raise ValueError("A frozen pair lacks an outcome")
                sections = {}
                for year in ("2024", "2025"):
                    annual = pairs.loc[pairs.date.str.startswith(year)]
                    month = annual.date.str[5:7].astype(int)
                    sections[year] = _summary(annual)
                    sections[f"{year}H1"] = _summary(annual.loc[month.le(6)])
                    sections[f"{year}H2"] = _summary(annual.loc[month.gt(6)])
                report["results"][str(amount)][str(horizon)][window] = sections
    report["full_pool_100k_t1_close"] = _full_pool_diagnostic()
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    report = evaluate(args.output)
    print({"archive_check": report["archive_check_100k_close"],
           "report": str(args.output / "report.json")})


if __name__ == "__main__":
    main()
