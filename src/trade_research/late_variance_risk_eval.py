"""Evaluate frozen late-variance pairs after raw-minute repricing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from .annual_cash_direct_eval import _month_bootstrap
from .market_study import _quality_keys, _quality_symbols, _week_bootstrap
from .strategy_scan import _stressed_returns


ROOT = Path("data/research")
HIGH = "high_variance"
LOW = "normal_variance_control"


def _archive_check(repriced: pd.DataFrame) -> dict:
    connection = duckdb.connect()
    connection.register("repriced", repriced.loc[
        repriced.target_notional.eq(100000)
    ])
    connection.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                            ).create_view("archived")
    checks = connection.execute("""
        SELECT COUNT(*) AS rows,
               COUNT(*) FILTER (
                   WHERE r.entry_status IS DISTINCT FROM o.entry_status
                      OR r.exit_status IS DISTINCT FROM o.exit_status
                      OR r.exit_date IS DISTINCT FROM o.exit_date
                      OR r.shares IS DISTINCT FROM o.shares
                      OR abs(r.entry_price-o.entry_price) > 1e-9
                      OR (r.entry_price IS NULL) != (o.entry_price IS NULL)
                      OR abs(r.exit_price-o.exit_price) > 1e-9
                      OR (r.exit_price IS NULL) != (o.exit_price IS NULL)
                      OR abs(r.net_return-o.net_return) > 1e-9
                      OR (r.net_return IS NULL) != (o.net_return IS NULL)
               ) AS mismatches
        FROM repriced r JOIN archived o USING (date, code, horizon)
    """).df().iloc[0].to_dict()
    checks = {key: int(value) for key, value in checks.items()}
    if (checks["rows"] != len(repriced.loc[
            repriced.target_notional.eq(100000)])
            or checks["mismatches"] != 0):
        raise ValueError("Raw-minute 100k repricing differs from archive")
    return checks


def _intervals(values: pd.Series, dates: pd.Series, seed: int) -> dict:
    return {
        "week_ci": _week_bootstrap(values, dates, seed),
        "month_ci": _month_bootstrap(values, dates, seed + 1),
    }


def _full_market_diagnostic(output_dir: Path) -> dict:
    """Show external validity without treating unmatched groups as causal."""
    connection = duckdb.connect()
    connection.read_parquet(str(output_dir / "all_candidates.parquet")
                            ).create_view("candidates")
    connection.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                            ).create_view("archived")
    connection.register("bad_days", _quality_keys(ROOT / "market_issues_ci"))
    connection.register("bad_symbols", _quality_symbols(ROOT / "market_issues_ci"))
    expected, archived = connection.execute("""
        SELECT (SELECT COUNT(*) FROM candidates
                WHERE surprise >= 1.5 OR surprise <= 1.0),
               (SELECT COUNT(*) FROM candidates x
                JOIN archived o USING (date, code)
                WHERE o.horizon = 1
                  AND (x.surprise >= 1.5 OR x.surprise <= 1.0))
    """).fetchone()
    if expected != archived:
        raise ValueError("Full-market diagnostic lacks archived T+1 outcomes")
    daily = connection.execute("""
        WITH scored AS (
            SELECT x.date,
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
            FROM candidates x JOIN archived o USING (date, code)
            WHERE o.horizon = 1
              AND (x.surprise >= 1.5 OR x.surprise <= 1.0)
        )
        SELECT date, variance_group, COUNT(*) AS stockdays,
               AVG(cash) AS cash
        FROM scored GROUP BY date, variance_group
    """).df()
    wide = daily.pivot(index="date", columns="variance_group",
                       values=["stockdays", "cash"])
    wide.columns = [f"{metric}_{group}" for metric, group in wide.columns]
    wide = wide.dropna(subset=["cash_high", "cash_low"]).reset_index()
    if wide.empty:
        raise ValueError("No dates with both full-market variance groups")
    wide["edge"] = wide.cash_low - wide.cash_high
    report = {}
    for year in ("2024", "2025"):
        for half, condition in (
            ("H1", wide.date.str[5:7].astype(int).le(6)),
            ("H2", wide.date.str[5:7].astype(int).gt(6)),
        ):
            frame = wide.loc[wide.date.str.startswith(year) & condition]
            report[f"{year}{half}"] = {
                "days": len(frame),
                "high_stockdays": int(frame.stockdays_high.sum()),
                "low_stockdays": int(frame.stockdays_low.sum()),
                "high_cash": float(frame.cash_high.mean()),
                "low_cash": float(frame.cash_low.mean()),
                "low_minus_high": float(frame.edge.mean()),
                "edge_intervals": _intervals(
                    frame.edge.reset_index(drop=True),
                    frame.date.reset_index(drop=True), 821,
                ),
            }
    return report


def _summarize(pairs: pd.DataFrame) -> dict:
    if pairs.empty:
        raise ValueError("Missing pairs for a requested period")
    dates = pd.Series(sorted(pairs.date.unique()))
    daily = pairs.groupby("date", sort=True).agg(
        high_cash=("cash_high", "mean"),
        low_cash=("cash_low", "mean"),
        high_stress=("stress_high", "mean"),
        low_stress=("stress_low", "mean"),
    ).reindex(dates)
    daily["edge"] = daily.low_cash - daily.high_cash
    daily["stress_edge"] = daily.low_stress - daily.high_stress
    result = {
        "pairs": int(len(pairs)), "days": len(dates),
        "high_entry_rate": float(pairs.entry_status_high.eq("filled").mean()),
        "low_entry_rate": float(pairs.entry_status_low.eq("filled").mean()),
        "high_clean_exit_rate": float(pairs.valid_high.mean()),
        "low_clean_exit_rate": float(pairs.valid_low.mean()),
        "high_cash": float(daily.high_cash.mean()),
        "low_cash": float(daily.low_cash.mean()),
        "low_minus_high_cash": float(daily.edge.mean()),
        "low_cash_stress15": float(daily.low_stress.mean()),
        "low_minus_high_stress15": float(daily.stress_edge.mean()),
        "edge_intervals": _intervals(daily.edge.reset_index(drop=True),
                                      dates, 819),
    }
    joint = pairs.loc[pairs.valid_high & pairs.valid_low].copy()
    result["joint_clean_pairs"] = len(joint)
    if not joint.empty:
        joint["risk_difference"] = (
            joint.net_return_high.le(-.02).astype(float)
            - joint.net_return_low.le(-.02).astype(float)
        )
        joint["absolute_difference"] = (
            joint.net_return_high.abs() - joint.net_return_low.abs()
        )
        risk_day = joint.groupby("date", sort=True).agg(
            risk_difference=("risk_difference", "mean"),
            absolute_difference=("absolute_difference", "mean"),
        )
        risk_dates = pd.Series(risk_day.index.tolist())
        result.update({
            "joint_clean_days": len(risk_day),
            "high_loss2pct_trade_rate": float(
                joint.net_return_high.le(-.02).mean()),
            "low_loss2pct_trade_rate": float(
                joint.net_return_low.le(-.02).mean()),
            "high_minus_low_loss2pct_equal_day": float(
                risk_day.risk_difference.mean()),
            "high_minus_low_abs_return_equal_day": float(
                risk_day.absolute_difference.mean()),
            "risk_intervals": _intervals(
                risk_day.risk_difference.reset_index(drop=True),
                risk_dates, 820),
        })
    return result


def evaluate(output_dir: Path = ROOT / "late_variance_risk") -> dict:
    signals = pd.read_parquet(output_dir / "selections.parquet")
    repriced = pd.read_parquet(output_dir / "repriced.parquet")
    expected = len(signals) * 2 * 2
    if (len(repriced) != expected
            or repriced.duplicated(["date", "code", "horizon",
                                    "target_notional"]).any()
            or set(repriced.target_notional) != {20000, 100000}
            or set(repriced.horizon) != {1, 5}):
        raise ValueError("Incomplete or duplicate raw-minute repricing")
    archive_check = _archive_check(repriced)
    labelled = repriced.merge(
        signals[["date", "code", "candidate", "pair_id"]],
        on=["date", "code"], validate="many_to_one",
    )
    if len(labelled) != expected:
        raise ValueError("Some repriced stock-days lack a frozen label")
    labelled["valid"] = (labelled.exit_status.eq("filled")
                         & labelled.quality_clean_exit)
    labelled["cash"] = labelled.net_return.where(labelled.valid, 0)
    labelled["stress"] = 0.0
    labelled.loc[labelled.valid, "stress"] = _stressed_returns(
        labelled.loc[labelled.valid], 15,
    )
    report: dict = {
        "archive_check": archive_check,
        "full_market_diagnostic_100k_t1": _full_market_diagnostic(output_dir),
        "slippage_stress_bps_each_side": 15,
        "note": "2025 already examined; 2026 not used; no order-book queue model",
        "results": {},
    }
    for amount in (20000, 100000):
        report["results"][str(amount)] = {}
        for horizon in (1, 5):
            rows = labelled.loc[
                labelled.target_notional.eq(amount)
                & labelled.horizon.eq(horizon)
            ]
            high = rows.loc[rows.candidate.eq(HIGH)]
            low = rows.loc[rows.candidate.eq(LOW)]
            pairs = high.merge(low, on=["date", "pair_id"],
                               suffixes=("_high", "_low"),
                               validate="one_to_one")
            if len(pairs) != len(signals) // 2:
                raise ValueError("Missing matched repriced member")
            periods = {}
            for year in ("2024", "2025"):
                yearly = pairs.loc[pairs.date.str.startswith(year)]
                for name, sample in (
                    (f"{year}H1", yearly.loc[
                        yearly.date.str[5:7].astype(int).le(6)]),
                    (f"{year}H2", yearly.loc[
                        yearly.date.str[5:7].astype(int).gt(6)]),
                    (year, yearly),
                ):
                    periods[name] = _summarize(sample)
            report["results"][str(amount)][str(horizon)] = periods
    (output_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "late_variance_risk")
    args = parser.parse_args()
    report = evaluate(args.output)
    print({"archive_check": report["archive_check"],
           "report": str(args.output / "report.json")})


if __name__ == "__main__":
    main()
