"""Build the frozen ex-dividend hypothesis inputs using 2024 onward history.

Only lagged daily returns and through-14:49 minute-prefix fields are projected.
The event calendar is joined later, after its independent source reconciliation.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
import json
from pathlib import Path

import duckdb
import pandas as pd

from .cash_dividend_catalog import ROOT as CATALOG
from .corporate_cash import save_json, sha


ROOT = Path("data/research/cash_ex_1449")
PREFIX = Path("data/research/minute_prefix_1449")


def register_history(connection: duckdb.DuckDBPyConnection) -> None:
    connection.execute("""
        CREATE TEMP VIEW traded AS
        SELECT date, code, preclose, close, isST
        FROM daily
        WHERE date BETWEEN '2024-01-01' AND '2025-12-31' AND tradestatus = 1
          AND (code LIKE 'sh.60%' OR code LIKE 'sz.00%')
    """)
    connection.execute("""
        CREATE TEMP VIEW history AS
        SELECT date, code, preclose, isST,
               ROW_NUMBER() OVER full_history - 1 AS prior_sessions,
               LAG(date) OVER full_history AS previous_traded_date,
               LAG(close) OVER full_history AS previous_close,
               COUNT(*) OVER prior20 AS prior20_count,
               COUNT(*) FILTER (WHERE close > 0 AND preclose > 0) OVER prior20 AS valid_prior20,
               EXP(SUM(CASE WHEN close > 0 AND preclose > 0
                            THEN LN(close / preclose) ELSE NULL END) OVER prior20) - 1 AS prior20_return
        FROM traded
        WINDOW full_history AS (PARTITION BY code ORDER BY date),
               prior20 AS (PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
    """)
    connection.execute("""
        CREATE TEMP VIEW calendar AS
        SELECT date, LAG(date) OVER (ORDER BY date) AS previous_market_date
        FROM (SELECT DISTINCT date FROM traded)
    """)


def build(output: Path = ROOT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((CATALOG / "manifest.json").read_text())
    daily_files = sorted(manifest["daily_sha256"])
    for name in daily_files:
        if sha(Path(name)) != manifest["daily_sha256"][name]:
            raise ValueError("Daily source changed since the event universe was frozen")
    prefix_files = sorted(str(p) for year in (2024, 2025)
                          for p in (PREFIX / str(year)).glob("part_*.parquet"))
    if not prefix_files:
        raise ValueError("No through-14:49 prefix files")
    destination = output / "base.parquet"
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.execute("SET memory_limit = '8GB'")
    try:
        connection.read_parquet(daily_files).create_view("daily")
        connection.read_parquet(prefix_files).create_view("prefix")
        register_history(connection)
        connection.execute("""
            CREATE TEMP VIEW prior_eligible AS
            SELECT h.*, c.previous_market_date,
                   LEFT(h.date, 4) || 'H' || CASE WHEN SUBSTR(h.date, 6, 2) <= '06'
                                                 THEN '1' ELSE '2' END AS half
            FROM history h JOIN calendar c USING (date)
            WHERE h.prior_sessions >= 60 AND h.isST = 0
              AND SUBSTR(h.date, 6, 5) <= '12-17'
              AND h.preclose > 0 AND h.previous_close > 0
              AND h.prior20_count = 20 AND h.valid_prior20 = 20
              AND h.prior20_return BETWEEN -.20 AND .20
        """)
        coverage = connection.execute("""
            SELECT h.half, COUNT(*) AS daily_rows,
                   COUNT(*) FILTER (WHERE p.code IS NOT NULL) AS prefix_rows,
                   COUNT(DISTINCT (h.date, h.code)) AS daily_keys
            FROM prior_eligible h LEFT JOIN prefix p USING (date, code)
            GROUP BY h.half ORDER BY h.half
        """).df().to_dict("records")
        escaped = str(destination).replace("'", "''")
        connection.execute(f"""
            COPY (
                SELECT h.date, h.code, h.half, h.preclose, h.previous_close,
                       h.previous_traded_date, h.previous_market_date, h.prior_sessions,
                       h.prior20_return, p.price_1449, p.amount_1449, p.return_last29,
                       p.price_1449 / h.preclose - 1 AS day_return,
                       ABS(h.preclose - h.previous_close) > .005 AS reference_gap
                FROM prior_eligible h JOIN prefix p USING (date, code)
                WHERE NOT p.quote_outside_traded_range AND p.price_1449 >= 5
                  AND p.amount_1449 BETWEEN 100000000 AND 1000000000
                  AND p.return_last29 BETWEEN -.01 AND .01
                  AND p.price_1449 / h.preclose - 1 BETWEEN -.03 AND .03
                ORDER BY h.date, h.code
            ) TO '{escaped}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """)
        connection.read_parquet(str(destination)).create_view("base")
        counts = connection.execute("""
            SELECT half, COUNT(*) AS rows, COUNT(DISTINCT (date, code)) AS keys,
                   COUNT(DISTINCT date) AS days, MIN(date) AS first_date, MAX(date) AS last_date
            FROM base GROUP BY half ORDER BY half
        """).df().to_dict("records")
    finally:
        connection.close()
    for row in coverage:
        row["coverage"] = row["prefix_rows"] / row["daily_rows"]
    result = {"rule_commit": "1599017", "coverage": coverage, "base_by_half": counts,
              "base_sha256": sha(destination), "catalog_universe_sha256": sha(CATALOG / "manifest.json"),
              "prefix_sha256": {p: sha(Path(p)) for p in prefix_files},
              "association_passed": len(coverage) == 4 and all(
                  r["daily_rows"] == r["daily_keys"] and r["coverage"] >= .999 for r in coverage)
                  and all(r["rows"] == r["keys"] for r in counts),
              "history_start": "2024-01-01", "last_minute_label": "1449",
              "strategy_returns_read": False, "holdout_read": False}
    save_json(output / "base_report.json", result)
    return result


def potential(output: Path = ROOT) -> dict:
    base_report = json.loads((output / "base_report.json").read_text())
    catalog_report = json.loads((CATALOG / "primary_supplement_report.json").read_text())
    if (sha(output / "base.parquet") != base_report["base_sha256"]
            or sha(CATALOG / "events_augmented.parquet") != catalog_report["augmented_events_sha256"]):
        raise ValueError("A frozen input or reconciled event source changed")
    if not base_report["association_passed"]:
        raise ValueError("Prefix association gate failed")
    base = pd.read_parquet(output / "base.parquet")
    events = pd.read_parquet(CATALOG / "events_augmented.parquet").rename(
        columns={"dividOperateDate": "date"})
    event_days = base.merge(events, on=["date", "code"], how="inner", validate="one_to_one")
    state = event_days.loc[event_days.day_return.between(-.03, -.005)].copy()
    # Even all distribution types and all cash amounts together cannot produce
    # more event-date signals than this ceiling in the observed catalog.
    selected = state.loc[state.action_type.eq("provisional_pure_cash")].copy()
    selected["cash_yield_exact"] = [Decimal(cash) / Decimal(str(previous))
        for cash, previous in zip(selected.dividCashPsBeforeTax, selected.previous_close)]
    selected = selected.loc[selected.cash_yield_exact.ge(Decimal(".01"))].copy()
    selected["cash_yield"] = selected.cash_yield_exact.map(float)
    selected = selected.sort_values(["date", "cash_yield_exact", "day_return", "code"],
                                    ascending=[True, False, True, True])
    selected["daily_rank"] = selected.groupby("date", sort=False).cumcount() + 1
    selected["metadata_visible_before_event"] = (
        selected.dividPlanDate.ge("2024-01-01")
        & selected.dividPlanDate.le(selected.previous_market_date)
        & selected.dividRegistDate.eq(selected.previous_market_date))
    selected = selected.drop(columns="cash_yield_exact")
    selected.to_parquet(output / "potential_events.parquet", index=False)
    summaries = []
    for half in ("2024H1", "2024H2", "2025H1", "2025H2"):
        all_types = state.loc[state.half.eq(half)]
        candidates = selected.loc[selected.half.eq(half)]
        attempted = candidates.loc[candidates.daily_rank.le(5)]
        summaries.append({"half": half,
            "all_action_state_rows": len(all_types), "all_action_state_days": int(all_types.date.nunique()),
            "all_action_capacity_ceiling": int(all_types.groupby("date").size().clip(upper=5).sum()),
            "provisional_cash_rows": len(candidates), "provisional_cash_days": int(candidates.date.nunique()),
            "provisional_top5_rows": len(attempted),
            "provisional_top5_visible_metadata": int(attempted.metadata_visible_before_event.sum()),
            "provisional_top5_primary_verified": int(attempted.primary_terms_verified.sum())})
    result = {"by_half": summaries, "base_sha256": base_report["base_sha256"],
        "events_sha256": catalog_report["augmented_events_sha256"],
        "potential_sha256": sha(output / "potential_events.parquet"),
        "all_action_support_ceiling_passed": all(r["all_action_state_days"] >= 30
            and r["all_action_capacity_ceiling"] >= 50 for r in summaries),
        "provisional_cash_support_ceiling_passed": all(r["provisional_cash_days"] >= 30
            and r["provisional_top5_rows"] >= 50 for r in summaries),
        "input_gate_passed": False, "primary_verification_complete": False,
        "strategy_returns_read": False, "holdout_read": False}
    save_json(output / "potential_report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", nargs="?", choices=("build", "potential"), default="build")
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    result = {"build": build, "potential": potential}[args.stage](args.output)
    print({k: v for k, v in result.items() if k != "prefix_sha256"})


if __name__ == "__main__":
    main()
