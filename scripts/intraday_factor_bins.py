"""Describe 2024 late-session factor bins before making another screen.

This is exploratory research, not a rule selector. It does not open 2025.
"""

import json
from pathlib import Path

import duckdb
import pandas as pd

from trade_research.market_study import _quality_keys, _quality_symbols, _week_bootstrap


ROOT = Path("data/research")
FACTORS = {
    "late_return": """
        CASE WHEN return_last30 < -.01 THEN 'below_minus_1pct'
             WHEN return_last30 < -.003 THEN 'minus_1_to_minus_0_3'
             WHEN return_last30 <= .003 THEN 'near_zero'
             WHEN return_last30 <= .01 THEN 'plus_0_3_to_1'
             ELSE 'above_plus_1pct' END
    """,
    "late_volume": """
        CASE WHEN volume_share_last30 < .08 THEN 'below_8pct'
             WHEN volume_share_last30 < .12 THEN '8_to_12pct'
             WHEN volume_share_last30 < .18 THEN '12_to_18pct'
             ELSE 'at_least_18pct' END
    """,
    "overnight_gap": """
        CASE WHEN overnight_gap < -.01 THEN 'below_minus_1pct'
             WHEN overnight_gap < 0 THEN 'minus_1_to_0'
             WHEN overnight_gap < .01 THEN '0_to_plus_1pct'
             ELSE 'at_least_plus_1pct' END
    """,
    "cutoff_return": """
        CASE WHEN return_1450 < 0 THEN 'loss'
             WHEN return_1450 < .015 THEN '0_to_1_5pct'
             WHEN return_1450 < .03 THEN '1_5_to_3pct'
             WHEN return_1450 < .05 THEN '3_to_5pct'
             ELSE 'at_least_5pct' END
    """,
    "prior_twenty": """
        CASE WHEN return20_prior_adjusted < -.1 THEN 'below_minus_10pct'
             WHEN return20_prior_adjusted < 0 THEN 'minus_10_to_0'
             WHEN return20_prior_adjusted < .1 THEN '0_to_plus_10pct'
             ELSE 'at_least_plus_10pct' END
    """,
    "amount": """
        CASE WHEN amount_1450 < 300000000 THEN '100m_to_300m'
             WHEN amount_1450 < 600000000 THEN '300m_to_600m'
             ELSE '600m_to_1b' END
    """,
}


def main() -> None:
    c = duckdb.connect()
    c.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")).create_view("s")
    c.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")).create_view("o")
    c.read_parquet(str(ROOT / "intraday_features" / "2024.parquet")).create_view("i")
    c.register("bad_days", _quality_keys(ROOT / "market_issues_ci"))
    c.register("bad_symbols", _quality_symbols(ROOT / "market_issues_ci"))
    c.execute("""
        CREATE TEMP TABLE base AS
        SELECT s.date, s.code, s.return_1450, s.return20_prior_adjusted,
               s.amount_1450, i.return_last30, i.volume_share_last30,
               s.open_1450 / NULLIF(s.preclose, 0) - 1 AS overnight_gap,
               o.horizon, o.entry_status, o.exit_status,
               o.exit_delay_sessions, o.net_return,
               NOT EXISTS (SELECT 1 FROM bad_symbols b
                           WHERE b.code = s.code)
               AND NOT EXISTS (
                   SELECT 1 FROM bad_days q WHERE q.code = s.code
                     AND q.date >= s.date AND q.date <= o.exit_date
               ) AS quality_clean_exit
        FROM s JOIN i USING (date, code) JOIN o USING (date, code)
        WHERE s.date BETWEEN '2024-01-01' AND '2024-12-17'
          AND o.horizon IN (1, 5)
          AND s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.amount_1450 BETWEEN 100000000 AND 1000000000
    """)
    output = []
    for name, expression in FACTORS.items():
        by_date = c.execute(f"""
            SELECT date, horizon, {expression} AS bucket,
                   COUNT(*) AS signals,
                   COUNT(*) FILTER (WHERE entry_status = 'filled') AS entries,
                   COUNT(*) FILTER (WHERE exit_status = 'filled'
                       AND quality_clean_exit) AS clean_exits,
                   COUNT(*) FILTER (WHERE exit_status = 'filled'
                       AND quality_clean_exit AND exit_delay_sessions = 0)
                       AS ontime_exits,
                   COALESCE(SUM(net_return) FILTER (
                       WHERE exit_status = 'filled' AND quality_clean_exit
                   ), 0) AS net_sum
            FROM base GROUP BY date, horizon, bucket
        """).df()
        by_date["half"] = by_date.date.str[5:7].astype(int).le(6).map(
            {True: "H1", False: "H2"}
        )
        by_date["cash_return"] = by_date.net_sum / by_date.signals
        for (half, horizon, bucket), frame in by_date.groupby(
            ["half", "horizon", "bucket"]
        ):
            if len(frame) < 20:
                continue
            ci = _week_bootstrap(frame.cash_return, frame.date, 20260924)
            output.append({
                "factor": name, "half": half, "horizon": int(horizon),
                "bucket": bucket, "days": len(frame),
                "signals": int(frame.signals.sum()),
                "entry_rate": float(frame.entries.sum() / frame.signals.sum()),
                "ontime_rate": float(frame.ontime_exits.sum()
                                     / max(frame.clean_exits.sum(), 1)),
                "cash_mean": float(frame.cash_return.mean()),
                "cash_week_ci": ci,
            })
    target = ROOT / "intraday_factor_bins_2024.json"
    target.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print({"rows": len(output), "output": str(target)})


if __name__ == "__main__":
    main()
