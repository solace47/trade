"""Test one fixed, point-in-time industry-information diffusion hypothesis.

Motivation: Xiang & Lu (2018), https://sysengi.cjoe.ac.cn/CN/Y2018/V38/I4/817,
report within-industry lead-lag effects linked to limited attention. This is
an adaptation: current 14:50 turnover only proxies attention, CSRC industries
only proxy economic peers, and the old paper does not establish 2024-2025
tradability.

Before reading outcomes: among mainboard industries with at least 20 usable
stocks, require the five most active peers' median 14:50 return >=1.5%, at
least four rising, and a median >=1% above the day's market median. Buy up to
five randomly chosen, liquid nonleader peers whose own 14:50 return is near
zero and whose late session is calm. Compare to same-day quiet stocks outside
the winning-industry condition, under the same minimum peer-count rule.
T+5 is primary, T+1/T+2 diagnostic; 2025 is previously viewed, not blind.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import pandas as pd

from .market_study import _quality_keys, _quality_symbols
from .residual_liquidity import _period
from .strategy_scan import _last_safe_entry


CAPACITY = 5
HORIZONS = (1, 2, 5)
MIN_INDUSTRY = 20
MIN_LEADER_RETURN = .015
MIN_LEADER_EXCESS = .01
MIN_POSITIVE_LEADERS = 4


def select(snapshot_dir: Path, intraday_dir: Path, interval_file: Path,
           output: Path) -> tuple[pd.DataFrame, dict]:
    c = duckdb.connect()
    c.execute("SET threads = 4")
    c.read_parquet(str(snapshot_dir / "*.parquet")).create_view("s")
    c.read_parquet(str(intraday_dir / "*.parquet")).create_view("i")
    c.read_parquet(str(interval_file)).create_view("h")
    c.execute("CREATE TEMP VIEW snapshots AS SELECT date FROM s")
    last_entry = {
        year: _last_safe_entry(c, f"{year}-01-01", f"{year + 1}-01-01")
        for year in (2024, 2025)
    }
    c.execute(f"""
        CREATE TEMP TABLE live AS
        SELECT s.date, s.code, h.industry, h.updateDate,
               s.return_1450, s.amount_1450, s.price_1450,
               s.return20_prior_adjusted, s.volume_ratio_est,
               i.return_last30
        FROM s JOIN h ON s.code = h.code
          AND s.date >= h.effective_date
          AND (h.next_effective_date IS NULL
               OR s.date < h.next_effective_date)
        JOIN i ON s.date = i.date AND s.code = i.code
        WHERE ((s.date BETWEEN '2024-01-01' AND '{last_entry[2024]}')
            OR (s.date BETWEEN '2025-01-01' AND '{last_entry[2025]}'))
          AND s.date > h.updateDate
          AND date_diff('day', CAST(h.updateDate AS DATE),
                        CAST(s.date AS DATE)) <= 370
          AND (s.code LIKE 'sh.60%' OR s.code LIKE 'sz.00%')
          AND s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.price_1450 >= 5 AND s.amount_1450 >= 30000000
          AND abs(s.price_1450 - i.price_1450) <= .005
    """)
    if c.execute("""
        SELECT COUNT(*) - COUNT(DISTINCT date || code) FROM live
    """).fetchone()[0]:
        raise ValueError("Point-in-time industry join duplicated a stock-day")
    c.execute("""
        CREATE TEMP TABLE ranked AS
        SELECT *, ROW_NUMBER() OVER (
            PARTITION BY date, industry ORDER BY amount_1450 DESC, code
        ) AS active_peer_rank
        FROM live
    """)
    c.execute("""
        CREATE TEMP TABLE industry_day AS
        SELECT date, industry, COUNT(*) AS industry_count,
               median(return_1450) FILTER (
                   WHERE active_peer_rank <= 5
               ) AS leader_median,
               COUNT(*) FILTER (
                   WHERE active_peer_rank <= 5 AND return_1450 > 0
               ) AS positive_leaders
        FROM ranked GROUP BY date, industry
    """)
    c.execute("""
        CREATE TEMP TABLE market_day AS
        SELECT date, median(return_1450) AS market_median
        FROM live GROUP BY date
    """)
    c.execute("""
        CREATE TEMP TABLE quiet AS
        SELECT r.date, r.code, r.industry, r.return_1450,
               r.amount_1450, r.return_last30, r.updateDate,
               g.industry_count, g.leader_median, g.positive_leaders,
               m.market_median
        FROM ranked r JOIN industry_day g USING (date, industry)
        JOIN market_day m USING (date)
        WHERE r.active_peer_rank > 5
          AND g.industry_count >= 20
          AND r.return_1450 BETWEEN -.005 AND .005
          AND abs(r.return_last30) <= .003
          AND r.amount_1450 BETWEEN 100000000 AND 500000000
          AND r.return20_prior_adjusted BETWEEN -.10 AND .10
          AND r.volume_ratio_est BETWEEN .5 AND 2
    """)
    candidate_condition = f"""
        industry_count >= {MIN_INDUSTRY}
        AND leader_median >= {MIN_LEADER_RETURN}
        AND leader_median - market_median >= {MIN_LEADER_EXCESS}
        AND positive_leaders >= {MIN_POSITIVE_LEADERS}
    """
    selections = []
    for name, condition in (
        ("industry_laggard", candidate_condition),
        ("same_day_quiet_random", f"NOT ({candidate_condition})"),
    ):
        chosen = c.execute(f"""
            SELECT date, code, industry, daily_rank,
                   leader_median, market_median, return_1450,
                   amount_1450, updateDate FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY date
                    ORDER BY md5('industry-v1' || date || code), code
                ) AS daily_rank
                FROM quiet WHERE {condition}
            ) WHERE daily_rank <= {CAPACITY}
        """).df()
        chosen["candidate"] = name
        selections.append(chosen)
    selected = pd.concat(selections, ignore_index=True)
    if selected.duplicated(["candidate", "date", "code"]).any():
        raise ValueError("Duplicate industry selection")
    report = {
        "hypothesis": "Active-peer industry gains diffuse to quiet nonleader stocks",
        "capacity": CAPACITY, "primary_horizon": 5,
        "diagnostic_horizons": [1, 2], "last_entry": last_entry,
        "candidate_condition": " ".join(candidate_condition.split()),
        "quiet_pool_rows": int(c.execute("SELECT COUNT(*) FROM quiet").fetchone()[0]),
        "candidate_pool_rows": int(c.execute(
            f"SELECT COUNT(*) FROM quiet WHERE {candidate_condition}"
        ).fetchone()[0]),
        "note": "Quarterly point-in-time CSRC classes; last known nonempty "
                "class carried forward <=370 days; 2025 is not blind",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output, index=False, compression="zstd")
    return selected, report


def evaluate(selected_path: Path, outcome_dir: Path, issues_dir: Path,
             report_path: Path, trades_path: Path) -> dict:
    selected = pd.read_parquet(selected_path)
    if selected.empty or not selected.date.str[:4].isin(("2024", "2025")).all():
        raise ValueError("Industry selections must be recent and nonempty")
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
               AND NOT EXISTS (SELECT 1 FROM bad_symbols b
                               WHERE b.code = r.code)
               AND NOT EXISTS (
                   SELECT 1 FROM bad_days q
                   WHERE q.code = r.code AND q.date >= r.date
                     AND q.date <= o.exit_date
               ) AS quality_clean_exit
        FROM selected r JOIN o USING (date, code)
        WHERE o.horizon IN (1, 2, 5)
    """).df()
    if len(trades) != len(selected) * len(HORIZONS):
        raise ValueError("A selected industry stock-day lacks an outcome")
    industry_age = (pd.to_datetime(selected.date)
                    - pd.to_datetime(selected.updateDate)).dt.days
    result = {"primary_horizon": 5,
              "control": "same-day quiet mainboard stocks outside "
                         "the winning-industry condition",
              "max_selected_industry_age_days": int(industry_age.max()),
              "results": {}}
    for year in ("2024", "2025"):
        annual = trades.loc[trades.date.str.startswith(year)]
        chosen = annual.loc[annual.candidate.eq("industry_laggard")]
        control = annual.loc[annual.candidate.eq("same_day_quiet_random")]
        result["results"][year] = {}
        for horizon in HORIZONS:
            sample = chosen.loc[chosen.horizon.eq(horizon)]
            reference = control.loc[control.horizon.eq(horizon)]
            result["results"][year][str(horizon)] = {}
            for period, rows in (
                ("H1", sample.loc[sample.date.str[5:7].astype(int).le(6)]),
                ("H2", sample.loc[sample.date.str[5:7].astype(int).gt(6)]),
                ("full", sample),
            ):
                if not rows.empty:
                    summary = _period(rows, reference)
                    summary["per_signal_cash_mean"] = float(rows.net_return.where(
                        rows.exit_status.eq("filled") & rows.quality_clean_exit,
                        0,
                    ).mean())
                    result["results"][year][str(horizon)][period] = summary
    report_path.parent.mkdir(parents=True, exist_ok=True)
    trades_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    trades.to_parquet(trades_path, index=False, compression="zstd")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--features", type=Path,
                        default=Path("data/research/intraday_features"))
    parser.add_argument("--intervals", type=Path,
                        default=Path("data/research/industry_intervals.parquet"))
    parser.add_argument("--selected", type=Path,
                        default=Path("data/research/industry_diffusion_selected.parquet"))
    parser.add_argument("--outcomes", type=Path,
                        default=Path("data/research/market_outcomes_ci"))
    parser.add_argument("--issues", type=Path,
                        default=Path("data/research/market_issues_ci"))
    parser.add_argument("--report", type=Path,
                        default=Path("data/research/industry_diffusion_report.json"))
    parser.add_argument("--trades", type=Path,
                        default=Path("data/research/industry_diffusion_trades.parquet"))
    parser.add_argument("--select-only", action="store_true")
    args = parser.parse_args()
    selected, prereg = select(args.snapshots, args.features, args.intervals,
                             args.selected)
    print({"selection": prereg, "rows": len(selected)}, flush=True)
    if not args.select_only:
        outcome = evaluate(args.selected, args.outcomes, args.issues,
                           args.report, args.trades)
        print({"report": str(args.report),
               "years": list(outcome["results"])}, flush=True)


if __name__ == "__main__":
    main()
