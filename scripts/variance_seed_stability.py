"""Post-hoc sensitivity to fixed, alternative daily hash rankings.

Always report seeds 0–9 together. Never choose a seed from its return.
"""

import json
from pathlib import Path

import duckdb
import pandas as pd

from trade_research.late_variance_risk import ROOT, select_inputs
from trade_research.market_study import _quality_keys, _quality_symbols


OUTPUT = ROOT / "late_variance_risk" / "seed_stability.json"
SEEDS = tuple(range(10))


def scored_candidates() -> pd.DataFrame:
    connection = duckdb.connect()
    connection.read_parquet(str(ROOT / "late_variance_risk" /
                                "all_candidates.parquet")).create_view("candidates")
    connection.read_parquet(str(ROOT / "market_outcomes_ci" / "*.parquet")
                            ).create_view("outcomes")
    connection.register("bad_days", _quality_keys(ROOT / "market_issues_ci"))
    connection.register("bad_symbols", _quality_symbols(ROOT / "market_issues_ci"))
    return connection.execute("""
        SELECT x.date, x.code,
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
        WHERE o.horizon = 1
          AND (x.surprise >= 1.5 OR x.surprise <= 1.0)
    """).df()


def pool_edges(cash: pd.DataFrame, selected_dates: set[str]) -> dict:
    """Compare all eligible stocks on all dates and on frozen pair dates."""
    daily = cash.groupby(["date", "variance_group"]).cash.mean().unstack()
    daily = daily.dropna(subset=["high", "low"])
    daily["edge"] = daily.low - daily.high
    dates = daily.index.to_series()
    report = {}
    for year in ("2024", "2025"):
        for half, condition in (("H1", dates.str[5:7].astype(int).le(6)),
                                ("H2", dates.str[5:7].astype(int).gt(6))):
            period = daily.loc[dates.str.startswith(year) & condition]
            paired_dates = period.loc[period.index.isin(selected_dates)]
            if period.empty or paired_dates.empty:
                raise ValueError("Missing full-pool or frozen-pair dates")
            report[f"{year}{half}"] = {
                "all_dates": len(period),
                "all_dates_edge": float(period.edge.mean()),
                "paired_dates": len(paired_dates),
                "paired_dates_pool_edge": float(paired_dates.edge.mean()),
            }
    return report


def one_seed(seed: int, cash: pd.DataFrame) -> dict:
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    connection.read_parquet(str(ROOT / "market_snapshots_ci" / "*.parquet")
                            ).create_view("snapshots")
    connection.read_parquet(str(ROOT / "intraday_features" / "*.parquet")
                            ).create_view("intraday")
    connection.read_parquet(str(ROOT / "late_variance" / "*.parquet")
                            ).create_view("variance")
    selected, audit = select_inputs(
        connection, rank_salt=f"rv-risk-stability-{seed}",
    )
    labelled = selected[["date", "code", "pair_id", "candidate"]].merge(
        cash, on=["date", "code"], validate="one_to_one",
    )
    if len(labelled) != len(selected):
        raise ValueError("Alternative ranking lost an archived outcome")
    if not ((labelled.candidate.eq("high_variance")
             & labelled.variance_group.eq("high"))
            | (labelled.candidate.eq("normal_variance_control")
               & labelled.variance_group.eq("low"))).all():
        raise ValueError("Alternative ranking has inconsistent labels")
    high = labelled.loc[labelled.candidate.eq("high_variance")]
    low = labelled.loc[labelled.candidate.eq("normal_variance_control")]
    pairs = high.merge(low, on=["date", "pair_id"],
                       suffixes=("_high", "_low"), validate="one_to_one")
    pairs["edge"] = pairs.cash_low - pairs.cash_high
    daily = pairs.groupby("date", sort=True).edge.mean().reset_index()
    result = {"seed": seed, "pairs": audit["paired"],
              "match_fraction": audit["match_fraction"], "periods": {}}
    for year in ("2024", "2025"):
        for half, condition in (
            ("H1", daily.date.str[5:7].astype(int).le(6)),
            ("H2", daily.date.str[5:7].astype(int).gt(6)),
        ):
            frame = daily.loc[daily.date.str.startswith(year) & condition]
            if frame.empty:
                raise ValueError("Alternative ranking lacks a half-year")
            result["periods"][f"{year}{half}"] = {
                "days": len(frame), "low_minus_high": float(frame.edge.mean()),
            }
    return result


def main() -> None:
    cash = scored_candidates()
    if cash.duplicated(["date", "code"]).any() or cash.cash.isna().any():
        raise ValueError("Candidate outcomes are incomplete or duplicated")
    frozen = pd.read_parquet(ROOT / "late_variance_risk" / "selections.parquet")
    date_context = pool_edges(cash, set(frozen.date))
    runs = [one_seed(seed, cash) for seed in SEEDS]
    periods = {}
    for period in ("2024H1", "2024H2", "2025H1", "2025H2"):
        values = [run["periods"][period]["low_minus_high"] for run in runs]
        periods[period] = {
            "positive_seeds": sum(value > 0 for value in values),
            "min": min(values), "median": float(pd.Series(values).median()),
            "max": max(values),
        }
    report = {
        "scope": "post-hoc random-ranking sensitivity, archived 100k T+1",
        "seed_set": list(SEEDS), "runs": runs, "summary": periods,
        "pool_date_context": date_context,
        "note": "All seeds reported; 2025 not blind; no rule chosen from seeds",
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                      encoding="utf-8")
    print({"output": str(OUTPUT), "summary": periods})


if __name__ == "__main__":
    main()
