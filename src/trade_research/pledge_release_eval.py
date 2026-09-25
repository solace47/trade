"""Score frozen pledge-release pairs from original 2024/2025 minute trades."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .annual_cash_direct_eval import _month_bootstrap
from .buyback_eval import _summary
from .buyback_inputs import CONTROL
from .hf_outcomes import Assumptions
from .pledge_release_inputs import EVENT
from .strategy_scan import _stressed_returns


def _check_pairs(pairs: pd.DataFrame, expected: dict) -> None:
    if (pairs.empty or len(pairs) != 2 * expected["matched_pairs"]
            or pairs.duplicated(["date", "code"]).any()
            or set(pairs.candidate) != {EVENT, CONTROL}
            or not pairs.date.str[:4].isin(("2024", "2025")).all()):
        raise ValueError("Frozen pledge pair membership changed")
    group = pairs.groupby(["date", "pair_code"]).candidate.agg(
        ["size", "nunique"])
    if (len(group) != expected["matched_pairs"]
            or not group["size"].eq(2).all()
            or not group["nunique"].eq(2).all()
            or pairs.planned_repledge.isna().any()
            or not pairs.planned_repledge.isin(("yes", "no")).all()
            or not pairs.groupby(["date", "pair_code"])
            .planned_repledge.nunique().eq(1).all()):
        raise ValueError("Pledge pair lost a control or source plan flag")
    events = pairs.loc[pairs.candidate.eq(EVENT)]
    if (not events.date.gt(events.notice_date).all()
            or events.groupby("date").size().gt(5).any()
            or events.date.nunique() != expected["matched_days"]):
        raise ValueError("Pledge pair violates disclosure timing or capacity")
    for year in ("2024", "2025"):
        sample = events.loc[events.date.str.startswith(year)]
        if (len(sample) != expected["by_year"][year]["pairs"]
                or sample.date.nunique() != expected["by_year"][year]["days"]):
            raise ValueError("Pledge pair list changed after input audit")


def _scored(raw: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = raw.merge(pairs[["date", "code", "candidate", "pair_code",
                            "planned_repledge"]], on=["date", "code"],
                     validate="many_to_one")
    if (len(rows) != len(pairs) * 4
            or rows.quality_clean_exit.isna().any()):
        raise ValueError("Frozen pledge pair lacks original-minute trade")
    rows["cash_return"] = np.where(
        rows.quality_clean_exit, rows.net_return, 0.0)
    rows["cash_stress10"] = 0.0
    clean = rows.quality_clean_exit
    if not np.allclose(_stressed_returns(
            rows.loc[clean], Assumptions().slippage_bps_each_side),
            rows.loc[clean, "net_return"], atol=1e-12):
        raise ValueError("Pledge raw net return disagrees with cost model")
    rows.loc[clean, "cash_stress10"] = _stressed_returns(
        rows.loc[clean], Assumptions().slippage_bps_each_side + 10)
    if not np.isfinite(rows[["cash_return", "cash_stress10"]].to_numpy()).all():
        raise ValueError("Nonfinite pledge cash return")
    return rows


def _same_events(main: pd.DataFrame, industry: pd.DataFrame,
                 year: str) -> dict:
    def wide(rows: pd.DataFrame) -> pd.DataFrame:
        sample = rows.loc[rows.date.str.startswith(year)
                          & rows.target_notional.eq(20_000)
                          & rows.horizon.eq(5)]
        return sample.pivot(index=["date", "pair_code"],
                            columns="candidate", values="cash_return")

    common = wide(main).join(wide(industry), how="inner",
                             lsuffix="_main", rsuffix="_industry")
    if common.empty or not np.allclose(
            common[f"{EVENT}_main"], common[f"{EVENT}_industry"]):
        raise ValueError("Industry diagnostic changed an event trade")
    common["main_edge"] = common[f"{EVENT}_main"] - common[f"{CONTROL}_main"]
    common["industry_edge"] = (common[f"{EVENT}_industry"]
                               - common[f"{CONTROL}_industry"])
    daily = common.groupby(level="date")[["main_edge", "industry_edge"]].mean()
    return {"pairs": len(common), "days": len(daily),
            "main_edge_mean": float(daily.main_edge.mean()),
            "industry_edge_mean": float(daily.industry_edge.mean()),
            "industry_edge_month_ci": _month_bootstrap(
                daily.industry_edge.reset_index(drop=True),
                pd.Series(daily.index.to_list()), 157)}


def _tail_diagnostic(rows: pd.DataFrame, year: str) -> dict:
    """Describe concentration after the frozen mean test; do not tune on it."""
    sample = rows.loc[rows.date.str.startswith(year)
                      & rows.target_notional.eq(20_000)
                      & rows.horizon.eq(5)]
    paired = sample.pivot(index=["date", "pair_code"], columns="candidate",
                          values="cash_return")
    spread = paired[EVENT] - paired[CONTROL]
    daily = spread.groupby(level="date").mean()
    if daily.empty or spread.isna().any():
        raise ValueError("Incomplete pledge tail diagnostic pairs")
    month = pd.Series(daily.index.str[:7], index=daily.index)
    month_mean = daily.groupby(month).mean()
    top_month = str(month_mean.idxmax())
    leave_one_out = {
        str(item): float(daily.loc[month.ne(item)].mean())
        for item in month_mean.index
    }
    return {
        "pair_median": float(spread.median()),
        "day_median": float(daily.median()),
        "positive_day_share": float(daily.gt(0).mean()),
        "highest_mean_month": top_month,
        "highest_mean_month_days": int(month.eq(top_month).sum()),
        "without_highest_mean_month": leave_one_out[top_month],
        "leave_one_month_out_range": [min(leave_one_out.values()),
                                      max(leave_one_out.values())],
    }


def _complete_pair_diagnostic(rows: pd.DataFrame, year: str) -> dict:
    """Check whether zero-valued unclean exits drive the frozen estimate."""
    sample = rows.loc[rows.date.str.startswith(year)
                      & rows.target_notional.eq(20_000)
                      & rows.horizon.eq(5)].copy()
    complete = sample.groupby(["date", "pair_code"])
    eligible = complete.quality_clean_exit.transform("all")
    return _summary(sample.loc[eligible], year, "full", EVENT)


def evaluate(root: Path, raw_path: Path, report_path: Path) -> dict:
    audit = json.loads((root / "input_audit.json").read_text(encoding="utf-8"))
    if (audit.get("outcomes_opened") is not False
            or not all(audit["coverage"][year]["input_gate_passed"]
                       for year in ("2024", "2025"))):
        raise ValueError("Pledge input gate was not frozen before outcomes")
    membership = {stem: pd.read_parquet(root / f"{stem}.parquet")
                  for stem in ("pairs", "industry_pairs")}
    _check_pairs(membership["pairs"], audit["main"])
    _check_pairs(membership["industry_pairs"], audit["same_industry"])
    raw = pd.read_parquet(raw_path)
    keys = pd.concat([p[["date", "code"]] for p in membership.values()]
                     ).drop_duplicates()
    settled = raw.loc[raw.exit_date.notna()]
    raw_keys = set(map(tuple, raw[["date", "code"]].itertuples(index=False,
                                                              name=None)))
    frozen_keys = set(map(tuple, keys.itertuples(index=False, name=None)))
    if (len(raw) != len(keys) * 4
            or raw_keys != frozen_keys
            or set(raw.target_notional) != {20_000.0, 100_000.0}
            or set(raw.horizon) != {1, 5}
            or not raw.exit_window.eq("close").all()
            or raw.duplicated(
                ["date", "code", "target_notional", "horizon"]).any()
            or not raw.date.str[:4].isin(("2024", "2025")).all()
            or not settled.date.str[:4].eq(
                settled.exit_date.str[:4]).all()):
        raise ValueError("Incomplete or out-of-window pledge minute outcomes")
    scored = {stem: _scored(raw, pairs)
              for stem, pairs in membership.items()}
    report = {"note": "2024/2025 exploratory; 2025 previously viewed, 2026 unopened",
              "raw_stock_days": len(keys), "results": {},
              "without_planned_repledge": {},
              "same_event_industry": {}, "posthoc_tail_diagnostic": {},
              "posthoc_complete_pairs": {}}
    for stem, rows in scored.items():
        report["results"][stem] = {}
        for size in (20_000, 100_000):
            report["results"][stem][str(size)] = {}
            for horizon in (1, 5):
                frame = rows.loc[rows.target_notional.eq(size)
                                 & rows.horizon.eq(horizon)]
                report["results"][stem][str(size)][str(horizon)] = {
                    year: {segment: _summary(frame, year, segment, EVENT)
                           for segment in ("full", "H1", "H2")}
                    for year in ("2024", "2025")
                }
    primary = scored["pairs"]
    no_repledge = primary.loc[primary.planned_repledge.eq("no")
                             & primary.target_notional.eq(20_000)
                             & primary.horizon.eq(5)]
    report["without_planned_repledge"] = {
        year: _summary(no_repledge, year, "full", EVENT)
        for year in ("2024", "2025")
    }
    report["same_event_industry"] = {
        year: _same_events(primary, scored["industry_pairs"], year)
        for year in ("2024", "2025")
    }
    report["posthoc_tail_diagnostic"] = {
        year: _tail_diagnostic(primary, year)
        for year in ("2024", "2025")
    }
    report["posthoc_complete_pairs"] = {
        year: _complete_pair_diagnostic(primary, year)
        for year in ("2024", "2025")
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/pledge"))
    parser.add_argument("--raw", type=Path, default=Path(
        "data/research/pledge/repriced.parquet"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/pledge/reprice_report.json"))
    args = parser.parse_args()
    report = evaluate(args.source, args.raw, args.report)
    print({"main_20k_t5": {
        year: report["results"]["pairs"]["20000"]["5"][year]["full"]
        for year in ("2024", "2025")},
        "without_repledge": report["without_planned_repledge"],
        "same_event_industry": report["same_event_industry"]})


if __name__ == "__main__":
    main()
