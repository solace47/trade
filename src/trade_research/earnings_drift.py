"""Pre-register a 14:50 test of delayed reaction to positive forecasts.

Motivation: Lan et al. (2024), https://doi.org/10.1016/j.irfa.2024.103460,
use the opening reaction to earnings announcements as a surprise proxy. Ding
et al. (2021), https://xbbjb.cufe.edu.cn/CN/Y2021/V0/I6/27, report attention
effects and a preliminary-announcement robustness check in older A-share data.
The BaoStock forecast type is only an adaptation, not their earnings-surprise
measure, and neither paper establishes a recent 14:50 trading return.

Before looking at outcomes: on the first trading day strictly after a positive
forecast was published, require an opening gain >=1% and no reversal by 14:50.
Rank by opening gain and take at most five liquid mainboard stocks per day.
Compare to unique same-day stocks without a forecast or express event, matched
on opening gain, 14:50 gain, turnover, and prior 20-session return. T+5 is
primary, T+1/T+2 diagnostic. 2025 has already been viewed in other studies,
so it is exploratory rather than blind; no 2026 outcomes enter this test.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .earnings_events import BATCH_SIZE, _validate, stock_codes
from .strategy_scan import _last_safe_entry


POSITIVE_TYPES = frozenset({"预增", "略增", "扭亏", "减亏", "续盈"})
CAPACITY = 5
FIRST_SIGNAL = "2024-01-01"
LAST_SIGNAL = "2025-12-31"


def load_complete_events(event_dir: Path, stock_basic: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reject a partial universe instead of silently testing available stocks."""
    codes = stock_codes(stock_basic)
    frames: dict[str, list[pd.DataFrame]] = {"forecast": [], "express": []}
    for index, offset in enumerate(range(0, len(codes), BATCH_SIZE)):
        batch = codes[offset:offset + BATCH_SIZE]
        manifest = event_dir / f"batch_{index:03d}.json"
        if not manifest.exists():
            raise FileNotFoundError(f"Missing earnings batch {index}")
        saved = json.loads(manifest.read_text(encoding="utf-8"))
        if saved.get("codes") != batch:
            raise ValueError(f"Earnings batch {index} has a different universe")
        for kind in frames:
            path = event_dir / f"{kind}_{index:03d}.parquet"
            frame = _validate(pd.read_parquet(path), kind, set(batch))
            if len(frame) != saved.get("rows", {}).get(kind):
                raise ValueError(f"Earnings batch {index} {kind} count differs")
            frames[kind].append(frame)
    return tuple(pd.concat(frames[kind], ignore_index=True) for kind in frames)  # type: ignore[return-value]


def next_session(events: pd.DataFrame, available_column: str,
                 sessions: np.ndarray) -> pd.DataFrame:
    """Date-only disclosures are traded no earlier than the next session."""
    source = events.loc[events[available_column].between(FIRST_SIGNAL, LAST_SIGNAL)].copy()
    if source.empty:
        return source.assign(date=pd.Series(dtype="str"))
    index = np.searchsorted(sessions, source[available_column].to_numpy(), side="right")
    source = source.loc[index < len(sessions)].copy()
    source["date"] = sessions[index[index < len(sessions)]]
    if not source.date.gt(source[available_column]).all():
        raise ValueError("Announcement joined to a session before it was public")
    return source


def event_pairs(forecast: pd.DataFrame, express: pd.DataFrame,
                sessions: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    forecasts = next_session(forecast, "profitForcastExpPubDate", sessions)
    expresses = next_session(express, "performanceExpUpdateDate", sessions)
    forecasts["positive"] = forecasts.profitForcastType.isin(POSITIVE_TYPES)
    grouped = forecasts.groupby(["date", "code"], as_index=False).agg(
        positive=("positive", "any"),
        count=("positive", "size"),
        positive_count=("positive", "sum"),
        event_pub_date=("profitForcastExpPubDate", "max"),
    )
    positive = grouped.loc[grouped.positive & grouped.positive_count.eq(grouped["count"]),
                           ["date", "code", "event_pub_date"]].copy()
    all_events = pd.concat((forecasts[["date", "code"]], expresses[["date", "code"]]),
                           ignore_index=True).drop_duplicates()
    if positive.duplicated(["date", "code"]).any() or all_events.duplicated(["date", "code"]).any():
        raise ValueError("Duplicate earnings event stock-day")
    return positive, all_events


def match_controls(candidates: pd.DataFrame, pool: pd.DataFrame) -> pd.DataFrame:
    """Greedy, unique same-day controls using fixed economically scaled distances."""
    controls = []
    outside = {date: rows.sort_values("code").reset_index(drop=True)
               for date, rows in pool.loc[~pool.has_event].groupby("date")}
    for date, selected in candidates.groupby("date", sort=True):
        choices = outside.get(date)
        if choices is None or len(choices) < len(selected):
            raise ValueError(f"No sufficient same-day earnings controls on {date}")
        choices = choices.copy()
        for signal in selected.itertuples():
            distance = (
                (choices.open_gap - signal.open_gap).abs() / .01
                + (choices.return_1450 - signal.return_1450).abs() / .02
                + np.abs(np.log(choices.amount_1450 / signal.amount_1450)) / .7
                + (choices.return20_prior_adjusted
                   - signal.return20_prior_adjusted).abs() / .10
            )
            nearest = int(np.argmin(distance.to_numpy()))
            row = choices.iloc[nearest].copy()
            row["pair_code"] = signal.code
            row["match_distance"] = float(distance.iloc[nearest])
            controls.append(row)
            choices = choices.drop(choices.index[nearest]).reset_index(drop=True)
    return pd.DataFrame(controls)


def select(snapshot_dir: Path, event_dir: Path, stock_basic: Path,
           output: Path) -> tuple[pd.DataFrame, dict]:
    forecast, express = load_complete_events(event_dir, stock_basic)
    c = duckdb.connect()
    c.execute("SET threads = 4")
    c.read_parquet(str(snapshot_dir / "*.parquet")).create_view("snapshots")
    sessions = np.array([row[0] for row in c.execute("""
        SELECT DISTINCT date FROM snapshots
        WHERE date BETWEEN '2024-01-01' AND '2025-12-31' ORDER BY date
    """).fetchall()], dtype="str")
    if not len(sessions) or not np.all(sessions[:-1] < sessions[1:]):
        raise ValueError("Missing or unsorted recent trading calendar")
    last_entry = {year: _last_safe_entry(c, f"{year}-01-01", f"{year+1}-01-01")
                  for year in (2024, 2025)}
    positive, all_events = event_pairs(forecast, express, sessions)
    positive = positive.loc[
        (positive.date.str.startswith("2024") & positive.date.le(last_entry[2024]))
        | (positive.date.str.startswith("2025") & positive.date.le(last_entry[2025]))
    ].copy()
    if positive.empty:
        raise ValueError("No positive forecast events in the permitted entry window")
    c.register("positive_events", positive)
    c.register("all_events", all_events)
    c.register("event_days", positive[["date"]].drop_duplicates())
    pool = c.execute("""
        SELECT s.date, s.code, s.price_1450, s.open_1450, s.preclose,
               s.return_1450, s.amount_1450, s.return20_prior_adjusted,
               s.open_1450 / s.preclose - 1 AS open_gap,
               p.event_pub_date, a.code IS NOT NULL AS has_event
        FROM snapshots s JOIN event_days d USING (date)
        LEFT JOIN positive_events p USING (date, code)
        LEFT JOIN all_events a USING (date, code)
        WHERE (s.code LIKE 'sh.60%' OR s.code LIKE 'sz.00%')
          AND s.isST = 0 AND s.listing_age_sessions >= 20
          AND NOT s.reference_gap AND NOT s.quote_outside_traded_range
          AND s.preclose > 0 AND s.open_1450 > 0
          AND s.open_1450 BETWEEN s.low_1450 - .005 AND s.high_1450 + .005
          AND s.price_1450 >= 5 AND s.amount_1450 >= 30000000
          AND s.return20_prior_adjusted IS NOT NULL
          AND s.open_1450 / s.preclose - 1 >= .01
          AND s.price_1450 >= s.open_1450
    """).df()
    if pool.duplicated(["date", "code"]).any():
        raise ValueError("Snapshot/event join duplicated a stock-day")
    candidates = (pool.loc[pool.event_pub_date.notna()]
                  .sort_values(["date", "open_gap", "code"],
                               ascending=[True, False, True])
                  .groupby("date", sort=False).head(CAPACITY).copy())
    if candidates.empty:
        raise ValueError("No event reactions passed the frozen selection rule")
    candidates["pair_code"] = candidates.code
    candidates["match_distance"] = np.nan
    candidates["candidate"] = "positive_forecast_reaction"
    controls = match_controls(candidates, pool)
    controls["candidate"] = "same_day_matched_nonannouncement"
    selected = pd.concat((candidates, controls), ignore_index=True)
    if len(controls) != len(candidates) or selected.duplicated(["candidate", "date", "code"]).any():
        raise ValueError("Earnings selection or matching is not one-to-one")
    if not candidates.date.gt(candidates.event_pub_date).all():
        raise ValueError("A selected forecast was not public before the signal day")
    report = {
        "hypothesis": "Positive forecast with positive opening response drifts after 14:50",
        "source": "BaoStock historical forecast publication dates; express excludes controls",
        "rule": "positive forecast type, first later trading day, mainboard non-ST, "
                "age>=20, price>=5, amount>=30m, opening gain>=1%, no 14:50 fade; "
                "rank opening gain descending, maximum five per day",
        "control": "unique same-day nonannouncement stocks with identical eligibility; "
                   "nearest by |gap|/.01 + |14:50 return|/.02 + "
                   "|log(amount ratio)|/.7 + |prior20 return|/.10",
        "primary_horizon": 5, "diagnostic_horizons": [1, 2],
        "last_entry": last_entry, "eligible_event_pairs": len(positive),
        "selected_stock_days": len(candidates),
        "selected_days": int(candidates.date.nunique()),
        "max_control_distance": float(controls.match_distance.max()),
        "note": "2025 exploratory, not blind; 2026 untouched",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(output, index=False, compression="zstd")
    return selected, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshots", type=Path,
                        default=Path("data/research/market_snapshots_ci"))
    parser.add_argument("--events", type=Path,
                        default=Path("data/research/earnings_events"))
    parser.add_argument("--stock-basic", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/stock_basic.parquet"))
    parser.add_argument("--selected", type=Path,
                        default=Path("data/research/earnings_drift_selected.parquet"))
    args = parser.parse_args()
    selected, report = select(args.snapshots, args.events,
                              args.stock_basic, args.selected)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
