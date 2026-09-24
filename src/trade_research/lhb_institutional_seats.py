"""Audit SSE institution-only trading seats without opening future returns.

The SSE publishes top-five branch amounts only after the trading session.
This feasibility audit uses them on the next session and checks whether
positive institution-seat net amounts have comparable listed peers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .buyback_inputs import _capacity_events, _universe
from .exchange_public_events import _check_saved_sse, trading_dates
from .margin_heterogeneity import _match_one


DAILY_REASONS = frozenset(("11", "12", "13", "14"))
MAX_CURRENT_GAP = .01


def _seat_amount(row: dict, side: str) -> float:
    names = row.get(f"branchName{side}")
    values = row.get(f"branchTxAmt{side}")
    if not isinstance(names, str) or not isinstance(values, str):
        raise ValueError("SSE top-five seat list is missing")
    branches = names.split(",")
    amounts = values.split(",")
    if (not 1 <= len(branches) <= 5 or len(branches) != len(amounts)
            or not all(branches)):
        raise ValueError("SSE seat names and values are misaligned")
    parsed = np.array([float(value) for value in amounts])
    if not np.isfinite(parsed).all() or (parsed < 0).any():
        raise ValueError("SSE seat amount is invalid")
    return float(sum(value for name, value in zip(branches, parsed)
                     if name.strip() == "机构专用"))


def _source(calendar_path: Path, sse_dir: Path) -> tuple[pd.DataFrame, dict]:
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    next_day = dict(zip(calendar[:-1], calendar[1:]))
    rows = []
    archived_days = 0
    for trade_day in calendar:
        if trade_day > "2025-12-31":
            break
        record = _check_saved_sse(
            sse_dir / f"sse_{trade_day.replace('-', '')}.json",
            trade_day.replace("-", ""))
        archived_days += 1
        if trade_day not in next_day:
            continue
        signal_day = next_day[trade_day]
        if (signal_day[:4] not in ("2024", "2025")
                or signal_day[5:] > "12-17"):
            continue
        for item in record["main"]:
            if (item.get("secType") != "A"
                    or item.get("refType") not in DAILY_REASONS):
                continue
            amount = float(item["secTxAmount"])
            if not np.isfinite(amount) or amount <= 0:
                raise ValueError("SSE disclosed stock turnover is invalid")
            buy = _seat_amount(item, "B")
            sell = _seat_amount(item, "S")
            rows.append({
                "date": signal_day, "trade_date": trade_day,
                "code": "sh." + item["secCode"],
                "ref_type": item["refType"],
                "disclosed_turnover": amount,
                "institution_buy": buy, "institution_sell": sell,
                "institution_net": buy - sell,
            })
    frame = pd.DataFrame(rows)
    if frame.empty or frame.duplicated(["date", "code", "ref_type"]).any():
        raise ValueError("SSE institutional seat keys are empty or duplicated")
    ambiguous = frame.duplicated(["date", "code"], keep=False)
    result = frame.loc[~ambiguous].copy()
    audit = {"archived_trade_days": archived_days,
             "daily_reason_rows": len(frame),
             "multi_reason_rows_excluded": int(ambiguous.sum()),
             "unique_reason_stock_days": len(result),
             "positive_net_rows": int(result.institution_net.gt(0).sum())}
    return result, audit


def audit(calendar_path: Path, sse_dir: Path, snapshot_dir: Path,
          daily_dir: Path, industry_path: Path, report_path: Path) -> dict:
    notices, source = _source(calendar_path, sse_dir)
    calendar = trading_dates(calendar_path, "2024-01-01", "2026-01-15")
    universe = _universe(snapshot_dir, daily_dir, industry_path,
                         notices, calendar)
    listed = universe.merge(notices, on=["date", "trade_date", "code"],
                            validate="one_to_one")
    positive = listed.loc[listed.institution_net.gt(0)]
    selected = _capacity_events(positive, calendar)
    controls = listed.loc[listed.institution_net.le(0)]
    controls_by_date = {day: group for day, group in controls.groupby("date")}
    matches = []
    for day, group in selected.groupby("date", sort=True):
        available = controls_by_date.get(day, pd.DataFrame()).copy()
        for _, signal in group.iterrows():
            if available.empty:
                continue
            choices = available.loc[
                available.ref_type.eq(signal.ref_type)
                & available.board.eq(signal.board)
                & available.size_bucket.eq(signal.size_bucket)
                & available.momentum_bucket.eq(signal.momentum_bucket)
                & (available.return_1450 - signal.return_1450).abs().le(
                    MAX_CURRENT_GAP)
            ]
            match = _match_one(signal, choices)
            if match is None:
                continue
            peer, _ = match
            matches.append({"date": day, "reason": signal.ref_type})
            available = available.loc[available.code.ne(peer.code)]
    matched = pd.DataFrame(matches, columns=["date", "reason"])
    report = {
        "source": source, "eligible_listed_stock_days": len(listed),
        "eligible_positive_net": len(positive),
        "capacity_selected": len(selected),
        "strict_same_day_pairs": len(matched),
        "pairs_by_year": {year: int(matched.date.str.startswith(year).sum())
                          for year in ("2024", "2025")},
        "note": "Input feasibility only; no future returns opened. "
                "Institution-only top-five seats are not all institutional flow.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2)
                           + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--sse", type=Path, default=Path(
        "data/research/lhb/sse_daily"))
    parser.add_argument("--snapshots", type=Path, default=Path(
        "data/research/market_snapshots_ci"))
    parser.add_argument("--daily", type=Path, default=Path(
        "data/baostock/market_2020_2026/daily"))
    parser.add_argument("--industry", type=Path, default=Path(
        "data/research/industry_intervals.parquet"))
    parser.add_argument("--report", type=Path, default=Path(
        "data/research/lhb/institutional_seat_audit.json"))
    args = parser.parse_args()
    print(audit(args.calendar, args.sse, args.snapshots, args.daily,
                args.industry, args.report))


if __name__ == "__main__":
    main()
