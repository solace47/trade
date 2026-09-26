"""Resolve the nine frozen accounting exceptions without changing selections."""

from __future__ import annotations

import argparse
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd
import pdfplumber

from scripts.audit_open_only_quality import _symbol
from .corporate_cash import (
    DAILY, MINUTES, REVIEWS, accounting_summary, curl, raw_fill_audit, save_json, sha,
)
from .fill_accounting import CALENDAR, KEY, daily_mean, verify_manifest
from .hf_outcomes import (
    Assumptions, EXECUTION_LABELS, _board_limit_rate, _fees, _fill, _limit_price,
    _window_quotes,
)
from .market_study import _quality_keys
from .quality_period import load_period_bad_symbols


ROOT = Path("data/research/holding_exceptions")
SOURCE = Path("data/research/corporate_cash/cash_accounted.parquet")
CLASSIFICATIONS = Path("data/research/opening_quality/day_classifications.parquet")
ISSUES = Path("data/research/market_issues_ci")
QUALITY = Path("data/research/quality_period_2024_2025.json")
BAD_DAYS = {("sh.603216", "2025-08-04"), ("sh.603661", "2025-02-19"),
            ("sz.003019", "2025-12-19")}
DELIST_URL = ("https://disc.static.szse.cn/disc/disk03/finalpage/2024-09-18/"
              "5e053ded-3391-4b82-94cf-ef8f0922f241.PDF")


def freeze(output: Path = ROOT) -> dict:
    verify_manifest(Path("data/research/fill_accounting"))
    report = json.loads(SOURCE.with_name("report.json").read_text())
    if (sha(SOURCE) != report["accounted_sha256"]
            or sha(REVIEWS) != report["review_sha256"]
            or sha(SOURCE.with_name("validated.json")) != report["validation_sha256"]):
        raise ValueError("Cash-accounted source changed")
    selected = pd.read_parquet(SOURCE)
    selected = selected.loc[selected.unknown_after_buy].sort_values(KEY)
    if selected.category.value_counts().to_dict() != {
            "exit_quality_unverified": 5, "corporate_action_unverified": 2,
            "no_exit_recorded": 2}:
        raise ValueError("The frozen nine-row cohort changed")
    files = [SOURCE, SOURCE.with_name("report.json"), SOURCE.with_name("validated.json"),
             REVIEWS, CLASSIFICATIONS, QUALITY, CALENDAR, *sorted(ISSUES.glob("shard_*.csv"))]
    for code in selected.code.unique():
        market, symbol = code.split(".")
        files += [MINUTES / market.upper() / f"{symbol}.parquet",
                  DAILY / f"{market}_{symbol}.parquet"]
    split = next(r for r in json.loads(REVIEWS.read_text()) if r["code"] == "sz.000715")
    files.append(Path("data/research/corporate_cash/source") /
                 f"sz.000715_{split['action_date']}_{Path(split['source_url']).stem}.pdf")
    manifest = {"rule_commit": "e3355ad", "keys": selected[KEY].to_dict("records"),
                "sha256": {str(path): sha(path) for path in files},
                "holdout_read": False, "adjusted_returns_read": False}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Cannot replace the frozen exception cohort")
    save_json(path, manifest)
    return manifest


def verify(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Exception input changed: {name}")
    return manifest


def load_quotes(code: str, first: str, last: str) -> tuple[dict, pd.DataFrame]:
    if not "2024-01-01" <= first <= last <= "2025-12-31":
        raise ValueError("Quote request outside research years")
    market, symbol = code.split(".")
    minute = pd.read_parquet(MINUTES / market.upper() / f"{symbol}.parquet",
                             columns=["timestamp", "volume", "turnover"], filters=[
                                 ("timestamp", ">=", pd.Timestamp(first)),
                                 ("timestamp", "<", pd.Timestamp(last) + pd.Timedelta(days=1))])
    minute["date"] = minute.timestamp.dt.strftime("%Y-%m-%d")
    minute["label"] = minute.timestamp.dt.strftime("%H%M")
    quotes = _window_quotes(minute.sort_values("timestamp"), EXECUTION_LABELS)
    daily = pd.read_parquet(DAILY / f"{market}_{symbol}.parquet", filters=[
        ("date", ">=", first), ("date", "<=", last)]).sort_values("date")
    if daily.date.duplicated().any():
        raise ValueError("Non-unique independent daily reference")
    return quotes, daily


def verified_opening_days() -> pd.DataFrame:
    classified = pd.read_parquet(CLASSIFICATIONS)
    fresh = pd.DataFrame([row for code, date in sorted(BAD_DAYS)
                          for row in _symbol((code, [date]))])
    old = classified.merge(fresh[["code", "date"]], on=["code", "date"],
                           validate="one_to_one")
    if len(old) != 3 or not fresh.classification.eq("opening_only_volume_matched").all():
        raise ValueError("A frozen opening-only day did not pass raw recheck")
    pd.testing.assert_frame_equal(old[fresh.columns].sort_values(["code", "date"]).reset_index(drop=True),
                                  fresh.sort_values(["code", "date"]).reset_index(drop=True))
    issues = pd.concat([pd.read_csv(p, dtype=str) for p in sorted(ISSUES.glob("shard_*.csv"))])
    marked = issues.merge(fresh[["code", "date"]], on=["code", "date"])
    if len(marked) != 3 or not marked.kind.eq("ohlc_disagreement").all():
        raise ValueError("Opening release would hide another issue")
    return fresh


def quality_clean(code: str, first: str, last: str, bad: pd.DataFrame,
                  bad_symbols: set[str], released: set[tuple[str, str]] = frozenset()) -> bool:
    if code in bad_symbols:
        return False
    overlaps = bad.loc[bad.code.eq(code) & bad.date.between(first, last)]
    return all((r.code, r.date) in released for r in overlaps.itertuples())


def new_share_count(shares: int, event: dict, date: str) -> int:
    if (shares <= 0 or Decimal(event["bonus_per_share"]) != 0
            or Decimal(event["reserve_per_share"]) != Decimal(".3")
            or event["new_share_trade_date"] is None
            or date < event["new_share_trade_date"]):
        raise ValueError("Unverified share distribution or shares not tradable")
    new = Decimal(str(shares)) * Decimal("1.3")
    if new != new.to_integral_value():
        raise ValueError("Fractional-share allocation is unverified")
    return int(new)


def recover(row: pd.Series, sold_shares: int, price: float, date: str,
            gross_dividend: float = 0, tax: float = 0) -> dict:
    """Restore economic proceeds; do not overwrite the old exit or strict score."""
    if (not np.isfinite([price, gross_dividend, tax]).all()
            or row.entry_status != "filled" or sold_shares < row.shares or price <= 0
            or not row.date < date <= "2025-12-31" or gross_dividend < tax or tax < 0):
        raise ValueError("Invalid recovered holding accounting")
    result = {"sold_shares": sold_shares, "accounting_exit_price": price,
              "accounting_exit_date": date, "unknown_after_buy": False}
    for slip in (5, 15):
        if not np.isfinite(row[f"buy_cost{slip}"]) or row[f"buy_cost{slip}"] <= 0:
            raise ValueError("Missing or invalid original purchase cash")
        value = sold_shares * price / .9995 * (1 - slip / 10000)
        proceeds = value - _fees(value, "sell", Assumptions(), date)
        result[f"verified_proceeds{slip}"] = proceeds
        result[f"known_return{slip}"] = (proceeds + gross_dividend - tax) / row[f"buy_cost{slip}"] - 1
    return result


def trace_unresolved(rows: pd.DataFrame, calendar: list[str], output: Path) -> dict:
    if set(rows.code) != {"sz.000861"} or set(rows.date) != {"2024-07-10"}:
        raise ValueError("Unfrozen unresolved holdings")
    source = output / "sz.000861_delisting.pdf"
    if not source.exists():
        content = curl(DELIST_URL)
        if not content.startswith(b"%PDF"):
            raise ValueError("Invalid primary delisting PDF")
        source.write_bytes(content)
    with pdfplumber.open(source) as document:
        notice = "".join((p.extract_text() or "").replace(" ", "").replace("\n", "")
                         for p in document.pages)
    if not all(s in notice for s in ("000861", "2024年9月18日", "不进入退市整理期")):
        raise ValueError("Primary delisting source does not match")
    quotes, daily = load_quotes("sz.000861", rows.date.min(), calendar[-1])
    days = {r["date"]: r for _, r in daily.iterrows()}
    attempts = []
    for row in rows.itertuples():
        for date in calendar[calendar.index(row.target_exit_date):]:
            quote, day = quotes.get(date), days.get(date)
            price, status = _fill(quote, day, row.code, "sell", row.shares, Assumptions())
            if price is not None:
                raise ValueError("The old unresolved trace missed a modeled fill")
            active = day is not None and int(day.tradestatus) == 1
            if active and quote is None:
                raise ValueError("Active-day four-minute data missing, not a trading failure")
            lower = (_limit_price(float(day.preclose), _board_limit_rate(
                row.code, int(day.isST), date), upper=False) if active else None)
            attempts.append({"date": date, "code": row.code,
                             "notional": row.target_notional, "status": status,
                             "active_daily": active,
                             "vwap": float(quote.vwap) if quote is not None else None,
                             "volume": int(quote.volume) if quote is not None else None,
                             "estimated_lower_limit": lower})
    pd.DataFrame(attempts).to_parquet(output / "unresolved_attempts.parquet", index=False)
    return {"source_url": DELIST_URL, "source_sha256": sha(source),
            "delisted_date": "2024-09-18", "no_delisting_consolidation_period": True,
            "active_sell_attempts": [r for r in attempts if r["active_daily"]],
            "attempt_counts": pd.Series([r["status"] for r in attempts]).value_counts().to_dict(),
            "terminal_value": None, "otc_execution_source_available": False}


def evaluate(output: Path = ROOT) -> dict:
    verify(output)
    original = pd.read_parquet(SOURCE).sort_values(KEY).reset_index(drop=True)
    rows = original.copy()
    rows["sold_shares"] = rows.shares.where(rows.exit_date.notna(), np.nan)
    rows["accounting_exit_date"] = rows.exit_date
    rows["accounting_exit_price"] = rows.exit_price
    rows["accounting_exit_delay_sessions"] = rows.exit_delay_sessions
    opening = verified_opening_days()
    opening.to_parquet(output / "opening_recheck.parquet", index=False)
    bad, bad_symbols = _quality_keys(ISSUES), set(load_period_bad_symbols(
        QUALITY, "2024-01-01", "2025-12-31").code)
    issues = rows.loc[rows.unknown_after_buy & rows.exit_date.notna()]
    raw = raw_fill_audit(issues)
    for i, row in issues.loc[issues.category.eq("exit_quality_unverified")].iterrows():
        if not quality_clean(row.code, row.date, row.exit_date, bad, bad_symbols, BAD_DAYS):
            raise ValueError("An additional holding-period quality problem remains")
        values = recover(row, int(row.shares), row.exit_price, row.exit_date)
        if not np.isclose(values["known_return5"], row.net_return, atol=1e-12, rtol=0):
            raise ValueError("Recovered quality P&L changed the original fill accounting")
        values["accounting_category"] = "verified_open_only_quality"
        for field, value in values.items():
            rows.at[i, field] = value
    calendar_table = pd.read_parquet(CALENDAR)
    calendar = sorted(calendar_table.loc[
        calendar_table.is_trading_day.eq("1")
        & calendar_table.calendar_date.between("2024-01-01", "2025-12-31"),
        "calendar_date"].tolist())
    event = next(r for r in json.loads(REVIEWS.read_text()) if r["code"] == "sz.000715")
    selected = rows.loc[rows.unknown_after_buy & rows.category.eq("corporate_action_unverified")]
    if set(selected.code) != {event["code"]} or len(selected) != 2:
        raise ValueError("Unfrozen share-distribution cohort")
    quotes, daily = load_quotes(event["code"], selected.date.min(), calendar[-1])
    days = {r["date"]: r for _, r in daily.iterrows()}
    active = daily.loc[daily.tradestatus.eq(1)]
    gap_dates = set(active.loc[(active.preclose - active.close.shift()).abs().gt(.005), "date"])
    share_audit = []
    for i, row in selected.iterrows():
        if not row.date <= event["record_date"] < row.target_exit_date:
            raise ValueError("Share entitlement was not held on the record date")
        shares = new_share_count(int(row.shares), event, row.target_exit_date)
        attempts = []
        for date in calendar[calendar.index(row.target_exit_date):]:
            price, status = _fill(quotes.get(date), days.get(date), row.code,
                                  "sell", shares, Assumptions())
            attempts.append({"date": date, "status": status})
            if price is None:
                continue
            if ({d for d in gap_dates if row.date < d <= date} != {event["action_date"]}
                    or not quality_clean(row.code, row.date, date, bad, bad_symbols)
                    or (pd.Timestamp(date) - pd.Timestamp(row.date)).days > 15):
                raise ValueError("Share-adjusted exit violates event, quality or tax window")
            gross = float(Decimal(str(row.shares)) * Decimal(event["cash_per_share"]))
            tax = gross * .2
            values = recover(row, shares, price, date, gross, tax)
            values.update({"dividend_gross": gross, "dividend_tax_accrued": tax,
                           "dividend_net": gross - tax, "dividend_record_date": event["record_date"],
                           "dividend_pay_date": event["pay_date"],
                           "dividend_tax_recognition_date": date,
                           "accounting_category": "verified_reserve_distribution",
                           "accounting_exit_delay_sessions": calendar.index(date)
                           - calendar.index(row.target_exit_date)})
            for field, value in values.items():
                rows.at[i, field] = value
            share_audit.append({"code": row.code, "date": row.date,
                                "notional": row.target_notional, "old_shares": int(row.shares),
                                "sold_shares": shares, "exit_date": date,
                                "price": price, "window_volume": int(quotes[date].volume),
                                "attempts": attempts})
            break
    unresolved = trace_unresolved(rows.loc[rows.unknown_after_buy], calendar, output)
    rows.to_parquet(output / "accounted.parquet", index=False)
    summary = accounting_summary(rows)
    scenarios = []
    for cell in summary["cells"]:
        if not cell["unknown_rows"]:
            continue
        part = rows.loc[rows.target_notional.eq(cell["notional"]) & rows.horizon.eq(cell["horizon"])
                        & rows.half.eq(cell["half"]) & rows.candidate.eq(cell["arm"])].copy()
        for slip in (5, 15):
            part["scenario"] = part[f"known_return{slip}"].fillna(-1)
            scenarios.append({"notional": cell["notional"], "horizon": cell["horizon"],
                              "half": cell["half"], "arm": cell["arm"], "slip": slip,
                              "known_contribution": cell[f"known_contribution{slip}"],
                              "zero_terminal_value_scenario": daily_mean(part, "scenario"),
                              "observed_complete_return": None})
    for column in ("entry_price", "shares", "exit_price", "exit_date", "old_score5", "old_score15",
                   "buy_cost5", "buy_cost15", "quality_clean_exit", "exit_status"):
        pd.testing.assert_series_equal(rows[column], original[column])
    if rows.loc[rows.unknown_after_buy, ["known_return5", "verified_proceeds5"]].notna().any().any():
        raise ValueError("Unresolved holding acquired an invented terminal value")
    result = {"restored_quality_rows": 5, "share_adjusted_rows": len(share_audit),
              "remaining_unknown_rows": int(rows.unknown_after_buy.sum()),
              **raw, "share_audit": share_audit, "unresolved": unresolved,
              **summary, "zero_terminal_value_scenarios": scenarios,
              "accounted_sha256": sha(output / "accounted.parquet"), "holdout_read": False,
              "qualification": "Fixed rejected cohort; complete economic per-trade means where "
                               "fully observed, not a shared-capital portfolio. Zero terminal "
                               "value is a scenario, not an observed OTC sale or total loss."}
    save_json(output / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "evaluate"))
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    result = freeze(args.output) if args.stage == "freeze" else evaluate(args.output)
    print({k: v for k, v in result.items() if k not in {
        "sha256", "raw_sha256", "keys", "cells", "pairs", "unresolved",
        "share_audit", "zero_terminal_value_scenarios"}})


if __name__ == "__main__":
    main()
