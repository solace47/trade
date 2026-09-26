"""Verify issuer cash distributions for the frozen accounting exceptions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd
import pdfplumber

from .absolute_ridge_1449_eval import apply_period_quality
from .fill_accounting import KEY, _paired_mean, daily_mean, verify_manifest
from .hf_outcomes import Assumptions, EXECUTION_LABELS, _fees, _fill, _window_quotes
from .ingest import _login, _rows


ROOT = Path("data/research/corporate_cash")
TRADES = Path("data/research/fill_accounting/extended_accounted.parquet")
DAILY = Path("data/baostock/market_2020_2026/daily")
API = "https://www.cninfo.com.cn/new/hisAnnouncement/query"
REVIEWS = Path("config/corporate_cash_reviews.json")
MINUTES = Path("data/hf/pilot/data/stock_1m")


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False) + "\n", encoding="utf-8")


def freeze(output: Path = ROOT) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    trades = pd.read_parquet(TRADES)
    selected = trades.loc[trades.category.eq("corporate_action_unverified")]
    events, covered = [], 0
    hashes = {str(TRADES): sha(TRADES)}
    for code, group in selected.groupby("code", sort=True):
        path = DAILY / f"{code.replace('.', '_')}.parquet"
        hashes[str(path)] = sha(path)
        daily = pd.read_parquet(path, filters=[
            ("date", ">=", "2024-01-01"), ("date", "<=", "2025-12-31")])
        active = daily.loc[daily.tradestatus.eq(1)].sort_values("date").copy()
        active["previous_close"] = active.close.shift()
        active["reference_change"] = active.preclose - active.previous_close
        for row in active.loc[active.reference_change.abs().gt(.005)].itertuples():
            affected = group.loc[group.date.lt(row.date) & group.exit_date.ge(row.date)]
            if affected.empty:
                continue
            covered += len(affected)
            events.append({
                "code": code, "action_date": row.date,
                "previous_close": row.previous_close, "preclose": row.preclose,
                "grid_rows": len(affected),
            })
    inputs = pd.DataFrame(events).sort_values(["code", "action_date"])
    if (len(selected) != 99 or len(inputs) != 40 or covered != len(selected)
            or inputs.duplicated(["code", "action_date"]).any()):
        raise ValueError("The frozen 99-row, 40-event cohort changed")
    manifest = {"rule_commit": "9913952", "grid_rows": len(selected),
                "events": inputs.to_dict("records"), "sha256": hashes,
                "holdout_read": False, "adjusted_returns_read": False}
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Cannot replace the frozen corporate-action cohort")
    save_json(path, manifest)
    inputs.to_parquet(output / "events.parquet", index=False)
    return manifest


def verify(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text())
    for name, expected in manifest["sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError(f"Frozen source changed: {name}")
    return manifest


def curl(url: str, data: dict | None = None) -> bytes:
    args = ["curl", "-fLsS", "--retry", "2", "--max-time", "20", url]
    if data is not None:
        args += ["-X", "POST", "-H",
                 "Content-Type: application/x-www-form-urlencoded; charset=UTF-8"]
        for name, value in data.items():
            args += ["--data-urlencode", f"{name}={value}"]
    return subprocess.run(args, check=True, capture_output=True).stdout


def fetch(output: Path = ROOT) -> dict:
    manifest = verify(output)
    cache = output / "source"
    cache.mkdir(exist_ok=True)
    stock_path = cache / "stock_list.json"
    if not stock_path.exists():
        stock_path.write_bytes(curl("https://www.cninfo.com.cn/new/data/szse_stock.json"))
    stock_map = {r["code"]: r for r in json.loads(stock_path.read_text())["stockList"]}
    records = []
    _login()
    try:
        for n, event in enumerate(manifest["events"], 1):
            code, date = event["code"], event["action_date"]
            tag = f"{code}_{date}"
            vendor_path = cache / f"{tag}_baostock.json"
            if not vendor_path.exists():
                vendor = _rows(bs.query_dividend_data(
                    code, year=date[:4], yearType="operate"))
                save_json(vendor_path, vendor.to_dict("records"))
            vendor = [r for r in json.loads(vendor_path.read_text())
                      if r["dividOperateDate"] == date]
            if len(vendor) != 1:
                raise ValueError(f"No unique vendor event: {tag}")
            security = stock_map[code[3:]]
            start = (pd.Timestamp(date) - pd.Timedelta(days=21)).strftime("%Y-%m-%d")
            notices = []
            page = 1
            while True:
                index_path = cache / f"{tag}_index_{page}.json"
                if not index_path.exists():
                    index_path.write_bytes(curl(API, {
                        "stock": f"{code[3:]},{security['orgId']}",
                        "tabName": "fulltext", "pageSize": 30, "pageNum": page,
                        "column": "sse" if code.startswith("sh.") else "szse",
                        "seDate": f"{start}~{date}", "searchkey": "",
                    }))
                payload = json.loads(index_path.read_text())
                notices.extend(payload.get("announcements") or [])
                total = payload.get("totalAnnouncement")
                if not isinstance(total, int) or total < 1:
                    raise ValueError(f"Empty issuer disclosure index: {tag}")
                if len(notices) >= total:
                    if len(notices) != total:
                        raise ValueError(f"Disclosure pagination mismatch: {tag}")
                    break
                page += 1
                if page > 10:
                    raise ValueError("Unexpectedly long event-only disclosure query")
            candidates = [r for r in notices if r["secCode"] == code[3:]
                          and re.search(r"(?:权益分派|利润分配|分红).*实施公告$",
                                        re.sub(r"<[^>]+>", "", r["announcementTitle"]))]
            if not candidates:
                raise ValueError(f"No issuer implementation announcement: {tag}")
            issuer_files = []
            for notice in candidates:
                url = "https://static.cninfo.com.cn/" + notice["adjunctUrl"]
                pdf_path = cache / f"{tag}_{Path(notice['adjunctUrl']).stem}.pdf"
                text_path = pdf_path.with_suffix(".txt")
                if not pdf_path.exists():
                    content = curl(url)
                    if not content.startswith(b"%PDF"):
                        raise ValueError("The issuer source is not a PDF")
                    pdf_path.write_bytes(content)
                if not text_path.exists():
                    with pdfplumber.open(pdf_path) as document:
                        text = "\n\n".join(page.extract_text() or "" for page in document.pages)
                    text_path.write_text(text, encoding="utf-8")
                issuer_files.append({
                    "title": re.sub(r"<[^>]+>", "", notice["announcementTitle"]),
                    "url": url, "pdf": str(pdf_path), "text": str(text_path),
                    "sha256": sha(pdf_path),
                })
            records.append({**event, "name": security["zwjc"],
                            "vendor": vendor[0], "issuer_sources": issuer_files})
            save_json(output / "source_review.json", records)
            print(f"{n}/40 {tag}: {len(issuer_files)} issuer documents", flush=True)
    finally:
        bs.logout()
    report = {"events": len(records), "issuer_pdfs": sum(
        len(r["issuer_sources"]) for r in records), "adjusted_returns_read": False}
    save_json(output / "fetch_report.json", report)
    return report


def reference_price(previous: float, event: dict) -> Decimal:
    """Use published ex-price cash, not the actual shareholder entitlement."""
    cash = Decimal(event["reference_cash_per_share"])
    factor = 1 + Decimal(event["bonus_per_share"]) + Decimal(event["reserve_per_share"])
    if cash < 0 or factor < 1 or Decimal(str(previous)) <= cash:
        raise ValueError("Invalid distribution reference terms")
    return ((Decimal(str(previous)) - cash) / factor).quantize(
        Decimal(".01"), rounding=ROUND_HALF_UP)


def verify_notice(text: str, event: dict) -> None:
    """Check the manually reviewed table against primary PDF text, not vendor data.

    This deliberately supports only the fixed 40 implementation notices. The
    merged A/B-share table and new-share date were also visually inspected.
    """
    text = re.sub(r"\s+", "", text)
    if event["code"][3:] not in text[:300]:
        raise ValueError("Wrong security in the issuer document")
    if event["code"].startswith("sh."):
        cash = re.search(r"[AＡ]股每股现金红利([\d.]+)元", text)
        if not cash or Decimal(cash[1]) != Decimal(event["cash_per_share"]):
            raise ValueError("A-share cash in primary header disagrees")
        header = text.split("相关日期", 1)[1].split("差异化", 1)[0]
        # Removing PDF whitespace joins adjacent dates (e.g. 6/6 + 2025/6/6).
        header = re.sub(r"(202[45]/)", r" \1", header)
        dates = re.findall(r"202[45]/\d{1,2}/\d{1,2}", header)[:3]
        dates = [pd.Timestamp(d).strftime("%Y-%m-%d") for d in dates]
        if dates != [event[k] for k in ("record_date", "action_date", "pay_date")]:
            raise ValueError("Primary A-share date table disagrees")
    else:
        cash = re.search(
            r"向全体股东(?:按)?每10股(?:派发现金(?:红利|股利)|派现金红利|派)([\d.]+)元", text)
        if not cash or Decimal(cash[1]) / 10 != Decimal(event["cash_per_share"]):
            raise ValueError("Actual per-ten-share cash in primary notice disagrees")
        for field, prefix in (("record_date", r"股权登记日(?:为)?[：:]?"),
                              ("action_date", r"除权(?:除)?息日(?:为)?[：:]?"),
                              ("pay_date", r"将于")):
            y, m, d = map(int, event[field].split("-"))
            if not re.search(prefix + fr"{y}年{m}月{d}日", text):
                raise ValueError(f"Primary notice disagrees on {field}")
    published_reference = float(event["reference_cash_per_share"])
    numbers = {float(n) for n in re.findall(r"(?<!\d)0\.\d+", text)}
    if published_reference not in numbers:
        raise ValueError("The reviewed ex-price cash is not in the primary notice")
    if Decimal(event["reserve_per_share"]):
        if (event["code"] != "sz.000715"
                or "每10股转增3股" not in text
                or "起始交易日为2025年5月29日" not in text):
            raise ValueError("Unreviewed share distribution")


def validate(output: Path = ROOT) -> dict:
    manifest = verify(output)
    verify_manifest(Path("data/research/fill_accounting"))
    reviews = json.loads(REVIEWS.read_text())
    collected = json.loads((output / "source_review.json").read_text())
    by_key = {(r["code"], r["action_date"]): r for r in collected}
    keys = {(r["code"], r["action_date"]) for r in reviews}
    if len(reviews) != 40 or len(keys) != 40 or keys != set(by_key):
        raise ValueError("Review coverage changed")
    hashes, checked = {str(REVIEWS): sha(REVIEWS)}, []
    for event in reviews:
        source = by_key[(event["code"], event["action_date"])]
        docs = source["issuer_sources"]
        if len(docs) != 1 or docs[0]["url"] != event["source_url"]:
            raise ValueError("No unique, matching issuer implementation source")
        path = Path(docs[0]["pdf"])
        hashes[str(path)] = sha(path)
        if hashes[str(path)] != event["source_sha256"]:
            raise ValueError("Issuer PDF changed after manual review")
        with pdfplumber.open(path) as document:
            text = "\n".join(page.extract_text() or "" for page in document.pages)
        verify_notice(text, event)
        vendor = source["vendor"]
        for field, key in (("record_date", "dividRegistDate"),
                           ("action_date", "dividOperateDate"),
                           ("pay_date", "dividPayDate")):
            if event[field] != vendor[key] or not "2024-01-01" <= event[field] <= "2025-12-31":
                raise ValueError("Vendor/issuer date conflict or out-of-period event")
        for field, key in (("cash_per_share", "dividCashPsBeforeTax"),
                           ("bonus_per_share", "dividStocksPs"),
                           ("reserve_per_share", "dividReserveToStockPs")):
            if Decimal(event[field]) != Decimal(vendor[key] or "0"):
                raise ValueError("Vendor/issuer distribution conflict")
        price = reference_price(source["previous_close"], event)
        if price != Decimal(str(source["preclose"])):
            raise ValueError(f"Unreconciled ex-reference price: {event['code']}")
        checked.append({**event, "reference_reconciled": True,
                        "pure_cash": not (Decimal(event["bonus_per_share"])
                                           or Decimal(event["reserve_per_share"]))})
    report = {"event_count": len(checked), "grid_rows": manifest["grid_rows"],
              "pure_cash_events": sum(r["pure_cash"] for r in checked),
              "sha256": hashes, "holdout_read": False,
              "adjusted_returns_read": False, "events": checked}
    save_json(output / "validated.json", report)
    return report


def raw_fill_audit(rows: pd.DataFrame) -> dict:
    """Recheck both original four-minute fills, shares and participation limits."""
    hashes, checks = {}, 0
    for code, group in rows.groupby("code", sort=True):
        market, symbol = code.split(".")
        path = MINUTES / market.upper() / f"{symbol}.parquet"
        hashes[str(path)] = sha(path)
        minute = pd.read_parquet(path, columns=["timestamp", "volume", "turnover"],
                                 filters=[("timestamp", ">=", pd.Timestamp(group.date.min())),
                                          ("timestamp", "<", pd.Timestamp(group.exit_date.max())
                                           + pd.Timedelta(days=1))])
        minute["date"] = minute.timestamp.dt.strftime("%Y-%m-%d")
        minute["label"] = minute.timestamp.dt.strftime("%H%M")
        quotes = _window_quotes(minute.sort_values("timestamp"), EXECUTION_LABELS)
        daily = pd.read_parquet(DAILY / f"{market}_{symbol}.parquet", filters=[
            ("date", ">=", group.date.min()), ("date", "<=", group.exit_date.max())])
        days = {r["date"]: r for _, r in daily.iterrows()}
        for row in group.itertuples():
            for side, day, expected in (("buy", row.date, row.entry_price),
                                        ("sell", row.exit_date, row.exit_price)):
                price, status = _fill(quotes.get(day), days.get(day), code,
                                      side, row.shares, Assumptions())
                if status != "filled" or not np.isclose(price, expected, atol=1e-12, rtol=0):
                    raise ValueError("Frozen raw-price/capacity check failed")
                checks += 1
    return {"fill_checks": checks, "raw_sha256": hashes}


def cash_entitlement(row: pd.Series, event: dict) -> tuple[float, float]:
    """Gross entitlement and accrued personal tax, kept separate from cash timing."""
    if (row.code != event["code"]
            or Decimal(event["bonus_per_share"]) or Decimal(event["reserve_per_share"])
            or row.shares <= 0 or row.entry_status != "filled"
            or not row.date <= event["record_date"] < row.exit_date
            or not row.date < event["action_date"] <= row.exit_date
            or event["record_date"] >= event["action_date"]
            or event["pay_date"] < event["action_date"]
            or not "2024-01-01" <= row.date < row.exit_date <= "2025-12-31"
            or event["pay_date"] > "2025-12-31"
            or (pd.Timestamp(row.exit_date) - pd.Timestamp(row.date)).days > 15):
        raise ValueError("Outside the frozen pure-cash, short-holding entitlement rule")
    gross = float(Decimal(str(row.shares)) * Decimal(event["cash_per_share"]))
    if not np.isfinite(gross) or gross < 0:
        raise ValueError("Invalid entitled dividend")
    return gross, gross * .2


def apply_cash(rows: pd.DataFrame, eligible: pd.DataFrame,
               events: list[dict]) -> pd.DataFrame:
    """Extend verified accounting without changing fills or the original scores."""
    rows = rows.sort_values(KEY).reset_index(drop=True).copy()
    rows["accounting_category"] = rows.category
    rows["corporate_cash_verified"] = False
    for column in ("dividend_gross", "dividend_tax_accrued", "dividend_net"):
        rows[column] = np.nan
    for column in ("dividend_record_date", "dividend_pay_date", "dividend_tax_recognition_date"):
        rows[column] = pd.Series(None, index=rows.index, dtype=object)
    eligible_keys = set(eligible.loc[eligible.quality_clean_exit, KEY]
                        .itertuples(index=False, name=None))
    for i, row in rows.loc[rows.category.eq("corporate_action_unverified")].iterrows():
        matches = [e for e in events if e["code"] == row.code
                   and row.date < e["action_date"] <= row.exit_date]
        if len(matches) != 1:
            raise ValueError("An unresolved position does not have one reviewed action")
        event = matches[0]
        if not event["pure_cash"] or tuple(row[KEY]) not in eligible_keys:
            continue
        gross, tax = cash_entitlement(row, event)
        rows.loc[i, ["dividend_gross", "dividend_tax_accrued", "dividend_net"]] = (
            gross, tax, gross - tax)
        rows.loc[i, ["dividend_record_date", "dividend_pay_date",
                     "dividend_tax_recognition_date"]] = (
            event["record_date"], event["pay_date"], row.exit_date)
        rows.loc[i, "corporate_cash_verified"] = True
        rows.loc[i, "unknown_after_buy"] = False
        rows.loc[i, "accounting_category"] = "verified_cash_distribution"
        for slip in (5, 15):
            sell_value = row.shares * row.exit_price / .9995 * (1 - slip / 10000)
            proceeds = sell_value - _fees(sell_value, "sell", Assumptions(), row.exit_date)
            rows.loc[i, f"verified_proceeds{slip}"] = proceeds
            rows.loc[i, f"known_return{slip}"] = (proceeds + gross - tax) / row[f"buy_cost{slip}"] - 1
    return rows


def accounting_summary(rows: pd.DataFrame) -> dict:
    cells, pairs = [], []
    for (amount, horizon, half, arm), part in rows.groupby(
            ["target_notional", "horizon", "half", "candidate"], sort=True):
        unknown = part.unknown_after_buy
        cell = {"notional": int(amount), "horizon": int(horizon), "half": half,
                "arm": arm, "signals": len(part), "unknown_rows": int(unknown.sum()),
                "unknown_buy_cost5": float(part.loc[unknown, "buy_cost5"].sum()),
                "cash_verified_rows": int(part.corporate_cash_verified.sum())}
        for slip in (5, 15):
            # Missing P&L stays missing. A zero-filled numerator is explicitly only
            # a component on the fixed signal-day denominator, never full return.
            component = part.assign(component=part[f"known_return{slip}"].fillna(0))
            cell[f"known_contribution{slip}"] = daily_mean(component, "component")
            cell[f"complete_cohort_mean{slip}"] = (
                None if unknown.any() else daily_mean(part, f"known_return{slip}"))
        cells.append(cell)
    for (amount, horizon, half), part in rows.groupby(
            ["target_notional", "horizon", "half"], sort=True):
        pairs.append({"notional": int(amount), "horizon": int(horizon), "half": half,
                      **{f"known_edge{slip}": _paired_mean(
                          part.assign(component=part[f"known_return{slip}"].fillna(0)),
                          "component") for slip in (5, 15)}})
    return {"cells": cells, "pairs": pairs}


def evaluate(output: Path = ROOT) -> dict:
    verify(output)
    verify_manifest(Path("data/research/fill_accounting"))
    validation = json.loads((output / "validated.json").read_text())
    for name, expected in validation["sha256"].items():
        if sha(Path(name)) != expected:
            raise ValueError("Reviewed source changed after validation")
    original = pd.read_parquet(TRADES).sort_values(KEY).reset_index(drop=True)
    selected = original.loc[original.category.eq("corporate_action_unverified")].copy()
    # Only probe quality here. Do not reinterpret these rows as already filled
    # and clean, or use an original missing net_return as a numerical value.
    probe = selected.copy()
    probe["exit_status"] = "filled"
    eligible, _ = apply_period_quality(probe)
    raw = raw_fill_audit(selected)
    result = apply_cash(original, eligible, validation["events"])
    for column in original.columns:
        if column in {"unknown_after_buy", "verified_proceeds5", "verified_proceeds15",
                      "known_return5", "known_return15"}:
            continue
        pd.testing.assert_series_equal(result[column], original[column])
    if (result.loc[result.unknown_after_buy, ["known_return5", "known_return15",
                                             "verified_proceeds5", "verified_proceeds15"]]
            .notna().any().any()):
        raise ValueError("Unverified proceeds or P&L were filled")
    result.to_parquet(output / "cash_accounted.parquet", index=False)
    summary = {"rows": len(result), "restored_rows": int(result.corporate_cash_verified.sum()),
               "remaining_unknown_rows": int(result.unknown_after_buy.sum()),
               "quality_rejected_corporate_rows": int((~eligible.quality_clean_exit).sum()),
               "review_sha256": sha(REVIEWS), "validation_sha256": sha(output / "validated.json"),
               "accounted_sha256": sha(output / "cash_accounted.parquet"),
               **raw, **accounting_summary(result), "holdout_read": False,
               "qualification": "Issuer-verified cash dividends; personal short-holder 20% tax "
                                "accrual. Sale proceeds exclude dividends; payment date and sale "
                                "tax-recognition date are distinct. No portfolio cash simulation."}
    save_json(output / "report.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fetch", "validate", "evaluate"))
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    result = {"freeze": freeze, "fetch": fetch, "validate": validate,
              "evaluate": evaluate}[args.stage](args.output)
    print({k: v for k, v in result.items()
           if k not in {"sha256", "raw_sha256", "events", "cells", "pairs"}})


if __name__ == "__main__":
    main()
