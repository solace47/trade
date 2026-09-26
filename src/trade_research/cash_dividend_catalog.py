"""Collect a frozen 2024–2025 cash-distribution event catalog, never returns.

Vendor implementation dates and action types are provisional until reconciled
with contemporaneous issuer implementation notices.
"""

from __future__ import annotations

import argparse
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import time

import baostock as bs
import duckdb
import pandas as pd
import pdfplumber

from .corporate_cash import DAILY, curl, save_json, sha
from .ingest import _login, _rows


ROOT = Path("data/research/cash_dividend_catalog")
OLD_CACHE = Path("data/research/corporate_cash/source")
REVIEWS = Path("config/cash_dividend_vendor_reviews.json")


def freeze(output: Path = ROOT) -> dict:
    files = sorted([*DAILY.glob("sh_60*.parquet"), *DAILY.glob("sz_00*.parquet")])
    if not files:
        raise ValueError("Historical mainboard source files are missing")
    connection = duckdb.connect()
    connection.execute("SET threads = 4")
    try:
        connection.read_parquet([str(p) for p in files]).create_view("daily")
        # No quote/return fields are projected; no 2026 listing is admitted on
        # the basis of its present-day membership or future trading history.
        jobs = connection.execute("""
            SELECT code, LEFT(date, 4) AS year, COUNT(*) AS active_days,
                   MIN(date) AS first_active, MAX(date) AS last_active
            FROM daily WHERE date BETWEEN '2024-01-01' AND '2025-12-31'
              AND tradestatus = 1
              AND (code LIKE 'sh.60%' OR code LIKE 'sz.00%')
            GROUP BY code, year ORDER BY code, year
        """).df()
    finally:
        connection.close()
    if jobs.empty or jobs.duplicated(["code", "year"]).any() or set(jobs.year) != {"2024", "2025"}:
        raise ValueError("Invalid historical code-year universe")
    output.mkdir(parents=True, exist_ok=True)
    job_path = output / "jobs.parquet"
    if job_path.exists():
        pd.testing.assert_frame_equal(pd.read_parquet(job_path), jobs)
    else:
        jobs.to_parquet(job_path, index=False)
    manifest = {"rule_commit": "ca9a9af", "code_years": len(jobs),
                "securities": int(jobs.code.nunique()), "jobs_sha256": sha(job_path),
                "daily_sha256": {str(p): sha(p) for p in files},
                "holdout_read": False, "strategy_returns_read": False}
    path = output / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != manifest:
        raise ValueError("Cannot replace the frozen event-catalog universe")
    save_json(path, manifest)
    return manifest


def jobs_for(output: Path) -> pd.DataFrame:
    manifest = json.loads((output / "manifest.json").read_text())
    if sha(output / "jobs.parquet") != manifest["jobs_sha256"]:
        raise ValueError("Frozen historical code-year keys changed")
    return pd.read_parquet(output / "jobs.parquet")


def reviewed_duplicates() -> dict:
    reviews = json.loads(REVIEWS.read_text())
    result = {}
    for row in reviews:
        path = Path(row["source_path"])
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(curl(row["source_url"]))
        if sha(path) != row["source_sha256"]:
            raise ValueError("Primary duplicate-resolution notice changed")
        with pdfplumber.open(path) as document:
            primary = re.sub(r"\s+", "", "".join(
                page.extract_text() or "" for page in document.pages)).replace("Ａ", "A")
        headline = primary.split("每股分配比例", 1)[1].split("相关日期", 1)[0]
        if not any(Decimal(value) == Decimal(row["cash_per_share"])
                   for value in re.findall(r"([0-9]+\.[0-9]+)元", headline)):
            raise ValueError("Reviewed cash amount is absent from primary headline")
        dates = [f"{int(d[:4])}/{int(d[5:7])}/{int(d[8:])}" for d in
                 (row["record_date"], row["action_date"], row["pay_date"])]
        if f"A股{dates[0]}－{dates[1]}{dates[2]}" not in primary:
            raise ValueError("Reviewed dates disagree with the primary A-share table")
        if f"/finalpage/{row['notice_date']}/" not in row["source_url"]:
            raise ValueError("Reviewed notice date disagrees with the disclosure URL")
        key = (row["code"], row["year"])
        result.setdefault(key, []).append(row)
    for rows in result.values():
        if len({r["action_date"] for r in rows}) != len(rows):
            raise ValueError("Conflicting primary reviews for the same event date")
    return result


def record_hash(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def apply_review(records: list[dict], review: dict) -> list[dict]:
    if "combine_record_sha256" in review:
        group = [r for r in records if r["dividOperateDate"] == review["action_date"]]
        replacement = review["combined_record"]
        if group == [replacement]:
            return records  # A cached, already reconciled response is idempotent.
        if (sorted(record_hash(r) for r in group) != sorted(review["combine_record_sha256"])
                or len(group) < 2):
            raise ValueError("Same-day distribution components differ from the reviewed originals")
        for field in ("dividCashPsBeforeTax", "dividStocksPs", "dividReserveToStockPs"):
            if sum(Decimal(r[field] or "0") for r in group) != Decimal(replacement[field] or "0"):
                raise ValueError("Combined distribution does not equal the reviewed components")
        for field in ("code", "dividOperateDate", "dividRegistDate", "dividPayDate", "dividPlanDate"):
            if any(r[field] != replacement[field] for r in group):
                raise ValueError("Cannot combine distributions with different security or dates")
        return [r for r in records if r["dividOperateDate"] != review["action_date"]] + [replacement]
    retained = [r for r in records if record_hash(r) != review["discard_incomplete_record_sha256"]]
    if len(records) - len(retained) > 1:
        raise ValueError("More duplicate copies than the reviewed source")
    return retained


def checked_response(records: list[dict], code: str, year: str,
                     reviews: list[dict] | dict | None = None) -> list[dict]:
    if not isinstance(records, list):
        raise ValueError("A completed query must have a list, including for no events")
    for row in records:
        if row.get("code") != code or not row.get("dividOperateDate", "").startswith(year + "-"):
            raise ValueError("Vendor returned a different security or implementation year")
        if pd.Timestamp(row["dividOperateDate"]).strftime("%Y-%m-%d") != row["dividOperateDate"]:
            raise ValueError("Invalid implementation date")
    for review in ([reviews] if isinstance(reviews, dict) else (reviews or [])):
        if (review["code"], review["year"]) != (code, year):
            raise ValueError("Duplicate review belongs to another code-year")
        retained = apply_review(records, review)
        matching = [row for row in retained if row["dividOperateDate"] == review["action_date"]]
        if len(matching) != 1:
            raise ValueError("The reviewed complete implementation record is missing")
        row = matching[0]
        for field, expected in (("dividPlanDate", "notice_date"),
                                ("dividRegistDate", "record_date"), ("dividPayDate", "pay_date")):
            if row[field] != review[expected]:
                raise ValueError("Retained vendor date disagrees with primary notice")
        for field, expected in (("dividCashPsBeforeTax", "cash_per_share"),
                                ("dividStocksPs", "bonus_per_share"),
                                ("dividReserveToStockPs", "reserve_per_share")):
            if Decimal(row[field] or "0") != Decimal(review[expected]):
                raise ValueError("Retained vendor distribution disagrees with primary notice")
        records = retained
    if len({r["dividOperateDate"] for r in records}) != len(records):
        raise ValueError("Multiple actions on the same date require manual reconciliation")
    return records


def fetch(output: Path = ROOT, pause_seconds: float = .05) -> dict:
    jobs = jobs_for(output)
    reviews = reviewed_duplicates()
    cache = output / "vendor"
    cache.mkdir(exist_ok=True)
    downloaded, reused, errors = 0, 0, []
    _login()
    try:
        for number, job in enumerate(jobs.itertuples(), 1):
            path = cache / f"{job.code}_{job.year}.json"
            error_path = path.with_suffix(".error.json")
            try:
                review = reviews.get((job.code, job.year))
                if path.exists():
                    checked_response(json.loads(path.read_text()), job.code, job.year, review)
                else:
                    previous = sorted(OLD_CACHE.glob(f"{job.code}_{job.year}-*_baostock.json"))
                    if previous:
                        rows = checked_response(json.loads(previous[0].read_text()), job.code, job.year, review)
                        for other in previous[1:]:
                            if json.loads(other.read_text()) != rows:
                                raise ValueError("Old responses for the same code-year disagree")
                        save_json(path, rows)
                        reused += 1
                    else:
                        for attempt in range(3):
                            try:
                                frame = _rows(bs.query_dividend_data(job.code, year=job.year,
                                                                    yearType="operate"))
                                raw = frame.to_dict("records")
                                try:
                                    rows = checked_response(raw, job.code, job.year, review)
                                except ValueError:
                                    conflict_dir = output / "source_conflicts"
                                    conflict_dir.mkdir(exist_ok=True)
                                    save_json(conflict_dir / f"{job.code}_{job.year}_raw.json", raw)
                                    raise
                                save_json(path, rows)
                                downloaded += 1
                                break
                            except ValueError:
                                raise  # Semantic conflicts cannot be repaired by repeating the query.
                            except Exception:
                                if attempt == 2:
                                    raise
                                bs.logout()
                                time.sleep(.5)
                                _login()
                        time.sleep(pause_seconds)
                error_path.unlink(missing_ok=True)
            except Exception as error:
                record = {"code": job.code, "year": job.year, "error": str(error)}
                save_json(error_path, record)
                errors.append(record)
            if number % 100 == 0 or number == len(jobs):
                progress = {"processed": number, "total": len(jobs), "downloaded": downloaded,
                            "reused": reused, "errors": len(errors)}
                save_json(output / "fetch_progress.json", progress)
                print(progress, flush=True)
    finally:
        bs.logout()
    result = {"jobs": len(jobs), "downloaded": downloaded, "reused": reused,
              "errors": errors, "complete": not errors, "holdout_read": False,
              "strategy_returns_read": False}
    save_json(output / "fetch_report.json", result)
    return result


def action_type(row: dict) -> str:
    try:
        cash = Decimal(row["dividCashPsBeforeTax"] or "0")
        bonus = Decimal(row["dividStocksPs"] or "0")
        reserve = Decimal(row["dividReserveToStockPs"] or "0")
    except (KeyError, InvalidOperation):
        return "invalid_vendor_fields"
    if not all(v.is_finite() and v >= 0 for v in (cash, bonus, reserve)):
        return "invalid_vendor_fields"
    if bonus or reserve:
        return "share_distribution"
    if cash <= 0:
        return "no_cash_distribution"
    if "转" in row.get("dividCashStock", "") or "送" in row.get("dividCashStock", ""):
        return "conflicting_vendor_action_type"
    return "provisional_pure_cash"


def assemble(output: Path = ROOT) -> dict:
    jobs = jobs_for(output)
    reviews = reviewed_duplicates()
    records, query_rows, hashes = [], [], {}
    for job in jobs.itertuples():
        path = output / "vendor" / f"{job.code}_{job.year}.json"
        if not path.exists() or path.with_suffix(".error.json").exists():
            raise ValueError(f"Incomplete code-year, not a zero-event response: {job.code} {job.year}")
        rows = checked_response(json.loads(path.read_text()), job.code, job.year,
                                 reviews.get((job.code, job.year)))
        hashes[str(path)] = sha(path)
        query_rows.append({"code": job.code, "year": job.year, "events": len(rows)})
        for row in rows:
            records.append({**row, "implementation_year": job.year, "action_type": action_type(row)})
    frame = pd.DataFrame(records)
    if frame.empty:
        raise ValueError("The entire event source is unexpectedly empty")
    frame = frame.sort_values(["code", "dividOperateDate"]).reset_index(drop=True)
    frame.to_parquet(output / "events.parquet", index=False)
    pd.DataFrame(query_rows).to_parquet(output / "query_coverage.parquet", index=False)
    result = {"code_year_queries": len(jobs), "event_rows": len(frame),
              "empty_code_years": sum(r["events"] == 0 for r in query_rows),
              "by_year_action": [{"year": year, "action_type": kind, "events": len(part),
                                  "securities": int(part.code.nunique())}
                                 for (year, kind), part in frame.groupby(
                                     ["implementation_year", "action_type"], sort=True)],
              "sha256": hashes, "events_sha256": sha(output / "events.parquet"),
              "duplicate_review_sha256": sha(REVIEWS),
              "manually_reviewed_code_years": len(reviews),
              "manually_reviewed_event_dates": sum(len(rows) for rows in reviews.values()),
              "issuer_notice_verified": False, "holdout_read": False,
              "strategy_returns_read": False}
    save_json(output / "catalog_report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("freeze", "fetch", "assemble"))
    parser.add_argument("--output", type=Path, default=ROOT)
    args = parser.parse_args()
    result = {"freeze": freeze, "fetch": fetch, "assemble": assemble}[args.stage](args.output)
    print({k: v for k, v in result.items() if k not in {"sha256", "daily_sha256"}})


if __name__ == "__main__":
    main()
