"""Freeze a single broker's 2024–2025 catalog and an outcome-free source sample."""

from __future__ import annotations

import argparse
import calendar
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import requests

from .corporate_cash import save_json, sha


ROOT = Path("data/research/analyst_revision_source/dongwu_catalog")
ENDPOINT = "https://reportapi.eastmoney.com/report/list"
BROKER = "80000031"
RULE_COMMIT = "e3ef572"


def periods():
    for year in (2024, 2025):
        for month in range(1, 13):
            yield (f"{year}-{month:02d}-01",
                   f"{year}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}")


def parameters(start: str, end: str, page: int) -> dict:
    return {"industryCode": "*", "pageSize": 100, "industry": "*", "rating": "*",
            "ratingChange": "*", "beginTime": start, "endTime": end, "pageNo": page,
            "qType": 0, "orgCode": BROKER, "code": "", "rcode": ""}


def validate_page(payload: dict, start: str, end: str, page: int) -> list[dict]:
    if int(payload["pageNo"]) != page:
        raise ValueError("Source page identity differs from the request")
    rows = payload["data"]
    if not isinstance(rows, list):
        raise ValueError("Source rows are not a list")
    for row in rows:
        date = row["publishDate"][:10]
        datetime.strptime(date, "%Y-%m-%d")
        if not start <= date <= end:
            raise ValueError("Report display date is outside the frozen month")
        if str(row["orgCode"]) != BROKER:
            raise ValueError("Source ignored the broker constraint")
        if not row["infoCode"] or not row["stockCode"]:
            raise ValueError("Report identity is missing")
    return rows


def finish_month(pages: list[dict], start: str, end: str) -> list[dict]:
    if not pages:
        raise ValueError("No catalog pages")
    hits = int(pages[0]["hits"])
    count = int(pages[0]["TotalPage"])
    if count < 1 or len(pages) != count:
        raise ValueError("Missing catalog pages")
    rows = []
    for number, payload in enumerate(pages, 1):
        if int(payload["hits"]) != hits or int(payload["TotalPage"]) != count:
            raise ValueError("Source totals changed during pagination")
        rows.extend(validate_page(payload, start, end, number))
    if len(rows) != hits:
        raise ValueError("Catalog row count differs from source total")
    ids = [row["infoCode"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate report identity; reconcile without dropping rows")
    return rows


def project_and_sample(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    catalog = []
    for row in rows:
        date = row["publishDate"][:10]
        if not "2024-01-01" <= date <= "2025-12-31":
            raise ValueError("Out-of-period report cannot enter the catalog")
        if str(row["orgCode"]) != BROKER:
            raise ValueError("Unexpected broker")
        catalog.append({"info_code": row["infoCode"], "stock_code": row["stockCode"],
            "broker_code": str(row["orgCode"]), "stock_name": row["stockName"],
            "title": row["title"], "display_date": date,
            "half": date[:4] + ("H1" if date[5:7] <= "06" else "H2"),
            "sample_hash": hashlib.sha256(row["infoCode"].encode()).hexdigest()})
    if len({r["info_code"] for r in catalog}) != len(catalog):
        raise ValueError("Report identity occurs across months")
    catalog.sort(key=lambda row: (row["display_date"], row["info_code"]))
    sample = []
    for half in ("2024H1", "2024H2", "2025H1", "2025H2"):
        members = sorted((r for r in catalog if r["half"] == half),
                         key=lambda r: (r["sample_hash"], r["info_code"]))
        if len(members) < 8:
            raise ValueError("Half-year has fewer than eight reports")
        sample.extend(members[:8])
    return catalog, sample


def build(root: Path = ROOT, fetch: bool = False) -> dict:
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.trust_env = False
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://data.eastmoney.com/"})
    files = []

    def read_page(start: str, end: str, number: int):
        path = raw / f"{start[:7]}_{number:03d}.json"
        params = parameters(start, end, number)
        if not path.exists():
            if not fetch:
                raise FileNotFoundError(path)
            response = session.get(ENDPOINT, params=params, timeout=40)
            response.raise_for_status()
            payload = response.json()
            validate_page(payload, start, end, number)
            path.write_bytes(response.content)
            save_json(path.with_suffix(".request.json"), {"url": response.url,
                "parameters": params, "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "sha256": sha(path)})
            time.sleep(0.2)
        payload = json.loads(path.read_text())
        manifest = json.loads(path.with_suffix(".request.json").read_text())
        if manifest["parameters"] != params or manifest["sha256"] != sha(path):
            raise ValueError("Frozen response or request changed")
        files.append({"path": str(path), "sha256": sha(path)})
        return payload

    rows, monthly = [], []
    for start, end in periods():
        first = read_page(start, end, 1)
        pages = [first] + [read_page(start, end, page) for page in range(2, int(first["TotalPage"]) + 1)]
        checked = finish_month(pages, start, end)
        rows.extend(checked)
        monthly.append({"month": start[:7], "reports": len(checked), "pages": len(pages)})
        print(monthly[-1], flush=True)
    catalog, sample = project_and_sample(rows)
    save_json(root / "catalog.json", catalog)
    save_json(root / "sample.json", sample)
    report = {"rule_commit": RULE_COMMIT, "broker_code": BROKER,
        "reports": len(catalog), "unique_stocks": len({r["stock_code"] for r in catalog}),
        "monthly": monthly, "source_files": files,
        "catalog_sha256": sha(root / "catalog.json"), "sample_sha256": sha(root / "sample.json"),
        "sample_rule": "eight_per_display_date_half_by_sha256_info_code",
        "projected_dynamic_forecasts": False, "market_data_read": False, "returns_read": False}
    save_json(root / "catalog_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fetch", action="store_true")
    args = parser.parse_args()
    result = build(args.root, args.fetch)
    print({k: result[k] for k in ("reports", "unique_stocks", "catalog_sha256", "sample_sha256")})


if __name__ == "__main__":
    main()
