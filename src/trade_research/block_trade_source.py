"""Archive complete SSE/SZSE agreement block trades by trade date.

Both exchanges publish these records after their trade date's 14:50 signal.
Raw rows include non-A-share securities; stock-universe filtering is downstream.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

from .exchange_public_events import trading_dates


SSE_PAGE = "https://www.sse.com.cn/disclosure/diclosure/block/deal/dzjyxx/"
SZSE_PAGE = "https://www.szse.cn/disclosure/deal/block/equity/index.html"
SSE_API = "https://query.sse.com.cn/commonQuery.do"
SZSE_API = "https://www.szse.cn/api/report/ShowReport/data"
SSE_SQL = "COMMON_SSE_XXPL_JYXXPL_DZJYXX_L_1"
SZSE_CATALOG = "1932_dzjyzqjy_after"
HEADERS = {"User-Agent": "Mozilla/5.0"}
SSE_PAGE_SIZE = 200


def _request(session: requests.Session, url: str, params: dict,
             referer: str) -> requests.Response:
    for attempt in range(5):
        try:
            response = session.get(url, params=params,
                                   headers={**HEADERS, "Referer": referer},
                                   timeout=25)
            response.raise_for_status()
            return response
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)
    raise AssertionError("Unreachable request retry state")


def _price_amount(value: object) -> float:
    number = float(str(value).replace(",", ""))
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"Invalid block trade quantity: {value!r}")
    return number


def _validate_trade(row: dict, day: str, exchange: str) -> None:
    if exchange == "sse":
        code, date = row.get("stockid"), row.get("tradedate")
        price, qty, amount = (row.get(k) for k in (
            "tradeprice", "tradeqty", "tradeamount"))
    else:
        code, date = row.get("zqdh"), row.get("cjrq")
        price, qty, amount = (row.get(k) for k in (
            "cjjg", "cjgsnew", "cjjenew"))
    if date != day or not isinstance(code, str) or not re.fullmatch(r"\d{6}", code):
        raise ValueError(f"Invalid {exchange} block trade code/date for {day}: {code}/{date}")
    p, q, a = map(_price_amount, (price, qty, amount))
    # Price (yuan) times quantity (10k shares) must equal amount (10k yuan).
    # Both displayed price and displayed 10k-share quantity may be rounded to
    # two decimals.  The price term matters for a high-priced small trade;
    # the quantity term matters for a low-priced fund.
    if abs(p * q - a) > max(0.2, p * 0.005 + q * 0.005 + a * 0.001):
        raise ValueError(f"Inconsistent {exchange} block trade value on {day}: {code}")


def _sse_response(text: str) -> dict:
    match = re.fullmatch(r"cb\((.*)\);?\s*", text, flags=re.DOTALL)
    if match is None:
        raise ValueError("Unexpected SSE block trade JSONP wrapper")
    payload = json.loads(match.group(1))
    page = payload.get("pageHelp")
    if (payload.get("actionErrors") or not isinstance(page, dict)
            or not isinstance(page.get("data"), list)
            or not isinstance(page.get("total"), int)):
        raise ValueError("SSE block trade endpoint returned an error")
    return page


def fetch_sse_day(day: str, session: requests.Session) -> list[dict]:
    rows: list[dict] = []
    page_no = 1
    while True:
        params = {"jsonCallBack": "cb", "isPagination": "true",
                  "pageHelp.pageSize": SSE_PAGE_SIZE,
                  "pageHelp.pageNo": page_no,
                  "pageHelp.beginPage": page_no,
                  "pageHelp.endPage": page_no,
                  "pageHelp.cacheSize": 1, "sqlId": SSE_SQL,
                  "stockId": "", "startDate": day, "endDate": day}
        page = _sse_response(_request(session, SSE_API, params, SSE_PAGE).text)
        if (page.get("pageNo") != page_no or page.get("total") < len(rows)):
            raise ValueError(f"SSE block trade pagination changed on {day}")
        block = page["data"]
        if len(block) > SSE_PAGE_SIZE:
            raise ValueError(f"SSE block trade exceeded page size on {day}")
        rows.extend(block)
        if len(rows) == page["total"]:
            break
        if not block or len(rows) > page["total"]:
            raise ValueError(f"Incomplete SSE block trade pages on {day}")
        page_no += 1
    for row in rows:
        _validate_trade(row, day, "sse")
    return rows


def _szse_tab(text: str) -> dict:
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("Unexpected SZSE block trade response")
    matches = [entry for entry in payload
               if entry.get("metadata", {}).get("tabkey") == "tab2"]
    if len(matches) != 1 or matches[0].get("error") or not isinstance(matches[0].get("data"), list):
        raise ValueError("SZSE agreement-trade tab is missing")
    return matches[0]


def fetch_szse_day(day: str, session: requests.Session) -> list[dict]:
    rows: list[dict] = []
    page_no = 1
    while True:
        params = {"SHOWTYPE": "JSON", "CATALOGID": SZSE_CATALOG,
                  "TABKEY": "tab2", "txtStart": day, "txtEnd": day,
                  "PAGENO": page_no}
        tab = _szse_tab(_request(session, SZSE_API, params, SZSE_PAGE).text)
        meta = tab["metadata"]
        total = meta.get("recordcount")
        if total == 0 and meta.get("pagecount") == 0 and not tab["data"]:
            break
        if (meta.get("pageno") != page_no or not isinstance(total, int)
                or not isinstance(meta.get("pagecount"), int)
                or not isinstance(meta.get("pagesize"), int)
                or len(tab["data"]) > meta["pagesize"]):
            raise ValueError(f"SZSE block trade pagination changed on {day}")
        rows.extend(tab["data"])
        if page_no == meta["pagecount"]:
            if len(rows) != total:
                raise ValueError(f"Incomplete SZSE block trade pages on {day}")
            break
        if not tab["data"] or page_no > meta["pagecount"]:
            raise ValueError(f"Unexpected SZSE empty page on {day}")
        page_no += 1
    for row in rows:
        _validate_trade(row, day, "szse")
    return rows


def fetch_day(day: str) -> dict:
    if not re.fullmatch(r"202[45]-\d\d-\d\d", day):
        raise ValueError("Block trade archive is limited to 2024–2025")
    with requests.Session() as session:
        sse = fetch_sse_day(day, session)
        szse = fetch_szse_day(day, session)
    if not sse and not szse:
        raise ValueError(f"Both exchange block trade lists are empty on {day}")
    return {"schema_version": 1, "trade_date": day,
            "sse": sse, "szse": szse}


def validate_saved(path: Path, day: str) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("schema_version") != 1 or record.get("trade_date") != day:
        raise ValueError(f"Invalid saved block trade envelope: {path}")
    for exchange in ("sse", "szse"):
        rows = record.get(exchange)
        if not isinstance(rows, list):
            raise ValueError(f"Missing saved {exchange} block trades: {path}")
        for row in rows:
            _validate_trade(row, day, exchange)
    if not record["sse"] and not record["szse"]:
        raise ValueError(f"Both saved exchanges are empty on {day}: {path}")
    return record


def archive(calendar: Path, output_dir: Path, start: str, end: str,
            workers: int = 2) -> dict:
    if workers not in (1, 2, 3):
        raise ValueError("Use 1–3 exchange request workers")
    dates = trading_dates(calendar, start, end)
    if not all("2024-01-01" <= d <= "2025-12-31" for d in dates):
        raise ValueError("Do not collect 2023 or 2026 block trades for this study")
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for day in dates:
        path = output_dir / f"{day}.json"
        if path.exists():
            validate_saved(path, day)
        else:
            pending.append(day)
    completed = 0
    for offset in range(0, len(pending), 25):
        batch = pending[offset:offset + 25]
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(fetch_day, day): day for day in batch}
            for future in as_completed(futures):
                day = futures[future]
                record = future.result()
                path = output_dir / f"{day}.json"
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(record, ensure_ascii=False) + "\n",
                                     encoding="utf-8")
                temporary.replace(path)
                completed += 1
        print(f"Block trade days: {completed}/{len(pending)} new; "
              f"{len(dates) - len(pending)} cached", flush=True)
    return {"expected_days": len(dates), "cached_days": len(dates) - len(pending),
            "new_days": len(pending)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calendar", type=Path, default=Path(
        "data/baostock/market_2020_2026/metadata/calendar.parquet"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("data/research/block_trade/daily"))
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    print(archive(args.calendar, args.output_dir, args.start, args.end,
                  args.workers))


if __name__ == "__main__":
    main()
