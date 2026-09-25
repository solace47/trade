"""Archive and audit first executed controller/chairman A-share purchases.

This module reads original announcement metadata and PDF text only. It must
never open post-signal market returns.
"""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import re

import pandas as pd


ACTOR = re.compile(r"控股股东|实际控制人|董事长")
EXCLUDE = re.compile(r"H股|B股|员工持股|更正|补充|律师|法律意见")
TRADE = re.compile(
    r"(?=((?:通过|采用|以|二级市场).{0,60}?"
    r"(?:集中竞价|交易系统|大宗交易).{0,90}?"
    r"增持.{0,40}?(?P<shares>\d[\d,.]*(?:万|亿)?股)))"
)
DAY = re.compile(r"20\d{2}年\d{1,2}月\d{1,2}日")
PROSPECTIVE = re.compile(r"拟|计划|将|不低于|不超过|预计")
FUTURE_DATE = re.compile(r"(?:拟|计划)(?:自|于|在|从).{0,8}$")
FUTURE_ACTION = re.compile(r"拟|将|预计|计划(?:自|于|在|通过|以|采用)")
AMOUNT = re.compile(
    r"(?:增持金额|成交总额|成交金额|耗资).{0,16}?"
    r"(?P<number>\d[\d,]*(?:\.\d+)?)\s*(?P<unit>亿|万)?元"
)


def strict_first_title(title: str) -> bool:
    return "首次增持" in title and bool(ACTOR.search(title)) and not EXCLUDE.search(title)


def screen(source_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for year in (2024, 2025):
        source = pd.read_parquet(source_dir / f"search_{year}.parquet")
        if (source.empty or source.pdf_url.duplicated().any()
                or not source.notice_date.str.startswith(str(year)).all()):
            raise ValueError("Incomplete or malformed first-increase search index")
        selected = source.loc[source.title.map(strict_first_title)].copy()
        if selected.empty or selected.pdf_url.duplicated().any():
            raise ValueError("No unique first-increase original candidates")
        month = selected.notice_date.str[:7].value_counts()
        report[str(year)] = {
            "a_share_search_pdfs": len(source),
            "conservative_titles": len(selected),
            "stocks": int(selected.code.nunique()),
            "notice_days": int(selected.notice_date.nunique()),
            "peak_month": month.idxmax(),
            "peak_month_share": float(month.max() / len(selected)),
        }
        selected.sort_values(["notice_date", "code", "pdf_url"]).to_parquet(
            output_dir / f"title_candidates_{year}.parquet", index=False,
            compression="zstd")
    (output_dir / "title_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def _quantity(token: str) -> float:
    value = token.removesuffix("股").replace(",", "")
    factor = 1
    if value.endswith("万"):
        value, factor = value[:-1], 10_000
    elif value.endswith("亿"):
        value, factor = value[:-1], 100_000_000
    return float(value) * factor


def _date(token: str) -> date:
    return date(*(int(part) for part in re.findall(r"\d+", token)))


def confirm_original(text: str, code: str, notice_date: str) -> dict:
    """Find a dated, completed market purchase; never infer one from a plan."""
    clean = re.sub(r"\s+", "", text)
    if code.split(".")[-1] not in clean or not ACTOR.search(clean):
        return {"status": "identity_unconfirmed"}
    notice = date.fromisoformat(notice_date)
    eligible = []
    for match in TRADE.finditer(clean):
        segment = match.group(1)
        prefix = clean[max(0, match.start() - 110):match.start()]
        if PROSPECTIVE.search(segment) or FUTURE_ACTION.search(prefix[-10:]):
            continue
        dates = [(item, _date(item.group())) for item in DAY.finditer(prefix)]
        dates = [(item, day) for item, day in dates
                 if 0 <= (notice - day).days <= 20]
        shares = _quantity(match.group("shares"))
        if not dates or shares <= 0:
            continue
        last_date, buy_date = dates[-1]
        if (FUTURE_DATE.search(prefix[max(0, last_date.start() - 20):
                                      last_date.start()])
                or FUTURE_ACTION.search(prefix[last_date.end():])):
            continue
        if ("H股" in clean and "A股" not in segment
                and not ("上海证券交易所" in segment
                         or "深圳证券交易所" in segment)):
            continue
        eligible.append((match.start(), buy_date, shares, segment))
    if not eligible:
        return {"status": "executed_a_share_or_date_unconfirmed"}
    position, buy_date, shares, segment = max(eligible, key=lambda item: item[0])
    after_trade = clean[position + len(segment):position + len(segment) + 120]
    amount = AMOUNT.search(after_trade)
    parsed_amount = None
    if amount and not PROSPECTIVE.search(after_trade[:amount.start()]):
        factor = {None: 1, "万": 10_000, "亿": 100_000_000}[amount.group("unit")]
        parsed_amount = float(amount.group("number").replace(",", "")) * factor
    return {"status": "ok", "buy_date": buy_date.isoformat(),
            "shares": shares, "parsed_actual_amount_yuan": parsed_amount,
            "trade_evidence": segment}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/first_insider_buy"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/first_insider_buy"))
    args = parser.parse_args()
    print(screen(args.source, args.output))


if __name__ == "__main__":
    main()
