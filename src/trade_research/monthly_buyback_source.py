"""Extract only report-date cumulative cash from monthly buyback originals.

The parser is conservative: missing, mixed, or contradictory cash figures
remain unresolved for original-PDF review and cannot create a signal.
"""

from __future__ import annotations

import calendar
from datetime import date
from decimal import Decimal
import re


CASH = re.compile(
    r"(?:成交总金额|成交金额|支付的资金总额|已支付的资金总额|"
    r"已支付的总金额|支付的总金额|支付的金额总额|"
    r"支付的金额|支付的总额|支付金额|已支付金额|"
    r"累计支付金额|累计回购金额)"
    r"(?:约为|合计为|为|约|合计|达|达到|人民币|：|:|共|了|的|总额|\s){0,10}"
    r"(?:人民币)?"
    r"(?P<number>(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)"
    r"(?P<unit>亿|万)?元"
)


def previous_month_end(notice_date: str) -> date:
    day = date.fromisoformat(notice_date)
    year, month = (day.year - 1, 12) if day.month == 1 else (day.year,
                                                              day.month - 1)
    return date(year, month, calendar.monthrange(year, month)[1])


def extract_cumulative_cash(text: str, notice_date: str) -> Decimal | None:
    """Read the unique cumulative RMB amount following last month's end date."""
    if not text or "集中竞价" not in text or "回购" not in text:
        return None
    clean = re.sub(r"\s+", "", text)
    report = previous_month_end(notice_date)
    anchor = f"截至{report.year}年{report.month}月{report.day}日"
    values = set()
    for match in re.finditer(re.escape(anchor), clean):
        window = clean[match.end():match.end() + 500]
        # Do not parse a planned amount if the as-of sentence does not state
        # an actual centralized-auction purchase.
        if "回购" not in window[:230]:
            continue
        for cash in CASH.finditer(window):
            if cash.start() > 300:
                continue
            value = Decimal(cash["number"].replace(",", ""))
            unit = cash["unit"]
            if unit == "万":
                value *= 10_000
            elif unit == "亿":
                value *= 100_000_000
            values.add(value)
    return next(iter(values)) if len(values) == 1 else None
