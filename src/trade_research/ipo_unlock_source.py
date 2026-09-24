"""Conservatively identify IPO pre-issue share unlocks in CNINFO PDFs.

This module reads disclosure metadata and original text only. A returned date
or share count is rejected when the first pages give conflicting values.
"""

from __future__ import annotations

from datetime import date
import re


TITLE_KIND = re.compile(
    r"首次公开发行前.{0,12}(?:限售|股份)|首发前已发行|"
    r"首次公开发行部分限售|首发限售股"
)
TITLE_FLOW = re.compile(r"限售.{0,15}(?:上市流通|解禁上市流通)")
TITLE_REJECT = re.compile(
    r"核查意见|保荐|法律意见|律师|财务顾问|独立董事|问询|回复|"
    r"上市公告书|募集说明书|更正|补充|进展|摘要|网下配售|"
    r"战略配售|股权激励|限制性股票激励|员工持股|可转债|"
    r"可转换公司债券|向特定对象发行|非公开发行|"
    r"发行股份购买资产|自愿承诺不减持"
)
PDF_PREISSUE = re.compile(r"首次公开发行前|首发前")
UNLOCK_DAY = re.compile(
    r"本次.{0,30}?上市流通(?:的)?(?:日期|日|时间).{0,20}?"
    r"(?P<year>20\d{2})年(?P<month>\d{1,2})月(?P<day>\d{1,2})日"
)
SHARES = (
    re.compile(r"本次股票上市流通总数为(?P<count>\d[\d,]*)股"),
    re.compile(r"本次.{0,110}?(?:股份数量|股份的数量)为"
               r"(?P<count>\d[\d,]*)股"),
)


def strict_title(title: str) -> bool:
    return bool(TITLE_KIND.search(title) and TITLE_FLOW.search(title)
                and not TITLE_REJECT.search(title))


def extract(text: str, code: str, notice_date: str) -> dict:
    clean = re.sub(r"\s+", "", text)
    evidence = clean[:3000]
    if (code.split(".")[-1] not in evidence
            or not PDF_PREISSUE.search(evidence)
            or "限售" not in evidence or "上市流通" not in evidence):
        return {"status": "identity_or_type_unconfirmed"}
    dates = set()
    for match in UNLOCK_DAY.finditer(evidence):
        try:
            dates.add(date(int(match["year"]), int(match["month"]),
                           int(match["day"])).isoformat())
        except ValueError:
            return {"status": "invalid_unlock_date"}
    if len(dates) != 1:
        return {"status": "missing_or_ambiguous_unlock_date",
                "date_candidates": sorted(dates)}
    unlock_date = next(iter(dates))
    if unlock_date <= notice_date:
        return {"status": "not_disclosed_before_unlock",
                "unlock_date": unlock_date}
    counts = set()
    for pattern in SHARES:
        counts.update(int(match["count"].replace(",", ""))
                      for match in pattern.finditer(evidence))
    if not counts or min(counts) <= 0 or len(counts) != 1:
        return {"status": "missing_or_ambiguous_share_count",
                "unlock_date": unlock_date,
                "share_candidates": sorted(counts)}
    return {"status": "ok", "unlock_date": unlock_date,
            "unlock_shares": next(iter(counts))}
