"""Extract annual profit and operating cash flow from original CNINFO PDFs.

The first financial table in an annual-report summary is used, before the
quarterly table. Values are kept in the report's own units; their ratio is
unitless. Extraction failures are never filled from later vendor data.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import pandas as pd
import pdfplumber


NUMBER = re.compile(r"^-?\d+(?:\.\d+)?$")
NUMBER_TOKEN = re.compile(r"(?<!\S)(-?\(?\d[\d,]*(?:\.\d+)?\)?)(?!\S)")
PROFIT_LABELS = ("归属于上市公司股东的净利润", "归属于母公司所有者的净利润",
                 "归属于母公司股东的净利润", "归属于本行股东的净利润",
                 "归属于本公司股东的净利润")
CASH_LABEL = "经营活动产生的现金流量净额"
ANNUAL_LABELS = (*PROFIT_LABELS, CASH_LABEL)


def _number(cell: object) -> float | None:
    if cell is None:
        return None
    value = re.sub(r"\s+", "", str(cell)).replace(",", "")
    value = value.replace("−", "-").replace("－", "-")
    if value.startswith("(") and value.endswith(")"):
        value = "-" + value[1:-1]
    return float(value) if NUMBER.fullmatch(value) else None


def _table_values(table: list[list[object]]):
    for offset, row in enumerate(table):
        numeric = [(position, value) for position, cell in enumerate(row)
                   if (value := _number(cell)) is not None]
        if not numeric:
            continue
        first_numeric = numeric[0][0]
        label = "".join(str(cell or "") for cell in row[:first_numeric])
        for following in table[offset + 1:offset + 7]:
            if any(_number(cell) is not None for cell in following):
                break
            label += "".join(str(cell or "") for cell in following)
        label = re.sub(r"\s+", "", label)
        yield label, numeric[0][1]


def _quarter_header(row: list[object]) -> bool:
    text = "".join(str(cell or "") for cell in row).replace("\n", "")
    return "第一季度" in text and "第二季度" in text


def _leading_table_label(table: list[list[object]]) -> str:
    """Read a label continued at the top of the following PDF page."""
    parts = []
    for row in table[:6]:
        if any(_number(cell) is not None for cell in row):
            break
        parts.append("".join(str(cell or "") for cell in row))
    return re.sub(r"\s+", "", "".join(parts))


def _text_values(lines: list[tuple[int, str]]):
    targets = set((*PROFIT_LABELS, CASH_LABEL))
    for index, (page, line) in enumerate(lines):
        match = NUMBER_TOKEN.search(line)
        if match is None:
            continue
        value = _number(match.group(1))
        if value is None:
            continue
        prefix = re.sub(r"\s+", "", line[:match.start()])
        before = []
        for _, previous in reversed(lines[max(0, index - 4):index]):
            if NUMBER_TOKEN.search(previous):
                break
            before.insert(0, re.sub(r"\s+", "", previous))
        after = []
        for _, following in lines[index + 1:index + 5]:
            if NUMBER_TOKEN.search(following):
                break
            after.append(re.sub(r"\s+", "", following))
        for left in range(len(before) + 1):
            for right in range(len(after) + 1):
                label = "".join(before[len(before) - left:]) + prefix
                label += "".join(after[:right])
                if label in targets:
                    yield label, value, page
                    break


def extract(pdf_path: Path) -> dict:
    values: dict[str, float] = {}
    pages: dict[str, int] = {}
    text_lines: list[tuple[int, str]] = []
    with pdfplumber.open(pdf_path) as pdf:
        quarterly_reached = False
        pending: tuple[str, float, int] | None = None
        for page_number, page in enumerate(pdf.pages[:60], start=1):
            next_pending = None
            text = page.extract_text() or ""
            for line in text.splitlines():
                if ("报告期分季度" in line
                        or ("第一季度" in line and "第二季度" in line)):
                    quarterly_reached = True
                    break
                if (not re.fullmatch(r"\s*\d+\s*", line)
                        and "年度报告摘要" not in line):
                    text_lines.append((page_number, line))
            for table in page.extract_tables():
                quarterly_at = next((i for i, row in enumerate(table)
                                     if _quarter_header(row)), len(table))
                annual_table = table[:quarterly_at]
                if pending is not None:
                    prefix, value, value_page = pending
                    label = prefix + _leading_table_label(annual_table)
                    if label in PROFIT_LABELS and "parent_profit_raw" not in values:
                        values["parent_profit_raw"] = value
                        pages["parent_profit_page"] = value_page
                    elif label == CASH_LABEL and "operating_cash_raw" not in values:
                        values["operating_cash_raw"] = value
                        pages["operating_cash_page"] = value_page
                    pending = None
                for label, value in _table_values(annual_table):
                    if ("parent_profit_raw" not in values
                            and any(field in label for field in PROFIT_LABELS)
                            and "扣除" not in label):
                        values["parent_profit_raw"] = value
                        pages["parent_profit_page"] = page_number
                    if "operating_cash_raw" not in values and CASH_LABEL in label:
                        values["operating_cash_raw"] = value
                        pages["operating_cash_page"] = page_number
                    if (len(label) >= 4 and any(target.startswith(label)
                            and target != label for target in ANNUAL_LABELS)):
                        next_pending = (label, value, page_number)
                if quarterly_at < len(table):
                    quarterly_reached = True
                    break
            if len(values) == 2:
                break
            if quarterly_reached or "报告期分季度" in text:
                break
            pending = next_pending
    for label, value, page_number in _text_values(text_lines):
        if "parent_profit_raw" not in values and label in PROFIT_LABELS:
            values["parent_profit_raw"] = value
            pages["parent_profit_page"] = page_number
        if "operating_cash_raw" not in values and label == CASH_LABEL:
            values["operating_cash_raw"] = value
            pages["operating_cash_page"] = page_number
        if len(values) == 2:
            break
    if len(values) != 2:
        raise ValueError(f"Annual profit/cash-flow table unreadable: {pdf_path}")
    if abs(pages["parent_profit_page"] - pages["operating_cash_page"]) > 1:
        raise ValueError(f"Annual profit/cash-flow tables too far apart: {pdf_path}")
    if values["parent_profit_raw"] == 0:
        raise ValueError(f"Annual parent profit is zero: {pdf_path}")
    return {**values, **pages,
            "cash_to_parent_profit": (values["operating_cash_raw"]
                                      / values["parent_profit_raw"])}


def _download(url: str, pdf_path: Path) -> None:
    if pdf_path.exists():
        return
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    temp = pdf_path.with_suffix(".tmp")
    command = ["curl", "-fLsS", "--max-time", "30", url, "-o", str(temp)]
    result = subprocess.run(command, capture_output=True, text=True,
                            check=False)
    if result.returncode or not temp.exists() or temp.stat().st_size < 1000:
        temp.unlink(missing_ok=True)
        raise RuntimeError(f"Original PDF download failed: {url}")
    with temp.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            temp.unlink()
            raise ValueError(f"Downloaded file is not a PDF: {url}")
    temp.replace(pdf_path)


def collect(index_path: Path, output: Path, pdf_cache: Path,
            universe_path: Path,
            max_reports: int | None = None,
            retry_failed: bool = False) -> dict:
    index = pd.read_parquet(index_path)
    index = index.loc[index.kind.eq("summary")].sort_values(
        ["report_year", "code"])
    allowed = set(pd.read_parquet(universe_path, columns=["code"]).code)
    index = index.loc[index.code.isin(allowed)]
    if index.duplicated(["report_year", "code"]).any():
        raise ValueError("Duplicate original summary in announcement index")
    output.parent.mkdir(parents=True, exist_ok=True)
    statuses = {}
    if output.exists():
        for line in output.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            statuses[(item["report_year"], item["code"])] = item["status"]
    prior_success = sum(value == "ok" for value in statuses.values())
    prior_failed = sum(value == "failed" for value in statuses.values())
    attempted = 0
    success = 0
    failures = 0
    with output.open("a", encoding="utf-8") as stream:
        for row in index.itertuples():
            key = (row.report_year, row.code)
            if statuses.get(key) == "ok" or (statuses.get(key) == "failed"
                                             and not retry_failed):
                continue
            if max_reports is not None and attempted >= max_reports:
                break
            attempted += 1
            pdf_path = pdf_cache / str(row.report_year) / f"{row.code}.pdf"
            item = {"report_year": row.report_year, "code": row.code,
                    "notice_date": row.notice_date, "pdf_url": row.pdf_url}
            try:
                _download(row.pdf_url, pdf_path)
                item.update(extract(pdf_path))
                item["status"] = "ok"
                success += 1
            except (OSError, ValueError, RuntimeError) as error:
                item["status"] = "failed"
                item["reason"] = str(error)[:200]
                failures += 1
            stream.write(json.dumps(item, ensure_ascii=False) + "\n")
            stream.flush()
    return {"attempted": attempted, "success": success,
            "failures": failures, "prior_success": prior_success,
            "prior_failed": prior_failed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, required=True, choices=(2023, 2024))
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pdf-cache", type=Path, default=Path(
        "data/research/annual_cash_quality/original_summaries"))
    parser.add_argument("--universe", type=Path, default=Path(
        "data/research/margin_universe_quintiles.parquet"))
    parser.add_argument("--max-reports", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    base = Path("data/research/annual_cash_quality")
    index = args.index or base / f"cninfo_original_{args.year}.parquet"
    output = args.output or base / f"original_cash_{args.year}.jsonl"
    print(collect(index, output, args.pdf_cache, args.universe,
                  args.max_reports, args.retry_failed))


if __name__ == "__main__":
    main()
