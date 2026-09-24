"""Screen new major-holder share-sale plans in original CNINFO notices."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd


ACTOR = re.compile(
    r"控股股东|实际控制人|董事长|持股5%以上|持股5%（含）以上|"
    r"第一大股东|大股东"
)
PLAN = re.compile(r"减持.{0,12}计划|计划.{0,12}减持|拟减持")
NOT_NEW = re.compile(
    r"进展|完成|完毕|结果|终止|届满|延期|延长|变更|更正|补充|"
    r"修订|调整|时间过半|期限过半|法律意见|律师|财务顾问|"
    r"核查意见|B股|H股|境内上市外资股|回购|已减持|"
    r"预披露的进展|股份质押|内部转让"
)


def strict_plan_title(title: str) -> bool:
    return bool(ACTOR.search(title) and PLAN.search(title)
                and not NOT_NEW.search(title))


FUTURE_TEXT = re.compile(
    r"拟减持|计划减持|拟通过.{0,50}减持|"
    r"计划通过.{0,50}减持|拟自.{0,70}减持|"
    r"计划自.{0,70}减持|拟在.{0,70}减持|"
    r"计划在.{0,70}减持"
)


def confirm_pdf_text(text: str, code: str) -> bool:
    clean = re.sub(r"\s+", "", text)
    return bool(code.split(".")[-1] in clean
                and ACTOR.search(clean)
                and "减持" in clean and "股份" in clean
                and ("减持计划" in clean or FUTURE_TEXT.search(clean)))


def screen(source_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for year in (2024, 2025):
        source = pd.read_parquet(source_dir / f"search_{year}.parquet")
        if (source.empty or source.pdf_url.duplicated().any()
                or not source.notice_date.str.startswith(str(year)).all()):
            raise ValueError("Malformed CNINFO major-holder-sale index")
        picked = source.loc[source.title.map(strict_plan_title)].copy()
        if picked.empty:
            raise ValueError("No original major-holder-sale plan candidates")
        month = picked.notice_date.str[:7].value_counts()
        report[str(year)] = {
            "a_share_search_pdfs": len(source),
            "conservative_titles": len(picked),
            "duplicate_code_notice_rows": int(picked.duplicated(
                ["code", "notice_date"], keep=False).sum()),
            "stocks": int(picked.code.nunique()),
            "notice_days": int(picked.notice_date.nunique()),
            "peak_month": month.idxmax(),
            "peak_month_share": float(month.max() / len(picked)),
        }
        picked.sort_values(["notice_date", "code", "pdf_url"]).to_parquet(
            output_dir / f"title_candidates_{year}.parquet", index=False,
            compression="zstd")
    (output_dir / "title_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path(
        "data/research/insider_sell"))
    parser.add_argument("--output", type=Path, default=Path(
        "data/research/insider_sell"))
    args = parser.parse_args()
    print(screen(args.source, args.output))


if __name__ == "__main__":
    main()
