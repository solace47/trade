"""Conservatively screen original issuer notices of the first actual buyback."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import pandas as pd


FIRST = re.compile(
    r"首次回购(?:公司|本公司|部分)?(?:A股|社会公众)?(?:股份|股票|A股)|"
    r"(?:股份|股票|A股).{0,5}首次回购"
)
NOT_ISSUER_A_SHARE = re.compile(
    r"限制性股票|激励计划|回购注销|解除限售|调整回购价格|"
    r"法律意见|律师|保荐机构|独立财务顾问|受托管理|"
    r"已获授|授予部分|行权期|首次减持|境内上市外资股|B股|H股|"
    r"更正|补充|修订|完成|完毕|终止"
)


def strict_first_title(title: str) -> bool:
    return bool(FIRST.search(title) and not NOT_ISSUER_A_SHARE.search(title))


def screen(source_dir: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for year in (2024, 2025):
        source = pd.read_parquet(source_dir / f"search_{year}.parquet")
        if (source.empty or source.pdf_url.duplicated().any()
                or not source.notice_date.str.startswith(str(year)).all()):
            raise ValueError("Malformed official first-buyback index")
        picked = source.loc[source.title.map(strict_first_title)].copy()
        if picked.empty:
            raise ValueError("No conservative issuer first-buyback notices")
        report[str(year)] = {
            "a_share_search_pdfs": len(source),
            "conservative_titles": len(picked),
            "code_date_duplicate_rows": int(picked.duplicated(
                ["code", "notice_date"], keep=False).sum()),
            "stocks": int(picked.code.nunique()),
            "notice_days": int(picked.notice_date.nunique()),
            "peak_month": picked.notice_date.str[:7].value_counts().idxmax(),
            "peak_month_share": float(
                picked.notice_date.str[:7].value_counts().max() / len(picked)),
        }
        picked.to_parquet(output_dir / f"title_candidates_{year}.parquet",
                          index=False, compression="zstd")
    (output_dir / "title_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=Path("data/research/first_buyback"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/research/first_buyback"))
    args = parser.parse_args()
    print(screen(args.source, args.output))


if __name__ == "__main__":
    main()
