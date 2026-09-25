from pathlib import Path

import pandas as pd
import pytest

from scripts.audit_pledge_release_pdfs import audit


def test_pdf_audit_rejects_truncated_title_subset(tmp_path: Path) -> None:
    rows = pd.DataFrame([
        {"pdf_url": "https://static.cninfo.com.cn/finalpage/2024-01-02/1.PDF",
         "notice_date": "2024-01-02", "code": "sh.600001",
         "title": "关于控股股东股份解除质押的公告"},
        {"pdf_url": "https://static.cninfo.com.cn/finalpage/2024-01-03/2.PDF",
         "notice_date": "2024-01-03", "code": "sh.600002",
         "title": "关于控股股东部分股份解除质押的公告"},
    ])
    rows.to_parquet(tmp_path / "search_2024.parquet", index=False)
    rows.head(1).to_parquet(tmp_path / "title_candidates_2024.parquet",
                            index=False)
    with pytest.raises(ValueError, match="incomplete"):
        audit(2024, tmp_path)
    assert not (tmp_path / "pdfs").exists()
