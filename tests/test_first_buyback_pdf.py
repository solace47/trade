from types import SimpleNamespace

import pytest

from scripts.audit_buyback_pdfs import _download, _first_trade_confirmed


def test_first_trade_needs_positive_executed_money() -> None:
    assert _first_trade_confirmed(
        "首次回购公司股份。成交总金额为2,999,255.00元（不含费用）。")
    assert _first_trade_confirmed(
        "首次回购公司股份。累计已回购金额2,000.33万元。")
    assert _first_trade_confirmed(
        "首次回购公司股份。已支付的资金总额为人民币3,066,691.27元。")
    assert _first_trade_confirmed(
        "首次回购公司股份。已使用资金总额为4,975,137元。")
    assert _first_trade_confirmed(
        "首次回购公司A股股份。累计已回购A股金额约1,999.09万元。")
    assert _first_trade_confirmed(
        "首次回购公司股份50,200股，最低成交价21.40元/股，回购总金额1,073,428元。")
    assert _first_trade_confirmed(
        "首次回购公司股份。交易总金额为3,137,700元。")
    assert not _first_trade_confirmed(
        "首次回购公司股份。预计回购金额2,000万元，尚未实施。")
    assert not _first_trade_confirmed(
        "首次回购公司股份。成交金额为0元。")
    assert not _first_trade_confirmed(
        "首次回购公司股份。拟回购总金额为2,000万元，尚未实施。")
    assert not _first_trade_confirmed(
        "首次回购公司B股股份。成交总金额为3,000,000元。")


def test_pdf_cache_rejects_html_even_when_large(tmp_path, monkeypatch) -> None:
    target = tmp_path / "notice.PDF"
    target.write_bytes(b"<html>" + b"x" * 2000)

    def fake_curl(command, **kwargs):
        (tmp_path / "notice.tmp").write_bytes(b"<html>" + b"x" * 2000)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("scripts.audit_buyback_pdfs.subprocess.run", fake_curl)
    with pytest.raises(RuntimeError, match="download failed"):
        _download("https://static.cninfo.com.cn/notice.PDF", target)
