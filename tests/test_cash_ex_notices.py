from copy import deepcopy

import pytest

from trade_research.cash_ex_notice_sources import extract_pages
from trade_research.cash_ex_notices import decision_checks, parse_terms, reviewed_terms


def sz_text(section="二、权益分派方案", cash="5.8", footer=False):
    pages = [f"证券代码：000963\n一、股东大会审议通过的方案\n每10股派5元人民币现金\n"
             f"{section}\n公司本次分配以现有总股本剔除已回购股份0股后的1000股为基数，"
             f"向全体股东每10股派{cash}" + ("\n1" if footer else ""),
             "元人民币现金（含税；扣税后每10股派5.22元），不送红股，不以资本公积金转增股本。\n"
             "三、股权登记日与除权除息日\n股权登记日为2024年5月27日，除权除息日为2024年5月28日。\n"
             "五、权益分派方法\n现金红利将于2024年5月28日直接划入资金账户。\n2"]
    return extract_pages(pages)


@pytest.mark.parametrize("section", ["一、权益分派方案", "二、本次实施的利润分配方案情况", "二、权益分配"])
def test_gross_implementation_cash_is_not_proposal_or_after_tax_cash(section):
    terms = parse_terms(sz_text(section), "sz.000963")
    assert terms["cash_per_share"] == "0.58"
    assert terms["reference_cash_per_share"] == "0.58"
    assert not terms["positive_share_distribution"]
    assert terms["issues"] == []


def test_page_counter_cannot_change_cash_amount_or_split_payment_date():
    text = sz_text(footer=True)
    assert parse_terms(text, "sz.000963")["cash_per_share"] == "0.58"
    assert "5.81" not in text.replace("\n", "")
    assert extract_pages(["现金红利将于\n1", "2024年5月28日\n2"]).replace("\n", "") == "现金红利将于2024年5月28日"
    assert extract_pages(["正文\n12"]) == "正文\n12"


def test_repeated_security_header_cannot_interrupt_per_ten_share_amount():
    text = extract_pages(["每10股\n1", "证券代码：002812 股票简称：某公司\n债券代码：128095\n派15.426097元\n2"])
    assert text.replace("\n", "") == "每10股派15.426097元"


def test_shanghai_positive_transfer_wording_is_not_pure_cash():
    text = ("证券代码：603927\n每股分配比例，每股转增比例 A股每股现金红利0.55元 每股转增股份0.4股\n"
            "相关日期 A股 2024/5/20 - 2024/5/21 2024/5/21\n差异化分红送转：否")
    terms = parse_terms(text, "sh.603927")
    assert terms["positive_share_distribution"]
    assert terms["share_values"] == ["0.4"]
    assert terms["action_date"] == "2024-05-21"


def test_shenzhen_table_dates_and_differentiated_reference():
    text = sz_text().replace("股权登记日为2024年5月27日，除权除息日为2024年5月28日。",
        "股权登记日 最后交易日 除权除息日 现金红利发放日\nA股 2024/5/27 -- 2024/5/28 2024/5/28\n差异化分红：是")
    text += "除权除息参考价=股权登记日（2024年5月27日）收盘价−0.579元/股。"
    terms = parse_terms(text, "sz.000963")
    assert terms["record_date"] == "2024-05-27"
    assert terms["action_date"] == terms["pay_date"] == "2024-05-28"
    assert terms["reference_cash_per_share"] == "0.579"


def check_fixture():
    row = {"date": "2024-05-28", "dividRegistDate": "2024-05-27", "dividPayDate": "2024-05-28",
           "dividCashPsBeforeTax": ".58", "action_type": "provisional_pure_cash", "previous_close": 31.93,
           "preclose": 31.35, "previous_market_date": "2024-05-27"}
    source = {"notice_date": "2024-05-20"}
    return row, parse_terms(sz_text(), "sz.000963"), source


def test_primary_correction_is_preserved_but_wrong_reference_blocks():
    row, parsed, source = check_fixture()
    row["dividCashPsBeforeTax"] = ".57"
    parsed["pay_date"] = "2024-05-27"
    issues, blocking, relevant = decision_checks(row, parsed, source)
    assert relevant and not blocking
    assert issues == ["cash_vendor_disagreement", "pay_date_vendor_disagreement"]
    row["preclose"] = 31.36
    assert "reference_price_disagreement" in decision_checks(row, parsed, source)[1]


def test_missing_reference_is_only_optional_for_proven_ineligible_cash_yield():
    row, parsed, source = check_fixture()
    parsed["reference_cash_per_share"] = None
    parsed["issues"] = ["reference_cash_template"]
    assert "reference_cash_template" in decision_checks(row, parsed, source)[1]
    row["previous_close"] = 100.0
    assert decision_checks(row, parsed, source)[1:] == ([], False)


def test_primary_action_date_mismatch_cannot_be_hidden_by_source_expected_date():
    row, parsed, source = check_fixture()
    parsed["action_date"] = "2024-05-29"
    source["expected_action_date"] = "2024-05-28"
    assert "action_date_vendor_disagreement" in decision_checks(row, parsed, source)[1]
    assert parsed["action_date"] == "2024-05-29"


def test_manual_reference_requires_exact_source_and_independent_arithmetic():
    source = {"code": "sh.600522", "expected_action_date": "2025-07-25", "announcement_id": "id",
              "notice_date": "2025-07-18", "source_url": "url", "source_sha256": "digest"}
    review = {**source, "evidence": ["明确股数"], "fields": {"reference_cash_per_share": "0.29834"},
              "resolves": ["multiple_reference_formulas"], "reason": "原件等式有冲突，按股数复算",
              "arithmetic": {"participating_shares": "3394039552", "total_shares": "3412949652",
                             "cash_per_share": ".3", "quantum": ".00001"}}
    parsed = {"issues": ["multiple_reference_formulas"], "reference_cash_per_share": None}
    assert reviewed_terms(parsed, "明确股数", source, review)["reference_cash_per_share"] == "0.29834"
    bad = deepcopy(review)
    bad["fields"]["reference_cash_per_share"] = "0.29865"
    with pytest.raises(ValueError, match="arithmetic"):
        reviewed_terms(parsed, "明确股数", source, bad)
    with pytest.raises(ValueError, match="exact source"):
        reviewed_terms(parsed, "明确股数", {**source, "source_sha256": "changed"}, review)
