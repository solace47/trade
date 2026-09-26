"""A failed event lookup must never masquerade as no cash distribution."""

import json
import hashlib

import pandas as pd
import pytest

from trade_research.cash_dividend_catalog import action_type, assemble, checked_response, record_hash
from trade_research.corporate_cash import sha


def event(**changes):
    return dict({"code": "sh.600001", "dividOperateDate": "2024-05-30",
                 "dividCashPsBeforeTax": ".2", "dividStocksPs": "0",
                 "dividReserveToStockPs": "", "dividCashStock": "10派2元（含税）"}, **changes)


def test_vendor_metadata_is_provisional_and_share_distributions_stay_separate():
    assert action_type(event()) == "provisional_pure_cash"
    assert action_type(event(dividReserveToStockPs=".3")) == "share_distribution"
    assert action_type(event(dividStocksPs=".1")) == "share_distribution"
    assert action_type(event(dividCashStock="10转3派2元")) == "conflicting_vendor_action_type"
    assert action_type(event(dividCashPsBeforeTax="NaN")) == "invalid_vendor_fields"
    assert action_type(event(dividCashPsBeforeTax="")) == "no_cash_distribution"


def test_response_must_match_security_and_implementation_year():
    assert checked_response([], "sh.600001", "2024") == []
    for rows in ([event(code="sh.600002")], [event(dividOperateDate="2026-05-30")],
                 [event(), event()], [event(dividOperateDate="2024-99-99")]):
        with pytest.raises(ValueError):
            checked_response(rows, "sh.600001", "2024")


def test_missing_and_error_responses_block_assembly_instead_of_becoming_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("trade_research.cash_dividend_catalog.reviewed_duplicates", lambda: {})
    jobs = pd.DataFrame({"code": ["sh.600001", "sh.600002"], "year": ["2024", "2024"]})
    jobs.to_parquet(tmp_path / "jobs.parquet", index=False)
    (tmp_path / "manifest.json").write_text(json.dumps({"jobs_sha256": sha(tmp_path / "jobs.parquet")}))
    cache = tmp_path / "vendor"
    cache.mkdir()
    (cache / "sh.600001_2024.json").write_text(json.dumps([event()]))
    with pytest.raises(ValueError, match="not a zero-event"):
        assemble(tmp_path)
    (cache / "sh.600002_2024.json").write_text("[]")
    error = cache / "sh.600002_2024.error.json"
    error.write_text('{"error":"paging failed"}')
    with pytest.raises(ValueError, match="not a zero-event"):
        assemble(tmp_path)
    error.unlink()
    report = assemble(tmp_path)
    assert report["code_year_queries"] == 2
    assert report["event_rows"] == 1
    assert report["empty_code_years"] == 1
    assert report["issuer_notice_verified"] is False
    assert report["strategy_returns_read"] is False


def test_only_an_exact_reviewed_incomplete_duplicate_can_be_removed():
    complete = event(dividPlanDate="2024-05-20", dividRegistDate="2024-05-29",
                     dividPayDate="2024-05-30")
    incomplete = dict(complete, dividPlanDate="")
    review = {"code": "sh.600001", "year": "2024", "action_date": "2024-05-30",
              "notice_date": "2024-05-20", "record_date": "2024-05-29", "pay_date": "2024-05-30",
              "cash_per_share": ".2", "bonus_per_share": "0", "reserve_per_share": "0",
              "discard_incomplete_record_sha256": hashlib.sha256(json.dumps(
                  incomplete, sort_keys=True, ensure_ascii=False).encode()).hexdigest()}
    assert checked_response([incomplete, complete], "sh.600001", "2024", review) == [complete]
    assert checked_response([complete], "sh.600001", "2024", review) == [complete]
    with pytest.raises(ValueError, match="disagrees with primary"):
        checked_response([incomplete, dict(complete, dividCashPsBeforeTax=".3")],
                         "sh.600001", "2024", review)
    with pytest.raises(ValueError, match="complete implementation record is missing"):
        checked_response([incomplete], "sh.600001", "2024", review)


def test_legitimate_same_day_distributions_are_combined_only_with_exact_primary_review():
    annual = event(dividCashPsBeforeTax=".286", dividPlanDate="2024-05-20",
                   dividRegistDate="2024-05-29", dividPayDate="2024-05-30")
    quarterly = dict(annual, dividCashPsBeforeTax=".091")
    combined = dict(annual, dividCashPsBeforeTax=".377")
    review = {"code": "sh.600001", "year": "2024", "action_date": "2024-05-30",
              "notice_date": "2024-05-20", "record_date": "2024-05-29", "pay_date": "2024-05-30",
              "cash_per_share": ".377", "bonus_per_share": "0", "reserve_per_share": "0",
              "combine_record_sha256": [record_hash(annual), record_hash(quarterly)],
              "combined_record": combined}
    assert checked_response([annual, quarterly], "sh.600001", "2024", [review]) == [combined]
    assert checked_response([combined], "sh.600001", "2024", [review]) == [combined]
    with pytest.raises(ValueError, match="reviewed originals"):
        checked_response([annual, annual], "sh.600001", "2024", [review])
    with pytest.raises(ValueError, match="does not equal"):
        checked_response([annual, quarterly], "sh.600001", "2024",
                         [dict(review, combined_record=dict(combined, dividCashPsBeforeTax=".286"))])
