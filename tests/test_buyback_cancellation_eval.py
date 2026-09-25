import pandas as pd
import pytest

from trade_research.buyback_cancellation_eval import (
    _same_day_purpose, _same_event_industry, _tag_pairs,
)


def test_purpose_comparison_uses_only_dates_with_both_groups() -> None:
    rows = pd.DataFrame([
        ("2024-01-03", "pure_cancel", "buyback_plan", .02),
        ("2024-01-03", "pure_cancel", "same_day_nonannouncer", .01),
        ("2024-01-03", "employee", "buyback_plan", .01),
        ("2024-01-03", "employee", "same_day_nonannouncer", .02),
        ("2024-02-03", "pure_cancel", "buyback_plan", .05),
        ("2024-02-03", "pure_cancel", "same_day_nonannouncer", .01),
    ], columns=["date", "decision", "candidate", "cash_return"])
    result = _same_day_purpose(rows, "2024")
    assert result["days"] == 1
    assert result["pure_minus_employee_edge"] == pytest.approx(.02)


def test_purpose_tag_requires_unchanged_one_to_one_control() -> None:
    pairs = pd.DataFrame([
        ("2024-01-03", "sh.600001", "sh.600001", "buyback_plan", "pdf-a"),
        ("2024-01-03", "sh.600002", "sh.600001",
         "same_day_nonannouncer", "pdf-a"),
    ], columns=["date", "code", "pair_code", "candidate", "pdf_url"])
    review = pd.DataFrame([{"date": "2024-01-03", "code": "sh.600001",
                            "pdf_url": "pdf-a", "decision": "pure_cancel"}])
    tagged = _tag_pairs(pairs, review, require_all=True)
    assert tagged.decision.tolist() == ["pure_cancel", "pure_cancel"]
    with pytest.raises(ValueError, match="one-to-one|same-day control"):
        _tag_pairs(pairs.iloc[:1], review, require_all=True)


def test_industry_comparison_keeps_identical_event_fills() -> None:
    def rows(control_return: float, event_return: float) -> pd.DataFrame:
        return pd.DataFrame([
            ("2024-01-03", "sh.600001", "buyback_plan", event_return),
            ("2024-01-03", "sh.600001", "same_day_nonannouncer",
             control_return),
        ], columns=["date", "pair_code", "candidate", "cash_return"]).assign(
            target_notional=20_000, horizon=5, decision="pure_cancel")

    result = _same_event_industry(rows(.01, .02), rows(.015, .02), "2024")
    assert result["pairs"] == 1
    assert result["industry_edge_mean"] == pytest.approx(.005)
    with pytest.raises(ValueError, match="changed or lost"):
        _same_event_industry(rows(.01, .02), rows(.015, .03), "2024")
