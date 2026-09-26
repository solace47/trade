"""Unknown proceeds cannot disappear from a purchased cohort."""

import numpy as np
import pandas as pd
import pytest

from trade_research import fill_accounting
from trade_research.fill_accounting import CATEGORIES, account_rows, summarize_cell


CALENDAR = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]


def trades():
    result = []
    for i, category in enumerate(CATEGORIES):
        bought = category != "not_bought"
        recorded = category not in {"not_bought", "no_exit_recorded"}
        filled = category in {"clean_ontime", "clean_delayed", "exit_quality_unverified"}
        delay = int(category == "clean_delayed") if recorded else np.nan
        result.append({
            "date": "2025-01-02", "code": f"sh.60000{i}",
            "target_notional": 20000, "horizon": 1,
            "entry_status": "filled" if bought else "estimated_upper_limit",
            "shares": 100 if bought else 0,
            "entry_price": 10.0 if bought else np.nan,
            "target_exit_date": "2025-01-03",
            "exit_date": ("2025-01-06" if delay == 1 else "2025-01-03")
                         if recorded else None,
            "exit_delay_sessions": delay,
            "exit_status": ("filled" if filled else "corporate_action_unadjusted"
                            if recorded else "volume_cap" if bought else "entry_not_filled"),
            "exit_price": 11.0 if recorded else np.nan,
            "net_return": 1094.439 / 1005.01 - 1 if filled else np.nan,
            "quality_clean_exit": category in {"clean_ontime", "clean_delayed"},
        })
    return pd.DataFrame(result)


def test_delayed_return_and_unverified_purchases_are_distinct():
    rows = account_rows(trades(), CALENDAR)
    assert rows.category.tolist() == list(CATEGORIES)
    assert rows.buy_cost5.sum() == pytest.approx(5 * 1005.01)
    assert rows.unknown_after_buy.sum() == 3
    assert rows.loc[rows.unknown_after_buy, "known_return5"].isna().all()
    assert rows.loc[rows.unknown_after_buy, "verified_proceeds5"].isna().all()
    summary = summarize_cell(rows)
    assert summary["old_score5"] == pytest.approx((1094.439 / 1005.01 - 1) / 6)
    assert summary["delay_contribution5"] == pytest.approx(summary["old_score5"])
    assert summary["delayed_extra_capital_sessions5"] == pytest.approx(1005.01)
    assert summary["unknown_buy_cost5"] == pytest.approx(3 * 1005.01)
    assert summary["complete_cohort_mean5"] is None
    assert summary["complete_cohort_mean15"] is None


def test_complete_cohort_counts_failed_buys_as_cash_but_keeps_delayed_returns():
    rows = account_rows(trades().iloc[:3], CALENDAR)
    summary = summarize_cell(rows)
    expected = (1094.439 / 1005.01 - 1) * 2 / 3
    assert summary["complete_cohort_mean5"] == pytest.approx(expected)
    assert summary["complete_cohort_mean15"] < expected


def test_input_row_order_does_not_change_cash_summation():
    first = account_rows(trades(), CALENDAR)
    second = account_rows(trades().iloc[::-1], CALENDAR)
    pd.testing.assert_frame_equal(first, second)
    assert summarize_cell(first) == summarize_cell(second)


@pytest.mark.parametrize("field,value", [
    ("exit_date", "2026-01-05"), ("exit_delay_sessions", 2),
    ("target_exit_date", "2025-01-06"), ("shares", 0),
    ("net_return", .9),
])
def test_bad_chronology_or_cash_identity_is_rejected(field, value):
    rows = trades()
    rows.loc[1, field] = value
    with pytest.raises(ValueError):
        account_rows(rows, CALENDAR)


def test_frozen_inputs_cannot_be_silently_replaced(tmp_path, monkeypatch):
    root = tmp_path / "research"
    source = root / "model"
    source.mkdir(parents=True)
    output = root / "audit"
    calendar = tmp_path / "calendar.parquet"
    pd.DataFrame({"date": CALENDAR}).to_parquet(calendar)
    pd.DataFrame({
        "date": ["2025-01-02"], "code": ["sh.600000"],
        "candidate": ["absolute_model"], "pair_id": [1],
    }).to_parquet(source / "selections.parquet")
    for name in ("repricing_signals.parquet", "repriced.parquet"):
        trades().to_parquet(source / name)
    for file in (source / "report.json", source / "input_audit.json",
                 root / "quality_period_2024_2025.json"):
        file.write_text("{}")
    issues = root / "market_issues_ci"
    issues.mkdir()
    for i in range(20):
        (issues / f"shard_{i}.csv").write_text("date,code\n")
    monkeypatch.setattr(fill_accounting, "ROOT", root)
    monkeypatch.setattr(fill_accounting, "CALENDAR", calendar)
    manifest = fill_accounting.freeze(source, output)
    assert fill_accounting.verify_manifest(output) == manifest
    assert fill_accounting.freeze(source, output) == manifest
    (source / "report.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="Frozen input changed"):
        fill_accounting.verify_manifest(output)
    with pytest.raises(ValueError, match="cannot be replaced"):
        fill_accounting.freeze(source, output)
