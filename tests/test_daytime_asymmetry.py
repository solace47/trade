import pandas as pd
import pytest

from trade_research.daytime_asymmetry import _comparable, _summarize_valid


def test_equal_stratum_and_date_weights_prevent_large_group_dominance() -> None:
    rows = []
    for date in ("2024-03-04", "2024-09-02", "2025-03-03", "2025-09-01"):
        for liquidity_third, count, up_return in ((1, 10, -.01), (2, 5, .01)):
            for group, gross in (("up", up_return), ("down", 0.0)):
                for stock in range(count):
                    rows.append({
                        "date": date, "board": "main",
                        "liquidity_third": liquidity_third,
                        "group": group, "code": f"{date}-{group}-{liquidity_third}-{stock}",
                        "gross_1449": gross, "paper_overnight": gross,
                        "remaining_day": 0.0,
                    })
    valid = pd.DataFrame(rows)
    _, comparable = _comparable(valid)
    assert len(comparable) == 8
    report = _summarize_valid(valid)
    assert report["valid_comparable_days"] == 4
    for half in ("2024H1", "2024H2", "2025H1", "2025H2"):
        assert report["by_half"][half]["gross_up_pp"] == pytest.approx(0.0)
    assert not report["raw_reprice_gate_passed"]


def test_comparison_requires_five_of_each_sign_per_stratum() -> None:
    rows = [
        {"date": "2024-03-04", "board": "main", "liquidity_third": 1,
         "group": group}
        for group, count in (("up", 5), ("down", 4))
        for _ in range(count)
    ]
    _, comparable = _comparable(pd.DataFrame(rows))
    assert comparable.empty
