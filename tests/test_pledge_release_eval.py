import pandas as pd

from trade_research.pledge_release_eval import _tail_diagnostic
from trade_research.pledge_release_inputs import EVENT
from trade_research.buyback_inputs import CONTROL


def test_posthoc_tail_diagnostic_uses_days_and_discloses_best_month() -> None:
    rows = []
    for date, codes, edge in (
        ("2024-01-02", ("sh.600001", "sh.600002"), 0.1),
        ("2024-02-01", ("sh.600003",), -0.1),
    ):
        for code in codes:
            for candidate, cash in ((EVENT, edge), (CONTROL, 0.0)):
                rows.append({"date": date, "pair_code": code,
                             "candidate": candidate, "cash_return": cash,
                             "target_notional": 20_000, "horizon": 5})
    result = _tail_diagnostic(pd.DataFrame(rows), "2024")
    assert result["highest_mean_month"] == "2024-01"
    assert result["highest_mean_month_days"] == 1
    assert abs(result["without_highest_mean_month"] + 0.1) < 1e-12
    assert abs(result["positive_day_share"] - 0.5) < 1e-12
    assert abs(result["day_median"]) < 1e-12
