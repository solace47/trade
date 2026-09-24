import pandas as pd
import pytest

from trade_research.fund_stake_eval import _execution_placebo, _summarize


def test_high_minus_low_stake_interaction_keeps_same_day_four_cells() -> None:
    rows = []
    for date, values in (
            ("2024-05-15", {(1, 1): 0.01, (1, 5): 0.02,
                            (3, 1): 0.00, (3, 5): 0.04}),
            ("2024-05-16", {(1, 1): 0.01, (1, 5): 0.10,
                            (3, 1): 0.01})):
        for (stake, quintile), value in values.items():
            rows.append({
                "date": date, "board": "sh_main", "size_bucket": 1,
                "cash_group": "low_cash_conversion",
                "ownership_tercile": stake, "quintile": quintile,
                "cash_return": value, "entry_filled": True,
                "clean_exit": True,
            })
    report = _summarize(pd.DataFrame(rows))
    section = report["groups"]["low_cash_conversion"]["2024"]["full"]
    assert section["four_cell_strata"] == 1
    assert section["four_cell_stock_days"] == 4
    assert section["days"] == 1
    assert section["low_stake_edge"] == pytest.approx(0.01)
    assert section["high_stake_edge"] == pytest.approx(0.04)
    assert section["interaction"] == pytest.approx(0.03)


def test_execution_placebo_preserves_cash_on_failed_exit() -> None:
    frame = pd.DataFrame([
        {"date": "2024-05-15", "board": "sh_main", "size_bucket": 1,
         "cash_group": "low_cash_conversion", "clean_exit": True,
         "raw_net_return": 0.02, "cash_return": 0.02},
        {"date": "2024-05-15", "board": "sh_main", "size_bucket": 1,
         "cash_group": "low_cash_conversion", "clean_exit": True,
         "raw_net_return": 0.04, "cash_return": 0.04},
        {"date": "2024-05-15", "board": "sh_main", "size_bucket": 1,
         "cash_group": "low_cash_conversion", "clean_exit": False,
         "raw_net_return": 0.99, "cash_return": 0.0},
    ])
    placebo = _execution_placebo(frame)
    assert placebo.cash_return.tolist() == pytest.approx([0.03, 0.03, 0.0])
