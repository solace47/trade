import json

import pandas as pd
import pytest

from trade_research.fund_attention_eval import _summarize, _validate_input_audit
from trade_research.fund_visibility_inputs import QUARTERS


def _stock(day: str, code: str, visible: bool, quintile: int,
           cash_return: float) -> dict:
    return {"date": day, "code": code, "board": "sh_main",
            "size_bucket": 1, "cash_group": "low_cash_conversion",
            "fund_visible": visible, "quintile": quintile,
            "cash_return": cash_return, "entry_filled": True,
            "clean_exit": True}


def test_four_cell_interaction_excludes_incomplete_stratum() -> None:
    scored = pd.DataFrame([
        _stock("2024-05-20", "a", False, 1, 0.01),
        _stock("2024-05-20", "b", False, 5, 0.04),
        _stock("2024-05-20", "c", True, 1, 0.02),
        _stock("2024-05-20", "d", True, 5, 0.03),
        _stock("2024-05-21", "e", False, 1, 0.90),
        _stock("2024-05-21", "f", False, 5, 0.90),
        _stock("2024-05-21", "g", True, 1, 0.90),
    ])
    report = _summarize(scored)
    assert report["total_strata"] == 2
    assert report["four_cell_strata"] == 1
    assert report["four_cell_stock_days"] == 4
    sample = report["groups"]["low_cash_conversion"]["2024"]["full"]
    assert sample["source_strata"] == 2
    assert sample["days"] == 1
    assert round(sample["absent_edge"], 8) == 0.03
    assert round(sample["visible_edge"], 8) == 0.01
    assert round(sample["interaction"], 8) == 0.02


def test_interaction_weights_dates_equally_after_stratum_means() -> None:
    rows = []
    for day, sizes, absent_high in (
            ("2024-05-20", (1, 2), 0.02),
            ("2024-05-21", (1,), -0.02)):
        for size in sizes:
            for visible, quintile, value in (
                    (False, 1, 0.0), (False, 5, absent_high),
                    (True, 1, 0.0), (True, 5, 0.0)):
                row = _stock(day, f"{day}-{size}-{visible}-{quintile}",
                             visible, quintile, value)
                row["size_bucket"] = size
                rows.append(row)
    sample = _summarize(pd.DataFrame(rows))[
        "groups"]["low_cash_conversion"]["2024"]["full"]
    assert sample["four_cell_strata"] == 3
    assert sample["days"] == 2
    assert round(sample["interaction"], 8) == 0.0


def test_outcome_gate_requires_all_eight_source_quarters(tmp_path) -> None:
    inputs = pd.DataFrame({"fund_visible": [False, True]})
    path = tmp_path / "fund_visibility_inputs.json"
    with pytest.raises(FileNotFoundError, match="Eight-quarter"):
        _validate_input_audit(inputs, path)
    audit = {"quarter_source_reports": {q: 1 for q in QUARTERS[:-1]},
             "source_reports": 7, "parsed_reports": 7,
             "rejected_reports": 0, "eligible_stock_days": 2,
             "visible_stock_days": 1}
    path.write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match="eight-quarter"):
        _validate_input_audit(inputs, path)
    audit["quarter_source_reports"][QUARTERS[-1]] = 1
    audit["source_reports"] = audit["parsed_reports"] = 8
    path.write_text(json.dumps(audit), encoding="utf-8")
    _validate_input_audit(inputs, path)
