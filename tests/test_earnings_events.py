from pathlib import Path

import pandas as pd
import pytest

from trade_research.earnings_events import (
    FIELDS, _select_available, _validate, stock_codes,
)


def test_stock_codes_include_delisted_shares(tmp_path: Path) -> None:
    source = tmp_path / "basic.parquet"
    pd.DataFrame({
        "code": ["sz.000001", "sh.600000", "sh.510050"],
        "type": ["1", "1", "5"], "status": ["1", "0", "1"],
    }).to_parquet(source, index=False)
    assert stock_codes(source) == ["sh.600000", "sz.000001"]


def test_future_fiscal_report_is_removed_until_published() -> None:
    row = ["sh.600000", "2026-01-01", "2025-12-31",
           "预增", "", "50", "30"]
    frame = pd.DataFrame([row], columns=FIELDS["forecast"])
    assert _select_available(frame, "forecast").empty
    with pytest.raises(ValueError, match="availability outside requested window"):
        _validate(frame, "forecast", {"sh.600000"})


def test_express_update_cannot_precede_publication() -> None:
    row = ["sh.600000", "2025-01-02", "2024-12-31", "2025-01-01"]
    row += ["1"] * (len(FIELDS["express"]) - len(row))
    frame = pd.DataFrame([row], columns=FIELDS["express"])
    with pytest.raises(ValueError, match="predates publication"):
        _validate(frame, "express", {"sh.600000"})


def test_express_version_is_available_only_after_update() -> None:
    row = ["sh.600000", "2025-12-30", "2025-09-30", "2026-01-05"]
    row += ["1"] * (len(FIELDS["express"]) - len(row))
    frame = pd.DataFrame([row], columns=FIELDS["express"])
    assert _select_available(frame, "express").empty
