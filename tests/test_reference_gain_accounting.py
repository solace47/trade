from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from trade_research.reference_gain_accounting import complete_mean, weekly_interval, distribution_cash


def test_unknown_trade_prevents_complete_mean():
    frame = pd.DataFrame({"date": ["2024-01-02", "2024-01-02", "2024-01-03"],
                          "value": [.1, np.nan, .2]})
    assert complete_mean(frame, "value") is None
    frame.loc[1, "value"] = 0.
    assert np.isclose(complete_mean(frame, "value"), .125)


def test_week_blocks_preserve_constant_and_reject_unknown_day():
    daily = pd.Series([.01] * 8, index=pd.bdate_range("2024-01-02", periods=8).strftime("%Y-%m-%d"))
    assert np.allclose(weekly_interval(daily), [.01, .01])
    daily.iloc[2] = np.nan
    assert weekly_interval(daily) is None


def test_cashless_reserve_requires_primary_evidence_but_no_cash_payment_date():
    event = pd.Series({"dividCashPsBeforeTax": "", "dividPayDate": "",
        "dividOperateDate": "2025-03-17", "dividReserveToStockPs": "0.450000",
        "dividStocksPs": "0.000000"})
    with pytest.raises(ValueError, match="Missing cash amount"):
        distribution_cash(event, last_date="2025-12-31")
    event["primary_terms_verified"] = True
    event["primary_reference_cash"], event["primary_reference_reserve"] = "0", "0.45"
    assert distribution_cash(event, last_date="2025-12-31") == 0
    event["primary_reference_reserve"] = "0.4"
    with pytest.raises(ValueError, match="Missing cash amount"):
        distribution_cash(event, last_date="2025-12-31")


def test_positive_cash_still_requires_known_payment_timing():
    event = pd.Series({"dividCashPsBeforeTax": "0.1", "dividPayDate": "",
        "dividOperateDate": "2025-03-17"})
    with pytest.raises(ValueError, match="Cash payment timing"):
        distribution_cash(event, last_date="2025-12-31")
    event["dividPayDate"] = "2026-01-02"
    with pytest.raises(ValueError, match="Cash payment timing"):
        distribution_cash(event, last_date="2025-12-31")
    assert distribution_cash(event, last_date="2025-12-31", allow_unsettled_payments=True) == Decimal(".1")
