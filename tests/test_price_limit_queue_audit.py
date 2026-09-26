import numpy as np
import pandas as pd

from trade_research.price_limit_queue_audit import conservative_returns, window_exposure


def test_mixed_reopening_window_is_not_verified_by_positive_volume():
    bars=pd.DataFrame({"high":[11.,10.8],"low":[11.,10.7],"volume":[1_000_000,100]})
    result=window_exposure(bars,11.,"buy")
    assert result["limit_touched"]
    assert not result["all_positive_bars_at_limit"]
    assert result["volume_in_wholly_away_minutes"]==100
    assert window_exposure(bars.iloc[:1],11.,"buy")["all_positive_bars_at_limit"]


def test_sell_limit_uses_low_and_zero_volume_does_not_prove_trade():
    bars=pd.DataFrame({"high":[9.2,9.2],"low":[9.,9.1],"volume":[0,100]})
    assert not window_exposure(bars,9.,"sell")["limit_touched"]
    bars.loc[0,"volume"]=100
    assert window_exposure(bars,9.,"sell")["limit_touched"]


def test_queue_unknown_stays_missing_in_the_complete_cohort():
    rows=pd.DataFrame({"date":["2025-01-02"]*2,"code":["a","b"],
        "horizon":[1,1],"tick_return15":[.1,-.1]})
    flags=rows[["date","code","horizon"]].assign(queue_allocation_unverified=[True,False])
    result=conservative_returns(rows,flags.iloc[::-1],"tick_return15")
    assert np.isnan(result.iloc[0])
    assert result.iloc[1]==-.1
