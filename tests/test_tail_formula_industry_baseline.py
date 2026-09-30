import numpy as np
import pandas as pd

from trade_research.tail_formula_industry_baseline import center


def frame(outcomes,industries):
    return pd.DataFrame(dict(date=['2024-01-03']*len(outcomes),
        code=[f'x{i:03}' for i in range(len(outcomes))],opportunity15=outcomes,industry=industries))


def test_industry_wide_success_alone_has_no_stock_specific_target():
    f=frame([1]*11+[0]*11,['A']*11+['B']*11)
    result=center(f)
    assert result.industry_baseline_used.all()
    np.testing.assert_array_equal(result.target,0)
    original=f.opportunity15-f.opportunity15.mean()
    assert original.iloc[0]==.5 and original.iloc[-1]==-.5


def test_unknown_and_small_industries_use_market_then_day_center():
    f=frame([1]*6+[0]*5+[1,1,0],["A"]*11+["B","B",None])
    r=center(f)
    assert r.industry_baseline_used.sum()==11
    fallback=r.loc[~r.industry_baseline_used]
    np.testing.assert_array_equal(fallback.baseline,f.opportunity15.mean())
    assert pd.isna(r.industry.iloc[-1]) and pd.isna(r.sector_mean.iloc[-1])
    assert r.sector_count.iloc[-1]==0
    assert abs(r.target.mean())<2e-12
    assert abs(r.residual_day_mean.iloc[0])>0


def test_industry_baseline_is_before_the_input_quality_intersection():
    f=frame([1]*6+[0]*6,['A']*12)
    full=center(f)
    # One excluded positive observation must remain in the full known baseline.
    selected=full.iloc[1:].copy()
    np.testing.assert_array_equal(selected.baseline,.5)
    incorrectly_intersected=center(f.iloc[1:].copy())
    np.testing.assert_array_equal(incorrectly_intersected.baseline,5/11)
    assert selected.target.iloc[0] != incorrectly_intersected.target.iloc[0]
