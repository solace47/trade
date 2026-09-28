import numpy as np
import pandas as pd

from trade_research.tail_formula_market_turnover import relative_turnover


def test_pool_uses_validity_and_current_date_not_future_labels():
    f = pd.DataFrame(dict(date=['2025-01-02']*3+['2025-01-03'], code=['a','b','c','a'],
        formula_input_valid=[True,True,False,True], S02=[1.,3.,999.,100.], S03=[0.,2.,999.,50.],
        known15=[True,False,True,True]))
    out, _ = relative_turnover(f)
    np.testing.assert_allclose(out.loc[:1, 'MT01'], [.5,1.5])
    np.testing.assert_allclose(out.loc[:1, 'MT02'], [0.,2.])
    assert out.loc[2, ['MT01', 'MT02']].isna().all()
    changed = f.copy()
    changed['known15'] = ~changed.known15
    changed.loc[3, ['S02','S03']] = [1e9,1e8]
    rebuilt, _ = relative_turnover(changed)
    pd.testing.assert_frame_equal(out.loc[:2], rebuilt.loc[:2].assign(known15=out.loc[:2, 'known15']), check_exact=True)


def test_zero_reference_stays_unknown_not_zero_signal():
    f = pd.DataFrame(dict(date=['2025-01-02'], code=['a'], formula_input_valid=[True], S02=[1.], S03=[0.]))
    out, _ = relative_turnover(f)
    assert out.prior_formula_input_valid.all() and not out.formula_input_valid.any()
    assert out[['MT01','MT02']].isna().all().all()
