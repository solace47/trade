import numpy as np
import pandas as pd

from audit_tail_formula_daily_order import pandas_auc, score_bins


def test_visible_tie_bins_preserve_unknown_labels_and_invalid_keys():
    f=pd.DataFrame(dict(date=['2024-01-02']*6,code=[f'sh.60000{i}' for i in range(6)],
        half=['2024H1']*6,board=['main']*6,decision_shares=[100]*6,
        formula_input_valid=[True]*5+[False],score=[0.,0.,1.,1.,1.,np.nan],
        known15=[True,False,True,False,True,False]))
    a=score_bins(f)
    assert a.bucket.tolist()==[2,2,7,7,7,-1]
    assert len(a)==len(f)
    f['known15']=~f.known15
    pd.testing.assert_frame_equal(a,score_bins(f),check_exact=True)


def test_auc_retains_undefined_dates_and_half_credit_for_ties():
    f=pd.DataFrame(dict(date=['2024-01-02']*4+['2024-01-03']*2,
        score_integer=[0,0,2,3,1,2],known15=[True,True,True,False,True,True],
        opportunity15=[1.,0.,1.,np.nan,1.,1.]))
    d=pandas_auc(f)
    assert d.date.tolist()==['2024-01-02','2024-01-03']
    assert d.auc.iloc[0]==.75 and np.isnan(d.auc.iloc[1])
    assert d.visible.tolist()==[4,2] and d.known.tolist()==[3,2]
