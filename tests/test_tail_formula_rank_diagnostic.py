import numpy as np
import pandas as pd
from review_tail_formula_order_risk import daily_metrics, verify_metrics


def test_order_metrics_match_brute_pairs_and_score_ties():
    f=pd.DataFrame(dict(date=['2024-01-02']*6,score=[2.,1.,1.,3.,0.,2.],utility=[1,0,-3,-3,0,1],positive_before=[1,0,0,0,1,1],space_before=[1,0,0,0,0,1]))
    d=daily_metrics(f);verify_metrics(f,d)
    for field in ['positive_before','space_before','utility']:
        pairs=[(i,j) for i in range(len(f)) for j in range(len(f)) if f[field][i]>f[field][j]]
        numerator=sum(2 if f.score[i]>f.score[j] else 1 if f.score[i]==f.score[j] else 0 for i,j in pairs)
        assert d[field+'_pairs'].iloc[0]==len(pairs)
        assert d[field+'_numerator2'].iloc[0]==numerator
        assert d[field+'_auc'].iloc[0]==numerator/(2*len(pairs))


def test_undefined_single_class_dates_are_kept_as_nan():
    f=pd.DataFrame(dict(date=['2024-01-02','2024-01-03','2024-01-03'],score=[1.,2.,2.],utility=[0,1,1],positive_before=[0,1,1],space_before=[0,1,1]))
    d=daily_metrics(f);verify_metrics(f,d)
    assert list(d.date)==['2024-01-02','2024-01-03']
    assert list(d.rows)==[1,2]
    for field in ['positive_before','space_before','utility']:
        assert d[field+'_pairs'].eq(0).all()
        assert np.isnan(d[field+'_auc']).all()
