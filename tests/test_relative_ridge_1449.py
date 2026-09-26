import numpy as np
import pandas as pd
import pytest

from trade_research.absolute_ridge import FEATURES
from trade_research.downside_ridge_1449 import fit_score
from trade_research.relative_ridge_1449 import relative_targets


def test_center_is_full_same_day_and_unknown_loss_stays_in_training_scenario():
    frame=pd.DataFrame({'date':['2024-01-02']*3+['2024-01-03']*2,
                        'code':['a','b','c','a','b'],'downside_score':[-1.,.03,.12,.04,.06]},index=[7,1,3,8,4])
    target,center=relative_targets(frame)
    assert target.index.equals(frame.index)
    np.testing.assert_allclose(center,[0.,0.,0.,.05,.05],atol=1e-15)
    np.testing.assert_allclose(target,[-.15,.03,.12,-.01,.01],atol=1e-15)
    np.testing.assert_allclose(target.groupby(frame.date).mean(),[0.,0.],atol=1e-15)
    assert frame.loc[7,'downside_score']==-1.


def test_no_second_clip_that_would_reintroduce_daily_mean():
    frame=pd.DataFrame({'date':['2024-01-02']*4,'code':['a','b','c','d'],'downside_score':[-1.,-1.,-1.,.15]})
    target,_=relative_targets(frame)
    assert np.isclose(target.iloc[-1],.225)
    assert abs(target.mean())<1e-15
    assert abs(target.clip(-.15,.15).mean())>.01


@pytest.mark.parametrize('scores',[[float('nan'),0.],[float('inf'),0.]])
def test_actual_missing_scores_are_not_silently_dropped(scores):
    with pytest.raises(ValueError):
        relative_targets(pd.DataFrame({'date':['2024-01-02']*2,'code':['a','b'],'downside_score':scores}))


def test_no_target_clip_is_explicit_and_default_model_still_clips():
    frame=pd.DataFrame({name:[1.,1.,1.] for name in FEATURES})
    target=pd.Series([.225,.225,.225])
    old,_=fit_score(frame,target)
    new,_=fit_score(frame,target,clip_target=False)
    assert np.isclose(old['intercept'],.15)
    assert np.isclose(new['intercept'],.225)
    assert np.allclose(old['coefficients'],0)
    assert np.allclose(new['coefficients'],0)
