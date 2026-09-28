import numpy as np
import pandas as pd

from trade_research.tail_formula_pairwise import make_pairs,derivatives
from trade_research.tail_formula_score_center import center
from trade_research.tail_formula_score_scale import standardize


def test_pairwise_loss_and_centered_scores_ignore_date_offsets():
    f=pd.DataFrame(dict(date=['a']*3+['b']*4,code=['1','2','3','1','2','3','4'],
        opportunity15=[0,1,0,1,0,1,0],score=[-.3,.4,.2,.8,.1,.6,-.2],formula_input_valid=True))
    pairs,_=make_pairs(f)
    shifted=f.copy();shifted['score']+=shifted.date.map({'a':17.,'b':-43.})
    a=derivatives(f.score.to_numpy(),pairs.positive,pairs.negative,pairs.weight)
    b=derivatives(shifted.score.to_numpy(),pairs.positive,pairs.negative,pairs.weight)
    for x,y in zip(a,b):np.testing.assert_allclose(x,y,rtol=0,atol=2e-14)
    np.testing.assert_allclose(center(f)[0].centered_score,center(shifted)[0].centered_score,atol=2e-14)
    np.testing.assert_allclose(standardize(f)[0].standardized_score,standardize(shifted)[0].standardized_score,atol=2e-13)
    # A raw common threshold need not preserve decisions under that freedom.
    assert not f.score.gt(.5).equals(shifted.score.gt(.5))
