import numpy as np
import pandas as pd

from trade_research.tail_formula_narrow_breakout import replay_helper


def test_actual_nr7_helper_excludes_current_bar_strict_ties_and_zero_ranges():
    # Seven completed bars: the last is strictly smallest. The eighth is
    # incomplete at decision time and can change without affecting inputs.
    ranges=np.array([8,7,6,5,4,3,2,100],float)/100
    d=pd.DataFrame(dict(low=np.full(8,10.),high=10.+ranges,
                        open=np.full(8,10.),close=np.full(8,10.),volume=np.ones(8)))
    before=replay_helper(d).iloc[-1]
    assert before.ready and before.nr7==1 and before.previous_high_cents==1002
    d.loc[7,['open','high','low','close','volume']]=[999,999,1,1,0]
    pd.testing.assert_series_equal(replay_helper(d).iloc[-1],before)
    d.loc[5,'high']=10.02
    assert replay_helper(d).iloc[-1].nr7==0
    d.loc[0,'high']=10.
    assert not replay_helper(d).iloc[-1].ready and np.isnan(replay_helper(d).iloc[-1].nr7)
    d.loc[0,'high']=10.08;d.loc[0,'volume']=0
    assert not replay_helper(d).iloc[-1].ready
