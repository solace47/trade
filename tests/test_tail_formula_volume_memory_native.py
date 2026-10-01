import numpy as np

from freeze_tail_formula_volume_memory_native import HELPER,replay
from trade_research.tail_formula_volume_memory import finite_states


def test_literal_helper_is_single_argument_round_and_strictly_prior():
    assert 'ROUND(C*100,0)' not in HELPER
    close=np.round(10+np.sin(np.arange(95)/9),2)
    volume=1000.+37*np.arange(95)
    expected,_,ready=finite_states(close,volume,np.ones(95,bool))
    actual=replay(np.r_[close,1e6],np.r_[volume,1e12])[1:]
    np.testing.assert_allclose(actual[ready],expected[ready],rtol=0,atol=2e-9)
    assert not np.isfinite(actual[:78]).any()
    np.testing.assert_array_equal(actual,replay(np.r_[close,.01],np.r_[volume,1.])[1:])
