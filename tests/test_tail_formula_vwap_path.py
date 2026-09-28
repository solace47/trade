import numpy as np
from trade_research.tail_formula_vwap_path import measure,native_value


def test_future_tail_change_cancels_out_of_earlier_cumulative_inputs():
    p=np.full((1,30),10.);v=np.ones((1,29))*100;a=v*10
    first=measure(p,v,a,np.array([4000.]),np.array([40000.]),np.array([2.]))
    v2=v.copy();a2=a.copy();v2[0,-1]+=100;a2[0,-1]+=1100
    second=measure(p,v2,a2,np.array([4100.]),np.array([41100.]),np.array([2.]))
    np.testing.assert_array_equal(first['premium'][:,:-1],second['premium'][:,:-1])
    assert second['premium'][0,-1]<first['premium'][0,-1]


def test_same_final_average_can_have_different_earlier_paths():
    p=np.full((1,30),10.);v=np.ones((1,29))*100;a=v*10;b=a.copy()
    b[0,0]-=100;b[0,-1]+=100
    x=measure(p,v,a,[4000.],[40000.],[2.]);y=measure(p,v,b,[4000.],[40000.],[2.])
    assert x['VP01'][0]==0 and y['VP01'][0]>0
    assert x['premium'][0,-1]==y['premium'][0,-1]


def test_zero_cumulative_is_unknown_and_native_matches_open_prefix():
    p=np.full((1,30),10.);v=np.zeros((1,29));a=v.copy()
    assert np.isnan(measure(p,v,a,[0.],[0.],[2.])['VP01'][0])
    full_p=np.linspace(9.77,10.,230).round(2);full_v=np.arange(230.)+100;full_a=full_v*(full_p-.01)
    x=measure(full_p[-30:][None,:],full_v[-29:][None,:],full_a[-29:][None,:],[full_v.sum()],[full_a.sum()],[2.])
    for outside in [.01,1000000.]:
        value,guard=native_value(full_p,full_v/100,full_a,2.,outside)
        assert guard==29
        np.testing.assert_allclose(value,x['VP01'][0],rtol=0,atol=2e-12)
