import numpy as np

from trade_research.tail_formula_histogram_tree import (
    best_shared_split,cut_grid,encode_bins,symmetric_tree,asymmetric_tree)
from trade_research.tail_formula_additive import leaf_indices


def test_shared_cut_matches_exhaustive_weighted_loss_and_support():
    rng=np.random.default_rng(73)
    x=rng.integers(-10,11,size=(96,3),dtype='int32')
    grids=cut_grid(x,bins=8);b=encode_bins(x,grids)
    leaves=np.repeat([0,1],48).astype('int32');w=rng.uniform(.1,2,96)
    residual=rng.normal(size=96)+.4*x[:,1]
    base=sum(np.sum(w[leaves==k]*(residual[leaves==k]-np.average(residual[leaves==k],weights=w[leaves==k]))**2) for k in [0,1])
    candidates=[]
    for j,cuts in enumerate(grids):
        for i,cut in enumerate(cuts):
            masks=[(leaves==k)&cond for k in [0,1] for cond in [x[:,j]<=cut,x[:,j]>cut]]
            if any(mask.sum()<8 for mask in masks):continue
            loss=sum(np.sum(w[m]*(residual[m]-np.average(residual[m],weights=w[m]))**2) for m in masks)
            candidates.append((base-loss,j,i))
    expected=sorted(candidates,key=lambda z:(-z[0],z[1],z[2]))[0]
    got=best_shared_split(b,grids,leaves,residual,w,minimum=8)
    assert got[:2]==expected[1:]
    np.testing.assert_allclose(got[2],expected[0],rtol=0,atol=1e-11)
    # A leaf too small to split vetoes the global shared cut.
    tiny=np.zeros(96,dtype='int32');tiny[-7:]=1
    assert best_shared_split(b,grids,tiny,residual,w,minimum=8) is None


def test_quantile_grid_and_raw_integer_tree_boundaries():
    x=np.column_stack([np.arange(32),np.repeat([0,5,10,15],8)]).astype('int32')
    grids=cut_grid(x,bins=4);assert grids[0].tolist()==[7,15,23]
    query=np.column_stack([np.arange(-5,38),np.arange(-5,38)])
    b=encode_bins(query,grids)
    for j,cuts in enumerate(grids):
        for k,cut in enumerate(cuts):np.testing.assert_array_equal(b[:,j]<=k,query[:,j]<=cut)
    w=np.ones(32);y=np.where(x[:,0]>15,2.,-1.)+np.where(x[:,1]>5,1.,-1.)
    for fit in [symmetric_tree,asymmetric_tree]:
        t=fit(encode_bins(x,grids),grids,y,w,depth=3,minimum=3)
        leaf=leaf_indices(x,t)
        for node in set(leaf):
            assert (leaf==node).sum()>=3
            np.testing.assert_allclose(t['value'][node],np.average(y[leaf==node],weights=w[leaf==node]))
        assert np.isfinite(np.asarray(t['value'])[leaf_indices(query,t)]).all()


def test_ties_and_constant_inputs_produce_deterministic_supported_trees():
    x=np.column_stack([np.arange(40),np.arange(40)]).astype('int32')
    grids=cut_grid(x,bins=4);b=encode_bins(x,grids);w=np.ones(40);y=(x[:,0]>19).astype(float)
    best=best_shared_split(b,grids,np.zeros(40,dtype='int32'),y,w,minimum=5)
    assert best[:2]==(0,1)
    z=np.zeros((40,2),dtype='int32');g=cut_grid(z);assert all(len(c)==0 for c in g)
    for fit in [symmetric_tree,asymmetric_tree]:
        tree=fit(encode_bins(z,g),g,y,w,minimum=5)
        assert tree['feature']==[-2] and tree['n_node_samples']==[40]
