"""Deterministic squared-loss trees on a fixed training-only integer cut grid."""
import numpy as np


def cut_grid(x, bins=64):
    grids=[]
    for column in x.T:
        cuts=np.unique(np.floor(np.quantile(column,np.arange(1,bins)/bins,method='linear')).astype('int32'))
        grids.append(cuts[(cuts>=column.min())&(cuts<column.max())])
    return grids


def encode_bins(x, grids):
    assert x.shape[1]==len(grids)
    out=np.empty(x.shape,dtype='uint8',order='F')
    for j,cuts in enumerate(grids):
        assert len(cuts)<256 and (np.diff(cuts)>0).all()
        out[:,j]=np.searchsorted(cuts,x[:,j],side='left')
    return out


def best_shared_split(binned, grids, leaves, residual, weight, minimum=300):
    """A cut must split every current leaf into two sufficiently large children."""
    nleaves=int(leaves.max())+1
    totals=np.bincount(leaves,weights=weight,minlength=nleaves)
    sums=np.bincount(leaves,weights=weight*residual,minlength=nleaves)
    assert (totals>0).all()
    parent=float(np.sum(sums*sums/totals)); best=None; best_gain=1e-15
    for feature,cuts in enumerate(grids):
        width=len(cuts)+1
        if width==1:
            continue
        cell=leaves*width+binned[:,feature]
        count=np.bincount(cell,minlength=nleaves*width).reshape(nleaves,width)
        mass=np.bincount(cell,weights=weight,minlength=nleaves*width).reshape(nleaves,width)
        signal=np.bincount(cell,weights=weight*residual,minlength=nleaves*width).reshape(nleaves,width)
        nl=np.cumsum(count,axis=1)[:,:-1];nr=count.sum(axis=1)[:,None]-nl
        wl=np.cumsum(mass,axis=1)[:,:-1];wr=mass.sum(axis=1)[:,None]-wl
        sl=np.cumsum(signal,axis=1)[:,:-1];sr=signal.sum(axis=1)[:,None]-sl
        valid=((nl>=minimum)&(nr>=minimum)&(wl>0)&(wr>0)).all(axis=0)
        gain=np.full(len(cuts),-np.inf)
        k=np.flatnonzero(valid)
        gain[k]=np.sum(sl[:,k]**2/wl[:,k]+sr[:,k]**2/wr[:,k],axis=0)-parent
        index=int(np.argmax(gain))
        # Feature order then ascending integer cut is the deterministic tie rule.
        if gain[index]>best_gain:
            best_gain=float(gain[index]);best=(feature,index,best_gain)
    return best


def _node(residual,weight):
    mean=float(np.average(residual,weights=weight))
    return len(residual),float(weight.sum()),mean,float(np.average((residual-mean)**2,weights=weight))


def symmetric_tree(binned,grids,residual,weight,depth=3,minimum=300):
    leaves=np.zeros(len(residual),dtype='int32');splits=[]
    for level in range(depth):
        best=best_shared_split(binned,grids,leaves,residual,weight,minimum)
        if best is None:
            break
        feature,index,gain=best;splits.append((feature,index,gain))
        leaves=2*leaves+(binned[:,feature]>index)
    levels=len(splits);count=2**(levels+1)-1
    tree={k:[] for k in ['feature','threshold','children_left','children_right','n_node_samples',
        'weighted_n_node_samples','value','impurity']}
    for node in range(count):
        level=(node+1).bit_length()-1;offset=node-(2**level-1)
        mask=(leaves>>(levels-level))==offset
        n,w,v,imp=_node(residual[mask],weight[mask])
        leaf=level==levels
        feature,index,_=splits[level] if not leaf else (-2,-2,0)
        values=[feature,float(grids[feature][index]) if not leaf else -2.,
            2*node+1 if not leaf else -1,2*node+2 if not leaf else -1,n,w,v,imp]
        for key,value in zip(tree,values):tree[key].append(value)
    tree['shared_splits']=[dict(feature=j,cut_index=k,gain=g) for j,k,g in splits]
    return tree


def asymmetric_tree(binned,grids,residual,weight,depth=3,minimum=300):
    tree={k:[] for k in ['feature','threshold','children_left','children_right','n_node_samples',
        'weighted_n_node_samples','value','impurity']}
    def grow(indices,level):
        node=len(tree['feature']);n,w,v,imp=_node(residual[indices],weight[indices])
        for key,value in zip(tree,[-2,-2.,-1,-1,n,w,v,imp]):tree[key].append(value)
        if level==depth or len(indices)<2*minimum:
            return node
        part=binned[indices]
        best=best_shared_split(part,grids,np.zeros(len(indices),dtype='int32'),residual[indices],weight[indices],minimum)
        if best is None:return node
        feature,index,_=best;lower=part[:,feature]<=index
        tree['feature'][node]=feature;tree['threshold'][node]=float(grids[feature][index])
        tree['children_left'][node]=grow(indices[lower],level+1)
        tree['children_right'][node]=grow(indices[~lower],level+1)
        return node
    grow(np.arange(len(residual)),0)
    return tree
