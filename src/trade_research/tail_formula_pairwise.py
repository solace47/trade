"""Date-balanced binary opportunity ranking with fixed within-date pairs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.tree import DecisionTreeRegressor

from . import tail_formula_additive as base
from . import tail_formula_before1000_model as adapter
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_pairwise'
PROTOCOL = Path('config')/(STEM+'_protocol.json')
PARAMETERS = dict(loss='within_date_pairwise_binary_logistic', n_estimators=64,
    learning_rate=.05, max_depth=3, min_samples_leaf=300, subsample=1.,
    random_state=20260927, criterion='friedman_mse', pair_passes=8,
    sigmoid_scale=1., newton_leaf_clip=2.)


def setup(fold):
    adapter.STEM = STEM; adapter.setup(fold)
    for name in ['2024','recent']:
        p = json.loads((Path('config')/(STEM+'_'+name+'_protocol.json')).read_text())
        assert p['pairwise_protocol_sha256'] == sha(PROTOCOL) and p['parameters'] == PARAMETERS
    main = json.loads(PROTOCOL.read_text())
    assert main['parameters'] == PARAMETERS and not main['new_2026_prices_allowed']


def make_pairs(frame, passes=8, seed=20260927):
    """Every record in an informative date participates on every fixed pass."""
    assert frame.index.equals(pd.RangeIndex(len(frame)))
    assert frame.opportunity15.isin([0,1]).all()
    rng = np.random.default_rng(seed); parts = []; days = []
    for day, group in frame.groupby('date', sort=True):
        positive = group.index[group.opportunity15.eq(1)].to_numpy()
        negative = group.index[group.opportunity15.eq(0)].to_numpy()
        n = max(len(positive),len(negative)); informative = bool(len(positive) and len(negative))
        days.append(dict(date=day, positive=len(positive), negative=len(negative),
            pairs=passes*n if informative else 0, informative=informative))
        if not informative:
            continue
        for turn in range(passes):
            a = np.resize(rng.permutation(positive), n)
            b = np.resize(rng.permutation(negative), n)
            parts.append(pd.DataFrame(dict(date=day, turn=turn, positive=a, negative=b, weight=1/(passes*n))))
    pairs = pd.concat(parts,ignore_index=True) if parts else pd.DataFrame(columns=['date','turn','positive','negative','weight'])
    return pairs, pd.DataFrame(days)


def derivatives(score, positive, negative, weight):
    rho = expit(score[negative]-score[positive])
    force = weight*rho
    gradient = np.bincount(positive, weights=force, minlength=len(score))-np.bincount(negative, weights=force, minlength=len(score))
    curvature = weight*rho*(1-rho)
    loss = np.sum(weight*np.logaddexp(0.,score[negative]-score[positive]))
    return gradient, curvature, float(loss)


def leaf_step(node, leaves, gradient, curvature, positive, negative, clip=2.):
    # Pairs wholly inside the leaf do not change margin under a leaf shift.
    crossing = (leaves[positive] == node) != (leaves[negative] == node)
    numerator = float(gradient[leaves == node].sum())
    denominator = float(curvature[crossing].sum())
    if denominator == 0:
        assert abs(numerator) < 1e-10
        return 0., numerator, denominator
    return float(np.clip(numerator/denominator,-clip,clip)), numerator, denominator


def model():
    root=base.ROOT; assert not (root/'model_report.json').exists()
    p=json.loads(base.PROTOCOL.read_text()); start,end,_=relative.training_scope()
    t=base.training(start=start,end=end); x=base.encode(t)
    assert (len(t),t.date.nunique()) == (p['expected_rows'],241)
    assert t.next_date.max()<end and t.opportunity15.isin([0,1]).all()
    pairs,days=make_pairs(t); assert len(pairs)>0
    positive=pairs.positive.to_numpy(dtype='int64'); negative=pairs.negative.to_numpy(dtype='int64'); weight=pairs.weight.to_numpy()
    w=1/t.groupby('date').code.transform('size').to_numpy()
    score=np.zeros(len(t)); trees=[]; trace=[]; rng=np.random.RandomState(20260927)
    for iteration in range(64):
        gradient,curvature,loss=derivatives(score,positive,negative,weight)
        residual=gradient/w
        np.testing.assert_allclose(pd.Series(gradient).groupby(t.date).sum(),0,rtol=0,atol=2e-12)
        estimator=DecisionTreeRegressor(criterion='friedman_mse',max_depth=3,min_samples_leaf=300,random_state=rng).fit(x,residual,sample_weight=w)
        q=estimator.tree_; leaves=estimator.apply(x)
        tree=dict(feature=q.feature.tolist(),threshold=q.threshold.tolist(),children_left=q.children_left.tolist(),
            children_right=q.children_right.tolist(),n_node_samples=q.n_node_samples.tolist(),
            weighted_n_node_samples=q.weighted_n_node_samples.tolist(),gradient_value=q.value.reshape(-1).tolist(),
            value=q.value.reshape(-1).tolist(),impurity=q.impurity.tolist(),leaf_numerator={},leaf_curvature={})
        for node in np.flatnonzero(q.children_left<0):
            value,numerator,denominator=leaf_step(node,leaves,gradient,curvature,positive,negative)
            tree['value'][node]=value;tree['leaf_numerator'][str(node)]=numerator;tree['leaf_curvature'][str(node)]=denominator
        trees.append(tree);score+=.05*np.asarray(tree['value'])[leaves]
        loss_after=float(np.sum(weight*np.logaddexp(0.,score[negative]-score[positive])))
        trace.append(dict(iteration=iteration,loss_before=loss,loss_after=loss_after,
            gradient_sum=float(gradient.sum()),clipped_leaves=sum(abs(tree['value'][n])==2 for n in np.flatnonzero(q.children_left<0))))
        assert np.isfinite(score).all() and np.isfinite(loss_after)
    root.mkdir(parents=True,exist_ok=True);pairs.to_parquet(root/'training_pairs.parquet',index=False,compression='zstd')
    days.to_parquet(root/'training_days.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(base.PROTOCOL),feature_report_sha256=sha(base.FEATURES/'feature_report.json'),
        label_report_sha256=sha(base.SOURCE/'full_label_report.json'),rows=len(t),days=t.date.nunique(),
        last_observation=t.next_date.max(),parameters=PARAMETERS,feature_names=list(base.EXPRESSIONS),
        variant='same_date_binary_pairwise',learning_rate=.05,bias=0.,trees=trees,trace=trace,
        training_start=start,training_end=end,training_pairs_sha256=sha(root/'training_pairs.parquet'),
        training_days_sha256=sha(root/'training_days.parquet'),pairs=len(pairs),informative_days=int(days.informative.sum()),
        constant_label_days=days.loc[~days.informative,'date'].tolist(),same_day_score_offsets_cancel_in_training_loss=True,
        no_cross_sectional_ranking_needed_at_inference=True,scores_are_not_win_probabilities=True,
        new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()),new_2025H2_score_groups_read=False,
        new_2026_prices_read=False,no_exit_rules=True)
    exported=base.predict(x,r);np.testing.assert_allclose(score,exported,rtol=0,atol=2e-12)
    r['thresholds']=[dict(id=i,training_quantile=q,threshold=float(np.quantile(exported,q))) for i,q in enumerate(base.QUANTILES)]
    save_json(root/'model_report.json',r)
    return {k:v for k,v in r.items() if k not in ['trees','trace']}


if __name__=='__main__':
    from .tail_formula_offset_logit48 import verify_scores
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['model','scores','verify_scores','freeze','verify','analyze'])
    p.add_argument('--fold',choices=['2024','recent','combined'],default='2024');a=p.parse_args();setup(a.fold)
    if a.stage=='analyze':
        assert all((Path('data/research')/(STEM+'_'+f)/'selection_verification.json').exists() for f in ['2024','recent','2025'])
        root=linkage.COMBINED if a.fold=='combined' else base.ROOT
        result=evaluation.analyze(root,linkage.PROTOCOL if a.fold=='combined' else base.PROTOCOL)
    elif a.fold=='combined':
        assert a.stage in ['freeze','verify'];result=linkage.combine() if a.stage=='freeze' else linkage.verify_combined()
    elif a.stage=='model':
        result=model()
    elif a.stage=='verify_scores':
        result=verify_scores()
    elif a.stage in ['freeze','verify']:
        result=getattr(study,a.stage)()
    else:
        result=base.scores()
    print(json.dumps(result,ensure_ascii=False,indent=2))
