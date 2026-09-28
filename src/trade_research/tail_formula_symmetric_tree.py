"""Compare level-shared and ordinary branches on the same training cut grid."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as inputs
from . import tail_formula_prior_day as adapter
from . import tail_formula_recent as selection
from . import tail_formula_relative as relative
from . import tail_formula_histogram_tree as histogram
from .corporate_cash import save_json,sha

STEM='tail_formula_symmetric_tree'
ROOT=Path('data/research')/STEM
PROTOCOL=Path('config')/(STEM+'_protocol.json')


def policy():
    p=json.loads(PROTOCOL.read_text())
    for file,digest in p['source_hashes'].items():assert sha(Path(file))==digest
    assert p['bins']==64 and p['variants']==['asymmetric','symmetric']
    assert p['new_2026_prices_allowed'] is False
    return p


def setup(variant,fold):
    p=policy();assert variant in p['variants']
    stem=STEM+'_'+variant
    if fold=='combined':
        adapter.STEM=stem;adapter.COMBINED_PROTOCOL=Path('config')/(stem+'_combined_protocol.json')
        adapter.setup(fold)
        return json.loads(adapter.COMBINED_PROTOCOL.read_text())
    protocol=Path('config')/(stem+'_'+fold+'_protocol.json')
    config=json.loads(protocol.read_text());assert config['master_protocol_sha256']==sha(PROTOCOL)
    assert config['variant']==variant and config['expected_features']==48
    base.ROOT=Path('data/research')/(stem+'_'+fold);base.PROTOCOL=protocol;relative.PROTOCOL=protocol
    base.FEATURES=inputs.ROOT;base.EXPRESSIONS=inputs.EXPRESSIONS;base.HEADER=inputs.HEADER;base.SOURCE=labels.ROOT
    selection.ROOT=base.ROOT;selection.PROTOCOL=protocol
    return config


def model(variant):
    config=json.loads(base.PROTOCOL.read_text());master=policy()
    assert not (base.ROOT/'model_report.json').exists()
    train=relative.training('relative')
    assert (len(train),train.date.nunique())==(config['expected_training_rows'],config['expected_training_days'])
    x=base.encode(train);y=train.target.to_numpy();w=1/train.groupby('date').code.transform('size').to_numpy()
    grids=histogram.cut_grid(x,master['bins']);binned=histogram.encode_bins(x,grids)
    bias=float(np.average(y,weights=w));score=np.full(len(y),bias);trees=[]
    fit=histogram.symmetric_tree if variant=='symmetric' else histogram.asymmetric_tree
    for stage in range(64):
        tree=fit(binned,grids,y-score,w,depth=3,minimum=300)
        trees.append(tree);score+=.05*np.asarray(tree['value'])[base.leaf_indices(x,tree)]
    r=dict(protocol_sha256=sha(base.PROTOCOL),master_protocol_sha256=sha(PROTOCOL),
        feature_report_sha256=sha(inputs.ROOT/'feature_report.json'),label_report_sha256=sha(labels.ROOT/'full_label_report.json'),
        rows=len(train),days=train.date.nunique(),last_observation=train.next_date.max(),
        parameters=config['parameters'],feature_names=list(inputs.EXPRESSIONS),variant='relative',
        tree_structure=variant,training_integer_cut_grid=[a.tolist() for a in grids],quantile_bins=64,
        learning_rate=.05,bias=bias,trees=trees,training_start=config['training_start'],training_end=config['training_end'],
        new_2025_score_groups_read=bool(train.date.ge('2025-01-01').any()),
        new_2025H2_score_groups_read=bool(train.date.ge('2025-07-01').any()),
        deterministic_feature_then_cut_ties=True,no_random_feature_or_date_sampling=True,
        original_inputs_not_quantized_in_export=True,new_2026_prices_read=False,no_exit_rules=True)
    np.testing.assert_allclose(score,base.predict(x,r),rtol=0,atol=2e-12)
    r['thresholds']=[dict(id=i,training_quantile=q,threshold=float(np.quantile(score,q))) for i,q in enumerate(base.QUANTILES)]
    base.ROOT.mkdir(parents=True,exist_ok=True);save_json(base.ROOT/'model_report.json',r)
    return {k:v for k,v in r.items() if k not in ['trees','training_integer_cut_grid']}


def verify_model(variant):
    config=json.loads(base.PROTOCOL.read_text());r=json.loads((base.ROOT/'model_report.json').read_text())
    assert r['tree_structure']==variant and r['master_protocol_sha256']==sha(PROTOCOL)
    assert r['parameters']==config['parameters'] and r['feature_names']==list(inputs.EXPRESSIONS)
    train=relative.training('relative');x=base.encode(train)
    # Reconstruct linear quantiles from order-statistic indices, without np.quantile.
    positions=(len(x)-1)*np.arange(1,64)/64;low=np.floor(positions).astype(int);high=np.ceil(positions).astype(int)
    grids=[]
    for column in x.T:
        ordered=np.sort(column);fraction=positions-low
        cuts=np.unique(np.floor(ordered[low]*(1-fraction)+ordered[high]*fraction).astype('int32'))
        grids.append(cuts[(cuts>=ordered[0])&(cuts<ordered[-1])].tolist())
    assert grids==r['training_integer_cut_grid']
    shared_levels=0
    for tree in r['trees']:
        levels={0:0};by_level={}
        for node,feature in enumerate(tree['feature']):
            if feature<0:continue
            cut=tree['threshold'][node];assert cut in grids[feature]
            depth=levels[node];by_level.setdefault(depth,[]).append((feature,cut))
            levels[tree['children_left'][node]]=levels[tree['children_right'][node]]=depth+1
        if variant=='symmetric':
            assert len(tree['feature'])==2**(len(by_level)+1)-1
            for level,values in by_level.items():
                assert len(values)==2**level and len(set(values))==1
                split=tree['shared_splits'][level]
                assert values[0]==(split['feature'],grids[split['feature']][split['cut_index']])
                assert split['gain']>1e-15
                shared_levels+=1
    del train,x
    proof=relative.verify_model('relative')
    proof.update(all_training_quantile_cuts_rebuilt_from_order_statistics=True,
        all_raw_integer_export_thresholds_on_training_grid=True,tree_structure=variant,
        shared_levels_checked=shared_levels,symmetric_level_conditions_identical=(variant=='symmetric'))
    save_json(base.ROOT/'model_verification.json',proof);return proof


def main():
    from .tail_formula_offset_logit48 import verify_scores
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['model','verify_model','scores','verify_scores','freeze','verify','analyze'])
    p.add_argument('--variant',choices=['asymmetric','symmetric'],required=True)
    p.add_argument('--fold',choices=['2024','recent','combined'],default='2024');a=p.parse_args()
    setup(a.variant,a.fold)
    if a.stage=='analyze':
        gate=json.loads((ROOT/'joint_selection_freeze.json').read_text())
        assert gate['passed'] and gate['protocol_sha256']==sha(PROTOCOL)
        for item in gate['selections']:
            folder=Path(item['root']);assert item['selection_report_sha256']==sha(folder/'selection_report.json')
            proof=json.loads((folder/'selection_verification.json').read_text())
            assert proof['passed'] and proof['selection_report_sha256']==sha(folder/'selection_report.json')
        result=evaluation.analyze(linkage.COMBINED if a.fold=='combined' else base.ROOT,
            linkage.PROTOCOL if a.fold=='combined' else base.PROTOCOL)
    elif a.fold=='combined':
        assert a.stage in ['freeze','verify'];result=linkage.combine() if a.stage=='freeze' else linkage.verify_combined()
    elif a.stage in ['model','verify_model']:
        result=globals()[a.stage](a.variant)
    elif a.stage in ['freeze','verify']:
        result=getattr(selection,a.stage)()
        if a.stage=='verify':
            m=json.loads((base.ROOT/'model_report.json').read_text())
            assert (base.ROOT/'frozen_numeric_core.tdx').read_text()==base.native_core(m,m['thresholds'][3]['threshold'],inputs.EXPRESSIONS,inputs.HEADER)
    else:
        result=verify_scores() if a.stage=='verify_scores' else base.scores()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
