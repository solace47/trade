"""A fixed arithmetic mean of verified one-year and three-year formulas."""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from . import tail_formula_additive as base
from . import tail_formula_float as original
from . import tail_formula_context_2024 as linkage
from . import tail_formula_long48 as history
from . import tail_formula_recent as study
from .corporate_cash import save_json, sha

STEM = 'tail_formula_horizon_blend'


def setup(fold):
    # Reuse the verified time-fold and combination adapters with new identities.
    history.STEM = STEM
    history.setup(fold)


def components():
    p = json.loads(base.PROTOCOL.read_text())
    assert p['weight']==.5 and p['model_max_depth']==3 and len(p['components'])==2
    assert p['feature_report_sha256']==sha(base.FEATURES/'feature_report.json')
    assert p['label_report_sha256']==sha(base.SOURCE/'full_label_report.json')
    result = []
    for item in p['components']:
        root = Path(item['root'])
        assert item['model_report_sha256']==sha(root/'model_report.json')
        assert item['model_verification_sha256']==sha(root/'model_verification.json')
        proof = json.loads((root/'model_verification.json').read_text())
        assert proof['passed'] and proof['model_report_sha256']==item['model_report_sha256']
        model = json.loads((root/'model_report.json').read_text())
        assert model['variant']=='relative' and model['feature_names']==list(original.EXPRESSIONS)
        assert model['learning_rate']==.05 and len(model['trees'])==64
        assert model['training_end']==p['training_end'] and model['last_observation']<p['evaluation_start']
        result.append(model)
    assert min(x['training_start'] for x in result)==p['training_start']
    return p,result


def model():
    p,parents = components(); root = base.ROOT
    if (root/'model_report.json').exists():
        raise ValueError('Do not replace the frozen horizon blend')
    root.mkdir(parents=True,exist_ok=True)
    train = base.training(start=p['training_start'],end=p['training_end'])
    x = base.encode(train)
    trees = []
    for parent in parents:
        for tree in parent['trees']:
            t = deepcopy(tree); t['value'] = (np.array(tree['value'])*.5).tolist(); trees.append(t)
    r = dict(protocol_sha256=sha(base.PROTOCOL),feature_report_sha256=sha(base.FEATURES/'feature_report.json'),
        label_report_sha256=sha(base.SOURCE/'full_label_report.json'),components=p['components'],
        component_training_starts=[m['training_start'] for m in parents],
        construction='fixed_arithmetic_mean_not_joint_GBR_fit',node_training_metadata_is_component_metadata=True,
        rows=len(train),rows_are_threshold_calibration_only=True,days=train.date.nunique(),
        last_observation=max([m['last_observation'] for m in parents]+[train.next_date.max()]),
        training_start=p['training_start'],training_end=p['training_end'],
        parameters=dict(n_estimators=128,max_depth=3,component_weight=.5,new_fit_performed=False),
        feature_names=list(original.EXPRESSIONS),variant='fixed_horizon_blend',learning_rate=.05,
        bias=float(np.mean([m['bias'] for m in parents])),trees=trees,
        new_2025_score_groups_read=bool(train.date.ge('2025-01-01').any()),new_2025H2_score_groups_read=False,
        new_2026_prices_read=False,no_exit_rules=True)
    prediction = base.predict(x,r)
    expected = .5*base.predict(x,parents[0])+.5*base.predict(x,parents[1])
    np.testing.assert_allclose(prediction,expected,atol=3e-15,rtol=0)
    r['thresholds'] = [dict(id=i,training_quantile=q,threshold=float(np.quantile(prediction,q))) for i,q in enumerate(base.QUANTILES)]
    save_json(root/'model_report.json',r)
    return {k:v for k,v in r.items() if k!='trees'}


def verify_model():
    p,parents = components(); root = base.ROOT; r = json.loads((root/'model_report.json').read_text())
    assert r['protocol_sha256']==sha(base.PROTOCOL) and r['components']==p['components']
    assert r['feature_report_sha256']==sha(base.FEATURES/'feature_report.json') and r['label_report_sha256']==sha(base.SOURCE/'full_label_report.json')
    assert r['training_start']==p['training_start'] and r['training_end']==p['training_end']
    assert r['variant']=='fixed_horizon_blend' and r['learning_rate']==.05 and len(r['trees'])==128
    assert r['bias']==(parents[0]['bias']+parents[1]['bias'])/2
    checks = 0
    for i,parent in enumerate(parents):
        for original_tree,combined_tree in zip(parent['trees'],r['trees'][64*i:64*(i+1)]):
            assert {k:v for k,v in original_tree.items() if k!='value'}=={k:v for k,v in combined_tree.items() if k!='value'}
            np.testing.assert_array_equal(np.array(original_tree['value'])/2,combined_tree['value'])
            checks += len(original_tree['value'])
    train = base.training(start=p['training_start'],end=p['training_end']); x = base.encode(train)
    assert r['rows']==len(train) and r['days']==train.date.nunique() and r['last_observation']==train.next_date.max()
    expected = (base.predict(x,parents[0])+base.predict(x,parents[1]))/2
    actual = base.predict(x,r)
    np.testing.assert_allclose(actual,expected,atol=3e-15,rtol=0)
    for q,item in zip(base.QUANTILES,r['thresholds']):
        assert item['training_quantile']==q
        np.testing.assert_allclose(item['threshold'],np.quantile(expected,q),atol=3e-15,rtol=0)
    # The generalized exporter must preserve all prior 64-tree cores exactly.
    core_hashes = {}
    for stem in ['tail_formula_float','tail_formula_long48']:
        for fold in ['2024','recent']:
            folder = Path('data/research')/(stem+'_'+fold)
            m = json.loads((folder/'model_report.json').read_text())
            selected = json.loads((folder/'selection_report.json').read_text())
            text = base.native_core(m,selected['chosen_threshold']['threshold'],original.EXPRESSIONS,original.HEADER)
            assert text==(folder/'frozen_numeric_core.tdx').read_text()
            core_hashes[str(folder/'frozen_numeric_core.tdx')] = sha(folder/'frozen_numeric_core.tdx')
    preview = base.native_core(r,r['thresholds'][3]['threshold'],original.EXPRESSIONS,original.HEADER)
    assert all(preview.count(f'T{i:02d}:=')==1 for i in range(1,129))
    score_line = next(x for x in preview.splitlines() if x.startswith('SC:='))
    assert score_line.endswith('+'+'+'.join(f'T{i:02d}' for i in range(1,129))+';')
    proof = dict(passed=True,model_report_sha256=sha(root/'model_report.json'),calibration_rows=len(train),
        component_nodes_checked=checks,all_component_structures_and_half_values_verified=True,
        all_calibration_scores_equal_component_average=True,all_training_quantiles_rebuilt=True,
        original_64_tree_cores_unchanged_sha256=core_hashes,all_128_native_terms_retained=True,
        source_model_residual_checks_reused=True,new_fit_performed=False,
        new_2026_prices_read=False,no_exit_rules=True,software_compilation_verified=False)
    save_json(root/'model_verification.json',proof); return proof


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold',choices=['2024','recent','combined'])
    p.add_argument('stage',choices=['model','verify_model','scores','verify_scores','freeze','verify','analyze'])
    a = p.parse_args(); setup(a.fold)
    if a.fold=='combined':
        assert a.stage in ['freeze','verify','analyze']
        result = (linkage.common_analysis(linkage.COMBINED,linkage.PROTOCOL) if a.stage=='analyze'
                  else getattr(linkage,'combine' if a.stage=='freeze' else 'verify_combined')())
    elif a.stage in ['model','verify_model']:
        result = globals()[a.stage]()
    elif a.stage=='verify_scores':
        result = history.verify_scores()
    elif a.stage in ['freeze','verify']:
        result = getattr(study,a.stage)()
    else:
        result = getattr(base,a.stage)()
    print(json.dumps(result,ensure_ascii=False,indent=2))
