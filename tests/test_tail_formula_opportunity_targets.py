import json
import numpy as np
import pandas as pd
import pytest

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_relative as target
from trade_research.research_io import save_json, sha
from find_existing_tail_formula_models import confirmed_target, MODEL_CORE


@pytest.fixture
def fixed_training(tmp_path, monkeypatch):
    # The invalid-input stock belongs to the known date-label population.
    # Unknown outcomes and observations after the fold boundary cannot enter.
    labels = pd.DataFrame(dict(
        date=['2023-01-03']*3 + ['2023-01-04', '2023-12-29'],
        code=['a','b','c','d','e'],
        next_date=['2023-01-04']*3 + ['2023-01-05', '2024-01-02'],
        known15=[True,True,False,True,True],
        opportunity15=[0.,1.,np.nan,1.,1.],
        known_no_trade=[False]*5,
    ))
    labels.to_parquet(tmp_path/'full_labels.parquet',index=False)
    save_json(tmp_path/'full_label_report.json',{'labels_sha256':sha(tmp_path/'full_labels.parquet')})
    save_json(tmp_path/'full_label_verification.json',dict(passed=True,
        label_report_sha256=sha(tmp_path/'full_label_report.json')))
    features=labels[['date','code']].copy()
    features['formula_input_valid']=[True,False,True,True,True]
    monkeypatch.setattr(base,'SOURCE',tmp_path)
    monkeypatch.setattr(base,'feature_inputs',lambda:features.copy())
    protocol=tmp_path/'fold.json'
    protocol.write_text(json.dumps({'training_start':'2023-01-01','training_end':'2024-01-01'}))
    monkeypatch.setattr(target,'PROTOCOL',protocol)


def test_absolute_keeps_market_level_and_known_population(fixed_training):
    t=target.training('absolute')
    assert t.code.tolist()==['a','d']
    np.testing.assert_array_equal(t.target,[0.,1.])
    assert t.next_date.max()<'2024-01-01'


def test_relative_centers_before_input_intersection(fixed_training):
    t=target.training('relative')
    assert t.code.tolist()==['a','d']
    # 'b' affects the same-day mean despite an invalid formula input.
    np.testing.assert_array_equal(t.target,[-.5,0.])


def test_unknown_or_after_boundary_are_not_zero_imputed(fixed_training):
    for variant in ['relative','absolute']:
        t=target.training(variant)
        assert not set(t.code).intersection(['c','e'])
        assert np.isfinite(t.target).all()


def target_chain(tmp_path):
    parent=tmp_path/'parent';copy=tmp_path/'copy';parent.mkdir();copy.mkdir()
    model={k:[] for k in MODEL_CORE};model.update(variant='relative',bias=0.,rows=2)
    for root in [parent,copy]:save_json(root/'model_report.json',model)
    save_json(parent/'model_verification.json',dict(passed=True,
        model_report_sha256=sha(parent/'model_report.json'),variant='relative',
        all_targets_integer_inputs_day_weights_residual_means_and_variances_rebuilt=True))
    proof=sha(parent/'model_verification.json')
    save_json(copy/'model_verification.json',dict(passed=True,
        model_report_sha256=sha(copy/'model_report.json'),reused_model_verification_sha256=proof,
        exact_input_target_weight_and_parameter_equivalence_before_proof_reuse=True,
        all_original_arithmetic_checks_reused=True))
    return parent,copy,{proof:parent}


def test_legacy_target_requires_full_hash_linked_equivalence(tmp_path):
    parent,copy,registry=target_chain(tmp_path);receipts={}
    assert confirmed_target(copy,registry,receipts)=='relative'
    assert len(receipts)==4
    p=copy/'model_report.json';m=json.loads(p.read_text());m['bias']=.5;save_json(p,m)
    v=copy/'model_verification.json';r=json.loads(v.read_text());r['model_report_sha256']=sha(p);save_json(v,r)
    with pytest.raises(AssertionError,match='different model core'):
        confirmed_target(copy,registry,{})


def test_model_variant_alone_cannot_identify_undeclared_target(tmp_path):
    parent,copy,registry=target_chain(tmp_path)
    v=copy/'model_verification.json';r=json.loads(v.read_text())
    r.pop('exact_input_target_weight_and_parameter_equivalence_before_proof_reuse')
    save_json(v,r)
    with pytest.raises(AssertionError,match='independent arithmetic'):
        confirmed_target(copy,registry,{})
