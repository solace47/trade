"""A fresh selection process must honor the protocol's extra input column."""
import json

import pandas as pd

import run_tail_formula_target as workflow
from trade_research import tail_formula_baseline as baseline


def test_frozen_selection_binds_the_declared_fifty_first_column(tmp_path, monkeypatch):
    features = tmp_path/'inputs'; features.mkdir()
    expressions = {**baseline.EXPRESSIONS, 'EXTRA': 'C'}
    frame = pd.DataFrame(dict(date=['2024-01-02']*2, code=['sh.600000','sh.600001'],
        half=['2024H1']*2, board=['main']*2, decision_shares=[100,100], formula_input_valid=[True,True]))
    for name in baseline.EXPRESSIONS: frame[name] = 0.
    frame['EXTRA'] = [-1.,1.]
    frame.to_parquet(features/'features.parquet', index=False)
    model = tmp_path/'model'; model.mkdir()
    report = dict(bias=0.,learning_rate=.05,feature_names=list(expressions),last_observation='2023-12-29',
        thresholds=[dict(training_quantile=.995,threshold=0.)],trees=[dict(
            children_left=[1,-1,-1],children_right=[2,-1,-1],feature=[50,-2,-2],
            threshold=[10000,-2,-2],value=[0.,-1.,1.])])
    (model/'model_report.json').write_text(json.dumps(report))
    registry = tmp_path/'registry.json'; registry.write_text(json.dumps(dict(reports=[],source_hashes={})))
    monkeypatch.setattr(workflow,'ROOT',tmp_path)
    monkeypatch.setattr(workflow.numeric,'EXPRESSIONS',baseline.EXPRESSIONS)
    monkeypatch.setattr(workflow,'committed_receipt',lambda name:dict(models=[dict(fold='h1',root=str(model))],source_hashes={}))
    monkeypatch.setattr(workflow,'check_sources',lambda sources:None)
    workflow.freeze(dict(expressions=expressions,features_root=str(features),
        folds=[dict(id='h1',evaluation_start='2024-01-01',evaluation_end='2025-01-01')],
        prior_registry=str(registry),prior_complete_extra=[],group_prefix='binding'))
    result = pd.read_parquet(tmp_path/'binding2024/selection.parquet')
    assert result.selected.tolist() == [False,True]
    assert not pd.read_parquet(tmp_path/'binding2025/selection.parquet').selected.any()
