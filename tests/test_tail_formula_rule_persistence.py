import json

import run_tail_formula_rule_persistence as persistence
from trade_research.research_io import save_json,sha


def test_empty_update_keeps_only_a_previous_nonempty_rule(tmp_path,monkeypatch):
    bank=tmp_path/'bank'
    bank.mkdir()
    folds=[]
    for index,(start,end,rule) in enumerate([
        ('2024-01-01','2024-07-01',None),('2024-07-01','2025-01-01',{'conditions':[[0,0,17]]}),
        ('2025-01-01','2025-07-01',None),('2025-07-01','2026-01-01',{'conditions':[[0,1,23]]})]):
        fold=dict(id=str(index),training_end=start,evaluation_start=start,evaluation_end=end)
        folds.append(fold)
        path=bank/str(index)
        path.mkdir()
        save_json(path/'search_trace.json',{})
        save_json(path/'model_report.json',dict(fold=fold,feature_names=['C06'],chosen_rule=rule,
                                              search_trace_sha256=sha(path/'search_trace.json')))
        save_json(path/'model_verification.json',dict(passed=True,model_report_sha256=sha(path/'model_report.json')))
    protocol=tmp_path/'protocol.json'
    execution=tmp_path/'execution.json'
    save_json(protocol,dict(original_bank=str(bank),folds=folds,feature_names=['C06'],policy='last_nonempty'))
    save_json(execution,{})
    monkeypatch.setattr(persistence,'ROOT',tmp_path/'result')
    monkeypatch.setattr(persistence,'PROTOCOL',protocol)
    monkeypatch.setattr(persistence,'EXECUTION',execution)
    monkeypatch.setattr(persistence,'checked',lambda:(json.loads(protocol.read_text()),{}))
    result=persistence.prepare()
    assert [x['source_index'] for x in result['bindings']]==[None,1,1,3]
    assert [x['carried_forward'] for x in result['bindings']]==[False,False,True,False]
    assert result['new_fits']==0
    carried=json.loads((persistence.ROOT/'2/model_report.json').read_text())
    assert carried['chosen_rule']=={'conditions':[[0,0,17]]}
    assert carried['source_fold']==folds[1]
    assert carried['source_training_metrics_not_current_training_fit']
    assert json.loads((persistence.ROOT/'0/model_report.json').read_text())['chosen_rule'] is None
