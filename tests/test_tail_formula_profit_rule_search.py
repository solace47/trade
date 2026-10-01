import numpy as np

from trade_research.tail_formula_profit_rule_search import learn, statistics


def protocol():
    return dict(max_conditions=2, beam_width=1, objective_integer_scale=1000000000,
        minimum_signal_days_each_training_half=2, minimum_known_opportunity_rows_each_training_half=2,
        minimum_reference_days_each_training_half=2, minimum_known_reference_rows_each_training_half=2)


def test_missing_return_is_not_zero_and_still_reduces_opportunity_lower_bound():
    dates = np.repeat(np.arange(4), 2)
    known = np.tile([True, False], 4)
    result = statistics(np.ones(8, dtype=bool), dates, np.array([0, 0, 1, 1]), known, known,
                        known, np.tile([.02, np.nan], 4))
    assert [h['reference'] for h in result] == [.02, .02]
    assert [h['reference_rows'] for h in result] == [2, 2]
    assert [h['lower'] for h in result] == [.5, .5]


def test_search_can_expand_parent_that_fails_final_opportunity_gate():
    x = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 1))
    dates = np.repeat(np.arange(4), 4)
    known = np.ones(16, dtype=bool)
    success = (x[:, 0] == 1) & (x[:, 1] == 1)
    reference = np.where(success, .03, -.01)
    atoms = [(0, 1, 0), (1, 1, 0)]
    trace, _, chosen = learn(x, atoms, dates, np.array([0, 0, 1, 1]), known, success,
                            known, reference, protocol())
    assert chosen['conditions'] == ((0, 1, 0), (1, 1, 0))
    assert all(h['lower'] == .5 for r in trace if r['depth'] == 1 for h in r['halves'])
    assert all(h['reference'] == .03 and h['lower'] == 1 for h in chosen['halves'])


def test_robust_variant_rejects_positive_mean_supported_by_rare_jumps():
    x=np.tile([[1],[0]],(30,1))
    dates=np.repeat(np.arange(30),2)
    halves=np.repeat([0,1],15)
    known=np.ones(len(x),dtype=bool)
    reference=np.full(len(x),-.001)
    reference[[0,30]]=.12
    atoms=[(0,1,0)]
    _,_,mean_choice=learn(x,atoms,dates,halves,known,known,known,reference,protocol())
    _,_,robust_choice=learn(x,atoms,dates,halves,known,known,known,reference,
                          dict(protocol(),robust_objective=True))
    assert mean_choice is not None
    assert robust_choice is None


def test_single_discovery_segment_does_not_require_a_fictitious_second_segment():
    x=np.tile([[1],[0]],(4,1))
    days=np.repeat(np.arange(4),2)
    known=np.ones(len(x),dtype=bool)
    reference=np.tile([.02,-.01],4)
    settings=dict(protocol(),training_segments=1,robust_objective=True)
    trace,_,choice=learn(x,[(0,1,0)],days,np.zeros(4,dtype=int),known,known,known,reference,settings)
    assert choice is not None and choice['conditions']==((0,1,0),)
    assert trace[0]['halves'][1]['days']==0


def test_single_validation_rejects_without_choosing_profitable_alternative(tmp_path,monkeypatch):
    import pandas as pd
    import run_tail_formula_profit_rule_search as runner
    names=list(runner.original.EXPRESSIONS)
    visible=pd.DataFrame(np.zeros((8,len(names))),columns=names)
    visible['C06']=np.tile([1.,0.],4)
    visible['date']=np.repeat(['2024-07-01','2024-07-02','2024-07-03','2024-07-04'],2)
    visible['code']=np.tile(['sh.600001','sh.600002'],4)
    visible['formula_input_valid']=True
    labels=visible[['date','code']].copy()
    labels['next_date']='2024-07-05'
    labels.loc[6,'next_date']='2025-01-01'
    labels['known15']=True
    labels['opportunity15']=np.tile([False,True],4)
    labels['target_valid']=True
    labels['net']=100*np.log1p(np.tile([-.01,.05],4))
    path=tmp_path/'targets.parquet'
    labels.to_parquet(path,index=False)
    monkeypatch.setattr(runner,'TARGETS',path)
    chosen={'conditions':[[names.index('C06'),1,10000]]}
    settings=dict(protocol(),training_segments=1,feature_names=names)
    accepted,receipt=runner.chronological_validation_once(
        dict(split='2024-07-01',training_end='2025-01-01'),chosen,visible,labels,settings)
    assert accepted is None and receipt['validation_rules_examined']==1
    assert receipt['no_alternative_after_failure'] and receipt['independent_SQL_verified']
    assert receipt['statistics']['known']==3 and receipt['statistics']['reference_days']==3
    assert receipt['statistics']['days']==4
    assert abs(receipt['statistics']['reference']+.01)<1e-12
