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
