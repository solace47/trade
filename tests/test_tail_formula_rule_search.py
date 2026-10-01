import numpy as np

from trade_research.tail_formula_rule_search import learn, statistics


def protocol():
    return dict(atomic_training_quantiles=[.1, .5, .9], max_conditions=3,
        beam_width=16, objective_integer_scale=1000000000,
        minimum_signal_days_each_training_half=2,
        minimum_known_opportunity_rows_each_training_half=2)


def test_unknown_selected_rows_reduce_lower_bound_without_becoming_known_failures():
    dates = np.repeat(np.arange(4), 2)
    known = np.tile([True, False], 4)
    selected = np.ones(8, dtype=bool)
    result = statistics(selected, dates, np.array([0, 0, 1, 1]), known, known, 2, 2)
    assert [r['lower'] for r in result] == [.5, .5]
    assert [r['known'] for r in result] == [2, 2]
    assert [r['rows'] for r in result] == [4, 4]


def test_two_conditions_find_joint_success_when_individual_conditions_cannot_pass_gate():
    x = np.tile([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 1))
    date_ids = np.repeat(np.arange(4), 4)
    known = np.ones(len(x), dtype=bool)
    success = (x[:, 0] == 1) & (x[:, 1] == 1)
    _, trace, _, chosen = learn(x, date_ids, np.array([0, 0, 1, 1]), known, success, protocol())
    assert chosen['conditions'] == ((0, 1, 0), (1, 1, 0))
    assert all(len({a[0] for a in r['conditions']}) == len(r['conditions']) for r in trace)
    assert all(r['lower'] == 1 for r in chosen['halves'])
