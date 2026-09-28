import pandas as pd

from trade_research.tail_formula_relative import training_size_buckets


def test_tied_sizes_stay_together_without_code_tie_breaks():
    f = pd.DataFrame({'date': ['2024-01-02'] * 10,
                      'code': [str(i) for i in range(10)], 'S01': [2.] * 10})
    assert training_size_buckets(f, 5).size_bucket.tolist() == [2] * 10
    assert training_size_buckets(f, 1).size_bucket.tolist() == [0] * 10


def test_membership_precedes_unknown_label_intersection():
    f = pd.DataFrame({'date': ['2024-01-02'] * 10,
                      'code': [str(i) for i in range(10)], 'S01': list(range(10))})
    buckets = training_size_buckets(f, 5)
    assert buckets.size_bucket.tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    # Even if only two labels are known, their original visible-universe
    # group is retained rather than ranking the selected known rows again.
    known = pd.DataFrame({'date': ['2024-01-02'] * 2, 'code': ['4', '5']})
    assert known.merge(buckets, on=['date', 'code']).size_bucket.tolist() == [2, 2]
