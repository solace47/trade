import numpy as np
import pandas as pd

from run_tail_formula_phase_split import candidates


def test_positive_gate_invalid_inputs_and_boundary_ties():
    rows = pd.DataFrame(dict(
        date=['2024-01-02'] * 10 + ['2024-01-03'] * 6 + ['2024-01-04'] * 3,
        code=[str(i) for i in range(19)],
        score=[10., 9., 8., 7., 6., 5., 0., -1., np.nan, 100.] + [1.] * 6 + [0., -1., np.nan],
        formula_input_valid=[True] * 9 + [False] + [True] * 9,
    ))
    chosen = candidates(rows)
    # Invalid high scores cannot displace valid candidates, six tied positive
    # stocks cannot silently become an arbitrary five, and no positive day stays empty.
    assert chosen.loc[chosen.flag, 'code'].tolist() == ['0', '1', '2', '3', '4']
    assert not chosen.code.isin(['6', '7', '8', '9', '16', '17', '18']).any()
    assert not chosen.loc[chosen.date.eq('2024-01-03'), 'flag'].any()


def test_integer_rounding_can_create_an_overfull_boundary():
    rows = pd.DataFrame(dict(date=['2024-01-02'] * 6, code=list('abcdef'),
        score=[5., 4., 3., 2., 1.0000001, 1.0000002], formula_input_valid=[True] * 6))
    # Both lower scores round to the same integer. The whole day must then be empty.
    assert not candidates(rows).flag.any()
