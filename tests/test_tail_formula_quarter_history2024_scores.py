import ast

import duckdb
import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_quarter_history2024_scores as score
from trade_research import tail_formula_quarter_minimum as minimum


def test_export_replays_negative_bias_minimum_and_integer_split():
    def component(bias, left, right):
        tree = dict(children_left=[1, -1, -1], children_right=[2, -1, -1],
                    feature=[0, -2, -2], threshold=[10000.9, -2, -2], value=[0, left, right])
        return dict(bias=bias, learning_rate=.05, trees=[tree] * 64)

    m = dict(full=component(-.125, 2, -1),
             components=[component(-.2, -1, 2), component(.4, 2, -.5),
                         component(-.01, .2, .3), component(.1, 0, 0)],
             thresholds=dict(full=dict(threshold=.02), minimum=dict(threshold=-.01)))
    cores = score.native_cores(m)
    definitions = score.full_definitions(cores['full'], m)
    lower = minimum.verify_native(cores['minimum'],
        dict(components=m['components'], thresholds=[m['thresholds']['minimum']]))
    c = duckdb.connect()
    encoded = pd.DataFrame(dict(date=['2024-01-02'] * 3, code=['a', 'b', 'c'], X01=[9999, 10000, 10001]))
    c.register('encoded', encoded)
    sql = lambda x: minimum.expression_sql(ast.parse(x, mode='eval').body)
    c.sql('SELECT date,code,' + ','.join(sql(definitions[f'T{i:02d}']) + f' AS T{i:02d}' for i in range(1, 65))
          + ' FROM encoded').create_view('full_leaves')
    full = c.sql('SELECT ' + sql(definitions['SC']) + ' FROM full_leaves ORDER BY code').fetchnumpy()
    actual = next(iter(full.values()))
    np.testing.assert_allclose(actual, [6.275, 6.275, -3.325], rtol=0, atol=1e-12)
    np.testing.assert_allclose(actual, base.predict(encoded[['X01']].to_numpy(), m['full']), rtol=0, atol=1e-12)
    minimum.score_views(c, lower)
    low = c.sql('SELECT score FROM rebuilt ORDER BY code').fetchnumpy()['score']; c.close()
    np.testing.assert_allclose(low, [-3.4, -3.4, -1.2], rtol=0, atol=1e-12)
    np.testing.assert_allclose(low, minimum.predict(encoded[['X01']].to_numpy(), m), rtol=0, atol=1e-12)
