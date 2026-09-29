import ast

import duckdb
import numpy as np
import pytest

from trade_research import tail_formula_quarter_minimum as study


def test_minimum_retains_negative_component_and_integer_branch_boundary():
    tree = dict(feature=[0, -2, -2], threshold=[2.5, -2, -2], children_left=[1, -1, -1],
                children_right=[2, -1, -1], value=[0, -4, 8])
    components = [dict(bias=bias, learning_rate=.05, trees=[tree]) for bias in [1, -.2, 3, .5]]
    x = np.array([[2], [3]], dtype='int32')
    np.testing.assert_array_equal(study.predict(x, dict(components=components)), [-.4, .2])
    expr = study.base.native_tree(tree)
    study.check_tree(ast.parse(expr, mode='eval').body, tree)
    sql = study.expression_sql(ast.parse(expr, mode='eval').body)
    c = duckdb.connect()
    actual = c.sql('SELECT ' + sql + ' FROM (VALUES (2),(3)) t(X01)').fetchnumpy()
    np.testing.assert_array_equal(next(iter(actual.values())), [-.2, .4])
    c.close()
    with pytest.raises(AssertionError):
        study.check_tree(ast.parse(expr.replace('X01<=2', 'X01<=3'), mode='eval').body, tree)


def test_nested_minimum_preserves_ties_and_rejects_unknown_native_function():
    sql = study.expression_sql(ast.parse('MIN(QMS1,MIN(QMS2,MIN(QMS3,QMS4)))', mode='eval').body)
    c = duckdb.connect()
    got = c.sql('SELECT ' + sql + ' FROM (VALUES (1,1,2,3),(-2,3,0,-1)) t(QMS1,QMS2,QMS3,QMS4)').fetchall()
    assert got == [(1,), (-2,)]
    c.close()
    with pytest.raises(KeyError):
        study.expression_sql(ast.parse('ABS(QMS1)', mode='eval').body)
