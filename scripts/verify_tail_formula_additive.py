"""Independent SQL replay of all model scores and training thresholds."""
import json
import math
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from trade_research.research_io import save_json, sha

ROOT = Path('data/research/tail_formula_minute_open')
FEATURES = SOURCE = ROOT / 'inputs'
PROTOCOL = Path('config/tail_formula_minute_open_model_protocol.json')

def load(name):
    return json.loads((ROOT/name).read_text())


def tree_sql(tree,node=0):
    left=tree['children_left'][node]
    if left<0:
        # Scientific literals force DOUBLE rather than decimal arithmetic.
        return format(.05*tree['value'][node],'.17e')
    right=tree['children_right'][node]
    feature=tree['feature'][node]+1
    cut=math.floor(tree['threshold'][node])
    return f'(CASE WHEN X{feature:02d}<={cut} THEN {tree_sql(tree,left)} ELSE {tree_sql(tree,right)} END)'


def connection():
    c=duckdb.connect()
    c.execute('SET threads=4')
    columns=['date','code','half','board','decision_shares','formula_input_valid',*load('model_report.json')['feature_names']]
    assert len({name.casefold() for name in columns})==len(columns)
    # Read exact Parquet field names before DuckDB's case-insensitive binding.
    c.register('features',pd.read_parquet(FEATURES/'features.parquet',columns=columns))
    c.read_parquet(str(ROOT/'scores.parquet')).create_view('scores')
    return c


def native_definitions(fr, expected_expressions, definition_protocol=None):
    expressions=fr.get('expressions',fr.get('native_expressions'))
    assert isinstance(expressions,dict) and expressions
    if 'expressions' in fr and 'native_expressions' in fr:
        assert list(fr['expressions'].items())==list(fr['native_expressions'].items())
    if expected_expressions is not None:
        q=json.loads(PROTOCOL.read_text())
        assert expected_expressions and len(expected_expressions)==q['expected_features']
        if definition_protocol is None:
            assert all(name in expressions and expression==expressions[name]
                       for name,expression in expected_expressions.items())
        else:
            # Both arms were declared before preparation; a renamed memory arm
            # need not contain the control definitions in its report dictionary.
            assert sha(definition_protocol)==q['master_protocol_sha256']==fr['protocol_sha256']
            arms=json.loads(definition_protocol.read_text())['arms']
            assert list(arms[q['arm']].items())==list(expected_expressions.items())
        expressions=expected_expressions
    return expressions


def scores(expected_expressions=None, encoding_multiplier=1, definition_protocol=None):
    r=load('score_report.json');m=load('model_report.json');v=load('model_verification.json')
    assert encoding_multiplier in [1,10] and json.loads(PROTOCOL.read_text()).get('encoding_multiplier',1)==encoding_multiplier
    assert v['passed'] and v['model_report_sha256']==sha(ROOT/'model_report.json')
    for key,path in [('protocol_sha256',PROTOCOL),('model_report_sha256',ROOT/'model_report.json'),
                     ('feature_report_sha256',FEATURES/'feature_report.json'),('scores_sha256',ROOT/'scores.parquet')]:
        assert r[key]==sha(path)
    fr=json.loads((FEATURES/'feature_report.json').read_text())
    assert fr['features_sha256']==sha(FEATURES/'features.parquet')
    expressions=native_definitions(fr,expected_expressions,definition_protocol)
    assert m['feature_names']==list(expressions)
    c=connection()
    multiplier='' if encoding_multiplier==1 else '10*'
    enc=','.join(f'floor({multiplier}least(greatest(100*{n}+10000+.000001,0),999999))::INT AS X{i:02d}'
                 for i,n in enumerate(m['feature_names'],1))
    c.sql('SELECT date,code,'+enc+' FROM features WHERE formula_input_valid').create_view('encoded')
    score=format(m['bias'],'.17e')+'+'+'+'.join(tree_sql(t) for t in m['trees'])
    c.sql('SELECT date,code,'+score+' AS rebuilt_score FROM encoded').create_view('rebuilt')
    expected=c.sql('''SELECT f.date,f.code,f.half,f.board,f.decision_shares,f.formula_input_valid,
        r.rebuilt_score AS score FROM features f LEFT JOIN rebuilt r USING(date,code) ORDER BY date,code''').df()
    actual=c.sql('SELECT * FROM scores ORDER BY date,code').df()
    pd.testing.assert_frame_equal(actual.drop(columns='score'),expected.drop(columns='score'),check_exact=True,check_dtype=False)
    np.testing.assert_allclose(actual.score,expected.score,rtol=0,atol=2e-11,equal_nan=True)
    assert actual.score.notna().equals(actual.formula_input_valid)
    checks=0
    for t in m['thresholds']:
        np.testing.assert_array_equal(actual.score.gt(t['threshold']),expected.score.gt(t['threshold']))
        checks+=len(actual)
    protocol=json.loads(PROTOCOL.read_text())
    if 'selection_score_cut' in m:
        assert m['selection_score_cut']==protocol['selection_score_cut']
        np.testing.assert_array_equal(actual.score.gt(m['selection_score_cut']),
                                      expected.score.gt(m['selection_score_cut']))
        checks+=len(actual)
    start,end=m.get('training_start'),m.get('training_end','2025-01-01')
    assert start==protocol.get('training_start') and end==protocol.get('training_end','2025-01-01')
    from datetime import date
    assert date.fromisoformat(end).isoformat()==end and (start is None or date.fromisoformat(start).isoformat()==start)
    where=f"next_date<'{end}'"+(f" AND date>='{start}'" if start else '')
    split=m.get('training_split')
    if split is not None:
        assert split==protocol['training_split'] and date.fromisoformat(split).isoformat()==split
        assert start is not None and start<split<end
        # The constrained-half experiment purges labels not yet observable
        # at the intermediate boundary in both its ordinary and constrained arm.
        where+=f" AND (date>='{split}' OR next_date<'{split}')"
    reference=m.get('training_reference_required')
    if reference is not None:
        assert reference==protocol['training_reference_required']=='mark_0959_return15'
        where+=' AND isfinite(mark_0959_return15)'
    c.execute(f'''CREATE VIEW training_keys AS SELECT date,code FROM read_parquet('{SOURCE}/full_labels.parquet')
        WHERE {where} AND known15''')
    train=c.sql('SELECT rebuilt_score FROM rebuilt JOIN training_keys USING(date,code) ORDER BY date,code').df()
    if split is not None:
        assert len(train)==m['rows']==protocol['expected_rows']
    if reference is not None:
        assert len(train)==m['rows']==protocol['expected_training_rows']
    for t in m['thresholds']:
        np.testing.assert_allclose(np.quantile(train.rebuilt_score,t['training_quantile']),t['threshold'],rtol=0,atol=2e-11)
    result=dict(passed=True,score_report_sha256=sha(ROOT/'score_report.json'),rows=len(actual),
        valid=int(actual.formula_input_valid.sum()),max_score_difference=float((actual.score-expected.score).abs().max()),
        threshold_flag_checks=checks,all_integer_encodings_tree_scores_and_training_quantiles_rebuilt=True,
        training_start=start,training_end=end,new_2025_score_groups_read=m.get('new_2025_score_groups_read',False),
        new_2025H2_score_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
    if encoding_multiplier!=1:
        result['encoding_multiplier']=encoding_multiplier
    if 'selection_score_cut' in m:
        result['fixed_selection_score_cut_verified']=m['selection_score_cut']
    if split is not None:
        result.update(training_split=split,purged_training_rows=len(train),intermediate_observation_boundary_verified=True)
    if reference is not None:
        result.update(training_reference_required=reference,reference_finite_training_rows=len(train),
                      inference_pool_unchanged=True)
    save_json(ROOT/'score_verification.json',result)
    return result


def verify_scores(expected_expressions=None, encoding_multiplier=1, definition_protocol=None):
    global ROOT, FEATURES, SOURCE, PROTOCOL
    from trade_research import tail_formula_additive as base
    ROOT, FEATURES, SOURCE, PROTOCOL = base.ROOT, base.FEATURES, base.SOURCE, base.PROTOCOL
    return scores(expected_expressions, encoding_multiplier, definition_protocol)
