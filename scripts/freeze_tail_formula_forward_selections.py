"""Score and independently freeze both 2026 Q1 arms before observing outcomes."""
import argparse
import json

import numpy as np
import pandas as pd

from verify_tail_formula_additive import tree_sql
from trade_research import tail_formula_additive as base
from trade_research import tail_formula_float as original
from trade_research.corporate_cash import save_json, sha
from trade_research.tail_formula_forward import ROOT, MODEL, PROTOCOL, checked_model, policy
from trade_research.tail_formula_forward_inputs import OUT


def inputs():
    freeze = checked_model()
    proof = json.loads((OUT/'feature_verification.json').read_text())
    r = json.loads((OUT/'feature_report.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256']==sha(OUT/'feature_report.json')
    assert r['features_sha256']==sha(OUT/'features.parquet') and r['protocol_sha256']==sha(PROTOCOL)
    m = json.loads((MODEL/'model_report.json').read_text())
    assert m['feature_names']==list(original.EXPRESSIONS)
    return freeze,m,pd.read_parquet(OUT/'features.parquet')


def scores():
    if (ROOT/'score_report.json').exists():
        raise ValueError('Do not replace frozen forward scores')
    freeze,m,f = inputs(); base.EXPRESSIONS = original.EXPRESSIONS
    out = f[['date','code','half','board','decision_shares','formula_input_valid']].copy()
    out['score'] = np.nan
    valid = f.formula_input_valid
    out.loc[valid,'score'] = base.predict(base.encode(f.loc[valid]),m)
    out.to_parquet(ROOT/'scores.parquet',index=False,compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL),model_freeze_report_sha256=sha(ROOT/'model_freeze_report.json'),
        feature_report_sha256=sha(OUT/'feature_report.json'),scores_sha256=sha(ROOT/'scores.parquet'),
        rows=len(out),valid=int(valid.sum()),new_2026_prices_read=True,
        no_2026_after_q1_signal_prices_read=True,next_morning_stock_outcomes_read=False,no_exit_rules=True)
    save_json(ROOT/'score_report.json',r)
    return r


def verify_scores():
    freeze,m,f = inputs()
    r = json.loads((ROOT/'score_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['scores_sha256']==sha(ROOT/'scores.parquet')
    assert r['feature_report_sha256']==sha(OUT/'feature_report.json')
    assert r['model_freeze_report_sha256']==sha(ROOT/'model_freeze_report.json')
    c = base.conn(); c.register('f',f)
    encode = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS X{i:02d}' for i,n in enumerate(m['feature_names'],1))
    c.sql('SELECT date,code,'+encode+' FROM f WHERE formula_input_valid').create_view('encoded')
    formula = format(m['bias'],'.17e')+'+'+'+'.join(tree_sql(t) for t in m['trees'])
    c.sql('SELECT date,code,'+formula+' AS score FROM encoded').create_view('rebuilt')
    expected = c.sql('''SELECT f.date,f.code,f.half,f.board,f.decision_shares,f.formula_input_valid,r.score
        FROM f LEFT JOIN rebuilt r USING(date,code) ORDER BY date,code''').df(); c.close()
    actual = pd.read_parquet(ROOT/'scores.parquet')
    pd.testing.assert_frame_equal(actual.drop(columns='score'),expected.drop(columns='score'),check_exact=True)
    np.testing.assert_allclose(actual.score,expected.score,rtol=0,atol=2e-11,equal_nan=True)
    threshold = freeze['chosen_threshold']['threshold']
    np.testing.assert_array_equal(actual.score.gt(threshold),expected.score.gt(threshold))
    valid = actual.formula_input_valid
    np.testing.assert_array_equal(np.floor(actual.loc[valid,'score']*1e6+.5),np.floor(expected.loc[valid,'score']*1e6+.5))
    r = dict(passed=True,score_report_sha256=sha(ROOT/'score_report.json'),rows=len(actual),
        all_integer_encodings_and_tree_scores_independently_rebuilt=True,
        all_threshold_and_six_decimal_rank_values_match=True,new_2026_prices_read=True,
        next_morning_stock_outcomes_read=False,no_exit_rules=True)
    save_json(ROOT/'score_verification.json',r)
    return r


def freeze():
    policy(); model = checked_model()
    assert not (ROOT/'labels/label_report.json').exists()
    assert not (ROOT/'selection_report.json').exists()
    proof = json.loads((ROOT/'score_verification.json').read_text())
    assert proof['passed'] and proof['score_report_sha256']==sha(ROOT/'score_report.json')
    r = json.loads((ROOT/'score_report.json').read_text())
    assert r['scores_sha256']==sha(ROOT/'scores.parquet')
    f = pd.read_parquet(ROOT/'scores.parquet')
    selected = f.formula_input_valid & f.score.gt(model['chosen_threshold']['threshold'])
    pool = f.loc[selected,['date','code','score']].copy()
    pool['score_integer'] = np.floor(pool.score*1e6+.5).astype('int64')
    pool['rank'] = pool.groupby('date').score_integer.rank(method='min',ascending=False)
    pool.to_parquet(ROOT/'ranking.parquet',index=False,compression='zstd')
    reports = {}
    for arm in ['all','shortlist']:
        folder = ROOT/arm; folder.mkdir(exist_ok=True)
        assert not (folder/'selection_report.json').exists()
        out = f[['date','code','half','board','decision_shares']].copy()
        out['selected'] = selected if arm=='all' else False
        if arm=='shortlist':
            out.loc[pool.index,'selected'] = pool['rank'].le(5)
        out.to_parquet(folder/'selection.parquet',index=False,compression='zstd')
        count = out.loc[out.selected].groupby('date').size()
        report = dict(protocol_sha256=sha(PROTOCOL),score_report_sha256=sha(ROOT/'score_report.json'),
            model_freeze_report_sha256=sha(ROOT/'model_freeze_report.json'),
            selection_sha256=sha(folder/'selection.parquet'),chosen_threshold=model['chosen_threshold'],
            arm=arm,selected=int(out.selected.sum()),days=len(count),max_daily=int(count.max()) if len(count) else 0,
            by_month={month:int((out.selected & out.date.str.startswith(month)).sum()) for month in ['2026-01','2026-02','2026-03']},
            boundary_ties_retained=arm=='shortlist',new_2026_prices_read=True,
            no_2026_after_q1_signal_prices_read=True,next_morning_stock_outcomes_read=False,
            strict_blind=False,native_source_parity_verified=False,no_exit_rules=True)
        save_json(folder/'selection_report.json',report); reports[arm]=sha(folder/'selection_report.json')
    r = dict(protocol_sha256=sha(PROTOCOL),score_report_sha256=sha(ROOT/'score_report.json'),
        arm_selection_report_sha256=reports,ranking_sha256=sha(ROOT/'ranking.parquet'),
        both_arms_frozen_before_outcomes=True,new_2026_prices_read=True,next_morning_stock_outcomes_read=False,
        strict_blind=False,no_exit_rules=True)
    save_json(ROOT/'selection_report.json',r)
    return r


def verify():
    p = policy(); model = checked_model()
    r = json.loads((ROOT/'selection_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['ranking_sha256']==sha(ROOT/'ranking.parquet')
    assert r['score_report_sha256']==sha(ROOT/'score_report.json')
    sp = json.loads((ROOT/'score_verification.json').read_text())
    assert sp['passed'] and sp['score_report_sha256']==r['score_report_sha256']
    c = base.conn(); c.read_parquet(str(ROOT/'scores.parquet')).create_view('scores')
    cut = format(model['chosen_threshold']['threshold'],'.17e')
    c.execute('CREATE VIEW ranked AS SELECT date,code,score,floor(score*1e6+.5)::BIGINT AS score_integer,'+
        'rank() OVER(PARTITION BY date ORDER BY floor(score*1e6+.5) DESC)::DOUBLE AS rank FROM scores WHERE formula_input_valid AND score>'+cut)
    expected = c.sql('SELECT * FROM ranked ORDER BY date,code').df()
    pd.testing.assert_frame_equal(pd.read_parquet(ROOT/'ranking.parquet'),expected,check_exact=True)
    rows = {}
    for arm in ['all','shortlist']:
        folder = ROOT/arm; a = json.loads((folder/'selection_report.json').read_text())
        assert r['arm_selection_report_sha256'][arm]==sha(folder/'selection_report.json')
        assert a['selection_sha256']==sha(folder/'selection.parquet') and a['chosen_threshold']==model['chosen_threshold']
        condition = 'r.rank IS NOT NULL' if arm=='all' else 'coalesce(r.rank<=5,false)'
        expected = c.sql('SELECT s.date,s.code,s.half,s.board,s.decision_shares,'+condition+
            ' AS selected FROM scores s LEFT JOIN ranked r USING(date,code) ORDER BY date,code').df()
        pd.testing.assert_frame_equal(pd.read_parquet(folder/'selection.parquet'),expected,check_exact=True)
        assert expected.date.between(p['signal_first'],p['signal_last']).all()
        assert int(expected.selected.sum())==a['selected']
        proof = dict(passed=True,selection_report_sha256=sha(folder/'selection_report.json'),rows=len(expected),
            all_selection_and_rank_flags_rebuilt=True,new_2026_prices_read=True,next_morning_stock_outcomes_read=False,no_exit_rules=True)
        save_json(folder/'selection_verification.json',proof); rows[arm]=a['selected']
    c.close()
    proof = dict(passed=True,selection_report_sha256=sha(ROOT/'selection_report.json'),selected=rows,
        both_arms_independently_verified=True,new_2026_prices_read=True,next_morning_stock_outcomes_read=False,no_exit_rules=True)
    save_json(ROOT/'selection_verification.json',proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['scores','verify_scores','freeze','verify'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
