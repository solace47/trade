"""Freeze the Q1 intersection and same-quality original before subset outcomes."""
import argparse
import json

import numpy as np
import pandas as pd

from verify_tail_formula_additive import tree_sql
from trade_research import tail_formula_size_agreement_q1 as study
from trade_research.corporate_cash import save_json, sha
from trade_research.tail_formula_forward import checked_model as original_model


def inputs():
    r = study.checked_model(); original_model()
    proof = json.loads((study.OUT/'feature_verification.json').read_text())
    f = json.loads((study.OUT/'feature_report.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256']==sha(study.OUT/'feature_report.json')
    assert f['features_sha256']==sha(study.OUT/'features.parquet')
    m = json.loads((study.MODEL/'model_report.json').read_text())
    assert m['feature_names']==list(study.extended.EXPRESSIONS) and r['score_cuts'][1]==m['thresholds'][3]['threshold']
    return r,m,pd.read_parquet(study.OUT/'features.parquet')


def scores():
    freeze,m,f = inputs(); root=study.ROOT
    assert not (root/'score_report.json').exists()
    study.base.EXPRESSIONS = study.extended.EXPRESSIONS
    out=f[['date','code','half','board','decision_shares','formula_input_valid']].copy(); out['score']=np.nan
    valid=f.formula_input_valid
    out.loc[valid,'score']=study.base.predict(study.base.encode(f.loc[valid]),m)
    out.to_parquet(root/'scores.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(study.PROTOCOL),model_freeze_report_sha256=sha(root/'model_freeze_report.json'),
        feature_report_sha256=sha(study.OUT/'feature_report.json'),scores_sha256=sha(root/'scores.parquet'),
        rows=len(out),valid=int(valid.sum()),new_2026_stock_prices_read=False,new_intersection_outcomes_read=False,no_exit_rules=True)
    save_json(root/'score_report.json',r); return r


def verify_scores():
    freeze,m,f=inputs(); root=study.ROOT; r=json.loads((root/'score_report.json').read_text())
    assert r['scores_sha256']==sha(root/'scores.parquet') and r['feature_report_sha256']==sha(study.OUT/'feature_report.json')
    assert r['model_freeze_report_sha256']==sha(root/'model_freeze_report.json') and r['protocol_sha256']==sha(study.PROTOCOL)
    c=study.base.conn();c.register('f',f)
    enc=','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS X{i:02d}' for i,n in enumerate(m['feature_names'],1))
    c.sql('SELECT date,code,'+enc+' FROM f WHERE formula_input_valid').create_view('x')
    expression=format(m['bias'],'.17e')+'+'+'+'.join(tree_sql(t) for t in m['trees'])
    c.sql('SELECT date,code,'+expression+' AS score FROM x').create_view('sc')
    expected=c.sql('SELECT f.date,f.code,f.half,f.board,f.decision_shares,f.formula_input_valid,sc.score FROM f LEFT JOIN sc USING(date,code) ORDER BY date,code').df(); c.close()
    actual=pd.read_parquet(root/'scores.parquet')
    pd.testing.assert_frame_equal(actual.drop(columns='score'),expected.drop(columns='score'),check_exact=True)
    np.testing.assert_allclose(actual.score,expected.score,rtol=0,atol=2e-11,equal_nan=True)
    np.testing.assert_array_equal(actual.score.gt(freeze['score_cuts'][1]),expected.score.gt(freeze['score_cuts'][1]))
    result=dict(passed=True,score_report_sha256=sha(root/'score_report.json'),rows=len(actual),
        all_50_input_encodings_and_64_tree_scores_rebuilt=True,all_threshold_flags_equal=True,
        new_2026_stock_prices_read=False,new_intersection_outcomes_read=False,no_exit_rules=True)
    save_json(root/'score_verification.json',result); return result


def freeze():
    inputs(); root=study.ROOT; r=study.checked_model()
    assert not any((root/arm/'analysis_report.json').exists() for arm in ['all','control'])
    proof=json.loads((root/'score_verification.json').read_text())
    assert proof['passed'] and proof['score_report_sha256']==sha(root/'score_report.json')
    oldr=json.loads((study.OLD/'all/selection_report.json').read_text())
    assert oldr['selection_sha256']==sha(study.OLD/'all/selection.parquet')
    old=pd.read_parquet(study.OLD/'all/selection.parquet'); s=pd.read_parquet(root/'scores.parquet')
    pd.testing.assert_frame_equal(old.drop(columns='selected'),s[old.columns.drop('selected')],check_exact=True)
    reports={}
    for arm in ['all','control']:
        folder=root/arm; folder.mkdir(exist_ok=True); assert not (folder/'selection_report.json').exists()
        chosen=old.copy(); chosen['selected'] &= s.formula_input_valid
        if arm=='all':
            chosen['selected'] &= s.score.gt(r['score_cuts'][1])
        assert (~chosen.selected | old.selected).all()
        chosen.to_parquet(folder/'selection.parquet',index=False,compression='zstd')
        counts=chosen.loc[chosen.selected].groupby('date').size()
        report=dict(protocol_sha256=sha(study.PROTOCOL),score_report_sha256=sha(root/'score_report.json'),
            model_freeze_report_sha256=sha(root/'model_freeze_report.json'),original_selection_report_sha256=sha(study.OLD/'all/selection_report.json'),
            selection_sha256=sha(folder/'selection.parquet'),arm=arm,selected=int(chosen.selected.sum()),days=len(counts),
            by_month={month:int((chosen.selected & chosen.date.str.startswith(month)).sum()) for month in ['2026-01','2026-02','2026-03']},
            original_selected=oldr['selected'],unchanged_original_selection=chosen.equals(old),strict_blind=False,
            prior_q1_original_outcomes_exposed=True,new_intersection_outcomes_read=False,new_2026_stock_prices_read=False,no_exit_rules=True)
        save_json(folder/'selection_report.json',report); reports[arm]=report
    return reports


def verify():
    inputs();root=study.ROOT;r=study.checked_model();c=study.base.conn()
    sp=json.loads((root/'score_verification.json').read_text())
    assert sp['passed'] and sp['score_report_sha256']==sha(root/'score_report.json')
    cut=r['score_cuts'][1]; reports={}
    for arm in ['all','control']:
        folder=root/arm; report=json.loads((folder/'selection_report.json').read_text())
        assert report['protocol_sha256']==sha(study.PROTOCOL) and report['score_report_sha256']==sha(root/'score_report.json')
        assert report['selection_sha256']==sha(folder/'selection.parquet')
        condition=f's.score>{cut:.17e}' if arm=='all' else 'true'
        expected=c.sql(f'''SELECT o.date,o.code,o.half,o.board,o.decision_shares,o.selected AND s.formula_input_valid AND {condition} AS selected
            FROM read_parquet('{study.OLD}/all/selection.parquet') o JOIN read_parquet('{root}/scores.parquet') s USING(date,code) ORDER BY o.date,o.code''').df()
        pd.testing.assert_frame_equal(expected,pd.read_parquet(folder/'selection.parquet'),check_exact=True)
        assert report['selected']==int(expected.selected.sum()) and report['days']==expected.loc[expected.selected,'date'].nunique()
        result=dict(passed=True,selection_report_sha256=sha(folder/'selection_report.json'),rows=len(expected),
            all_selection_flags_independently_rebuilt=True,strict_blind=False,new_intersection_outcomes_read=False,new_2026_stock_prices_read=False,no_exit_rules=True)
        save_json(folder/'selection_verification.json',result); reports[arm]=result
    c.close();return reports


def analyze():
    from trade_research.tail_formula_1000_analysis import analyze as analysis
    from verify_tail_formula_1000_analysis import analysis_check
    root=study.ROOT;study.checked_model()
    for arm in ['all','control']:
        proof=json.loads((root/arm/'selection_verification.json').read_text())
        assert proof['passed'] and proof['selection_report_sha256']==sha(root/arm/'selection_report.json')
    folder=root/'all'
    for name in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        target=folder/name
        if not target.exists():
            target.symlink_to((study.OLD/'all'/name).resolve())
        assert sha(target)==sha(study.OLD/'all'/name)
    c=study.base.conn()
    missing=c.sql(f'''SELECT count(*) FROM read_parquet('{folder}/selection.parquet') s
        ANTI JOIN read_parquet('{folder}/full_labels.parquet') l USING(date,code) WHERE selected''').fetchone()[0]
    c.close();assert missing==0, 'Never silently omit a selected result'
    report=analysis(folder,study.PROTOCOL); proof=analysis_check(folder)
    return dict(analysis=report,verification=proof)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['scores','verify_scores','freeze','verify','analyze'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
