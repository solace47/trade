"""Date-equal training-score quantile for unchanged original 48-input models."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_float as inputs
from .corporate_cash import save_json, sha

STEM = 'tail_formula_date_quantile'
PROTOCOL = Path('config') / (STEM + '_protocol.json')


def root(fold):
    return Path('data/research') / (STEM + '_' + fold)


def config():
    p = json.loads(PROTOCOL.read_text())
    assert p['training_quantile'] == .995 and not p['model_refitted'] and not p['new_2026_prices_allowed']
    for file,digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    return p


def source(fold):
    p = config()['fold_sources'][fold]
    path = Path(p['root'])
    for file,digest in p['source_hashes'].items():
        assert sha(path / file) == digest
    for stage in ['model','score','selection']:
        v = json.loads((path / (stage+'_verification.json')).read_text())
        assert v['passed'] and v[stage+'_report_sha256'] == sha(path / (stage+'_report.json'))
    m = json.loads((path / 'model_report.json').read_text())
    r = json.loads((path / 'score_report.json').read_text())
    assert r['scores_sha256'] == sha(path / 'scores.parquet')
    assert r['model_report_sha256'] == sha(path / 'model_report.json')
    assert m['feature_names'] == list(inputs.EXPRESSIONS) and len(m['trees']) == 64
    assert (m['training_start'],m['training_end']) == (p['training_start'],p['training_end'])
    assert m['last_observation'] < p['training_end'] == p['evaluation_start']
    assert p['evaluation_end'] <= '2026-01-01'
    return p,path,m


def date_quantile(frame,q):
    assert 0 < q <= 1 and len(frame) and not frame[['date','code']].duplicated().any()
    assert np.isfinite(frame.score).all()
    f = frame.copy()
    f['weight'] = 1/f.groupby('date').code.transform('size')
    curve = f.groupby('score',sort=True).weight.sum().reset_index()
    curve['cumulative_weight'] = curve.weight.cumsum()
    total = frame.date.nunique()
    np.testing.assert_allclose(curve.weight.sum(),total,rtol=0,atol=2e-10)
    indices = np.flatnonzero(curve.cumulative_weight.to_numpy() >= q*total)
    # q=1 can be microscopically above the sum after floating accumulation.
    index = int(indices[0]) if len(indices) else len(curve)-1
    assert len(indices) or q == 1
    return float(curve.score.iloc[index]),curve


def calibrate(fold):
    p,prior,m = source(fold)
    out = root(fold)
    assert not (out / 'threshold_report.json').exists()
    c = base.conn()
    known = c.sql(f"SELECT date,code,next_date FROM read_parquet('{labels.ROOT}/full_labels.parquet') " +
        f"WHERE known15 AND date>='{p['training_start']}' AND next_date<'{p['training_end']}'").df()
    c.close()
    scores = pd.read_parquet(prior / 'scores.parquet')
    t = scores.loc[scores.formula_input_valid,['date','code','score']].merge(known,on=['date','code'],validate='one_to_one')
    assert len(t) == m['rows'] and t.date.nunique() == m['days']
    cut,curve = date_quantile(t,.995)
    out.mkdir(parents=True,exist_ok=True)
    curve.to_parquet(out / 'weighted_curve.parquet',index=False,compression='zstd')
    (out / 'frozen_numeric_core.tdx').write_text(base.native_core(m,cut,inputs.EXPRESSIONS,inputs.HEADER))
    r = dict(protocol_sha256=sha(PROTOCOL),source_model_report_sha256=sha(prior / 'model_report.json'),
        source_score_report_sha256=sha(prior / 'score_report.json'),label_report_sha256=sha(labels.ROOT / 'full_label_report.json'),
        weighted_curve_sha256=sha(out / 'weighted_curve.parquet'),core_sha256=sha(out / 'frozen_numeric_core.tdx'),
        rows=len(t),days=t.date.nunique(),last_observation=t.next_date.max(),threshold=cut,
        original_threshold=m['thresholds'][3]['threshold'],training_quantile=.995,
        score_groups=len(curve),model_refitted=False,original_scores_unchanged=True,
        software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(out / 'threshold_report.json',r)
    return r


def verify_threshold(fold):
    p,prior,m = source(fold)
    out = root(fold)
    r = json.loads((out / 'threshold_report.json').read_text())
    for key,path in [('protocol',PROTOCOL),('source_model_report',prior / 'model_report.json'),
        ('source_score_report',prior / 'score_report.json'),('label_report',labels.ROOT / 'full_label_report.json'),
        ('weighted_curve',out / 'weighted_curve.parquet'),('core',out / 'frozen_numeric_core.tdx')]:
        assert r[key+'_sha256'] == sha(path)
    c = base.conn()
    c.execute(f'''CREATE VIEW training AS SELECT s.date,s.code,s.score,l.next_date,
        1./count(*) OVER(PARTITION BY s.date) AS weight FROM read_parquet('{prior}/scores.parquet') s
        JOIN read_parquet('{labels.ROOT}/full_labels.parquet') l USING(date,code)
        WHERE s.formula_input_valid AND l.known15 AND s.date>='{p['training_start']}' AND l.next_date<'{p['training_end']}' ''')
    c.execute('CREATE VIEW grouped AS SELECT score,sum(weight) AS weight FROM training GROUP BY score')
    curve = c.sql('SELECT *,sum(weight) OVER(ORDER BY score ROWS UNBOUNDED PRECEDING) AS cumulative_weight FROM grouped ORDER BY score').df()
    actual = pd.read_parquet(out / 'weighted_curve.parquet')
    np.testing.assert_array_equal(actual.score,curve.score)
    np.testing.assert_allclose(actual[['weight','cumulative_weight']],curve[['weight','cumulative_weight']],rtol=0,atol=2e-10)
    rows,days,last = c.sql('SELECT count(*),count(DISTINCT date),max(next_date) FROM training').fetchone()
    assert (rows,days,last) == (r['rows'],r['days'],r['last_observation']) and rows == m['rows'] and days == m['days']
    c.register('curve',curve)
    cut = c.sql('SELECT min(score) FROM curve WHERE cumulative_weight>=.995*'+str(days)).fetchone()[0]
    assert cut == r['threshold']
    boundary = curve.loc[curve.score.eq(cut)].iloc[0]
    previous = curve.loc[curve.score.lt(cut),'cumulative_weight']
    below = float(previous.iloc[-1]) if len(previous) else 0.
    assert below < .995*days <= boundary.cumulative_weight
    # The nearest CDF boundary is separated from summation tolerance.
    assert min(.995*days-below,boundary.cumulative_weight-.995*days) > 2e-10
    weights = c.sql('SELECT date,sum(weight) AS weight FROM training GROUP BY date').df()
    np.testing.assert_allclose(weights.weight,1,rtol=0,atol=2e-12)
    c.close()
    assert (out / 'frozen_numeric_core.tdx').read_text() == base.native_core(m,cut,inputs.EXPRESSIONS,inputs.HEADER)
    assert r['original_threshold'] == m['thresholds'][3]['threshold'] and r['score_groups'] == len(curve)
    proof = dict(passed=True,threshold_report_sha256=sha(out / 'threshold_report.json'),rows=rows,days=days,
        all_original_training_keys_scores_equal_day_weights_grouped_cdf_and_left_inverse_independently_rebuilt=True,
        previous_cumulative_weight=below,chosen_cumulative_weight=float(boundary.cumulative_weight),target_weight=.995*days,
        original_model_reused_native_constant_only_changed=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(out / 'threshold_verification.json',proof)
    return proof


def freeze(fold):
    config()
    out = root(fold)
    assert not (out / 'selection_report.json').exists()
    assert not any((root(f) / 'analysis_report.json').exists() for f in ['2024','recent','2025'])
    out.mkdir(parents=True,exist_ok=True)
    if fold == '2025':
        a,b = [pd.read_parquet(root(f) / 'selection.parquet') for f in ['2024','recent']]
        pd.testing.assert_frame_equal(a.drop(columns='selected'),b.drop(columns='selected'),check_exact=True)
        assert not (a.selected & b.selected).any()
        selection = a.copy();selection['selected'] |= b.selected
        details = dict(fold_selection_report_sha256={f:sha(root(f) / 'selection_report.json') for f in ['2024','recent']})
        for f,h in details['fold_selection_report_sha256'].items():
            v = json.loads((root(f) / 'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == h
    else:
        p,prior,_ = source(fold)
        v = json.loads((out / 'threshold_verification.json').read_text())
        assert v['passed'] and v['threshold_report_sha256'] == sha(out / 'threshold_report.json')
        r = json.loads((out / 'threshold_report.json').read_text())
        f = pd.read_parquet(prior / 'scores.parquet')
        selection = f[['date','code','half','board','decision_shares']].copy()
        selection['selected'] = f.formula_input_valid & f.date.ge(p['evaluation_start']) & f.date.lt(p['evaluation_end']) & f.score.gt(r['threshold'])
        details = dict(threshold_report_sha256=sha(out / 'threshold_report.json'),threshold=r['threshold'])
    selection.to_parquet(out / 'selection.parquet',index=False,compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL),selection_sha256=sha(out / 'selection.parquet'),**details,
        selected=int(selection.selected.sum()),days=selection.loc[selection.selected,'date'].nunique(),
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(out / 'selection_report.json',r)
    return r


def verify(fold):
    config()
    out = root(fold)
    r = json.loads((out / 'selection_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['selection_sha256'] == sha(out / 'selection.parquet')
    c = base.conn()
    if fold == '2025':
        parts = []
        for f,h in r['fold_selection_report_sha256'].items():
            v = json.loads((root(f) / 'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == h == sha(root(f) / 'selection_report.json')
            parts.append(f"SELECT * FROM read_parquet('{root(f)}/selection.parquet')")
        expected = c.sql('SELECT date,code,half,board,decision_shares,bool_or(selected) AS selected FROM ('+
            ' UNION ALL '.join(parts)+') GROUP BY date,code,half,board,decision_shares ORDER BY date,code').df()
    else:
        p,prior,_ = source(fold)
        assert r['threshold_report_sha256'] == sha(out / 'threshold_report.json')
        t = json.loads((out / 'threshold_report.json').read_text());assert t['threshold'] == r['threshold']
        expected = c.sql('SELECT date,code,half,board,decision_shares,formula_input_valid AND '+
            f"date>='{p['evaluation_start']}' AND date<'{p['evaluation_end']}' AND score>"+format(r['threshold'],'.17e')+
            f" AS selected FROM read_parquet('{prior}/scores.parquet') ORDER BY date,code").df()
    c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(out / 'selection.parquet'),expected,check_exact=True)
    assert r['selected'] == int(expected.selected.sum()) and r['days'] == expected.loc[expected.selected,'date'].nunique()
    proof = dict(passed=True,selection_report_sha256=sha(out / 'selection_report.json'),rows=len(expected),
        all_original_scores_and_strict_weighted_cut_flags_rebuilt=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(out / 'selection_verification.json',proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['calibrate','verify_threshold','freeze','verify','analyze'])
    parser.add_argument('--fold',choices=['2024','recent','2025'],required=True)
    args = parser.parse_args()
    if args.stage == 'analyze':
        for f in ['2024','recent','2025']:
            v = json.loads((root(f) / 'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == sha(root(f) / 'selection_report.json')
        result = evaluation.analyze(root(args.fold),PROTOCOL)
    else:
        result = globals()[args.stage](args.fold)
    print(json.dumps(result,ensure_ascii=False,indent=2))
