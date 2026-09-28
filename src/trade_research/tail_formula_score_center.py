"""Subtract each day's visible-pool mean from fixed 48-input model scores."""
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

STEM = 'tail_formula_score_center'
PROTOCOL = Path('config') / (STEM + '_protocol.json')


def root(fold):
    return Path('data/research') / (STEM + '_' + fold)


def config():
    p = json.loads(PROTOCOL.read_text())
    assert p['training_quantile'] == .995 and not p['model_refitted']
    assert not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    return p


def source(fold):
    p = config()['fold_sources'][fold]
    path = Path(p['root'])
    for file, digest in p['source_hashes'].items():
        assert sha(path / file) == digest
    for stage in ['model', 'score', 'selection']:
        v = json.loads((path / (stage + '_verification.json')).read_text())
        assert v['passed'] and v[stage + '_report_sha256'] == sha(path / (stage + '_report.json'))
    m = json.loads((path / 'model_report.json').read_text())
    s = json.loads((path / 'score_report.json').read_text())
    assert s['scores_sha256'] == sha(path / 'scores.parquet')
    assert s['model_report_sha256'] == sha(path / 'model_report.json')
    assert m['feature_names'] == list(inputs.EXPRESSIONS) and len(m['trees']) == 64
    assert (m['training_start'], m['training_end']) == (p['training_start'], p['training_end'])
    assert m['last_observation'] < p['training_end'] == p['evaluation_start']
    assert p['evaluation_end'] <= '2026-01-01'
    return p, path, m


def center(scores):
    """Every visible valid peer participates, independently of outcome availability."""
    f = scores.copy()
    assert f[['date', 'code']].duplicated().sum() == 0
    valid = f.formula_input_valid
    assert np.isfinite(f.loc[valid, 'score']).all() and f.loc[~valid, 'score'].isna().all()
    stats = f.loc[valid].groupby('date', sort=True).agg(
        members=('code', 'size'), mean_score=('score', 'mean')).reset_index()
    f = f.merge(stats, on='date', how='left', validate='many_to_one')
    f['centered_score'] = (f.score - f.mean_score).where(valid)
    return f, stats


def native_structure(model, threshold):
    original = base.native_core(model, 0, inputs.EXPRESSIONS, inputs.HEADER)
    assert original.endswith('CORE:SC>0;\n')
    return ('{仅数值结构，不能当完整条件选股公式使用。YJSC64辅助指标输出1为原硬过滤且48项有效的标记，'
            '输出2为该标记乘本期原SC；完整成员与客户端历史一致性尚未核准。}\n' +
            original.removesuffix('CORE:SC>0;\n') +
            "CSN:=INSUM('沪深Ａ股','YJSC64',1,0);\n" +
            "CSM:=INSUM('沪深Ａ股','YJSC64',2,0)/MAX(CSN,1);\n" +
            'CORE:CSN>0 AND SC-CSM>' + format(threshold, '.17g') + ';\n')


def scores(fold):
    p, prior, model = source(fold)
    out = root(fold)
    assert not (out / 'score_report.json').exists()
    old = pd.read_parquet(prior / 'scores.parquet')
    f, daily = center(old)
    c = base.conn()
    known = c.sql(f"SELECT date,code,next_date FROM read_parquet('{labels.ROOT}/full_labels.parquet') " +
                  f"WHERE date>='{p['training_start']}' AND next_date<'{p['training_end']}' AND known15").df()
    c.close()
    t = f.loc[f.formula_input_valid].merge(known, on=['date', 'code'], validate='one_to_one')
    assert len(t) == model['rows'] and t.date.nunique() == model['days']
    cut = float(np.quantile(t.centered_score, .995))
    out.mkdir(parents=True, exist_ok=True)
    f.to_parquet(out / 'scores.parquet', index=False, compression='zstd')
    daily.to_parquet(out / 'daily_means.parquet', index=False, compression='zstd')
    (out / 'centered_numeric_structure.tdx').write_text(native_structure(model, cut))
    r = dict(protocol_sha256=sha(PROTOCOL), source_score_report_sha256=sha(prior / 'score_report.json'),
        source_model_report_sha256=sha(prior / 'model_report.json'), scores_sha256=sha(out / 'scores.parquet'),
        daily_means_sha256=sha(out / 'daily_means.parquet'), structure_sha256=sha(out / 'centered_numeric_structure.tdx'),
        rows=len(f), valid=int(f.formula_input_valid.sum()), days=len(daily),
        minimum_members=int(daily.members.min()), maximum_members=int(daily.members.max()),
        training_rows=len(t), training_days=t.date.nunique(), threshold=cut, training_quantile=.995,
        label_report_sha256=sha(labels.ROOT / 'full_label_report.json'), model_refitted=False,
        mean_pool_does_not_use_future_outcomes=True, native_complete_formula_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(out / 'score_report.json', r)
    return r


def verify_scores(fold):
    p, prior, model = source(fold)
    out = root(fold)
    r = json.loads((out / 'score_report.json').read_text())
    for key, path in [('protocol', PROTOCOL), ('source_score_report', prior / 'score_report.json'),
        ('source_model_report', prior / 'model_report.json'), ('scores', out / 'scores.parquet'),
        ('daily_means', out / 'daily_means.parquet'), ('structure', out / 'centered_numeric_structure.tdx'),
        ('label_report', labels.ROOT / 'full_label_report.json')]:
        assert r[key + '_sha256'] == sha(path)
    c = base.conn()
    c.execute(f"CREATE VIEW original AS SELECT * FROM read_parquet('{prior}/scores.parquet')")
    c.execute('CREATE VIEW means AS SELECT date,count(*) AS members,sum(score)/count(*) AS mean_score '
              'FROM original WHERE formula_input_valid GROUP BY date')
    expected = c.sql('SELECT original.*,members,mean_score,CASE WHEN formula_input_valid '
                     'THEN score-mean_score END AS centered_score FROM original LEFT JOIN means USING(date) ORDER BY date,code').df()
    actual = pd.read_parquet(out / 'scores.parquet')
    original_columns = list(pd.read_parquet(prior / 'scores.parquet').columns)
    pd.testing.assert_frame_equal(actual[original_columns], expected[original_columns], check_exact=True)
    np.testing.assert_array_equal(actual.members, expected.members)
    for name in ['mean_score', 'centered_score']:
        np.testing.assert_allclose(actual[name], expected[name], rtol=0, atol=2e-12, equal_nan=True)
    daily = c.sql('SELECT * FROM means ORDER BY date').df()
    pd.testing.assert_frame_equal(pd.read_parquet(out / 'daily_means.parquet'), daily,
                                  check_dtype=False, rtol=0, atol=2e-12)
    c.register('centered', expected)
    train = c.sql(f"SELECT s.date,s.code,s.centered_score,l.next_date FROM centered s JOIN " +
        f"read_parquet('{labels.ROOT}/full_labels.parquet') l USING(date,code) " +
        f"WHERE s.formula_input_valid AND l.known15 AND s.date>='{p['training_start']}' " +
        f"AND l.next_date<'{p['training_end']}' ORDER BY s.date,s.code").df()
    c.register('training', train)
    cut = c.sql('SELECT quantile_cont(centered_score,.995) FROM training').fetchone()[0]
    c.close()
    np.testing.assert_allclose(cut, r['threshold'], rtol=0, atol=2e-12)
    assert len(train) == r['training_rows'] == model['rows'] and train.date.nunique() == r['training_days']
    # Check actual deployment decisions too: a numerically close cut may still flip ties.
    evaluation_mask = actual.date.ge(p['evaluation_start']) & actual.date.lt(p['evaluation_end'])
    np.testing.assert_array_equal(actual.loc[evaluation_mask, 'centered_score'].gt(r['threshold']),
                                  expected.loc[evaluation_mask, 'centered_score'].gt(cut))
    residual = actual.loc[actual.formula_input_valid].groupby('date').centered_score.mean()
    assert residual.abs().max() < 2e-12
    assert (out / 'centered_numeric_structure.tdx').read_text() == native_structure(model, r['threshold'])
    assert len(actual) == r['rows'] and len(daily) == r['days']
    assert int(actual.formula_input_valid.sum()) == r['valid']
    assert int(daily.members.min()) == r['minimum_members'] and int(daily.members.max()) == r['maximum_members']
    proof = dict(passed=True, score_report_sha256=sha(out / 'score_report.json'), rows=len(actual),
        all_visible_pool_means_centered_scores_training_quantiles_and_strict_flags_independently_rebuilt=True,
        original_score_values_and_validity_unchanged=True, max_daily_centered_mean=float(residual.abs().max()),
        numeric_structure_arithmetic_bound_to_original_model=True, native_complete_formula_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(out / 'score_verification.json', proof)
    return proof


def freeze(fold):
    config()
    out = root(fold)
    assert not (out / 'selection_report.json').exists()
    assert not any((root(f) / 'analysis_report.json').exists() for f in ['2024', 'recent', '2025'])
    out.mkdir(parents=True, exist_ok=True)
    if fold == '2025':
        a, b = [pd.read_parquet(root(f) / 'selection.parquet') for f in ['2024', 'recent']]
        pd.testing.assert_frame_equal(a.drop(columns='selected'), b.drop(columns='selected'), check_exact=True)
        assert not (a.selected & b.selected).any()
        selection = a.copy()
        selection['selected'] |= b.selected
        details = dict(fold_selection_report_sha256={f: sha(root(f) / 'selection_report.json') for f in ['2024', 'recent']})
        for f, digest in details['fold_selection_report_sha256'].items():
            v = json.loads((root(f) / 'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == digest
    else:
        p, _, _ = source(fold)
        r = json.loads((out / 'score_report.json').read_text())
        v = json.loads((out / 'score_verification.json').read_text())
        assert v['passed'] and v['score_report_sha256'] == sha(out / 'score_report.json')
        assert r['scores_sha256'] == sha(out / 'scores.parquet')
        f = pd.read_parquet(out / 'scores.parquet')
        selection = f[['date', 'code', 'half', 'board', 'decision_shares']].copy()
        selection['selected'] = (f.formula_input_valid & f.date.ge(p['evaluation_start']) &
                                 f.date.lt(p['evaluation_end']) & f.centered_score.gt(r['threshold']))
        details = dict(score_report_sha256=sha(out / 'score_report.json'), threshold=r['threshold'])
    selection.to_parquet(out / 'selection.parquet', index=False, compression='zstd')
    counts = selection.loc[selection.selected].groupby('date').size()
    r = dict(protocol_sha256=sha(PROTOCOL), selection_sha256=sha(out / 'selection.parquet'), **details,
        selected=int(counts.sum()), days=len(counts), model_refitted=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(out / 'selection_report.json', r)
    return r


def verify(fold):
    config()
    out = root(fold)
    r = json.loads((out / 'selection_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['selection_sha256'] == sha(out / 'selection.parquet')
    c = base.conn()
    if fold == '2025':
        parts = []
        for f in ['2024', 'recent']:
            v = json.loads((root(f) / 'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == r['fold_selection_report_sha256'][f] == sha(root(f) / 'selection_report.json')
            parts.append(f"SELECT * FROM read_parquet('{root(f)}/selection.parquet')")
        expected = c.sql('SELECT date,code,half,board,decision_shares,bool_or(selected) AS selected FROM (' +
            ' UNION ALL '.join(parts) + ') GROUP BY date,code,half,board,decision_shares ORDER BY date,code').df()
    else:
        p, _, _ = source(fold)
        assert r['score_report_sha256'] == sha(out / 'score_report.json')
        sr = json.loads((out / 'score_report.json').read_text())
        assert sr['threshold'] == r['threshold']
        expected = c.sql('SELECT date,code,half,board,decision_shares,formula_input_valid AND '
            f"date>='{p['evaluation_start']}' AND date<'{p['evaluation_end']}' AND " +
            'centered_score>' + format(r['threshold'], '.17e') +
            f" AS selected FROM read_parquet('{out}/scores.parquet') ORDER BY date,code").df()
    c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(out / 'selection.parquet'), expected, check_exact=True)
    assert int(expected.selected.sum()) == r['selected'] and expected.loc[expected.selected, 'date'].nunique() == r['days']
    proof = dict(passed=True, selection_report_sha256=sha(out / 'selection_report.json'), rows=len(expected),
        all_selection_flags_independently_rebuilt=True, new_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(out / 'selection_verification.json', proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    parser.add_argument('--fold', choices=['2024', 'recent', '2025'], required=True)
    args = parser.parse_args()
    if args.stage == 'analyze':
        for f in ['2024', 'recent', '2025']:
            v = json.loads((root(f) / 'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == sha(root(f) / 'selection_report.json')
        result = evaluation.analyze(root(args.fold), PROTOCOL)
    else:
        result = globals()[args.stage](args.fold)
    print(json.dumps(result, ensure_ascii=False, indent=2))
