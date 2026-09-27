"""A fixed top-five-with-ties shortlist within the original 48-input selections."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as inputs
from .corporate_cash import save_json, sha
from .tail_formula_1000_daily import analyze as common_analysis

PROTOCOL = Path('config/tail_formula_shortlist_protocol.json')


def root(fold):
    return Path('data/research/tail_formula_shortlist_' + fold)


def config():
    p = json.loads(PROTOCOL.read_text())
    assert p['top_positions'] == 5 and p['score_integer_scale'] == 1000000
    assert not p['model_refitted'] and p['do_not_fill_below_original_cut']
    assert p['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
    return p


def source(fold):
    cfg = config()['folds'][fold]
    prior = Path(cfg['source_root'])
    for name, digest in cfg['source_sha256'].items():
        assert sha(prior / name) == digest
    for kind in ['model', 'score', 'selection']:
        proof = json.loads((prior / f'{kind}_verification.json').read_text())
        assert proof['passed'] and proof[f'{kind}_report_sha256'] == sha(prior / f'{kind}_report.json')
    score = json.loads((prior / 'score_report.json').read_text())
    selected = json.loads((prior / 'selection_report.json').read_text())
    model = json.loads((prior / 'model_report.json').read_text())
    assert score['scores_sha256'] == sha(prior / 'scores.parquet')
    assert score['model_report_sha256'] == selected['model_report_sha256'] == sha(prior / 'model_report.json')
    assert selected['selection_sha256'] == sha(prior / 'selection.parquet')
    assert selected['core_sha256'] == sha(prior / 'frozen_numeric_core.tdx')
    assert selected['chosen_threshold']['training_quantile'] == .995
    assert model['variant'] == 'relative' and model['feature_names'] == list(inputs.EXPRESSIONS)
    assert model['last_observation'] < cfg['evaluation_start']
    return cfg, prior


def freeze(fold):
    path = root(fold)
    if (path / 'selection_report.json').exists():
        raise ValueError('Do not replace the frozen shortlist')
    assert not any((root(f) / 'analysis_report.json').exists() for f in ['2024', 'recent', '2025'])
    path.mkdir(parents=True, exist_ok=True)
    if fold == '2025':
        verify('2024'); verify('recent')
        a = pd.read_parquet(root('2024') / 'selection.parquet')
        b = pd.read_parquet(root('recent') / 'selection.parquet')
        pd.testing.assert_frame_equal(a.drop(columns='selected'), b.drop(columns='selected'), check_exact=True)
        assert not (a.selected & b.selected).any()
        out = a.copy(); out['selected'] |= b.selected
        details = dict(fold_selection_report_sha256={f: sha(root(f) / 'selection_report.json') for f in ['2024', 'recent']})
    else:
        cfg, prior = source(fold)
        out = pd.read_parquet(prior / 'selection.parquet')
        scores = pd.read_parquet(prior / 'scores.parquet', columns=['date', 'code', 'score'])
        pd.testing.assert_frame_equal(out[['date', 'code']], scores[['date', 'code']], check_exact=True)
        pool = scores.loc[out.selected].copy()
        assert np.isfinite(pool.score).all() and pool.score.gt(0).all()
        pool['score_integer'] = np.floor(pool.score * 1000000 + .5).astype('int64')
        pool['rank'] = pool.groupby('date').score_integer.rank(method='min', ascending=False)
        out['selected'] = False
        out.loc[pool.index, 'selected'] = pool['rank'].le(5)
        assert out.loc[out.selected, 'date'].ge(cfg['evaluation_start']).all()
        assert out.loc[out.selected, 'date'].lt(cfg['evaluation_end']).all()
        pool.to_parquet(path / 'ranking.parquet', index=False, compression='zstd')
        details = dict(source_selection_report_sha256=sha(prior / 'selection_report.json'),
                       source_score_report_sha256=sha(prior / 'score_report.json'),
                       ranking_sha256=sha(path / 'ranking.parquet'))
    out.to_parquet(path / 'selection.parquet', index=False, compression='zstd')
    counts = out.loc[out.selected].groupby('date').size()
    report = dict(protocol_sha256=sha(PROTOCOL), selection_sha256=sha(path / 'selection.parquet'),
                  selected=int(out.selected.sum()), days=len(counts), max_daily=int(counts.max()),
                  boundary_tie_days_above_five=int(counts.gt(5).sum()), **details,
                  model_refitted=False, top_positions=5, all_boundary_ties_retained=True,
                  original_results_previously_seen=True, new_group_outcomes_read=False,
                  year_2025_is_exploratory=True, new_2026_prices_read=False, no_exit_rules=True,
                  native_ranking_parity_verified=False, not_a_complete_native_formula=True)
    save_json(path / 'selection_report.json', report)
    return report


def verify(fold):
    config(); path = root(fold)
    r = json.loads((path / 'selection_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['selection_sha256'] == sha(path / 'selection.parquet')
    c = base.conn()
    if fold == '2025':
        queries = []
        for f in ['2024', 'recent']:
            proof = json.loads((root(f) / 'selection_verification.json').read_text())
            assert proof['passed'] and proof['selection_report_sha256'] == r['fold_selection_report_sha256'][f] == sha(root(f) / 'selection_report.json')
            queries.append(f"SELECT * FROM read_parquet('{root(f)}/selection.parquet')")
        expected = c.sql('SELECT date,code,half,board,decision_shares,bool_or(selected) AS selected FROM (' +
                         ' UNION ALL '.join(queries) + ') GROUP BY date,code,half,board,decision_shares ORDER BY date,code').df()
    else:
        cfg, prior = source(fold)
        assert r['source_selection_report_sha256'] == sha(prior / 'selection_report.json')
        assert r['source_score_report_sha256'] == sha(prior / 'score_report.json')
        assert r['ranking_sha256'] == sha(path / 'ranking.parquet')
        c.execute(f"CREATE VIEW ranked AS SELECT s.date,s.code,score, floor(score*1000000+.5)::BIGINT AS score_integer," +
                  f"rank() OVER(PARTITION BY s.date ORDER BY floor(score*1000000+.5) DESC)::DOUBLE AS rank " +
                  f"FROM read_parquet('{prior}/selection.parquet') s JOIN read_parquet('{prior}/scores.parquet') p USING(date,code) WHERE s.selected")
        rank = c.sql('SELECT * FROM ranked ORDER BY date,code').df()
        pd.testing.assert_frame_equal(pd.read_parquet(path / 'ranking.parquet'), rank, check_exact=True)
        expected = c.sql(f"SELECT s.* EXCLUDE(selected), s.selected AND coalesce(r.rank<=5,false) AS selected " +
                         f"FROM read_parquet('{prior}/selection.parquet') s LEFT JOIN ranked r USING(date,code) ORDER BY date,code").df()
        old = pd.read_parquet(prior / 'selection.parquet')
        assert not (expected.selected & ~old.selected).any()
        assert set(expected.loc[expected.selected, 'date']) == set(old.loc[old.selected, 'date'])
    c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(path / 'selection.parquet'), expected, check_exact=True)
    counts = expected.loc[expected.selected].groupby('date').size()
    assert int(counts.sum()) == r['selected'] and len(counts) == r['days'] and int(counts.max()) == r['max_daily']
    assert int(counts.gt(5).sum()) == r['boundary_tie_days_above_five']
    proof = dict(passed=True, selection_report_sha256=sha(path / 'selection_report.json'), rows=len(expected),
                 all_shortlist_flags_independently_rebuilt=True, new_group_outcomes_read=False,
                 new_2026_prices_read=False, no_exit_rules=True, native_ranking_parity_verified=False)
    save_json(path / 'selection_verification.json', proof)
    return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold', choices=['2024', 'recent', '2025'])
    p.add_argument('stage', choices=['freeze', 'verify', 'analyze'])
    a = p.parse_args()
    if a.stage == 'analyze':
        assert (root('2025') / 'selection_verification.json').exists()
        result = common_analysis(root(a.fold), PROTOCOL)
    else:
        result = globals()[a.stage](a.fold)
    print(json.dumps(result, ensure_ascii=False, indent=2))
