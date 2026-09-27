"""Apply the frozen original models after removal of the amount floor."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as original
from . import tail_formula_liquidity_inputs as inputs
from .corporate_cash import save_json, sha

ROOT, PROTOCOL = inputs.ROOT, inputs.PROTOCOL
OLD = Path('data/research/tail_formula_float_2025')
SELECTION_COLUMNS = ['date','code','half','board','decision_shares']


def checked_features():
    inputs.checked_sources()
    report = json.loads((inputs.OUT/'feature_report.json').read_text())
    proof = json.loads((inputs.OUT/'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(inputs.OUT/'feature_report.json')
    assert report['features_sha256'] == sha(inputs.OUT/'features.parquet')
    return pd.read_parquet(inputs.OUT/'features.parquet')


def scores():
    if (ROOT/'score_report.json').exists():
        raise ValueError('Do not replace frozen out-of-pool scores')
    f = checked_features()
    base.EXPRESSIONS = original.EXPRESSIONS
    out = f[SELECTION_COLUMNS+['formula_input_valid']].copy()
    out['score'] = np.nan
    out['threshold'] = np.nan
    out['fold'] = ''
    records = {}
    for fold in ['2024','recent']:
        cfg, source = inputs.source(fold)
        model = json.loads((source/'model_report.json').read_text())
        selection = json.loads((source/'selection_report.json').read_text())
        mask = f.date.ge(cfg['evaluation_start']) & f.date.lt(cfg['evaluation_end'])
        valid = mask & f.formula_input_valid
        out.loc[valid, 'score'] = base.predict(base.encode(f.loc[valid]), model)
        out.loc[mask, 'threshold'] = selection['chosen_threshold']['threshold']
        out.loc[mask, 'fold'] = fold
        records[fold] = dict(model_report_sha256=sha(source/'model_report.json'),
            source_selection_report_sha256=sha(source/'selection_report.json'),
            threshold=selection['chosen_threshold']['threshold'], rows=int(mask.sum()), valid=int(valid.sum()))
    assert out.fold.ne('').all() and out.threshold.notna().all()
    out['selected'] = out.formula_input_valid & out.score.gt(out.threshold)
    out.to_parquet(ROOT/'scores.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(inputs.OUT/'feature_report.json'),
        feature_verification_sha256=sha(inputs.OUT/'feature_verification.json'), scores_sha256=sha(ROOT/'scores.parquet'),
        rows=len(out), selected=int(out.selected.sum()), folds=records, model_refitted=False,
        original_quantile_thresholds_unchanged=True, low_amount_model_extrapolation_unproven=True,
        outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT/'score_report.json', report)
    return report


def node_sql(tree, node):
    left, right = tree['children_left'][node], tree['children_right'][node]
    if left < 0:
        return format(tree['value'][node], '.17g')
    return (f'(CASE WHEN X{tree["feature"][node]:02d}<={tree["threshold"][node]:.17g}'
            f' THEN {node_sql(tree,left)} ELSE {node_sql(tree,right)} END)')


def verify_scores():
    f = checked_features()
    report = json.loads((ROOT/'score_report.json').read_text())
    for key,path in [('protocol_sha256',PROTOCOL), ('feature_report_sha256',inputs.OUT/'feature_report.json'),
        ('feature_verification_sha256',inputs.OUT/'feature_verification.json'),('scores_sha256',ROOT/'scores.parquet')]:
        assert report[key] == sha(path)
    got = pd.read_parquet(ROOT/'scores.parquet')
    pd.testing.assert_frame_equal(got[SELECTION_COLUMNS+['formula_input_valid']], f[SELECTION_COLUMNS+['formula_input_valid']], check_exact=True)
    c = base.conn(); c.register('features', f)
    columns = ','.join(f'floor(least(greatest(100*{name}+10000+.000001,0),999999))::INT AS X{i:02d}'
                       for i,name in enumerate(original.EXPRESSIONS))
    c.execute('CREATE VIEW encoded AS SELECT date,code,'+columns+' FROM features WHERE formula_input_valid')
    expected = pd.Series(np.nan, index=f.index, dtype=float)
    threshold = pd.Series(np.nan, index=f.index, dtype=float)
    assigned = pd.Series('', index=f.index)
    for fold in ['2024','recent']:
        cfg, source = inputs.source(fold)
        model = json.loads((source/'model_report.json').read_text())
        selected = json.loads((source/'selection_report.json').read_text())
        frozen = report['folds'][fold]
        assert frozen['model_report_sha256'] == sha(source/'model_report.json')
        assert frozen['source_selection_report_sha256'] == sha(source/'selection_report.json')
        cut = selected['chosen_threshold']['threshold']
        assert cut == frozen['threshold']
        expression = format(model['bias'], '.17g')
        for tree in model['trees']:
            expression = f'({expression}+{model["learning_rate"]:.17g}*({node_sql(tree,0)}))'
        result = c.execute(f'SELECT date,code,{expression} AS score FROM encoded WHERE date>=? AND date<? ORDER BY date,code',
                            [cfg['evaluation_start'],cfg['evaluation_end']]).df()
        mask = f.date.ge(cfg['evaluation_start']) & f.date.lt(cfg['evaluation_end'])
        valid = mask & f.formula_input_valid
        pd.testing.assert_frame_equal(f.loc[valid,['date','code']].reset_index(drop=True), result[['date','code']], check_exact=True)
        expected.loc[valid] = result.score.to_numpy()
        threshold.loc[mask] = cut; assigned.loc[mask] = fold
        assert frozen['rows'] == int(mask.sum()) and frozen['valid'] == int(valid.sum())
    c.close()
    np.testing.assert_allclose(expected, got.score, atol=2e-12, rtol=0, equal_nan=True)
    pd.testing.assert_series_equal(threshold, got.threshold, check_names=False, check_exact=True)
    pd.testing.assert_series_equal(assigned, got.fold, check_names=False, check_exact=True)
    chosen = f.formula_input_valid & expected.gt(threshold)
    assert chosen.equals(got.selected) and int(chosen.sum()) == report['selected']
    proof = dict(passed=True, score_report_sha256=sha(ROOT/'score_report.json'), rows=len(f),
        all_scores_independently_rebuilt_in_sql=True, all_frozen_thresholds_and_selection_flags_rebuilt=True,
        maximum_score_difference=float(np.nanmax(np.abs(expected-got.score))),
        original_models_not_refitted=True, outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT/'score_verification.json', proof)
    return proof


def checked_scores():
    report = json.loads((ROOT/'score_report.json').read_text())
    proof = json.loads((ROOT/'score_verification.json').read_text())
    assert proof['passed'] and proof['score_report_sha256'] == sha(ROOT/'score_report.json')
    assert report['protocol_sha256'] == sha(PROTOCOL) and report['scores_sha256'] == sha(ROOT/'scores.parquet')
    p = json.loads(PROTOCOL.read_text())
    assert p['original_selection_report_sha256'] == sha(OLD/'selection_report.json')
    r = json.loads((OLD/'selection_report.json').read_text())
    v = json.loads((OLD/'selection_verification.json').read_text())
    assert v['passed'] and v['selection_report_sha256'] == sha(OLD/'selection_report.json')
    assert r['selection_sha256'] == sha(OLD/'selection.parquet')
    return pd.read_parquet(ROOT/'scores.parquet'), pd.read_parquet(OLD/'selection.parquet')


def freeze():
    score, old = checked_scores()
    new = score[SELECTION_COLUMNS+['selected']]
    assert old.merge(new, on=['date','code']).empty
    result = {}
    for kind in ['added','combined']:
        assert not (ROOT/kind/'selection_report.json').exists()
        assert not (ROOT/kind/'analysis_report.json').exists()
    for kind in ['added','combined']:
        prior = old.copy()
        if kind == 'added':
            prior['selected'] = False
        out = pd.concat([prior,new], ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
        path = ROOT/kind; path.mkdir(exist_ok=True)
        out.to_parquet(path/'selection.parquet', index=False, compression='zstd')
        counts = out.loc[out.selected].groupby('date').size()
        r = dict(protocol_sha256=sha(PROTOCOL), score_report_sha256=sha(ROOT/'score_report.json'),
            original_selection_report_sha256=sha(OLD/'selection_report.json'), selection_sha256=sha(path/'selection.parquet'),
            selection_scope=kind, rows=len(out), selected=int(out.selected.sum()), days=len(counts),
            by_half=out.groupby('half').agg(rows=('code','size'),selected=('selected','sum')).reset_index().to_dict('records'),
            two_halves_fixed_together=True, original_models_and_thresholds_unchanged=True, model_refitted=False,
            new_group_outcomes_read=False, year_2025_is_exploratory=True,
            new_2026_prices_read=False, no_exit_rules=True, software_compilation_verified=False)
        save_json(path/'selection_report.json', r); result[kind] = r
    return result


def verify():
    score, old = checked_scores()
    c = base.conn(); c.register('added_scores',score); c.register('original',old)
    proofs = {}
    for kind in ['added','combined']:
        path = ROOT/kind; r = json.loads((path/'selection_report.json').read_text())
        for key,p in [('protocol_sha256',PROTOCOL),('score_report_sha256',ROOT/'score_report.json'),
            ('original_selection_report_sha256',OLD/'selection_report.json'),('selection_sha256',path/'selection.parquet')]:
            assert r[key] == sha(p)
        old_flag = 'selected' if kind == 'combined' else 'false'
        expected = c.sql(f'''SELECT date,code,half,board,decision_shares,{old_flag} AS selected FROM original
            UNION ALL SELECT date,code,half,board,decision_shares,formula_input_valid AND score>threshold AS selected
            FROM added_scores ORDER BY date,code''').df()
        assert not expected.duplicated(['date','code']).any()
        pd.testing.assert_frame_equal(pd.read_parquet(path/'selection.parquet'), expected, check_exact=True)
        assert len(expected) == r['rows'] and int(expected.selected.sum()) == r['selected']
        assert expected.loc[expected.selected,'date'].nunique() == r['days']
        assert expected.groupby('half').agg(rows=('code','size'),selected=('selected','sum')).reset_index().to_dict('records') == r['by_half']
        v = dict(passed=True, selection_report_sha256=sha(path/'selection_report.json'), rows=len(expected),
            all_keys_and_selection_flags_independently_rebuilt=True, all_original_keys_preserved=True,
            source_models_and_thresholds_unchanged=True, outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
        save_json(path/'selection_verification.json',v); proofs[kind] = v
    c.close()
    return proofs


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['scores','verify_scores','freeze','verify'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
