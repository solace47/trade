"""Fixed, outcome-free input support gate on the frozen corrected 48-input tree."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import tail_formula_additive as base
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as original
from .corporate_cash import save_json, sha

STEM = 'tail_formula_training_support'
ROOT = Path('data/research') / STEM
MASTER = Path('config') / (STEM + '_protocol.json')
DATA = Path('data/research')


def checked():
    p = json.loads(MASTER.read_text())
    for path, digest in p['references'].items():
        assert sha(Path(path)) == digest
    assert p['support_quantile'] == .99 and p['scale_floor'] == 1e-12 and p['expected_features'] == 48
    base.FEATURES = original.ROOT
    base.EXPRESSIONS = original.EXPRESSIONS
    base.HEADER = original.HEADER
    return p


def control(fold):
    return DATA / ('tail_formula_before1000_model_' + fold)


def support_root(fold):
    return ROOT / fold


def selection_root(arm, fold):
    assert arm in ['supported', 'rejected']
    return DATA / f'{STEM}_{arm}_{fold}'


def protocol(arm, fold):
    return Path('config') / f'{STEM}_{arm}_{"combined" if fold == "2025" else fold}_protocol.json'


def date_weights(dates):
    s = pd.Series(dates)
    weights = 1 / s.groupby(s).transform('size').to_numpy() / s.nunique()
    return weights / weights.sum()


def weighted_quantile(values, weights, quantile=.99):
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weights[order] / np.sum(weights))
    position = min(int(np.searchsorted(cumulative, quantile, side='left')), len(order) - 1)
    return float(values[order[position]])


def distances(x, means, scales):
    result = np.zeros(len(x))
    for i, (mean, scale) in enumerate(zip(means, scales)):
        result = np.maximum(result, np.abs((x[:, i] - mean) / scale))
    return result


def fit_numeric(x, dates):
    assert np.isfinite(x).all()
    w = date_weights(dates)
    means = np.average(x, axis=0, weights=w)
    scales = np.sqrt(np.average((x - means) ** 2, axis=0, weights=w))
    constant = scales <= 1e-12
    scales[constant] = 1
    d = distances(x, means, scales)
    return dict(means=means.tolist(), scales=scales.tolist(), constant_indices=np.flatnonzero(constant).tolist(),
                quantile=.99, threshold=weighted_quantile(d, w), rows=len(x), days=len(set(dates)),
                training_weight_at_or_below_threshold=float(w[d <= weighted_quantile(d, w)].sum()),
                training_weight_strictly_below_threshold=float(w[d < weighted_quantile(d, w)].sum()))


def fit(fold):
    p = checked()
    scope = next(v for v in p['folds'] if v['fold'] == fold)
    root = support_root(fold)
    root.mkdir(parents=True, exist_ok=True)
    assert not (root / 'support_report.json').exists()
    f = base.feature_inputs()
    train = f.loc[f.formula_input_valid & f.date.ge(scope['start']) & f.date.lt(scope['end'])]
    m = fit_numeric(base.encode(train), train.date.to_numpy())
    m.update(protocol_sha256=sha(MASTER), feature_report_sha256=sha(original.ROOT / 'feature_report.json'),
             feature_names=list(original.EXPRESSIONS), start=scope['start'], end=scope['end'],
             last_input_date=train.date.max(), no_label_join=True, no_outcomes_used=True,
             new_2026_prices_read=False, no_exit_rules=True)
    out = f[['date', 'code', 'formula_input_valid']].copy()
    out['distance'] = np.nan
    out.loc[f.formula_input_valid, 'distance'] = distances(base.encode(f.loc[f.formula_input_valid]), m['means'], m['scales'])
    out.to_parquet(root / 'support.parquet', index=False, compression='zstd')
    m['support_sha256'] = sha(root / 'support.parquet')
    save_json(root / 'support_report.json', m)
    return {k: v for k, v in m.items() if k not in ['means', 'scales', 'feature_names']}


def load(fold):
    p = checked()
    root = support_root(fold)
    m = json.loads((root / 'support_report.json').read_text())
    assert m['protocol_sha256'] == sha(MASTER)
    assert m['feature_report_sha256'] == sha(original.ROOT / 'feature_report.json')
    assert m['support_sha256'] == sha(root / 'support.parquet')
    assert m['feature_names'] == list(original.EXPRESSIONS) and len(m['means']) == len(m['scales']) == 48
    assert m['quantile'] == p['support_quantile'] and min(m['scales']) > 0
    assert np.isfinite([*m['means'], *m['scales'], m['threshold']]).all()
    scope = next(v for v in p['folds'] if v['fold'] == fold)
    assert m['start'] == scope['start'] and m['end'] == scope['end']
    assert m['last_input_date'] < m['end'] and m['no_label_join']
    return m


def support_expressions(m):
    exprs = [(f'SG{i:02d}', f'ABS((X{i:02d}-({mean:.17e}))/({scale:.17e}))')
             for i, (mean, scale) in enumerate(zip(m['means'], m['scales']), 1)]
    result = 'SG01'
    for i in range(2, 49):
        result = f'MAX({result},SG{i:02d})'
    return exprs + [('SD', result)]


def verify_support(fold):
    m = load(fold)
    root = support_root(fold)
    names = list(original.EXPRESSIONS)
    c = base.conn()
    c.register('features', pq.read_table(original.ROOT / 'features.parquet', columns=['date', 'code', 'formula_input_valid', *names]))
    encoded = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::DOUBLE AS X{i:02d}'
                       for i, n in enumerate(names, 1))
    c.sql('SELECT date,code,' + encoded + ' FROM features WHERE formula_input_valid').create_view('encoded')
    c.sql(f"""WITH rows AS(SELECT *,1./count(*) OVER(PARTITION BY date) AS rw FROM encoded
        WHERE date>='{m['start']}' AND date<'{m['end']}')
        SELECT * EXCLUDE(rw),rw/sum(rw) OVER() AS w FROM rows""").create_view('training')
    columns = [f'X{i:02d}' for i in range(1, 49)]
    means = np.array(c.sql('SELECT ' + ','.join(f'sum(w*{x})/sum(w)' for x in columns) + ' FROM training').fetchone())
    scales = np.sqrt(c.sql('SELECT ' + ','.join(f'sum(w*({x}-({mean:.17e}))*({x}-({mean:.17e})))/sum(w)'
                       for x, mean in zip(columns, means)) + ' FROM training').fetchone())
    constant = scales <= 1e-12
    scales[constant] = 1
    np.testing.assert_allclose(means, m['means'], rtol=0, atol=2e-7)
    np.testing.assert_allclose(scales, m['scales'], rtol=0, atol=2e-7)
    assert np.flatnonzero(constant).tolist() == m['constant_indices']
    # SQL replays the exact scalar arithmetic destined for the native core.
    expressions = support_expressions(m)
    c.sql('SELECT date,code,' + ','.join(expr + ' AS ' + name for name, expr in expressions[:-1])
          + ' FROM encoded').create_view('standardized')
    distance_sql = expressions[-1][1].replace('MAX(', 'greatest(')
    c.sql('SELECT date,code,' + distance_sql + ' AS distance FROM standardized').create_view('distances')
    expected = c.sql('''SELECT f.date,f.code,f.formula_input_valid,d.distance FROM features f
        LEFT JOIN distances d USING(date,code) ORDER BY date,code''').df()
    actual = pd.read_parquet(root / 'support.parquet')
    pd.testing.assert_frame_equal(actual.drop(columns='distance'), expected.drop(columns='distance'), check_exact=True)
    np.testing.assert_allclose(actual.distance, expected.distance, rtol=0, atol=2e-10, equal_nan=True)
    np.testing.assert_array_equal(actual.distance.le(m['threshold']), expected.distance.le(m['threshold']))
    c.register('replayed', expected[['date', 'code', 'distance']])
    dist = c.sql('SELECT t.date,t.code,t.w,d.distance FROM training t JOIN replayed d USING(date,code) ORDER BY date,code').df()
    f = base.feature_inputs()
    train = f.loc[f.formula_input_valid & f.date.ge(m['start']) & f.date.lt(m['end'])]
    pd.testing.assert_frame_equal(dist[['date', 'code']], train[['date', 'code']].reset_index(drop=True), check_exact=True)
    np.testing.assert_allclose(dist.w, date_weights(train.date.to_numpy()), rtol=0, atol=2e-15)
    c.register('quantile_input', dist[['distance', 'w']])
    cut = c.sql('''WITH cumulative AS(SELECT distance,sum(w) OVER(ORDER BY distance ROWS UNBOUNDED PRECEDING)/sum(w) OVER() AS mass
        FROM quantile_input) SELECT min(distance) FROM cumulative WHERE mass>=.99''').fetchone()[0]
    np.testing.assert_allclose(cut, m['threshold'], rtol=0, atol=2e-10)
    assert dist.distance.le(cut).equals(dist.distance.le(m['threshold']))
    below, through = c.sql(f'''SELECT sum(w) FILTER(WHERE distance<{cut:.17e}),
        sum(w) FILTER(WHERE distance<={cut:.17e}) FROM quantile_input''').fetchone()
    assert below < .99 <= through
    np.testing.assert_allclose([below, through], [m['training_weight_strictly_below_threshold'], m['training_weight_at_or_below_threshold']], rtol=0, atol=2e-12)
    assert len(dist) == m['rows'] and dist.date.nunique() == m['days']
    assert train.date.max() == m['last_input_date']
    c.close()
    proof = dict(passed=True, support_report_sha256=sha(root / 'support_report.json'), rows=len(actual),
                 valid=int(actual.formula_input_valid.sum()), training_rows=len(dist), training_days=dist.date.nunique(),
                 all_training_members_weights_and_statistics_verified=True, all_distances_threshold_flags_verified=True,
                 weighted_quantile_sql_verified=True, max_distance_difference=float((actual.distance - expected.distance).abs().max()),
                 no_label_join=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'support_verification.json', proof)
    return proof


def core_text(fold, arm, m):
    old = (control(fold) / 'frozen_numeric_core.tdx').read_text()
    match = re.search(r'\nCORE:([^;]+);\n$', old)
    assert match
    head = old[:match.start()+1]
    extra = '\n'.join(f'{n}:={e};' for n, e in support_expressions(m))
    comparator = '<=' if arm == 'supported' else '>'
    return head + extra + f'\nCORE:({match.group(1)}) AND SD{comparator}{m["threshold"]:.17e};\n'


def freeze(fold):
    m = load(fold)
    source = control(fold)
    root = support_root(fold)
    proof = json.loads((root / 'support_verification.json').read_text())
    assert proof['passed'] and proof['support_report_sha256'] == sha(root / 'support_report.json')
    old = json.loads((source / 'selection_report.json').read_text())
    original_selection = pd.read_parquet(source / 'selection.parquet')
    d = pd.read_parquet(root / 'support.parquet')
    pd.testing.assert_frame_equal(original_selection[['date', 'code']], d[['date', 'code']], check_exact=True)
    assert d.loc[original_selection.selected, 'distance'].notna().all()
    records = []
    for arm in ['supported', 'rejected']:
        dest = selection_root(arm, fold)
        dest.mkdir(parents=True, exist_ok=True)
        assert not (dest / 'selection_report.json').exists()
        selection = original_selection.copy()
        selection['selected'] &= d.distance.le(m['threshold']) if arm == 'supported' else d.distance.gt(m['threshold'])
        selection.to_parquet(dest / 'selection.parquet', index=False, compression='zstd')
        core = dest / 'frozen_numeric_core.tdx'
        core.write_text(core_text(fold, arm, m))
        for name in ['model_report.json', 'model_verification.json']:
            (dest / name).symlink_to((source / name).resolve())
        r = dict(protocol_sha256=sha(protocol(arm, fold)), model_report_sha256=sha(source / 'model_report.json'),
                 selection_sha256=sha(dest / 'selection.parquet'), core_sha256=sha(core),
                 support_report_sha256=sha(root / 'support_report.json'),
                 original_selection_report_sha256=sha(source / 'selection_report.json'),
                 chosen_threshold=old['chosen_threshold'], support_threshold=m['threshold'],
                 selected=int(selection.selected.sum()),
                 by_half=selection.groupby('half').selected.agg(['size', 'sum']).reset_index().to_dict('records'),
                 evaluation_start=old['evaluation_start'], evaluation_end=old['evaluation_end'],
                 new_group_outcomes_read=False, year_2025_is_exploratory=True,
                 new_2026_prices_read=False, no_exit_rules=True, software_compilation_verified=False)
        save_json(dest / 'selection_report.json', r)
        records.append(dict(arm=arm, selected=r['selected'], days=selection.loc[selection.selected, 'date'].nunique()))
    return records


def verify(fold):
    m = load(fold)
    root = support_root(fold)
    source = control(fold)
    c = base.conn()
    c.read_parquet(str(source / 'selection.parquet')).create_view('original')
    c.read_parquet(str(root / 'support.parquet')).create_view('support')
    arms = []
    records = []
    for arm in ['supported', 'rejected']:
        dest = selection_root(arm, fold)
        r = json.loads((dest / 'selection_report.json').read_text())
        for key, path in [('protocol_sha256', protocol(arm, fold)), ('selection_sha256', dest / 'selection.parquet'),
                          ('core_sha256', dest / 'frozen_numeric_core.tdx'), ('model_report_sha256', source / 'model_report.json'),
                          ('support_report_sha256', root / 'support_report.json'),
                          ('original_selection_report_sha256', source / 'selection_report.json')]:
            assert r[key] == sha(path)
        comparator = '<=' if arm == 'supported' else '>'
        expected = c.sql(f'''SELECT a.date,a.code,a.half,a.board,a.decision_shares,
            a.selected AND b.distance{comparator}{m['threshold']:.17e} AS selected
            FROM original a JOIN support b USING(date,code) ORDER BY date,code''').df()
        # Invalid inputs are false in the original list, preserving Boolean false in SQL.
        actual = pd.read_parquet(dest / 'selection.parquet')
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        assert int(expected.selected.sum()) == r['selected']
        core = (dest / 'frozen_numeric_core.tdx').read_text()
        assert core == core_text(fold, arm, m)
        names = re.findall(r'^([A-Z][A-Z0-9]*):=', core, re.M)
        assert len(names) == len(set(names))
        terms = re.findall(r'^SG(\d{2}):=ABS\(\(X\1-\(([^)]+)\)\)/\(([^)]+)\)\);$', core, re.M)
        assert [i for i, _, _ in terms] == [f'{i:02d}' for i in range(1, 49)]
        assert [float(a) for _, a, _ in terms] == m['means']
        assert [float(b) for _, _, b in terms] == m['scales']
        proof = dict(passed=True, selection_report_sha256=sha(dest / 'selection_report.json'), rows=len(actual),
                     all_selection_flags_and_native_support_constants_verified=True,
                     original_model_unchanged=True, no_new_group_outcomes_read=True,
                     new_2026_prices_read=False, no_exit_rules=True)
        save_json(dest / 'selection_verification.json', proof)
        arms.append(actual)
        records.append(dict(arm=arm, selected=r['selected'], passed=True))
    assert not (arms[0].selected & arms[1].selected).any()
    np.testing.assert_array_equal(arms[0].selected | arms[1].selected, pd.read_parquet(source / 'selection.parquet').selected)
    c.close()
    save_json(root / 'partition_verification.json', dict(passed=True,
        supported_selection_sha256=sha(selection_root('supported', fold) / 'selection_report.json'),
        rejected_selection_sha256=sha(selection_root('rejected', fold) / 'selection_report.json'),
        original_selection_sha256=sha(source / 'selection_report.json'), mutually_exclusive_and_exhaustive=True))
    return records


def combined(arm, verify_only=False):
    checked()
    linkage.ROOT = selection_root(arm, '2024')
    linkage.H2 = selection_root(arm, 'recent')
    linkage.COMBINED = selection_root(arm, '2025')
    linkage.PROTOCOL = protocol(arm, '2025')
    linkage.H2_MODEL_SHA = None
    linkage.H2_SELECTION_SHA = None
    linkage.H2_OUTCOMES_PREVIOUSLY_SEEN = False
    return linkage.verify_combined() if verify_only else linkage.combine()


def analyze(fold, arm):
    joint = json.loads((ROOT / 'joint_selection_freeze.json').read_text())
    assert joint['passed'] and joint['protocol_sha256'] == sha(MASTER) and len(joint['selections']) == 6
    root = selection_root(arm, fold)
    frozen = next(v for v in joint['selections'] if v['root'] == str(root))
    assert frozen['selection_report_sha256'] == sha(root / 'selection_report.json')
    return evaluation.analyze(root, protocol(arm, fold))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit', 'verify_support', 'freeze', 'verify', 'combine', 'verify_combined', 'analyze'])
    parser.add_argument('--fold', choices=['2024', 'recent', '2025'], default='2024')
    parser.add_argument('--arm', choices=['supported', 'rejected'])
    args = parser.parse_args()
    if args.stage in ['combine', 'verify_combined']:
        result = combined(args.arm, args.stage == 'verify_combined')
    elif args.stage == 'analyze':
        result = analyze(args.fold, args.arm)
    else:
        assert args.fold != '2025'
        result = globals()[args.stage](args.fold)
    print(json.dumps(result, ensure_ascii=False, indent=2))
