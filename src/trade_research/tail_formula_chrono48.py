"""Separate one-year fitting, half-year calibration, and next-half evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as original
from . import tail_formula_long48 as history
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha
from .tail_formula_1000_daily import analyze as common_analysis

STEM = 'tail_formula_chrono48'
LABELS = Path('data/research/tail_formula_1000')
KEYS = ['date', 'code', 'half', 'board', 'decision_shares']


def setup(fold):
    original.setup(fold)
    base.ROOT = Path('data/research') / (STEM + '_' + fold)
    base.PROTOCOL = Path('config') / (STEM + '_' + fold + '_protocol.json')
    base.FEATURES = history.ROOT; base.SOURCE = history.ROOT
    relative.PROTOCOL = base.PROTOCOL
    p = json.loads(base.PROTOCOL.read_text())
    assert p['inputs_protocol_sha256'] == sha(history.PROTOCOL)
    for field, path in [('feature_report_sha256', history.ROOT / 'feature_report.json'),
                        ('label_report_sha256', history.ROOT / 'full_label_report.json'),
                        ('calibration_label_report_sha256', LABELS / 'full_label_report.json')]:
        assert p[field] == sha(path)
    assert p['training_end'] == p['calibration_start'] < p['calibration_end'] == p['evaluation_start']
    assert p['training_quantiles'] == base.QUANTILES
    return p


def model():
    p = json.loads(base.PROTOCOL.read_text())
    if 'reuse_model_root' not in p:
        return relative.model('relative')
    assert not (base.ROOT / 'model_report.json').exists()
    source = Path(p['reuse_model_root'])
    assert sha(source / 'model_report.json') == p['reuse_model_report_sha256']
    proof = json.loads((source / 'model_verification.json').read_text())
    assert proof['passed'] and proof['model_report_sha256'] == sha(source / 'model_report.json')
    r = json.loads((source / 'model_report.json').read_text())
    assert r['training_start'] == p['training_start'] and r['training_end'] == p['training_end']
    assert r['feature_names'] == list(base.EXPRESSIONS) and r['variant'] == 'relative'
    r.update(protocol_sha256=sha(base.PROTOCOL), feature_report_sha256=sha(history.ROOT / 'feature_report.json'),
             label_report_sha256=sha(history.ROOT / 'full_label_report.json'),
             reused_model_report_sha256=sha(source / 'model_report.json'), model_refitted=False)
    base.ROOT.mkdir(parents=True, exist_ok=True)
    save_json(base.ROOT / 'model_report.json', r)
    return {k: v for k, v in r.items() if k != 'trees'}


def verify_model():
    p = json.loads(base.PROTOCOL.read_text())
    if 'reuse_model_root' in p:
        old = json.loads((Path(p['reuse_model_root']) / 'model_report.json').read_text())
        new = json.loads((base.ROOT / 'model_report.json').read_text())
        for key in ['trees', 'bias', 'thresholds', 'parameters', 'feature_names', 'rows', 'days', 'last_observation']:
            assert old[key] == new[key]
    return relative.verify_model('relative')


def calibration_labels():
    p = json.loads(base.PROTOCOL.read_text())
    r = json.loads((LABELS / 'full_label_report.json').read_text())
    v = json.loads((LABELS / 'full_label_verification.json').read_text())
    assert v['passed'] and v['label_report_sha256'] == sha(LABELS / 'full_label_report.json')
    assert r['labels_sha256'] == sha(LABELS / 'full_labels.parquet')
    c = base.conn()
    f = c.execute('''SELECT date,code,next_date,known15,opportunity15,known_no_trade,
        sustained_return15,adverse_return15,mark_1000_return15
        FROM read_parquet(?) WHERE date>=? AND next_date<? ORDER BY date,code''',
        [str(LABELS / 'full_labels.parquet'), p['calibration_start'], p['calibration_end']]).df()
    c.close()
    assert len(f) and f.date.ge(p['training_end']).all() and f.next_date.lt(p['evaluation_start']).all()
    return f


def counts(f):
    f = f.copy()
    f['success'] = f.known15 & f.opportunity15.eq(1)
    f['unknown'] = ~f.known15 & ~f.known_no_trade
    for name, column, compare in [('one', 'sustained_return15', lambda x: x.ge(.01)),
                                   ('bad', 'adverse_return15', lambda x: x.le(-.03))]:
        known = f.known15 & np.isfinite(f[column])
        f[name + '_success'] = known & compare(f[column])
        f[name + '_unknown'] = ~known & ~f.known_no_trade
    f['mark'] = f.mark_1000_return15.where(f.known15 & np.isfinite(f.mark_1000_return15))
    d = f.groupby('date').agg(rows=('code', 'size'), known=('known15', 'sum'),
        success=('success', 'sum'), unknown=('unknown', 'sum'), no_trade=('known_no_trade', 'sum'),
        one_success=('one_success', 'sum'), one_unknown=('one_unknown', 'sum'),
        bad_success=('bad_success', 'sum'), bad_unknown=('bad_unknown', 'sum'), mean1000=('mark', 'mean'))
    d['rate'] = d.success / d.known.replace(0, np.nan)
    d['lower'] = d.success / d.rows
    d['upper'] = (d.success + d.unknown) / d.rows
    d['one_lower'] = d.one_success / d.rows
    d['bad_upper'] = (d.bad_success + d.bad_unknown) / d.rows
    return d


def bootstrap_lower(d, p):
    if not len(d) or d.lower_delta.isna().any():
        return None
    a = d[['lower_delta']].copy()
    a['week'] = pd.to_datetime(a.index).to_period('W-SUN')
    blocks = a.groupby('week').lower_delta.agg(['sum', 'count'])
    if len(blocks) < 2:
        return None
    b = p['bootstrap']
    choices = np.random.default_rng(b['seed']).integers(len(blocks), size=(b['replicates'], len(blocks)))
    means = blocks['sum'].to_numpy()[choices].sum(axis=1) / blocks['count'].to_numpy()[choices].sum(axis=1)
    return float(np.quantile(means, b['lower_quantile']))


def clean(x):
    return float(x) if pd.notna(x) and np.isfinite(x) else None


def summarize(d, p):
    s = dict(rows=int(d.rows.sum()), known=int(d.known.sum()), unknown=int(d.unknown.sum()),
        no_trade=int(d.no_trade.sum()), days=len(d), rate=clean(d.rate.mean()), lower=clean(d.lower.mean()),
        delta=clean(d.delta.mean()), mean_selected=clean(d.rows.mean()), p95_selected=clean(d.rows.quantile(.95)),
        one_lower=clean(d.one_lower.mean()), bad_upper=clean(d.bad_upper.mean()), mean1000=clean(d.mean1000.mean()),
        lower_delta=clean(d.lower_delta.mean()), lower_delta_bootstrap=bootstrap_lower(d, p),
        conditional_gap_days=int((d.rate.isna() | d.delta.isna() | d.mean1000.isna()).sum()))
    g = p['quality_gates']
    tests = dict(days=s['days'] >= g['minimum_days'], known=s['known'] >= g['minimum_known'],
        conditional_complete=s['conditional_gap_days'] == 0)
    for name, bound, op in [('rate', 'minimum_rate', 'ge'), ('lower', 'minimum_complete_lower', 'ge'),
        ('delta', 'minimum_same_day_delta', 'ge'), ('mean_selected', 'maximum_mean_selected', 'le'),
        ('p95_selected', 'maximum_p95_selected', 'le'), ('one_lower', 'minimum_one_percent_complete_lower', 'ge'),
        ('bad_upper', 'maximum_bad3_complete_upper', 'le'), ('mean1000', 'minimum_known_mean1000_exclusive', 'gt')]:
        value = s[name]
        tests[name] = value is not None and (value >= g[bound] if op == 'ge' else value <= g[bound] if op == 'le' else value > g[bound])
    tests['conservative_weekly_increment'] = s['lower_delta_bootstrap'] is not None and s['lower_delta_bootstrap'] > 0
    s['gates'] = tests; s['eligible'] = all(tests.values())
    return s


def choose(summaries):
    a = [s for s in summaries if s['eligible']]
    return sorted(a, key=lambda s: (-s['lower'], -s['delta'], -s['known'], s['id']))[0] if a else None


def calibrate():
    root = base.ROOT; p = json.loads(base.PROTOCOL.read_text())
    assert not (root / 'calibration_report.json').exists()
    proof = json.loads((root / 'score_verification.json').read_text())
    assert proof['passed'] and proof['score_report_sha256'] == sha(root / 'score_report.json')
    m = json.loads((root / 'model_report.json').read_text())
    assert m['last_observation'] < p['calibration_start']
    labels = calibration_labels(); baseline = counts(labels)
    f = pd.read_parquet(root / 'scores.parquet', filters=[('date', '>=', p['calibration_start']), ('date', '<', p['calibration_end'])])
    joined = f.merge(labels, on=['date', 'code'], validate='one_to_one')
    parts = []; summaries = []
    for cut in m['thresholds']:
        q = joined.loc[joined.formula_input_valid & joined.score.gt(cut['threshold'])]
        d = counts(q)
        d['base_rate'] = baseline.rate.reindex(d.index)
        d['base_upper'] = baseline.upper.reindex(d.index)
        d['delta'] = d.rate - d.base_rate
        d['lower_delta'] = d.lower - d.base_upper
        summaries.append(dict(**cut, **summarize(d, p)))
        d['cut_id'] = cut['id']; parts.append(d.reset_index())
    pd.concat(parts, ignore_index=True).to_parquet(root / 'calibration_days.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(base.PROTOCOL), model_report_sha256=sha(root / 'model_report.json'),
        score_report_sha256=sha(root / 'score_report.json'), calibration_label_report_sha256=sha(LABELS / 'full_label_report.json'),
        calibration_days_sha256=sha(root / 'calibration_days.parquet'), summaries=summaries, chosen_threshold=choose(summaries),
        first_signal=labels.date.min(), last_signal=labels.date.max(), last_observation=labels.next_date.max(),
        calibration_start=p['calibration_start'], calibration_end=p['calibration_end'],
        evaluation_group_outcomes_read=False, year_2025_is_exploratory=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'calibration_report.json', r)
    return r


def freeze():
    root = base.ROOT; p = json.loads(base.PROTOCOL.read_text())
    v = json.loads((root / 'calibration_verification.json').read_text())
    assert v['passed'] and v['calibration_report_sha256'] == sha(root / 'calibration_report.json')
    r = json.loads((root / 'calibration_report.json').read_text())
    m = json.loads((root / 'model_report.json').read_text()); f = pd.read_parquet(root / 'scores.parquet')
    results = {}
    for name, cut in [('calibrated', r['chosen_threshold']), ('control', m['thresholds'][3])]:
        output = root if name == 'calibrated' else Path(str(root) + '_control')
        assert not (output / 'selection_report.json').exists()
        output.mkdir(parents=True, exist_ok=True)
        out = f[KEYS].copy()
        out['selected'] = (f.date.ge(p['evaluation_start']) & f.date.lt(p['evaluation_end']) &
                           f.formula_input_valid & f.score.gt(cut['threshold'])) if cut else False
        out.to_parquet(output / 'selection.parquet', index=False, compression='zstd')
        core = base.native_core(m, cut['threshold'], base.EXPRESSIONS, base.HEADER) if cut else 'CORE:0;\n'
        (output / 'frozen_numeric_core.tdx').write_text(core)
        report = dict(protocol_sha256=sha(base.PROTOCOL), model_root=str(root), arm=name,
            model_report_sha256=sha(root / 'model_report.json'), score_report_sha256=sha(root / 'score_report.json'),
            calibration_report_sha256=sha(root / 'calibration_report.json'), selection_sha256=sha(output / 'selection.parquet'),
            chosen_threshold=cut, core_sha256=sha(output / 'frozen_numeric_core.tdx'), selected=int(out.selected.sum()),
            by_half=out.groupby('half').selected.agg(['size', 'sum']).reset_index().to_dict('records'),
            evaluation_start=p['evaluation_start'], evaluation_end=p['evaluation_end'],
            evaluation_group_outcomes_read=False, year_2025_is_exploratory=True, new_2026_prices_read=False,
            no_exit_rules=True, software_compilation_verified=False)
        save_json(output / 'selection_report.json', report); results[name] = report
    return results


def combine():
    protocol = Path('config') / (STEM + '_combined_protocol.json')
    p = json.loads(protocol.read_text()); results = {}
    for suffix in ['', '_control']:
        outroot = Path('data/research') / (STEM + '_2025' + suffix)
        assert not (outroot / 'selection_report.json').exists()
        outroot.mkdir(parents=True, exist_ok=True); frames = []; folds = []
        for fold, pp in zip(['2024', 'recent'], p['fold_protocols']):
            root = Path('data/research') / (STEM + '_' + fold + suffix)
            assert not (root / 'analysis_report.json').exists()
            s = json.loads((root / 'selection_report.json').read_text())
            v = json.loads((root / 'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == sha(root / 'selection_report.json')
            assert s['protocol_sha256'] == sha(Path(pp)) and s['selection_sha256'] == sha(root / 'selection.parquet')
            frames.append(pd.read_parquet(root / 'selection.parquet'))
            core = outroot / ('frozen_numeric_core_' + fold + '.tdx')
            core.write_bytes((root / 'frozen_numeric_core.tdx').read_bytes())
            folds.append(dict(root=str(root), selection_report_sha256=sha(root / 'selection_report.json'),
                              core_sha256=sha(core), selected=s['selected']))
        pd.testing.assert_frame_equal(frames[0][KEYS], frames[1][KEYS], check_exact=True)
        assert not (frames[0].selected & frames[1].selected).any()
        out = frames[0].copy(); out['selected'] = frames[0].selected | frames[1].selected
        out.to_parquet(outroot / 'selection.parquet', index=False, compression='zstd')
        r = dict(protocol_sha256=sha(protocol), folds=folds, selection_sha256=sha(outroot / 'selection.parquet'),
            selected=int(out.selected.sum()), arm='control' if suffix else 'calibrated',
            by_half=out.groupby('half').selected.agg(['size', 'sum']).reset_index().to_dict('records'),
            evaluation_group_outcomes_read=False, year_2025_is_exploratory=True,
            new_2026_prices_read=False, no_exit_rules=True, software_compilation_verified=False)
        save_json(outroot / 'selection_report.json', r); results[r['arm']] = r
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold', choices=['2024', 'recent', 'combined'])
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'calibrate', 'freeze', 'combine', 'analyze'])
    p.add_argument('--control', action='store_true'); a = p.parse_args()
    if a.fold == 'combined':
        assert a.stage in ['combine', 'analyze']
        root = Path('data/research') / (STEM + '_2025' + ('_control' if a.control else ''))
        protocol = Path('config') / (STEM + '_combined_protocol.json')
    else:
        setup(a.fold); root = Path(str(base.ROOT) + ('_control' if a.control else '')); protocol = base.PROTOCOL
    if a.stage == 'analyze':
        for suffix in ['', '_control']:
            combined = Path('data/research') / (STEM + '_2025' + suffix)
            proof = json.loads((combined / 'selection_verification.json').read_text())
            assert proof['passed'] and proof['selection_report_sha256'] == sha(combined / 'selection_report.json')
        assert json.loads((root / 'selection_report.json').read_text())['selected'] > 0, 'No candidates: do not invent evaluation returns'
        result = common_analysis(root, protocol)
    elif a.stage == 'verify_scores':
        result = history.verify_scores()
    elif a.stage == 'scores':
        result = base.scores()
    else:
        result = globals()[a.stage]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
