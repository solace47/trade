"""Fixed agreement between two models trained on disjoint calendar halves."""
import argparse
import json
from pathlib import Path
import re

import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_float as inputs
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

STEM = 'tail_formula_half_consensus'
PROTOCOL = Path('config') / (STEM + '_protocol.json')
PARAMETERS = dict(n_estimators=64, max_depth=3, min_samples_leaf=300,
                  learning_rate=.05, subsample=1., random_state=20260927, loss='squared_error')


def policy():
    p = json.loads(PROTOCOL.read_text())
    assert p['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
    assert p['label_report_sha256'] == sha(labels.ROOT / 'full_label_report.json')
    assert p['label_verification_sha256'] == sha(labels.ROOT / 'full_label_verification.json')
    assert p['training_quantile'] == .995 and p['window_end'] == '09:59'
    assert p['arms'] == ['older', 'newer', 'intersection']
    assert p['no_threshold_or_split_scan'] and not p['new_2026_prices_allowed']
    for prefix in ['primary', 'secondary']:
        assert p[prefix + '_control_selection_report_sha256'] == sha(
            Path(p[prefix + '_control_root']) / 'selection_report.json')
    return p


def model_root(name):
    return Path('data/research') / (STEM + '_model_' + name)


def setup(name):
    p = policy(); spec = p['models'][name]
    config = Path('config') / (STEM + '_' + name + '_protocol.json')
    q = json.loads(config.read_text())
    assert q['parent_protocol_sha256'] == sha(PROTOCOL)
    assert all(q[k] == v for k, v in spec.items())
    assert q['parameters'] == PARAMETERS and q['model_max_depth'] == 3
    base.ROOT = model_root(name); base.PROTOCOL = relative.PROTOCOL = config
    base.FEATURES = inputs.ROOT; base.EXPRESSIONS = inputs.EXPRESSIONS; base.HEADER = inputs.HEADER
    base.SOURCE = labels.ROOT
    return spec


def verify_model(name):
    spec = setup(name)
    r = json.loads((base.ROOT / 'model_report.json').read_text())
    assert r['rows'] == spec['expected_rows'] and r['days'] == spec['expected_days']
    assert r['feature_names'] == list(inputs.EXPRESSIONS) and len(r['feature_names']) == 48
    assert all(r['parameters'][k] == v for k, v in PARAMETERS.items())
    return relative.verify_model('relative')


def folder(fold, arm):
    assert fold in ['2024', 'recent', '2025'] and arm in policy()['arms']
    return Path('data/research') / (STEM + '_' + fold) / arm


def checked_component(name):
    spec = setup(name); root = model_root(name)
    reports = {}
    for kind in ['model', 'score']:
        r = json.loads((root / (kind + '_report.json')).read_text())
        v = json.loads((root / (kind + '_verification.json')).read_text())
        assert v['passed'] and v[kind + '_report_sha256'] == sha(root / (kind + '_report.json'))
        assert r['protocol_sha256'] == sha(base.PROTOCOL)
        assert r['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
        reports[kind] = r
    m, s = reports['model'], reports['score']
    assert m['rows'] == spec['expected_rows'] and m['days'] == spec['expected_days']
    assert m['feature_names'] == list(inputs.EXPRESSIONS) and m['variant'] == 'relative'
    assert m['label_report_sha256'] == sha(labels.ROOT / 'full_label_report.json')
    assert all(m[k] == spec[k] for k in ['training_start', 'training_end'])
    assert m['last_observation'] < spec['training_end']
    assert all(m['parameters'][k] == v for k, v in PARAMETERS.items())
    assert m['thresholds'][3]['training_quantile'] == .995
    assert s['model_report_sha256'] == sha(root / 'model_report.json')
    assert s['scores_sha256'] == sha(root / 'scores.parquet')
    return m, dict(name=name, root=str(root), model_report_sha256=sha(root / 'model_report.json'),
                   score_report_sha256=sha(root / 'score_report.json'))


def core_parts(text):
    prefix, rest = text.split('T01:=', 1)
    body, clause = ('T01:=' + rest).split('\nCORE:', 1)
    assert clause.endswith(';\n')
    return prefix, body + '\n'


def consensus_core(cores, cuts):
    (prefix, older), (other_prefix, newer) = map(core_parts, cores)
    assert prefix == other_prefix
    newer = re.sub(r'\bT(\d{2})\b', r'U\1', newer)
    newer = re.sub(r'\bSC\b', 'SSC', newer)
    result = prefix + older + newer + f'CORE:(SC>{cuts[0]:.17g}) AND (SSC>{cuts[1]:.17g});\n'
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', result)
    assert len(names) == len(set(names))
    return result


def checked_fold(fold):
    p = policy(); spec = p['folds'][fold]
    values = [checked_component(name) for name in spec['components']]
    a, b = [x[0] for x in values]
    assert a['training_end'] == b['training_start']
    assert a['last_observation'] < b['training_start']
    assert b['training_end'] == spec['evaluation_start']
    assert max(a['last_observation'], b['last_observation']) < spec['evaluation_start']
    return spec, values


def freeze_fold(fold):
    spec, components = checked_fold(fold); arms = policy()['arms']
    assert not any((folder(f, arm) / 'analysis_report.json').exists()
                   for f in ['2024', 'recent', '2025'] for arm in arms)
    assert not any((folder(fold, arm) / 'selection_report.json').exists() for arm in arms)
    frames = [pd.read_parquet(Path(receipt['root']) / 'scores.parquet') for _, receipt in components]
    pd.testing.assert_frame_equal(frames[0].drop(columns='score'), frames[1].drop(columns='score'), check_exact=True)
    cuts = [m['thresholds'][3]['threshold'] for m, _ in components]
    scope = frames[0].date.ge(spec['evaluation_start']) & frames[0].date.lt(spec['evaluation_end'])
    flags = [scope & f.formula_input_valid & f.score.gt(cut) for f, cut in zip(frames, cuts)]
    cores = [base.native_core(m, cut, inputs.EXPRESSIONS, inputs.HEADER) for (m, _), cut in zip(components, cuts)]
    source = frames[0][['date', 'code', 'half', 'board', 'decision_shares']]
    reports = []
    for arm, selected, core in zip(arms, [*flags, flags[0] & flags[1]], [*cores, consensus_core(cores, cuts)]):
        root = folder(fold, arm); root.mkdir(parents=True, exist_ok=True)
        out = source.copy(); out['selected'] = selected
        out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
        (root / 'frozen_numeric_core.tdx').write_text(core)
        r = dict(protocol_sha256=sha(PROTOCOL), fold=fold, arm=arm,
            components=[receipt for _, receipt in components], score_cuts=cuts,
            selection_sha256=sha(root / 'selection.parquet'), core_sha256=sha(root / 'frozen_numeric_core.tdx'),
            selected=int(selected.sum()), days=int(out.loc[selected, 'date'].nunique()),
            evaluation_start=spec['evaluation_start'], evaluation_end=spec['evaluation_end'],
            component_training_windows_disjoint=True, selection_outcomes_read=False,
            software_compilation_verified=False, year_2025_is_exploratory=True,
            new_2026_prices_read=False, no_exit_rules=True)
        save_json(root / 'selection_report.json', r); reports.append(r)
    return reports


def verify_fold(fold):
    spec, components = checked_fold(fold); cuts = [m['thresholds'][3]['threshold'] for m, _ in components]
    c = base.conn()
    a, b = [str(Path(receipt['root']) / 'scores.parquet') for _, receipt in components]
    expected = c.sql(f'''SELECT a.date,a.code,a.half,a.board,a.decision_shares,
        a.date>='{spec['evaluation_start']}' AND a.date<'{spec['evaluation_end']}'
        AND a.formula_input_valid AND a.score>{cuts[0]:.17e} AS older,
        b.date>='{spec['evaluation_start']}' AND b.date<'{spec['evaluation_end']}'
        AND b.formula_input_valid AND b.score>{cuts[1]:.17e} AS newer
        FROM read_parquet('{a}') a JOIN read_parquet('{b}') b USING(date,code) ORDER BY date,code''').df()
    c.close(); expected['intersection'] = expected.older & expected.newer
    cores = [base.native_core(m, cut, inputs.EXPRESSIONS, inputs.HEADER) for (m, _), cut in zip(components, cuts)]
    results = []
    for arm in policy()['arms']:
        root = folder(fold, arm); r = json.loads((root / 'selection_report.json').read_text())
        assert r['protocol_sha256'] == sha(PROTOCOL) and r['score_cuts'] == cuts
        assert r['components'] == [receipt for _, receipt in components]
        for kind, file in [('selection', 'selection.parquet'), ('core', 'frozen_numeric_core.tdx')]:
            assert r[kind + '_sha256'] == sha(root / file)
        wanted = expected[['date', 'code', 'half', 'board', 'decision_shares', arm]].rename(columns={arm: 'selected'})
        pd.testing.assert_frame_equal(pd.read_parquet(root / 'selection.parquet'), wanted, check_exact=True)
        assert int(wanted.selected.sum()) == r['selected'] and wanted.loc[wanted.selected, 'date'].nunique() == r['days']
        text = (root / 'frozen_numeric_core.tdx').read_text()
        if arm != 'intersection':
            assert text == cores[['older', 'newer'].index(arm)]
        else:
            prefix, body = text.split('T01:=', 1); older, newer = body.split('U01:=', 1)
            clause = f'CORE:(SC>{cuts[0]:.17g}) AND (SSC>{cuts[1]:.17g});\n'
            assert newer.endswith(clause)
            assert prefix + 'T01:=' + older + f'CORE:SC>{cuts[0]:.17g};\n' == cores[0]
            restored = re.sub(r'\bU(\d{2})\b', r'T\1', 'U01:=' + newer[:-len(clause)])
            restored = re.sub(r'\bSSC\b', 'SC', restored)
            assert prefix + restored + f'CORE:SC>{cuts[1]:.17g};\n' == cores[1]
        proof = dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'), rows=len(wanted),
            all_flags_rebuilt_by_sql=True, all_core_texts_restored=True, all_input_declarations_unchanged=True,
            component_training_windows_disjoint=True, selection_outcomes_read=False,
            new_2026_prices_read=False, no_exit_rules=True)
        save_json(root / 'selection_verification.json', proof); results.append(proof)
    return results


def combined(stage):
    reports = []
    for arm in policy()['arms']:
        receipts = []; frames = []
        for fold in ['2024', 'recent']:
            root = folder(fold, arm); r = json.loads((root / 'selection_report.json').read_text())
            v = json.loads((root / 'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == sha(root / 'selection_report.json')
            assert r['protocol_sha256'] == sha(PROTOCOL) and r['selection_sha256'] == sha(root / 'selection.parquet')
            receipts.append(dict(root=str(root), selection_report_sha256=sha(root / 'selection_report.json')))
            frames.append(pd.read_parquet(root / 'selection.parquet'))
        a, b = frames
        pd.testing.assert_frame_equal(a.drop(columns='selected'), b.drop(columns='selected'), check_exact=True)
        assert not (a.selected & b.selected).any()
        assert a.loc[a.selected, 'half'].eq('2025H1').all() and b.loc[b.selected, 'half'].eq('2025H2').all()
        out = a.copy(); out['selected'] = a.selected | b.selected
        root = folder('2025', arm)
        if stage == 'freeze':
            assert not (root / 'selection_report.json').exists()
            assert not any((folder('2025', x) / 'analysis_report.json').exists() for x in policy()['arms'])
            root.mkdir(parents=True, exist_ok=True); out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
            r = dict(protocol_sha256=sha(PROTOCOL), arm=arm, components=receipts,
                selection_sha256=sha(root / 'selection.parquet'), selected=int(out.selected.sum()),
                days=int(out.loc[out.selected, 'date'].nunique()), selection_outcomes_read=False,
                year_2025_is_exploratory=True, new_2026_prices_read=False, no_exit_rules=True)
            save_json(root / 'selection_report.json', r)
        else:
            r = json.loads((root / 'selection_report.json').read_text())
            assert r['protocol_sha256'] == sha(PROTOCOL) and r['components'] == receipts
            assert r['selection_sha256'] == sha(root / 'selection.parquet')
            pd.testing.assert_frame_equal(out, pd.read_parquet(root / 'selection.parquet'), check_exact=True)
            assert int(out.selected.sum()) == r['selected'] and out.loc[out.selected, 'date'].nunique() == r['days']
            r = dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'), rows=len(out),
                disjoint_half_selection_union_rebuilt=True, selection_outcomes_read=False,
                new_2026_prices_read=False, no_exit_rules=True)
            save_json(root / 'selection_verification.json', r)
        reports.append(r)
    return reports


if __name__ == '__main__':
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--model', choices=['2024h1', '2024h2', '2025h1'])
    p.add_argument('--fold', choices=['2024', 'recent', '2025'])
    p.add_argument('--arm', choices=['older', 'newer', 'intersection'])
    a = p.parse_args()
    if a.stage in ['freeze', 'verify']:
        assert a.fold is not None and a.model is None
        result = combined(a.stage) if a.fold == '2025' else (freeze_fold(a.fold) if a.stage == 'freeze' else verify_fold(a.fold))
    elif a.stage == 'analyze':
        from .tail_formula_boundary_evaluation import analyze
        assert a.fold == '2025' and a.arm is not None
        assert all((folder(f, arm) / 'selection_verification.json').exists()
                   for f in ['2024', 'recent', '2025'] for arm in policy()['arms'])
        result = analyze(folder(a.fold, a.arm), PROTOCOL)
    else:
        assert a.model is not None and a.fold is None
        setup(a.model)
    if a.stage == 'model':
        result = relative.model('relative')
    elif a.stage == 'verify_model':
        result = verify_model(a.model)
    elif a.stage == 'scores':
        result = base.scores()
    elif a.stage == 'verify_scores':
        result = verify_scores()
    print(json.dumps(result, ensure_ascii=False, indent=2))
