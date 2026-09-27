"""Freeze the intersection of two already verified relative-opportunity formulas."""
import argparse
import json
from pathlib import Path
import re

import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as inputs
from .corporate_cash import save_json, sha
from .tail_formula_1000_daily import analyze as common_analysis

PROTOCOL = Path('config/tail_formula_day_agreement_protocol.json')
STEM = 'tail_formula_day_agreement'


def folder(fold):
    return Path('data/research') / (STEM + '_' + ('2025' if fold == 'combined' else fold))


def checked(fold):
    p = json.loads(PROTOCOL.read_text())
    assert p['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
    assert p['full_label_report_sha256'] == sha(base.SOURCE / 'full_label_report.json')
    spec = p['folds'][fold]; result = []
    for item in spec['components']:
        root = Path(item['root'])
        for name, digest in item.items():
            if name.endswith('_sha256'):
                assert sha(root / (name.removesuffix('_sha256') + '.json')) == digest
        model = json.loads((root / 'model_report.json').read_text())
        assert model['variant'] == 'relative' and model['feature_names'] == list(inputs.EXPRESSIONS)
        assert model['feature_report_sha256'] == p['feature_report_sha256']
        assert len(model['trees']) == 64 and model['learning_rate'] == .05
        assert model['last_observation'] < spec['evaluation_start']
        for key in ['training_start', 'training_end']:
            assert model[key] == spec[key]
        reports = {}
        for kind in ['model', 'score', 'selection']:
            report = json.loads((root / (kind + '_report.json')).read_text())
            proof = json.loads((root / (kind + '_verification.json')).read_text())
            assert proof['passed'] and proof[kind + '_report_sha256'] == sha(root / (kind + '_report.json'))
            reports[kind] = report
        assert reports['score']['scores_sha256'] == sha(root / 'scores.parquet')
        assert reports['score']['model_report_sha256'] == item['model_report_sha256']
        selected = reports['selection']
        assert selected['selection_sha256'] == sha(root / 'selection.parquet')
        assert selected['core_sha256'] == sha(root / 'frozen_numeric_core.tdx')
        assert selected['model_report_sha256'] == item['model_report_sha256']
        assert selected['chosen_threshold'] == model['thresholds'][3]
        assert selected['chosen_threshold']['training_quantile'] == .995
        result.append((root, selected))
    assert len(result) == 2
    return spec, result


def compose(sources, cuts):
    prefix, a = sources[0].split('T01:=', 1)
    prefix2, b = sources[1].split('T01:=', 1)
    assert prefix == prefix2
    a = 'T01:=' + a.split('\nCORE:', 1)[0] + '\n'
    b = 'T01:=' + b.split('\nCORE:', 1)[0] + '\n'
    b = re.sub(r'\bT(\d{2})\b', r'U\1', b)
    b = re.sub(r'\bSC\b', 'DSC', b)
    core = prefix + a + b + f'CORE:(SC>{cuts[0]:.17g}) AND (DSC>{cuts[1]:.17g});\n'
    names = re.findall(r'\b([A-Z][A-Z0-9]*):=', core)
    assert len(names) == len(set(names))
    return core


def freeze(fold):
    spec, components = checked(fold); root = folder(fold)
    if (root / 'selection_report.json').exists():
        raise ValueError('Do not replace the frozen agreement selection')
    assert not any((folder(f) / 'analysis_report.json').exists() for f in ['2024', 'recent', 'combined'])
    a, b = [pd.read_parquet(path / 'selection.parquet') for path, _ in components]
    pd.testing.assert_frame_equal(a.drop(columns='selected'), b.drop(columns='selected'), check_exact=True)
    out = a.copy(); out['selected'] = a.selected & b.selected
    root.mkdir(parents=True, exist_ok=True)
    out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
    cuts = [r['chosen_threshold']['threshold'] for _, r in components]
    sources = [(path / 'frozen_numeric_core.tdx').read_text() for path, _ in components]
    (root / 'frozen_numeric_core.tdx').write_text(compose(sources, cuts))
    r = dict(protocol_sha256=sha(PROTOCOL), fold=fold, components=spec['components'],
        selection_sha256=sha(root / 'selection.parquet'), core_sha256=sha(root / 'frozen_numeric_core.tdx'),
        score_cuts=cuts, selected=int(out.selected.sum()), original_selected=int(a.selected.sum()),
        weighted_model_selected=int(b.selected.sum()), evaluation_start=spec['evaluation_start'],
        evaluation_end=spec['evaluation_end'], new_model_fitted=False, component_results_previously_seen=True,
        new_agreement_group_outcomes_read=False, year_2025_is_exploratory=True,
        new_2026_prices_read=False, no_exit_rules=True, software_compilation_verified=False)
    save_json(root / 'selection_report.json', r); return r


def verify(fold):
    spec, components = checked(fold); root = folder(fold)
    r = json.loads((root / 'selection_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['components'] == spec['components']
    assert r['selection_sha256'] == sha(root / 'selection.parquet')
    assert r['core_sha256'] == sha(root / 'frozen_numeric_core.tdx')
    cuts = [s['chosen_threshold']['threshold'] for _, s in components]
    assert r['score_cuts'] == cuts
    c = base.conn(); left, right = [str(path / 'scores.parquet') for path, _ in components]
    expected = c.sql(f'''SELECT a.date,a.code,a.half,a.board,a.decision_shares,
        a.date>='{spec['evaluation_start']}' AND a.date<'{spec['evaluation_end']}'
        AND a.formula_input_valid AND b.formula_input_valid
        AND a.score>{cuts[0]:.17e} AND b.score>{cuts[1]:.17e} AS selected
        FROM read_parquet('{left}') a JOIN read_parquet('{right}') b USING(date,code) ORDER BY date,code''').df()
    c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(root / 'selection.parquet'), expected, check_exact=True)
    assert int(expected.selected.sum()) == r['selected']
    assert r['selected'] <= min(r['original_selected'], r['weighted_model_selected'])
    # Reverse the composed text, independently of compose(), preserving both
    # original floating literals and each 64-tree sum's exact order.
    text = (root / 'frozen_numeric_core.tdx').read_text()
    prefix, body = text.split('T01:=', 1); a, b = body.split('U01:=', 1)
    clause = f'CORE:(SC>{cuts[0]:.17g}) AND (DSC>{cuts[1]:.17g});\n'
    assert b.endswith(clause)
    restored_a = prefix + 'T01:=' + a + f'CORE:SC>{cuts[0]:.17g};\n'
    restored_b = re.sub(r'\bU(\d{2})\b', r'T\1', 'U01:=' + b[:-len(clause)])
    restored_b = re.sub(r'\bDSC\b', 'SC', restored_b)
    restored_b = prefix + restored_b + f'CORE:SC>{cuts[1]:.17g};\n'
    for restored, (path, _) in zip([restored_a, restored_b], components):
        assert restored == (path / 'frozen_numeric_core.tdx').read_text()
    proof = dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'), rows=len(expected),
        all_selection_flags_rebuilt_from_both_verified_raw_scores=True,
        both_64_tree_core_texts_and_sum_orders_exactly_restored=True,
        new_agreement_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'selection_verification.json', proof); return proof


def combined(stage):
    root = folder('combined'); frames = []; identities = []
    for fold in ['2024', 'recent']:
        path = folder(fold); r = json.loads((path / 'selection_report.json').read_text())
        v = json.loads((path / 'selection_verification.json').read_text())
        assert v['passed'] and v['selection_report_sha256'] == sha(path / 'selection_report.json')
        assert r['protocol_sha256'] == sha(PROTOCOL) and r['selection_sha256'] == sha(path / 'selection.parquet')
        frames.append(pd.read_parquet(path / 'selection.parquet'))
        identities.append(dict(fold=fold, selection_report_sha256=sha(path / 'selection_report.json'),
                               core_sha256=sha(path / 'frozen_numeric_core.tdx')))
    a, b = frames
    pd.testing.assert_frame_equal(a.drop(columns='selected'), b.drop(columns='selected'), check_exact=True)
    assert not (a.selected & b.selected).any()
    if stage == 'freeze':
        assert not (root / 'selection_report.json').exists()
        assert not any((folder(f) / 'analysis_report.json').exists() for f in ['2024', 'recent', 'combined'])
        root.mkdir(parents=True, exist_ok=True); out = a.copy(); out['selected'] = a.selected | b.selected
        out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
        for item in identities:
            (root / ('frozen_numeric_core_' + item['fold'] + '.tdx')).write_bytes((folder(item['fold']) / 'frozen_numeric_core.tdx').read_bytes())
        r = dict(protocol_sha256=sha(PROTOCOL), folds=identities, selection_sha256=sha(root / 'selection.parquet'),
            selected=int(out.selected.sum()), component_results_previously_seen=True, new_model_fitted=False,
            year_2025_is_exploratory=True, new_agreement_group_outcomes_read=False,
            new_2026_prices_read=False, no_exit_rules=True, software_compilation_verified=False)
        save_json(root / 'selection_report.json', r); return r
    r = json.loads((root / 'selection_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['folds'] == identities
    assert r['selection_sha256'] == sha(root / 'selection.parquet')
    for item in identities:
        assert item['core_sha256'] == sha(root / ('frozen_numeric_core_' + item['fold'] + '.tdx'))
    c = base.conn(); c.register('a', a); c.register('b', b)
    expected = c.sql('''SELECT a.date,a.code,a.half,a.board,a.decision_shares,
        (a.date>='2025-01-01' AND a.date<'2025-07-01' AND a.selected) OR
        (a.date>='2025-07-01' AND a.date<'2026-01-01' AND b.selected) AS selected
        FROM a JOIN b USING(date,code) ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(root / 'selection.parquet'), expected, check_exact=True)
    assert r['selected'] == int(expected.selected.sum())
    proof = dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'), rows=len(expected),
        all_time_windows_and_selection_flags_rebuilt=True, new_agreement_group_outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'selection_verification.json', proof); return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold', choices=['2024', 'recent', 'combined'])
    p.add_argument('stage', choices=['freeze', 'verify', 'analyze'])
    a = p.parse_args()
    if a.stage == 'analyze':
        assert all((folder(f) / 'selection_verification.json').exists() for f in ['2024', 'recent', 'combined'])
        result = common_analysis(folder(a.fold), PROTOCOL)
    elif a.fold == 'combined':
        result = combined(a.stage)
    else:
        result = globals()[a.stage](a.fold)
    print(json.dumps(result, ensure_ascii=False, indent=2))
