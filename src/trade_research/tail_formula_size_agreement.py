"""Intersection of the frozen original and size-context opportunity models."""
import argparse
import json
from pathlib import Path
import re

import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_day_agreement as combine_adapter
from . import tail_formula_float as original
from . import tail_formula_size_context as extended
from .corporate_cash import save_json, sha
from .tail_formula_1000_daily import analyze as common_analysis

PROTOCOL = Path('config/tail_formula_size_agreement_protocol.json')
STEM = 'tail_formula_size_agreement'


def folder(fold):
    return Path('data/research') / (STEM + '_' + ('2025' if fold == 'combined' else fold))


def checked(fold):
    p = json.loads(PROTOCOL.read_text()); spec = p['folds'][fold]; components = []
    assert p['feature_report_sha256'] == sha(original.ROOT / 'feature_report.json')
    assert p['additional_feature_report_sha256'] == sha(extended.ROOT / 'feature_report.json')
    assert p['full_label_report_sha256'] == sha(base.SOURCE / 'full_label_report.json')
    for item, module in zip(spec['components'], [original, extended]):
        root = Path(item['root'])
        for name, digest in item.items():
            if name.endswith('_sha256'):
                assert sha(root / (name.removesuffix('_sha256') + '.json')) == digest
        input_proof = json.loads((module.ROOT / 'feature_verification.json').read_text())
        assert input_proof['passed'] and input_proof['feature_report_sha256'] == sha(module.ROOT / 'feature_report.json')
        reports = {}
        for kind in ['model', 'score', 'selection']:
            reports[kind] = json.loads((root / (kind + '_report.json')).read_text())
            proof = json.loads((root / (kind + '_verification.json')).read_text())
            assert proof['passed'] and proof[kind + '_report_sha256'] == sha(root / (kind + '_report.json'))
        m = reports['model']; selected = reports['selection']
        assert m['variant'] == 'relative' and m['feature_names'] == list(module.EXPRESSIONS)
        assert m['feature_report_sha256'] == sha(module.ROOT / 'feature_report.json')
        assert len(m['trees']) == 64 and m['learning_rate'] == .05 and m['last_observation'] < spec['evaluation_start']
        assert all(m[key] == spec[key] for key in ['training_start', 'training_end'])
        assert reports['score']['scores_sha256'] == sha(root / 'scores.parquet')
        assert reports['score']['model_report_sha256'] == item['model_report_sha256']
        assert selected['selection_sha256'] == sha(root / 'selection.parquet')
        assert selected['core_sha256'] == sha(root / 'frozen_numeric_core.tdx')
        assert selected['model_report_sha256'] == item['model_report_sha256']
        assert selected['chosen_threshold'] == m['thresholds'][3] and m['thresholds'][3]['training_quantile'] == .995
        components.append((root, selected))
    assert len(components) == 2
    assert list(extended.EXPRESSIONS.items())[:48] == list(original.EXPRESSIONS.items())
    return spec, components


def prefixes_and_bodies(sources):
    prefixes = []; bodies = []
    for source in sources:
        prefix, body = source.split('T01:=', 1)
        prefixes.append(prefix); bodies.append('T01:=' + body.split('\nCORE:', 1)[0] + '\n')
    # Every extra declaration is explicit and every shared declaration identical.
    names = [set(re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', p)) for p in prefixes]
    assert names[0] < names[1]
    stripped = ''.join(line for line in prefixes[1].splitlines(keepends=True) if line.split(':=', 1)[0] in names[0])
    assert stripped == prefixes[0]
    return prefixes, bodies


def freeze(fold):
    spec, components = checked(fold); root = folder(fold)
    assert not (root / 'selection_report.json').exists()
    assert not any((folder(f) / 'analysis_report.json').exists() for f in ['2024', 'recent', 'combined'])
    a, b = [pd.read_parquet(path / 'selection.parquet') for path, _ in components]
    pd.testing.assert_frame_equal(a.drop(columns='selected'), b.drop(columns='selected'), check_exact=True)
    out = a.copy(); out['selected'] = a.selected & b.selected
    root.mkdir(parents=True, exist_ok=True); out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
    cuts = [r['chosen_threshold']['threshold'] for _, r in components]
    prefixes, bodies = prefixes_and_bodies([(path / 'frozen_numeric_core.tdx').read_text() for path, _ in components])
    second = re.sub(r'\bT(\d{2})\b', r'U\1', bodies[1]); second = re.sub(r'\bSC\b', 'SSC', second)
    core = prefixes[1] + bodies[0] + second + f'CORE:(SC>{cuts[0]:.17g}) AND (SSC>{cuts[1]:.17g});\n'
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', core); assert len(names) == len(set(names))
    (root / 'frozen_numeric_core.tdx').write_text(core)
    r = dict(protocol_sha256=sha(PROTOCOL), fold=fold, components=spec['components'],
        selection_sha256=sha(root / 'selection.parquet'), core_sha256=sha(root / 'frozen_numeric_core.tdx'),
        score_cuts=cuts, selected=int(out.selected.sum()), original_selected=int(a.selected.sum()),
        size_model_selected=int(b.selected.sum()), evaluation_start=spec['evaluation_start'], evaluation_end=spec['evaluation_end'],
        new_model_fitted=False, component_results_previously_seen=True, new_agreement_group_outcomes_read=False,
        year_2025_is_exploratory=True, new_2026_prices_read=False, no_exit_rules=True, software_compilation_verified=False)
    save_json(root / 'selection_report.json', r); return r


def verify(fold):
    spec, components = checked(fold); root = folder(fold)
    r = json.loads((root / 'selection_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['components'] == spec['components']
    assert r['selection_sha256'] == sha(root / 'selection.parquet') and r['core_sha256'] == sha(root / 'frozen_numeric_core.tdx')
    cuts = [s['chosen_threshold']['threshold'] for _, s in components]; assert cuts == r['score_cuts']
    c = base.conn(); left, right = [str(path / 'scores.parquet') for path, _ in components]
    expected = c.sql(f'''SELECT a.date,a.code,a.half,a.board,a.decision_shares,
        a.date>='{spec['evaluation_start']}' AND a.date<'{spec['evaluation_end']}'
        AND a.formula_input_valid AND b.formula_input_valid AND a.score>{cuts[0]:.17e} AND b.score>{cuts[1]:.17e} AS selected
        FROM read_parquet('{left}') a JOIN read_parquet('{right}') b USING(date,code) ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(root / 'selection.parquet'), expected, check_exact=True)
    assert int(expected.selected.sum()) == r['selected'] <= min(r['original_selected'], r['size_model_selected'])
    text = (root / 'frozen_numeric_core.tdx').read_text(); prefix, body = text.split('T01:=', 1)
    a, b = body.split('U01:=', 1); clause = f'CORE:(SC>{cuts[0]:.17g}) AND (SSC>{cuts[1]:.17g});\n'
    assert b.endswith(clause)
    sources = [(path / 'frozen_numeric_core.tdx').read_text() for path, _ in components]
    original_prefix = sources[0].split('T01:=', 1)[0]
    assert prefix == sources[1].split('T01:=', 1)[0]
    shared_lines = [line for line in prefix.splitlines(keepends=True) if line in set(original_prefix.splitlines(keepends=True))]
    assert ''.join(shared_lines) == original_prefix
    assert original_prefix + 'T01:=' + a + f'CORE:SC>{cuts[0]:.17g};\n' == sources[0]
    restored = re.sub(r'\bU(\d{2})\b', r'T\1', 'U01:=' + b[:-len(clause)])
    restored = re.sub(r'\bSSC\b', 'SC', restored)
    assert prefix + restored + f'CORE:SC>{cuts[1]:.17g};\n' == sources[1]
    proof = dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'), rows=len(expected),
        all_selection_flags_rebuilt_from_both_verified_raw_scores=True,
        both_64_tree_core_texts_and_sum_orders_exactly_restored=True, all_original_input_declarations_unchanged=True,
        new_agreement_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'selection_verification.json', proof); return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold', choices=['2024', 'recent', 'combined']); p.add_argument('stage', choices=['freeze', 'verify', 'analyze'])
    a = p.parse_args(); combine_adapter.PROTOCOL = PROTOCOL; combine_adapter.STEM = STEM
    if a.stage == 'analyze':
        assert all((folder(f) / 'selection_verification.json').exists() for f in ['2024', 'recent', 'combined'])
        result = common_analysis(folder(a.fold), PROTOCOL)
    elif a.fold == 'combined':
        result = combine_adapter.combined(a.stage)
    else:
        result = globals()[a.stage](a.fold)
    print(json.dumps(result, ensure_ascii=False, indent=2))
