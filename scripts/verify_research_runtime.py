"""Read-only proof that cleanup preserves current inputs and old control scores."""
import argparse
import ast
import importlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_baseline as baseline
from trade_research import tail_formula_etf_quantity as study
from trade_research import tail_formula_relative as relative
from trade_research.research_io import check_runtime, check_sources, sha


def archived(revision, path):
    return subprocess.run(['git', 'show', revision + ':' + path],
                          text=True, capture_output=True, check=True).stdout


def functions(source):
    return {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}


def verify(*, committed=True):
    runtime = check_runtime(committed=committed)
    revision = runtime['archive_revision']
    if committed:
        p = study.checked()
    else:
        # Only this read-only cleanup check can inspect an uncommitted runtime.
        # Research entrypoints always require its committed receipt.
        guard = study.check_runtime
        try:
            study.check_runtime = lambda: check_runtime(committed=False)
            p = study.checked()
        finally:
            study.check_runtime = guard
    execution = json.loads(Path('config/tail_formula_etf_quantity_model_protocol.json').read_text())
    check_sources(execution['source_hashes'])
    frozen = json.loads(Path('config/tail_formula_baseline.json').read_text())
    assert frozen['META'] == study.META and frozen['EXPRESSIONS'] == study.CONTROL
    assert p['arms'] == {'control': study.CONTROL, 'memory': study.EXPRESSIONS}
    assert p['native_header'] == study.HEADER and p['etf_helper'] == study.ETF_HELPER
    assert len(study.CONTROL) == 50 and len(study.EXPRESSIONS) == 52

    # These numeric and native functions were moved without changing their AST.
    compared = []
    for path, names in {
        'src/trade_research/tail_formula_additive.py':
            ['feature_inputs', 'labels', 'training', 'encode', 'leaf_indices',
             'predict', 'scores', 'native_tree', 'native_core'],
        'src/trade_research/tail_formula_etf_quantity.py':
            ['native_series', 'stock_values', 'prepare'],
        'scripts/compare_tail_formula_same_dates.py': ['compare'],
        'scripts/compare_tail_formula_shared_unknowns.py':
            ['daily_bounds', 'check_signal_dates', 'checked_selection', 'compare'],
        'scripts/audit_tail_formula_reference_coverage.py': ['audit'],
    }.items():
        old, new = functions(archived(revision, path)), functions(Path(path).read_text())
        for name in names:
            assert ast.dump(old[name], include_attributes=False) == ast.dump(new[name], include_attributes=False), (path, name)
            compared.append(path + ':' + name)

    for file in sorted(Path('scripts').glob('*.py')):
        if file.stem != 'verify_research_runtime':
            importlib.import_module(file.stem)
    report = json.loads((study.INPUTS / 'feature_report.json').read_text())
    for name in ['feature', 'native_input']:
        proof = json.loads((study.INPUTS / (name + '_verification.json')).read_text())
        assert proof['passed'] and proof['feature_report_sha256'] == sha(study.INPUTS / 'feature_report.json')
    assert report['protocol_sha256'] == sha(study.PROTOCOL)
    assert report['features_sha256'] == sha(study.INPUTS / 'features.parquet')
    assert sha(study.INPUTS / 'YJETFL.tdx') == report['helper_sha256']
    assert sha(study.INPUTS / 'quantity_inputs.tdx') == report['adapter_sha256']
    assert (study.INPUTS / 'YJETFL.tdx').read_text() == study.ETF_HELPER
    adapter = study.EXTRA_HEADER + ''.join(f'{name}:={value};\n' for name, value in study.NEW_EXPRESSIONS.items())
    assert (study.INPUTS / 'quantity_inputs.tdx').read_text() == adapter

    original = baseline.original()
    current = pd.read_parquet(study.INPUTS / 'features.parquet', columns=[*study.META, *study.EXPRESSIONS])
    pd.testing.assert_frame_equal(current[[*study.META, *study.CONTROL]], original, check_exact=True)
    assert len(current) == 1815129 and int(current.formula_input_valid.sum()) == 1602413
    assert current.date.lt('2026-01-01').all()
    del original

    # Full old score tables, including invalid rows, must replay exactly.
    base.EXPRESSIONS = study.CONTROL
    controls = []
    for fold, parent in p['old_model_roots'].items():
        root = Path(parent)
        model = json.loads((root / 'model_report.json').read_text())
        proof = json.loads((root / 'model_verification.json').read_text())
        assert proof['passed'] and proof['model_report_sha256'] == sha(root / 'model_report.json')
        scores = pd.read_parquet(root / 'scores.parquet')
        value = scores[[*study.META]].merge(current, on=study.META, validate='one_to_one')
        assert len(value) == len(scores) and value[study.META].equals(scores[study.META])
        valid = value.formula_input_valid
        rebuilt = np.full(len(value), np.nan)
        rebuilt[valid] = base.predict(base.encode(value.loc[valid]), model)
        np.testing.assert_allclose(rebuilt, scores.score, rtol=0, atol=2e-11, equal_nan=True)
        for cut in model['thresholds']:
            np.testing.assert_array_equal(rebuilt > cut['threshold'], scores.score > cut['threshold'])
        controls.append(dict(fold=fold, rows=len(scores), all_scores_and_threshold_flags_equal=True))
        print(json.dumps(dict(verified_control=fold, rows=len(scores))), flush=True)

    # Compare the simplified target with the pre-cleanup function on a pool
    # containing invalid input, a known zero with no reference, and unknowns.
    old_functions = functions(archived(revision, 'src/trade_research/tail_formula_relative.py'))
    old_env = {'base': base, 'np': np, 'pd': pd, 'json': json, 'Path': Path}
    tree = ast.Module(body=[old_functions['training_scope'], old_functions['training']], type_ignores=[])
    exec(compile(ast.fix_missing_locations(tree), '<pre-cleanup-target>', 'exec'), old_env)
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix='trade-cleanup-') as directory:
        folder = Path(directory)
        fields = dict(date=['2023-12-27'] * 4, code=['a', 'b', 'c', 'd'],
                      next_date=['2023-12-28'] * 4, known15=[True, True, True, False],
                      opportunity15=[1., 0., 0., np.nan], known_no_trade=[False] * 4,
                      adverse_return15=[-.01, np.nan, -.02, np.nan])
        pd.DataFrame(fields).to_parquet(folder / 'full_labels.parquet', index=False)
        visible = pd.DataFrame(dict(date=fields['date'], code=fields['code'],
                                    formula_input_valid=[True, True, False, True]))
        target_protocol = folder / 'target.json'
        target_protocol.write_text(json.dumps({'training_start': '2023-01-01', 'training_end': '2024-01-01'}))
        saved = (base.SOURCE, base.feature_inputs, base.labels, relative.PROTOCOL)
        base.SOURCE = folder
        base.feature_inputs = lambda: visible.copy()
        base.labels = lambda where: pd.DataFrame(fields).drop(columns='adverse_return15')
        relative.PROTOCOL = target_protocol
        old_env['PROTOCOL'] = target_protocol
        try:
            before = old_env['training']('relative')
            after = relative.training('relative')
            pd.testing.assert_frame_equal(after, before, check_exact=True)
            np.testing.assert_allclose(after.target, [2 / 3, -1 / 3], rtol=0, atol=2e-12)
        finally:
            base.SOURCE, base.feature_inputs, base.labels, relative.PROTOCOL = saved

    return dict(passed=True, unchanged_numeric_functions=len(compared),
                rows=len(current), valid=int(current.formula_input_valid.sum()),
                controls=controls, target_population_and_known_zero_semantics_equal=True,
                new_fits=0, new_selection_economics=0, new_2026_prices_read=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--uncommitted', action='store_true', help='Read-only check before committing cleanup')
    args = parser.parse_args()
    print(json.dumps(verify(committed=not args.uncommitted), ensure_ascii=False, indent=2), flush=True)
