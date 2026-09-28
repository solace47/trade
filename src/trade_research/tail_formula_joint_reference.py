"""Keep frozen opportunity candidates only when a frozen reference score is positive."""
import argparse
import json
from pathlib import Path
import re

import pandas as pd

from . import tail_formula_absolute_zero as zero
from . import tail_formula_additive as base
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_float as inputs
from .corporate_cash import save_json, sha
from .tail_formula_context_2024 import normalized_selection_flags
from .tail_formula_rank_gate import compose_core

STEM = 'tail_formula_joint_reference'
PROTOCOL = Path('config') / (STEM + '_protocol.json')
FOLDS = ['2024', 'recent', '2025', 'removed', 'removed_dates']
META = ['date', 'code', 'half', 'board', 'decision_shares']


def root_for(arm, fold):
    return Path('data/research') / (STEM + '_' + arm + '_' + fold)


def binary_root(fold):
    return Path('data/research/tail_formula_endpoint_binary_' + fold)


def sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['arms'] == ['squared', 'huber'] and not p['model_refit_allowed']
    assert not p['threshold_search_allowed'] and not p['new_2026_prices_allowed']
    assert p['zero_protocol_sha256'] == sha(zero.PROTOCOL)
    zero.sources()
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for arm in p['arms']:
        for fold in ['2024', 'recent', '2025']:
            for root in [zero.root_for(arm, fold), binary_root(fold)]:
                r = json.loads((root / 'selection_report.json').read_text())
                v = json.loads((root / 'selection_verification.json').read_text())
                assert v['passed'] and v['selection_report_sha256'] == sha(root / 'selection_report.json')
                assert r['selection_sha256'] == sha(root / 'selection.parquet')
    for fold in p['folds']:
        root = binary_root(fold['name'])
        m = json.loads((root / 'model_report.json').read_text())
        assert m['variant'] == 'endpoint_binary' and m['feature_names'] == list(inputs.EXPRESSIONS)
        assert m['last_observation'] < m['training_end'] <= fold['start'] < fold['end']
        for stage in ['model', 'score']:
            v = json.loads((root / (stage + '_verification.json')).read_text())
            assert v['passed'] and v[stage + '_report_sha256'] == sha(root / (stage + '_report.json'))
        score = json.loads((root / 'score_report.json').read_text())
        assert score['scores_sha256'] == sha(root / 'scores.parquet')
        assert score['model_report_sha256'] == sha(root / 'model_report.json')
        assert score['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
        text = base.native_core(m, m['thresholds'][3]['threshold'], inputs.EXPRESSIONS, inputs.HEADER)
        assert text == (root / 'frozen_numeric_core.tdx').read_text()
    return p


def numeric_core(arm, fold):
    binary = (binary_root(fold) / 'frozen_numeric_core.tdx').read_text()
    absolute = (zero.root_for(arm, fold) / 'frozen_numeric_core.tdx').read_text()
    m = json.loads((binary_root(fold) / 'model_report.json').read_text())
    cut = m['thresholds'][3]['threshold']
    result = compose_core(binary, absolute, cut, 0.)
    # Reversibly recover both complete tree bodies, preserving every literal
    # and addition order; the shared input prefix must be exactly the same.
    prefix, body = result.split('T01:=', 1)
    first, second = ('T01:=' + body).split('U01:=', 1)
    assert prefix + first + f'CORE:SC>{cut:.17g};\n' == binary
    second = 'U01:=' + second.split('\nCORE:', 1)[0] + '\n'
    second = re.sub(r'\bU(\d{2})\b', r'T\1', second)
    second = re.sub(r'\bAC\b', 'SC', second)
    assert prefix + second + 'CORE:SC>0;\n' == absolute
    assert result.endswith(f'CORE:(SC>{cut:.17g}) AND (AC>0);\n')
    return result


def all_frames(arm, p):
    frames = {}
    for fold in ['2024', 'recent']:
        a = pd.read_parquet(binary_root(fold) / 'selection.parquet')
        b = pd.read_parquet(zero.root_for(arm, fold) / 'selection.parquet')
        pd.testing.assert_frame_equal(a[META], b[META], check_exact=True)
        f = a[META].copy()
        f['selected'] = a.selected & b.selected
        frames[fold] = normalized_selection_flags(f)
    a = pd.read_parquet(binary_root('2025') / 'selection.parquet')
    for f in frames.values():
        pd.testing.assert_frame_equal(a[META], f[META], check_exact=True)
    combined = a[META].copy()
    combined['selected'] = frames['2024'].selected | frames['recent'].selected
    frames['2025'] = combined
    kept_dates = set(combined.loc[combined.selected, 'date'])
    for fold, condition in [('removed', a.selected & ~combined.selected),
                            ('removed_dates', a.selected & ~a.date.isin(kept_dates))]:
        frames[fold] = a[META].copy()
        frames[fold]['selected'] = condition
    for name, f in frames.items():
        frames[name] = normalized_selection_flags(f)
    assert (frames['2025'].selected.astype(int) + frames['removed'].selected.astype(int)).eq(a.selected.astype(int)).all()
    return frames


def freeze():
    p = sources()
    for arm in p['arms']:
        for fold in FOLDS:
            root = root_for(arm, fold)
            assert not (root / 'selection_report.json').exists() and not (root / 'analysis_report.json').exists()
    result = []
    for arm in p['arms']:
        frames = all_frames(arm, p)
        for fold, frame in frames.items():
            root = root_for(arm, fold)
            root.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
            cores = {}
            for period in [fold] if fold in ['2024', 'recent'] else ['2024', 'recent']:
                name = 'frozen_numeric_core.tdx' if fold in ['2024', 'recent'] else 'frozen_numeric_core_' + period + '.tdx'
                if fold in ['removed', 'removed_dates']:
                    continue  # Diagnostic subsets are not inverted live formulas.
                path = root / name
                path.write_text(numeric_core(arm, period))
                cores[name] = sha(path)
            counts = frame.loc[frame.selected].groupby('date').size()
            r = dict(protocol_sha256=sha(PROTOCOL), arm=arm, fold=fold,
                source_selection_reports={str(source): sha(source / 'selection_report.json')
                    for period in ['2024', 'recent', '2025']
                    for source in [binary_root(period), zero.root_for(arm, period)]},
                selection_sha256=sha(root / 'selection.parquet'), core_hashes=cores,
                selected=int(frame.selected.sum()), signal_days=len(counts),
                top_dates={str(k): int(v) for k, v in counts.nlargest(5).items()},
                signal_dates=sorted(counts.index), diagnostic_only=fold in ['removed', 'removed_dates'],
                model_refitted=False, year_2025_is_exploratory=True, new_group_outcomes_read=False,
                new_2026_prices_read=False, no_exit_rules=True, software_compilation_verified=False)
            save_json(root / 'selection_report.json', r)
            result.append(dict(arm=arm, fold=fold, selected=r['selected'], days=r['signal_days']))
    return result


def verify():
    p = sources()
    results = []
    for arm in p['arms']:
        c = base.conn()
        for fold in p['folds']:
            period = fold['name']
            a = binary_root(period)
            z = Path('data/research/tail_formula_endpoint_absolute_' + arm + '_' + period)
            m = json.loads((a / 'model_report.json').read_text())
            cut = m['thresholds'][3]['threshold']
            c.read_parquet(str(a / 'scores.parquet')).create_view('binary_' + period)
            c.read_parquet(str(z / 'scores.parquet')).create_view('absolute_' + period)
            c.sql(f'''SELECT a.date,a.code,a.half,a.board,a.decision_shares,
                coalesce(a.formula_input_valid AND a.score>{cut:.17g} AND
                a.date>='{fold['start']}' AND a.date<'{fold['end']}',false) AS original,
                coalesce(original AND z.formula_input_valid AND z.score>0.,false) AS selected
                FROM binary_{period} a JOIN absolute_{period} z USING(date,code)
                ORDER BY a.date,a.code''').create_view('fold_' + period)
        c.sql('''SELECT a.date,a.code,a.half,a.board,a.decision_shares,
            a.original OR b.original AS original,a.selected OR b.selected AS selected
            FROM fold_2024 a JOIN fold_recent b USING(date,code) ORDER BY a.date,a.code''').create_view('combined')
        original = c.sql('SELECT date,code,original AS selected FROM combined ORDER BY date,code').df()
        saved = pd.read_parquet(binary_root('2025') / 'selection.parquet')
        pd.testing.assert_frame_equal(normalized_selection_flags(original),
            normalized_selection_flags(saved[['date', 'code', 'selected']]), check_exact=True)
        for fold in FOLDS:
            root = root_for(arm, fold)
            r = json.loads((root / 'selection_report.json').read_text())
            assert r['protocol_sha256'] == sha(PROTOCOL) and r['arm'] == arm and r['fold'] == fold
            assert r['selection_sha256'] == sha(root / 'selection.parquet')
            for source, digest in r['source_selection_reports'].items():
                assert digest == sha(Path(source) / 'selection_report.json')
            for name, digest in r['core_hashes'].items():
                period = fold if fold in ['2024', 'recent'] else name.removeprefix('frozen_numeric_core_').removesuffix('.tdx')
                assert digest == sha(root / name) and (root / name).read_text() == numeric_core(arm, period)
            table = 'fold_' + fold if fold in ['2024', 'recent'] else 'combined'
            condition = 'selected'
            if fold == 'removed':
                condition = 'original AND NOT selected'
            elif fold == 'removed_dates':
                condition = 'original AND date NOT IN (SELECT DISTINCT date FROM combined WHERE selected)'
            expected = c.sql(f'SELECT date,code,half,board,decision_shares,{condition} AS selected FROM {table} ORDER BY date,code').df()
            got = pd.read_parquet(root / 'selection.parquet')
            pd.testing.assert_frame_equal(normalized_selection_flags(expected), normalized_selection_flags(got), check_exact=True)
            assert int(expected.selected.sum()) == r['selected']
            assert sorted(expected.loc[expected.selected, 'date'].unique()) == r['signal_dates']
            assert len(r['signal_dates']) == r['signal_days']
            proof = dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'),
                rows=len(expected), all_flags_rebuilt_from_original_scores=True,
                source_models_scores_and_selection_hashes_checked=True,
                native_trees_and_thresholds_reversibly_recovered=bool(r['core_hashes']),
                removed_stock_days_and_whole_dates_preserved=True, no_training_period_selection=True,
                diagnostic_only=r['diagnostic_only'], new_2026_prices_read=False)
            save_json(root / 'selection_verification.json', proof)
            results.append(proof)
        c.close()
    return results


def analyze(arm, fold):
    p = sources()
    for a in p['arms']:
        for f in FOLDS:
            root = root_for(a, f)
            v = json.loads((root / 'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256'] == sha(root / 'selection_report.json')
    return evaluation.analyze(root_for(arm, fold), PROTOCOL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'verify', 'analyze'])
    parser.add_argument('--arm', choices=['squared', 'huber'], default='squared')
    parser.add_argument('--fold', choices=FOLDS, default='2025')
    args = parser.parse_args()
    result = analyze(args.arm, args.fold) if args.stage == 'analyze' else globals()[args.stage]()
    print(json.dumps(result, ensure_ascii=False, indent=2))
