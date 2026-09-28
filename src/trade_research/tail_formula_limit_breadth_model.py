"""Fixed-model comparison for price-limit market context on conservative labels."""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_context_2024 as linkage
from . import tail_formula_limit_breadth as inputs
from . import tail_formula_prior_day as adapter
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

CONTROL = Path('data/research/tail_formula_limit_breadth_control')
ORIGINAL = Path('data/research/tail_formula_before1000_model_2025')


def configure(fold):
    p = inputs.checked_sources()
    n = json.loads((inputs.ROOT / 'native_input_verification.json').read_text())
    assert n['passed'] and n['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
    assert n['feature_verification_sha256'] == sha(inputs.ROOT / 'feature_verification.json')
    assert sha(ORIGINAL / 'selection_report.json') == p['primary_control_selection_report_sha256']
    adapter.STEM = inputs.STEM; adapter.ROOT = inputs.ROOT
    adapter.EXPRESSIONS = inputs.EXPRESSIONS; adapter.HEADER = inputs.HEADER
    adapter.COMBINED_PROTOCOL = inputs.COMBINED_PROTOCOL
    if fold != 'control':
        adapter.setup(fold)
    base.SOURCE = labels.ROOT; base.native_core = inputs.native_core
    for f in ['2024', 'recent']:
        q = json.loads((Path('config') / (inputs.STEM + '_' + f + '_protocol.json')).read_text())
        assert q['inputs_protocol_sha256'] == sha(inputs.PROTOCOL)
        assert q['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
        assert q['label_report_sha256'] == sha(labels.ROOT / 'full_label_report.json')


def control(stage):
    old = pd.read_parquet(ORIGINAL / 'selection.parquet')
    f = pd.read_parquet(inputs.ROOT / 'features.parquet', columns=['date', 'code', 'formula_input_valid'])
    pd.testing.assert_frame_equal(old[['date', 'code']], f[['date', 'code']], check_exact=True)
    if stage == 'freeze':
        if (CONTROL / 'selection_report.json').exists():
            raise ValueError('Do not replace the fixed corrected-label control')
        assert not any((Path('data/research') / (inputs.STEM + '_' + fold) / 'analysis_report.json').exists()
                       for fold in ['2024', 'recent', '2025'])
        out = old.copy(); out['selected'] &= f.formula_input_valid
        CONTROL.mkdir(parents=True, exist_ok=True); out.to_parquet(CONTROL / 'selection.parquet', index=False, compression='zstd')
        r = dict(protocol_sha256=sha(inputs.COMBINED_PROTOCOL), feature_report_sha256=sha(inputs.ROOT / 'feature_report.json'),
            source_selection_report_sha256=sha(ORIGINAL / 'selection_report.json'), selection_sha256=sha(CONTROL / 'selection.parquet'),
            selected=int(out.selected.sum()), unchanged_original_selection=out.equals(old), window_end='09:59',
            no_group_outcomes_read=True, new_2026_prices_read=False, no_exit_rules=True)
        save_json(CONTROL / 'selection_report.json', r); return r
    r = json.loads((CONTROL / 'selection_report.json').read_text())
    assert r['protocol_sha256'] == sha(inputs.COMBINED_PROTOCOL)
    assert r['feature_report_sha256'] == sha(inputs.ROOT / 'feature_report.json')
    assert r['source_selection_report_sha256'] == sha(ORIGINAL / 'selection_report.json')
    assert r['selection_sha256'] == sha(CONTROL / 'selection.parquet')
    c = base.conn(); c.register('old', old); c.register('f', f)
    expected = c.sql('''SELECT old.date,old.code,old.half,old.board,old.decision_shares,
        old.selected AND f.formula_input_valid AS selected FROM old JOIN f USING(date,code) ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(CONTROL / 'selection.parquet'), expected, check_exact=True)
    assert r['unchanged_original_selection'] == expected.equals(old)
    proof = dict(passed=True, selection_report_sha256=sha(CONTROL / 'selection_report.json'),
        all_flags_independently_rebuilt=True, rows=len(expected), no_group_outcomes_read=True)
    save_json(CONTROL / 'selection_verification.json', proof); return proof


def verify_model():
    p = json.loads(base.PROTOCOL.read_text()); r = json.loads((base.ROOT / 'model_report.json').read_text())
    assert len(r['feature_names']) == 50 and r['feature_names'] == list(inputs.EXPRESSIONS)
    assert r['days'] == p['expected_training_days'] == 241
    assert all(r['parameters'][k] == v for k, v in p['parameters'].items())
    return relative.verify_model('relative')


if __name__ == '__main__':
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--fold', choices=['2024', 'recent', 'combined', 'control'], default='2024')
    a = p.parse_args(); configure(a.fold)
    if a.stage == 'analyze':
        assert all((Path('data/research') / (inputs.STEM + '_' + f) / 'selection_verification.json').exists()
                   for f in ['2024', 'recent', '2025', 'control'])
        root = CONTROL if a.fold == 'control' else linkage.COMBINED if a.fold == 'combined' else base.ROOT
        protocol = inputs.COMBINED_PROTOCOL if a.fold in ['combined', 'control'] else base.PROTOCOL
        result = evaluation.analyze(root, protocol)
    elif a.fold == 'control':
        assert a.stage in ['freeze', 'verify']; result = control(a.stage)
    elif a.fold == 'combined':
        assert a.stage in ['freeze', 'verify']
        result = linkage.combine() if a.stage == 'freeze' else linkage.verify_combined()
    elif a.stage == 'model':
        result = relative.model('relative')
    elif a.stage == 'verify_model':
        result = verify_model()
    elif a.stage == 'verify_scores':
        result = verify_scores()
    elif a.stage in ['freeze', 'verify']:
        result = getattr(study, a.stage)()
    else:
        result = getattr(base, a.stage)()
    print(json.dumps(result, ensure_ascii=False, indent=2))
