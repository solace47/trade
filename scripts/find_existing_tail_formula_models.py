"""Find prior model receipts matching a proposed fit before running it again.

This is a candidate lookup, not evidence of equal inputs or equal targets.
Actual reuse still requires the existing full-value and full-selection audits.
It reads model metadata only, skips 2026/Q1 paths, and never reads outcomes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

MODEL_CORE = ['trees','bias','parameters','thresholds','training_start','training_end',
              'feature_names','rows','days','last_observation']


def verification_registry():
    files=subprocess.check_output(['rg','--files','--no-ignore','-g','model_verification.json',
                                   'data/research'],text=True).splitlines()
    registry={}
    for file in sorted(files):
        path=Path(file)
        if any('2026' in p or 'q1' in p.lower() for p in path.parts):
            continue
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        registry.setdefault(digest,path.parent)
    return registry


def confirmed_target(root, registry, receipts, seen=()):
    """Resolve direct target arithmetic or an exact, hash-linked proof reuse."""
    root=Path(root)
    raw=(root/'model_report.json').read_bytes()
    proof_raw=(root/'model_verification.json').read_bytes()
    m=json.loads(raw);v=json.loads(proof_raw)
    digest=hashlib.sha256(proof_raw).hexdigest()
    assert digest not in seen, 'Cyclic model verification reuse'
    assert v['passed'] and v['model_report_sha256']==hashlib.sha256(raw).hexdigest()
    assert not m.get('new_2026_prices_read',False) and not v.get('new_2026_prices_read',False)
    receipts[str(root/'model_report.json')]=hashlib.sha256(raw).hexdigest()
    receipts[str(root/'model_verification.json')]=digest
    if v.get('all_targets_integer_inputs_day_weights_residual_means_and_variances_rebuilt'):
        assert v['variant']==m['variant'] and v['variant'] in ['relative','absolute']
        return v['variant']
    exact = v.get('exact_all_training_values_targets_weights_parameters_and_equations_before_reuse') or (
        v.get('exact_input_target_weight_and_parameter_equivalence_before_proof_reuse')
        and v.get('all_original_arithmetic_checks_reused'))
    assert exact, 'An undeclared target requires independent arithmetic evidence'
    parent=registry[v['reused_model_verification_sha256']]
    target=confirmed_target(parent,registry,receipts,seen+(digest,))
    source=json.loads((parent/'model_report.json').read_text())
    assert all(m[k]==source[k] for k in MODEL_CORE), 'Reused proof has a different model core'
    assert m['variant']==target
    return target


def find(protocol, variant=None, training_event=None):
    p = json.loads(protocol.read_text())
    parameters = p['parameters']
    rows = p.get('expected_training_rows')
    days = p.get('expected_training_days')
    names = p.get('feature_names')
    files = subprocess.run(['rg', '--files', '--no-ignore', '-g', 'model_report.json', 'data/research'],
        text=True, capture_output=True, check=True).stdout.splitlines()
    matches = []; skipped = []; examined = 0
    for file in sorted(files):
        path = Path(file)
        if not any(part.startswith('tail_formula_') for part in path.parts):
            continue
        if any('2026' in part or 'q1' in part.lower() for part in path.parts):
            continue
        try:
            data = path.read_bytes(); m = json.loads(data)
        except (OSError, json.JSONDecodeError):
            skipped.append(file); continue
        if m.get('new_2026_prices_read', False):
            continue
        if m.get('last_observation', '') > '2025-06-30':
            continue
        examined += 1
        if any(m.get('parameters', {}).get(k) != value for k, value in parameters.items()):
            continue
        if any(m.get(k) != p[k] for k in ['training_start', 'training_end'] if k in p):
            continue
        if rows is not None and m.get('rows') != rows:
            continue
        if days is not None and m.get('days') != days:
            continue
        if names is not None and m.get('feature_names') != names:
            continue
        if variant is not None and m.get('variant') != variant:
            continue
        if training_event is not None and m.get('training_event') != training_event:
            continue
        matches.append(dict(root=str(path.parent), model_report_sha256=hashlib.sha256(data).hexdigest(),
            variant=m.get('variant'), model_family=m.get('model_family'), training_event=m.get('training_event'),
            feature_report_sha256=m.get('feature_report_sha256'), label_report_sha256=m.get('label_report_sha256'),
            rows=m.get('rows'), days=m.get('days'), last_observation=m.get('last_observation')))
    return dict(protocol=str(protocol), existing_model_metadata_examined=examined,
        matches=matches, unreadable_or_in_progress_receipts=skipped,
        candidate_matches_are_not_proof_of_equivalent_inputs_or_targets=True,
        reuse_requires_full_training_values_and_complete_selection_equality=True,
        no_price_score_or_outcome_tables_read=True, no_2026_model_paths_read=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--variant')
    p.add_argument('--training-event')
    a = p.parse_args()
    print(json.dumps(find(a.protocol, a.variant, a.training_event), ensure_ascii=False, indent=2))
