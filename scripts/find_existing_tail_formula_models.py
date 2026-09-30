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
