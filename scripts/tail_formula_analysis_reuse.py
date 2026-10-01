"""Reuse only after every selection field and applicable label is identical."""
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess
import pandas as pd
from trade_research import tail_formula_additive as base
from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research.research_io import save_json, sha
from tail_formula_reports import checked_selection, checked_analysis

PROTOCOL = Path('config/tail_formula_minute_open_model_protocol.json')
inputs = SimpleNamespace(ROOT=Path('data/research/tail_formula_minute_open'))

def checked_joint():
    raise RuntimeError('Bind the current study and joint receipt before analysis')

def reuse(target, source):
    assert target.resolve() != source.resolve()
    assert not (target / 'analysis_report.json').exists()
    for root in [target, source]:
        report = json.loads((root / 'selection_report.json').read_text())
        proof = json.loads((root / 'selection_verification.json').read_text())
        assert proof['passed'] and proof['selection_report_sha256'] == sha(root / 'selection_report.json')
        assert report['selection_sha256'] == sha(root / 'selection.parquet')
    pd.testing.assert_frame_equal(pd.read_parquet(target / 'selection.parquet'),
                                  pd.read_parquet(source / 'selection.parquet'), check_exact=True)
    report = json.loads((source / 'analysis_report.json').read_text())
    proof = json.loads((source / 'analysis_verification.json').read_text())
    assert proof['passed'] and proof['analysis_report_sha256'] == sha(source / 'analysis_report.json')
    assert report['selection_report_sha256'] == sha(source / 'selection_report.json')
    assert report['daily_summary_sha256'] == sha(source / 'daily_summary.parquet')
    if 'full_label_verification_sha256' in report:
        assert report['full_label_verification_sha256'] == sha(source / 'full_label_verification.json')
    else:
        assert report['reference_label'] == '09:59'
        assert report['label_report_sha256'] == sha(source / 'full_label_report.json')
    labels = json.loads((source / 'full_label_report.json').read_text())
    label_proof = json.loads((source / 'full_label_verification.json').read_text())
    assert label_proof['passed'] and label_proof['label_report_sha256'] == sha(source / 'full_label_report.json')
    assert labels['labels_sha256'] == sha(source / 'full_labels.parquet')
    result = dict(passed=True, target=str(target), source=str(source),
                  selection_report_sha256=sha(target / 'selection_report.json'),
                  reused_selection_report_sha256=sha(source / 'selection_report.json'),
                  reused_analysis_report_sha256=sha(source / 'analysis_report.json'),
                  reused_analysis_verification_sha256=sha(source / 'analysis_verification.json'),
                  full_labels_sha256=sha(source / 'full_labels.parquet'),
                  all_selection_columns_and_keys_identical=True,
                  original_analysis_reused_without_recalculation=True,
                  scope='Same existing next-morning labels and all existing cost/quality scenarios only',
                  new_2026_prices_read=False, no_exit_rules=True)
    save_json(target / 'analysis_reuse.json', result)
    return result


def reuse_analysis(root, parent):
    """New full label file, but all labels joined by this complete list are equal."""
    frame = checked_selection(root); parent_frame, ar = checked_analysis(parent)
    assert frame.equals(parent_frame)
    evaluation.attach_labels(root)
    c = base.conn(); c.register('keys', frame[['date', 'code']])
    old = c.execute('SELECT o.* FROM read_parquet(?) o JOIN keys USING(date,code) ORDER BY date,code',
        [str(parent/'full_labels.parquet')]).df()
    new = c.execute('SELECT o.* FROM read_parquet(?) o JOIN keys USING(date,code) ORDER BY date,code',
        [str(root/'full_labels.parquet')]).df(); c.close()
    pd.testing.assert_frame_equal(new[old.columns], old, check_exact=True)
    assert len(new) == len(frame)
    proof = dict(passed=True, old_labels_sha256=sha(parent/'full_labels.parquet'),
        new_labels_sha256=sha(root/'full_labels.parquet'), rows=len(frame),
        every_complete_selection_key_and_applicable_label_field_exactly_identical=True,
        old_metadata_and_unknowns_preserved=True, no_new_economic_aggregation=True)
    save_json(root/'label_projection_reuse_verification.json', proof)
    reuse(root, parent)
    (root/'daily_summary.parquet').symlink_to((parent/'daily_summary.parquet').resolve())
    report = dict(ar)
    report.update(protocol_sha256=sha(PROTOCOL), selection_report_sha256=sha(root/'selection_report.json'),
        label_report_sha256=sha(root/'full_label_report.json'),
        reused_analysis_report_sha256=sha(parent/'analysis_report.json'),
        label_projection_reuse_verification_sha256=sha(root/'label_projection_reuse_verification.json'),
        all_joined_label_values_unchanged=True, no_new_economic_aggregation=True)
    save_json(root/'analysis_report.json', report)
    v = json.loads((parent/'analysis_verification.json').read_text())
    v.update(analysis_report_sha256=sha(root/'analysis_report.json'),
        reused_analysis_verification_sha256=sha(parent/'analysis_verification.json'),
        all_source_stats_reused_after_full_metadata_and_joined_labels_equal=True,
        no_new_economic_aggregation=True)
    save_json(root/'analysis_verification.json', v)
    return proof


def analyze():
    master, joint = checked_joint(); previous = [Path(master['original_control'])]
    previous.extend(Path(v) for v in master['controls'].values())
    records = []
    for item in joint['selections']:
        root = Path(item['root']); frame = checked_selection(root)
        parent = next((p for p in previous if frame.equals(checked_selection(p))), None)
        existed = (root/'analysis_report.json').exists()
        if not existed:
            if parent is not None:
                reuse_analysis(root, parent)
            else:
                evaluation.analyze(root, PROTOCOL)
                with (root/'analysis_check_command.log').open('w') as log:
                    subprocess.run(['.venv/bin/python', 'scripts/verify_tail_formula_before1000.py',
                        'analysis', '--root', str(root)], stdout=log, check=True)
        checked_analysis(root); previous.append(root)
        records.append(dict(group=item['group'], reused_source=str(parent) if parent else None,
            new_economic_aggregation_run=not existed and parent is None,
            analysis_report_sha256=sha(root/'analysis_report.json')))
        print(json.dumps(dict(completed=item['group'])), flush=True)
    result = dict(passed=True, joint_sha256=sha(inputs.ROOT/'joint_selection_freeze.json'), records=records,
        full_frame_equality_before_aggregation=True, no_duplicate_half_year_aggregation=True,
        all_joined_labels_proven_equal_before_any_reuse=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(inputs.ROOT/'analysis_dispatch_verification.json', result); return records
