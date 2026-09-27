"""Reuse verified statistics only when every frozen selection field is identical."""
import argparse
import json
from pathlib import Path

import pandas as pd

from trade_research.corporate_cash import save_json, sha


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
    assert report['full_label_verification_sha256'] == sha(source / 'full_label_verification.json')
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


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(reuse(a.target, a.source), ensure_ascii=False, indent=2))
