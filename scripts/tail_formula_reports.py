"""Read-only checks of complete selection, label and analysis receipts."""
import json
import pandas as pd
from trade_research.research_io import sha

def checked_selection(root):
    report = json.loads((root / 'selection_report.json').read_text())
    proof = json.loads((root / 'selection_verification.json').read_text())
    assert proof['passed'] and proof['selection_report_sha256'] == sha(root / 'selection_report.json')
    assert report['selection_sha256'] == sha(root / 'selection.parquet')
    return pd.read_parquet(root / 'selection.parquet')


def checked_analysis(root):
    frame = checked_selection(root)
    assert frame.date.lt('2026-01-01').all() and frame.loc[frame.selected, 'date'].ge('2024-01-01').all()
    r = json.loads((root / 'analysis_report.json').read_text()); v = json.loads((root / 'analysis_verification.json').read_text())
    assert v['passed'] and v['analysis_report_sha256'] == sha(root / 'analysis_report.json')
    assert r['selection_report_sha256'] == sha(root / 'selection_report.json')
    assert r['daily_summary_sha256'] == sha(root / 'daily_summary.parquet') and r['reference_label'] == '09:59'
    lr = json.loads((root / 'full_label_report.json').read_text()); lv = json.loads((root / 'full_label_verification.json').read_text())
    assert lv['passed'] and lv['label_report_sha256'] == r['label_report_sha256'] == sha(root / 'full_label_report.json')
    assert lr['labels_sha256'] == sha(root / 'full_labels.parquet')
    return frame, r
