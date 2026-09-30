"""Audit the accidentally repeated 2025 control without counting it as new evidence."""
import json
from pathlib import Path

import pandas as pd

from trade_research.corporate_cash import save_json, sha

ROOT = Path('data/research/tail_formula_feature_subsample')
PROTOCOL = Path('config/tail_formula_feature_subsample_reuse_protocol.json')
PROOF = ROOT / 'historical_control_reuse_verification.json'
OLD_ANNUAL = Path('data/research/tail_formula_subspace48_2025')
OLD_FOLDS = {'2025h1': Path('data/research/tail_formula_subspace48_2024'),
             '2025h2': Path('data/research/tail_formula_subspace48_recent')}


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['master_protocol_sha256'] == sha(Path('config/tail_formula_feature_subsample_protocol.json'))
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    return p


def verify():
    checked(); assert not PROOF.exists()
    receipts = {str(PROTOCOL): sha(PROTOCOL)}; records = []
    for fold, old in OLD_FOLDS.items():
        new = ROOT / 'norm' / fold
        reports = [json.loads((r / 'model_report.json').read_text()) for r in [new, old]]
        receipt_keys = {'protocol_sha256', 'feature_report_sha256', 'label_report_sha256'}
        assert set(reports[0]) == set(reports[1])
        assert {k for k in reports[0] if reports[0][k] != reports[1][k]} == receipt_keys
        for root in [new, old]:
            for kind in ['model', 'score']:
                r = json.loads((root / (kind + '_report.json')).read_text())
                v = json.loads((root / (kind + '_verification.json')).read_text())
                assert v['passed'] and v[kind + '_report_sha256'] == sha(root / (kind + '_report.json'))
                if kind == 'score':
                    assert r['model_report_sha256'] == sha(root / 'model_report.json')
                    assert r['scores_sha256'] == sha(root / 'scores.parquet')
            for name in ['model_report.json', 'model_verification.json', 'score_report.json',
                         'score_verification.json', 'scores.parquet']:
                receipts[str(root / name)] = sha(root / name)
        old_scores = pd.read_parquet(old / 'scores.parquet')
        new_scores = pd.read_parquet(new / 'scores.parquet', filters=[('date', '>=', '2024-01-01')])
        pd.testing.assert_frame_equal(new_scores.reset_index(drop=True), old_scores, check_exact=True)
        records.append(dict(fold=fold, source=str(old), repeated_fit=str(new),
            all_model_math_parameters_training_dates_and_thresholds_exact=True,
            all_2024_2025_score_values_metadata_and_validity_exact=True, score_rows=len(old_scores),
            repeated_fit_counted_as_new_evidence=False))
    save_json(PROOF, dict(passed=True, protocol_sha256=sha(PROTOCOL), source_hashes=receipts,
        records=records, historical_annual_root=str(OLD_ANNUAL),
        no_new_economic_aggregation=True, new_2026_prices_read=False, no_exit_rules=True))
    return dict(passed=True, proof_sha256=sha(PROOF), records=records)


def checked_proof():
    checked(); r = json.loads(PROOF.read_text())
    assert r['passed'] and r['protocol_sha256'] == sha(PROTOCOL)
    for file, digest in r['source_hashes'].items():
        assert sha(Path(file)) == digest
    return r


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
