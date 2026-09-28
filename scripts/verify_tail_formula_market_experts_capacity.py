"""Confirm the pooled 128-tree control exactly extends the original 64-tree fit."""
import json
from pathlib import Path

from trade_research.corporate_cash import save_json, sha
from trade_research.tail_formula_market_experts import ROOT, PROTOCOL, STEM


def verify():
    out = ROOT / 'capacity_control_verification.json'
    assert not out.exists(), 'Do not overwrite a frozen capacity-control proof'
    records = []
    for fold in ['2024', 'recent']:
        roots = [Path('data/research') / (STEM + '_pooled_' + fold),
                 Path('data/research') / ('tail_formula_before1000_model_' + fold)]
        reports = []
        for root in roots:
            proof = json.loads((root / 'model_verification.json').read_text())
            assert proof['passed'] and proof['model_report_sha256'] == sha(root / 'model_report.json')
            reports.append(json.loads((root / 'model_report.json').read_text()))
        extended, original = reports
        assert len(extended['trees']) == 128 and len(original['trees']) == 64
        assert extended['trees'][:64] == original['trees']
        for key in ['bias', 'learning_rate', 'feature_names', 'rows', 'days', 'last_observation',
                    'training_start', 'training_end', 'feature_report_sha256', 'label_report_sha256']:
            assert extended[key] == original[key], key
        for key, value in extended['parameters'].items():
            if key != 'n_estimators':
                assert value == original['parameters'][key], key
        records.append(dict(fold=fold, extended_model_sha256=sha(roots[0] / 'model_report.json'),
            original_model_sha256=sha(roots[1] / 'model_report.json'),
            every_original_tree_node_exactly_preserved=True))
    result = dict(passed=True, protocol_sha256=sha(PROTOCOL), verifier_sha256=sha(Path(__file__)),
        folds=records, model_rounds_are_the_only_pooled_fit_change=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    save_json(out, result)
    return result


if __name__ == '__main__':
    print(json.dumps(verify(), ensure_ascii=False, indent=2))
