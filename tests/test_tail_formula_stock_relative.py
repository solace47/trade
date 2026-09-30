import json
import numpy as np
import pandas as pd

from trade_research import tail_formula_stock_holdout as absolute
from trade_research import tail_formula_stock_relative as relative
from trade_research import tail_formula_additive as base
from trade_research import tail_formula_relative as model
from trade_research.corporate_cash import save_json, sha


def test_projection_and_calibration_namespaces_leave_absolute_artifacts_unchanged():
    assert absolute.model_root('2025h1', 0) == absolute.ROOT / 'models/2025h1/group0'
    assert relative.model_root('2025h1', 0) == relative.ROOT / 'models/2025h1/group0'
    assert relative.model_root('2025h1', 0) != absolute.model_root('2025h1', 0)
    assert relative.INPUTS == absolute.INPUTS
    assert absolute.LABEL_FIELDS == ['date', 'code', 'next_date', 'known15', 'opportunity15', 'known_no_trade']
    assert relative.LABEL_FIELDS == absolute.LABEL_FIELDS + ['adverse_return15']
    assert absolute.PROTOCOL.name == 'tail_formula_stock_holdout_input_protocol.json'
    assert relative.shared.PROTOCOL == relative.PROTOCOL


def test_existing_seven_fields_support_training_and_center_before_valid_intersection(tmp_path, monkeypatch):
    labels = pd.DataFrame(dict(date=['2024-06-03'] * 3, code=['sh.600001', 'sh.600002', 'sh.600003'],
        next_date=['2024-06-04'] * 3, known15=[True] * 3, opportunity15=[1., 0., 1.],
        known_no_trade=[False] * 3, adverse_return15=[np.nan] * 3))
    labels.to_parquet(tmp_path / 'full_labels.parquet', index=False)
    save_json(tmp_path / 'full_label_report.json', dict(labels_sha256=sha(tmp_path / 'full_labels.parquet')))
    save_json(tmp_path / 'full_label_verification.json', dict(passed=True,
        label_report_sha256=sha(tmp_path / 'full_label_report.json')))
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps(dict(training_start='2024-01-01', training_end='2025-01-01')))
    features = labels[['date', 'code']].copy(); features['formula_input_valid'] = [True, True, False]
    monkeypatch.setattr(base, 'SOURCE', tmp_path)
    monkeypatch.setattr(base, 'feature_inputs', lambda: features)
    monkeypatch.setattr(model, 'PROTOCOL', protocol)
    actual = model.training('relative')
    assert actual.code.tolist() == ['sh.600001', 'sh.600002']
    np.testing.assert_allclose(actual.target, [1/3, -2/3], rtol=0, atol=1e-12)
