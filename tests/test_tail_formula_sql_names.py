"""Source fields must not shadow case-distinct formula inputs in SQL."""
import importlib.util
import json
from pathlib import Path

import pandas as pd


def test_exact_formula_fields_precede_sql_binding(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('score_name_check', scripts / 'verify_tail_formula_additive.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'FEATURES', tmp_path)
    (tmp_path / 'model_report.json').write_text(json.dumps({'feature_names': ['IP01', 'IP02']}))
    frame = pd.DataFrame({'date': ['2024-01-02'], 'code': ['sh.600000'],
        'half': ['2024H1'], 'board': ['main'], 'decision_shares': [100],
        'formula_input_valid': [True], 'ip01': [2968.56], 'IP01': [.08], 'IP02': [-.22]})
    frame.to_parquet(tmp_path / 'features.parquet', index=False)
    frame[['date', 'code']].to_parquet(tmp_path / 'scores.parquet', index=False)
    c = module.connection()
    assert c.sql('SELECT IP01,IP02 FROM features').fetchone() == (.08, -.22)
    assert c.sql('SELECT floor(100*IP01+10000+.000001)::INT FROM features').fetchone() == (10008,)
    c.close()
