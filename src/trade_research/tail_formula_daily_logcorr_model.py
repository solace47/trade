import json
from pathlib import Path
import sys

import pandas as pd

from . import tail_formula_daily_logcorr as inputs
from . import tail_formula_range_change_model as engine
from .corporate_cash import save_json, sha

engine.inputs = inputs

if __name__ == '__main__':
    if 'analyze' in sys.argv:
        joint = json.loads((inputs.ROOT / 'joint_selection_freeze.json').read_text())
        assert joint['passed'] and joint['protocol_sha256'] == sha(inputs.PROTOCOL)
        constant = Path('data/research/tail_formula_exchange_context_constant_2025')
        original = Path('data/research/tail_formula_before1000_model_2025')
        for root in [constant, original]:
            proof = json.loads((root / 'selection_verification.json').read_text())
            report = json.loads((root / 'selection_report.json').read_text())
            assert proof['passed'] and proof['selection_report_sha256'] == sha(root / 'selection_report.json')
            assert report['selection_sha256'] == sha(root / 'selection.parquet')
        pd.testing.assert_frame_equal(pd.read_parquet(constant / 'selection.parquet'),
                                      pd.read_parquet(original / 'selection.parquet'), check_exact=True)
        record = dict(passed=True, constant_selection_report_sha256=sha(constant / 'selection_report.json'),
            original_selection_report_sha256=sha(original / 'selection_report.json'),
            full_annual_frame_equal=True, original_analysis_also_represents_same_dimension_constant_control=True)
        path = inputs.ROOT / 'constant_control_verification.json'
        if path.exists():
            assert json.loads(path.read_text()) == record
        else:
            save_json(path, record)
    engine.main()
