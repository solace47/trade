"""Frozen original fifty inputs and their already verified source tables."""
import json
from pathlib import Path
import pandas as pd

DEFINITION = json.loads(Path('config/tail_formula_baseline.json').read_text())
META = DEFINITION['META']
CONTROL = EXPRESSIONS = DEFINITION['EXPRESSIONS']
HEADER = DEFINITION['HEADER']
INPUTS = Path(DEFINITION['BASE_INPUTS'])
ROOT = Path(DEFINITION['BASE_ROOT'])
WINDOW_CLOCKS = DEFINITION['WINDOW_CLOCKS']
SEQUENCE_GUARD = DEFINITION['SEQUENCE_GUARD']

def original():
    # 2024 is present in both sources and must agree, rather than being
    # silently deduplicated. Historical stocks absent in 2024 are retained.
    a = pd.read_parquet('data/research/tail_formula_stock_2024/inputs/features.parquet',
                        columns=[*META, *CONTROL])
    b = pd.read_parquet(INPUTS / 'features.parquet', columns=[*META, *CONTROL])
    pd.testing.assert_frame_equal(a.loc[a.date.ge('2024-01-01')].reset_index(drop=True),
                                  b.loc[b.date.lt('2025-01-01')].reset_index(drop=True), check_exact=True)
    f = pd.concat([a.loc[a.date.lt('2024-01-01')], b], ignore_index=True)
    assert len(f) == 1815129 and not f.duplicated(['date', 'code']).any()
    assert f.date.ge('2023-01-01').all() and f.date.lt('2026-01-01').all()
    return f.sort_values(['date', 'code']).reset_index(drop=True)
