"""Reuse the audited collector for only the two frozen Q1 size indices."""
import argparse
import json

import pandas as pd

import collect_tail_formula_forward_indices as source
from trade_research import tail_formula_size_agreement_q1 as study


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['daily', 'minutes'])
    a = parser.parse_args(); p = study.policy(); study.checked_model()
    source.ROOT = study.ROOT; source.OUT = study.ROOT / 'indices'; source.PROTOCOL = study.PROTOCOL
    source.CODES = p['symbols']; source.policy = study.policy; source.checked_model = study.checked_model
    original_calendar = source.calendar
    def fixed_calendar(first, last):
        if (first, last) == (p['signal_first'], p['signal_last']):
            old = pd.read_parquet(study.OLD / 'inputs/features.parquet', columns=['date'])
            days = sorted(old.date.unique())
            assert len(days) == 56 and set(days) <= set(original_calendar(first, last))
            return days
        assert (first, last) == (p['warmup_first'], p['signal_last'])
        return original_calendar(first, last)
    source.calendar = fixed_calendar
    print(json.dumps(getattr(source, a.stage)(), ensure_ascii=False, indent=2))
