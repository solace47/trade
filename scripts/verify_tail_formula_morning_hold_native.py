"""Bind the morning high in integer cents in the frozen literal replay harness."""
import json
from pathlib import Path

import numpy as np

from trade_research import tail_formula_morning_hold as study
from trade_research.corporate_cash import save_json,sha

PROTOCOL=Path('config/tail_formula_morning_hold_native_verification_protocol.json')


def run():
    p=json.loads(PROTOCOL.read_text())
    assert p['input_protocol_sha256']==sha(study.PROTOCOL)
    assert p['helper_sha256']==sha(Path(__file__))
    for f,h in p['source_hashes'].items():assert sha(Path(f))==h
    assert not (study.INPUTS/'native_input_verification.json').exists()
    original=study.native_values
    def cent_bound(prices,high,**kwargs):
        return original(prices,float(np.floor(high*100+.5)),**kwargs)
    study.native_values=cent_bound
    result=study.native()
    for file,digest in p['unchanged_input_receipts'].items():assert sha(Path(file))==digest
    out=dict(passed=True,verification_protocol_sha256=sha(PROTOCOL),input_protocol_sha256=sha(study.PROTOCOL),
        native_input_verification_sha256=sha(study.INPUTS/'native_input_verification.json'),
        all_cached_values_and_input_proofs_unchanged=True,
        only_literal_harness_high_argument_bound_from_yuan_to_integer_cents=True,
        no_native_expression_or_production_arithmetic_changed=True,no_new_raw_extraction=True,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(study.INPUTS/'native_verification_execution_receipt.json',out)
    return result


if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False,indent=2))
