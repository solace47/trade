"""Check dated classification against the existing cash engine and modern labels."""
import json

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research.corporate_cash import save_json, sha
from trade_research.fixed_return_ranges import dated_economic_return
from trade_research.tail_formula_1000_analysis import BUY_COLUMNS, OBS_COLUMNS, classify as modern_classify
from trade_research.tail_formula_long48_inputs import ROOT, PROTOCOL, policy
from trade_research.tail_formula_long48_labels import classify


def main():
    policy(); source = 'data/research/tail_formula_1000/full_labels.parquet'
    columns = list(dict.fromkeys(BUY_COLUMNS+OBS_COLUMNS))
    c = base.conn()
    r = c.sql(f"SELECT {','.join(columns)} FROM read_parquet('{source}') WHERE known15 ORDER BY sha256(date||code||'long48-tax-v1') LIMIT 256").df()
    c.close()
    modern = modern_classify(r); dated = classify(r)
    pd.testing.assert_frame_equal(modern,dated,check_exact=True)
    checks = 0
    cases = [('2022-04-27','2022-04-28'),('2022-04-28','2022-04-29'),('2022-04-29','2022-05-05'),
             ('2023-08-24','2023-08-25'),('2023-08-25','2023-08-28'),('2023-08-28','2023-08-29')]
    for first,last in cases:
        example = r.assign(date=first,next_date=last); got = classify(example)
        assert got.known15.all()
        for bps in [5,15]:
            buy = example.entry_vwap+np.maximum(example.entry_vwap*bps/10000,.005)
            for name,price in [('sustained',example.sustained_close),('any_close',example.max_close),
                               ('mark_1000',example.price_1000),('adverse',example.min_low)]:
                sale = price-np.maximum(price*bps/10000,.005)
                expected = dated_economic_return(example.decision_shares,example.decision_shares,buy,sale,
                    0.,0.,example.date,example.next_date)
                np.testing.assert_allclose(got[f'{name}_return{bps}'],expected,rtol=0,atol=3e-15)
                checks += len(expected)
    from pathlib import Path
    proof = dict(passed=True,protocol_sha256=sha(PROTOCOL),classification_source_sha256=sha(Path('src/trade_research/tail_formula_long48_labels.py')),
        modern_identity_rows=len(r),modern_all_columns_exact=True,synthetic_date_pairs=cases,
        dated_cash_engine_comparisons=checks,synthetic_checks_are_not_strategy_results=True,
        source_sha256=sha(Path(source)),new_2026_prices_read=False)
    save_json(ROOT/'tax_verification.json',proof); print(json.dumps(proof,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
