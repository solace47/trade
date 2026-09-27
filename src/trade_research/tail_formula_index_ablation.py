"""Remove direct index predictors while preserving stock-relative index context."""
import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as previous
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json,sha

ROOT=Path('data/research/tail_formula_index_ablation')
PROTOCOL=Path('config/tail_formula_index_ablation_protocol.json')
COMBINED_PROTOCOL=Path('config/tail_formula_index_ablation_combined_protocol.json')
REMOVED=[f'{prefix}{i:02d}' for prefix in ['I','J'] for i in range(1,5)]
EXPRESSIONS={k:v for k,v in previous.EXPRESSIONS.items() if k not in REMOVED}
HEADER=previous.HEADER+''.join(f'{name}:={previous.EXPRESSIONS[name]};\n' for name in REMOVED if name.startswith('J'))


def source():
    config=json.loads(PROTOCOL.read_text())
    r=json.loads((previous.ROOT/'feature_report.json').read_text())
    proof=json.loads((previous.ROOT/'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256']==sha(previous.ROOT/'feature_report.json')
    assert sha(previous.ROOT/'feature_report.json')==config['previous_feature_report_sha256']
    assert r['features_sha256']==sha(previous.ROOT/'features.parquet') and config['removed']==REMOVED
    assert len(EXPRESSIONS)==40 and all(f'R{i:02d}' in EXPRESSIONS for i in range(1,5))
    return r


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace frozen ablation inputs')
    old=source();ROOT.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(previous.ROOT/'features.parquet',ROOT/'features.parquet')
    assert sha(ROOT/'features.parquet')==old['features_sha256']
    r=dict(protocol_sha256=sha(PROTOCOL),previous_feature_report_sha256=sha(previous.ROOT/'feature_report.json'),
        features_sha256=sha(ROOT/'features.parquet'),rows=old['rows'],valid=old['valid'],removed_direct_inputs=REMOVED,
        expressions=EXPRESSIONS,native_header=HEADER,feature_values_and_validity_unchanged=True,
        outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True,native_source_parity_verified=False)
    save_json(ROOT/'feature_report.json',r)
    return {k:v for k,v in r.items() if k not in ['expressions','native_header']}


def verify_features():
    old=source();r=json.loads((ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL)
    assert r['previous_feature_report_sha256']==sha(previous.ROOT/'feature_report.json')
    assert r['features_sha256']==sha(ROOT/'features.parquet')==sha(previous.ROOT/'features.parquet')
    assert (r['rows'],r['valid'])==(old['rows'],old['valid'])
    # Independently select names by their documented families, not the removal expression above.
    expected={k:v for k,v in old['expressions'].items() if k[:1] not in ('I','J')}
    assert list(r['expressions'].items())==list(expected.items()) and len(expected)==40
    expected_header=old['native_header']+''.join(f'J{i:02d}:={old["expressions"][f"J{i:02d}"]};\n' for i in range(1,5))
    assert r['native_header']==expected_header
    import re
    declarations=re.findall(r'([A-Z][A-Z0-9]*):=',expected_header+''.join(f'{k}:={v};\n' for k,v in expected.items()))
    assert len(declarations)==len(set(declarations)), 'No duplicate intermediate definitions'
    for i in range(1,5):
        assert declarations.index(f'J{i:02d}')<declarations.index(f'R{i:02d}')
    proof=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=r['rows'],valid=r['valid'],
        original_parquet_byte_identical=True,all_40_inputs_and_J_intermediates_verified=True,
        outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True,native_client_values_verified=False)
    save_json(ROOT/'feature_verification.json',proof);return proof


def setup(fold):
    if fold=='combined':
        linkage.ROOT=Path('data/research/tail_formula_index_ablation_2024')
        linkage.H2=Path('data/research/tail_formula_index_ablation_recent')
        linkage.COMBINED=Path('data/research/tail_formula_index_ablation_2025')
        linkage.PROTOCOL=COMBINED_PROTOCOL
        linkage.H2_SELECTION_SHA=None;linkage.H2_MODEL_SHA=None;linkage.H2_OUTCOMES_PREVIOUSLY_SEEN=False
    else:
        previous.setup(fold)
        root=Path('data/research/tail_formula_index_ablation_'+fold);protocol=Path('config/tail_formula_index_ablation_'+fold+'_protocol.json')
        base.ROOT=root;base.PROTOCOL=protocol;relative.PROTOCOL=protocol;study.ROOT=root;study.PROTOCOL=protocol
        base.FEATURES=ROOT;base.EXPRESSIONS=EXPRESSIONS;base.HEADER=HEADER


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['features','model','verify_model','scores','freeze','verify','analyze',
        'verify_features'])
    p.add_argument('--fold',choices=['2024','recent','combined'],default='2024');a=p.parse_args()
    if a.stage in ['features','verify_features']:
        r=globals()[a.stage]()
    else:
        setup(a.fold)
        if a.fold=='combined':
            assert a.stage in ['freeze','verify','analyze']
            r=(linkage.common_analysis(linkage.COMBINED,linkage.PROTOCOL) if a.stage=='analyze'
                else getattr(linkage,'combine' if a.stage=='freeze' else 'verify_combined')())
        elif a.stage in ['model','verify_model']:
            r=getattr(relative,a.stage)('relative')
        elif a.stage in ['freeze','verify']:
            r=getattr(study,a.stage)()
        else:
            r=getattr(base,a.stage)()
    print(json.dumps(r,ensure_ascii=False,indent=2))
