"""Keep only the direction of direct index predictors, without changing the pool."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as previous
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json,sha

ROOT=Path('data/research/tail_formula_index_direction')
PROTOCOL=Path('config/tail_formula_index_direction_protocol.json')
COMBINED_PROTOCOL=Path('config/tail_formula_index_direction_combined_protocol.json')
SOURCE_NAMES=[f'{prefix}{i:02d}' for prefix in ['I','J'] for i in range(1,5)]
NEW_EXPRESSIONS={'M'+name:f'IF({name}>0,1,0)' for name in SOURCE_NAMES}
EXPRESSIONS={**{k:v for k,v in previous.EXPRESSIONS.items() if k not in SOURCE_NAMES},**NEW_EXPRESSIONS}
HEADER=previous.HEADER+''.join(f'{name}:={previous.EXPRESSIONS[name]};\n' for name in SOURCE_NAMES)


def source():
    config=json.loads(PROTOCOL.read_text())
    r=json.loads((previous.ROOT/'feature_report.json').read_text())
    proof=json.loads((previous.ROOT/'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256']==sha(previous.ROOT/'feature_report.json')
    assert sha(previous.ROOT/'feature_report.json')==config['previous_feature_report_sha256']
    assert r['features_sha256']==sha(previous.ROOT/'features.parquet') and config['source_names']==SOURCE_NAMES
    assert len(EXPRESSIONS)==48 and all(f'R{i:02d}' in EXPRESSIONS for i in range(1,5))
    return r


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace frozen index-direction inputs')
    old=source();f=pd.read_parquet(previous.ROOT/'features.parquet')
    for name in SOURCE_NAMES:
        f['M'+name]=f[name].gt(0).astype(float).where(np.isfinite(f[name]))
    assert np.isfinite(f.loc[f.formula_input_valid,list(EXPRESSIONS)]).all().all()
    ROOT.mkdir(parents=True,exist_ok=True);f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),previous_feature_report_sha256=sha(previous.ROOT/'feature_report.json'),
        features_sha256=sha(ROOT/'features.parquet'),rows=old['rows'],valid=old['valid'],
        expressions=EXPRESSIONS,native_header=HEADER,previous_48_values_and_validity_unchanged=True,
        direct_continuous_index_inputs_replaced=SOURCE_NAMES,
        direction_counts={name:{'positive':int(f['M'+name].eq(1).sum()),'nonpositive':int(f['M'+name].eq(0).sum()),
            'unknown':int(f['M'+name].isna().sum())} for name in SOURCE_NAMES},
        outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True,native_source_parity_verified=False)
    save_json(ROOT/'feature_report.json',r)
    return {k:v for k,v in r.items() if k not in ['expressions','native_header']}


def verify_features():
    old_report=source();r=json.loads((ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL)
    assert r['previous_feature_report_sha256']==sha(previous.ROOT/'feature_report.json')
    assert r['features_sha256']==sha(ROOT/'features.parquet')
    assert (r['rows'],r['valid'])==(old_report['rows'],old_report['valid'])
    old=pd.read_parquet(previous.ROOT/'features.parquet');f=pd.read_parquet(ROOT/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns],old,check_exact=True)
    expected_expressions={k:v for k,v in old_report['expressions'].items() if k[:1] not in ('I','J')}
    for prefix in ['I','J']:
        for i in range(1,5):
            name=f'{prefix}{i:02d}';expected_expressions['M'+name]=f'IF({name}>0,1,0)'
    assert list(r['expressions'].items())==list(expected_expressions.items()) and len(expected_expressions)==48
    expected_header=old_report['native_header']+''.join(f'{prefix}{i:02d}:={old_report["expressions"][f"{prefix}{i:02d}"]};\n'
        for prefix in ['I','J'] for i in range(1,5))
    assert r['native_header']==expected_header
    declarations=re.findall(r'([A-Z][A-Z0-9]*):=',expected_header+''.join(f'{k}:={v};\n' for k,v in expected_expressions.items()))
    assert len(declarations)==len(set(declarations))
    for i in range(1,5):
        assert declarations.index(f'J{i:02d}')<declarations.index(f'R{i:02d}')
    c=base.conn();c.register('old',old)
    fields=','.join(f'CASE WHEN isfinite({name}) THEN CASE WHEN {name}>0 THEN 1.0 ELSE 0.0 END END AS M{name}' for name in SOURCE_NAMES)
    expected=c.sql('SELECT date,code,'+fields+' FROM old ORDER BY date,code').df();c.close()
    pd.testing.assert_frame_equal(f[expected.columns],expected,check_exact=True,check_dtype=False)
    valid=f.formula_input_valid
    expected_encoded=np.floor(expected.loc[valid,list(NEW_EXPRESSIONS)].to_numpy()*100+10000+.000001).astype('int32')
    native_encoded=np.where(expected.loc[valid,list(NEW_EXPRESSIONS)].eq(1).to_numpy(),10100,10000)
    np.testing.assert_array_equal(expected_encoded,native_encoded)
    assert np.isfinite(f.loc[valid,list(EXPRESSIONS)]).all().all() and len(f)==r['rows'] and int(valid.sum())==r['valid']
    for name in SOURCE_NAMES:
        assert r['direction_counts'][name]=={'positive':int(expected['M'+name].eq(1).sum()),
            'nonpositive':int(expected['M'+name].eq(0).sum()),'unknown':int(expected['M'+name].isna().sum())}
    proof=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=r['rows'],valid=r['valid'],
        previous_48_values_and_validity_unchanged=True,all_8_direction_inputs_and_encodings_rebuilt=True,
        all_48_expressions_and_index_intermediates_verified=True,
        outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True,native_client_values_verified=False)
    save_json(ROOT/'feature_verification.json',proof);return proof


def setup(fold):
    if fold=='combined':
        linkage.ROOT=Path('data/research/tail_formula_index_direction_2024')
        linkage.H2=Path('data/research/tail_formula_index_direction_recent')
        linkage.COMBINED=Path('data/research/tail_formula_index_direction_2025')
        linkage.PROTOCOL=COMBINED_PROTOCOL
        linkage.H2_SELECTION_SHA=None;linkage.H2_MODEL_SHA=None;linkage.H2_OUTCOMES_PREVIOUSLY_SEEN=False
    else:
        previous.setup(fold)
        root=Path('data/research/tail_formula_index_direction_'+fold);protocol=Path('config/tail_formula_index_direction_'+fold+'_protocol.json')
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
