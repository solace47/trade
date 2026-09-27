"""A fixed recent-large-advance research universe, separate from input quality."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_large_gain as source
from . import tail_formula_prior_day as adapter
from .corporate_cash import save_json, sha

STEM='tail_formula_gain_population'
ROOT=Path('data/research')/STEM
PROTOCOL=Path('config')/(STEM+'_protocol.json')
COMBINED_PROTOCOL=Path('config')/(STEM+'_combined_protocol.json')
POPULATION=Path('data/research')/(STEM+'_population')
EXPRESSIONS=source.EXPRESSIONS
HEADER=source.HEADER
original_native_core=base.native_core


def native_core(*args,**kwargs):
    text=original_native_core(*args,**kwargs)
    assert text.count('CORE:SC>')==1
    return text.replace('CORE:SC>','CORE:HG01>=1 AND SC>')


def checked_source():
    p=json.loads(PROTOCOL.read_text());r=json.loads((source.ROOT/'feature_report.json').read_text())
    v=json.loads((source.ROOT/'feature_verification.json').read_text())
    assert p['source_feature_report_sha256']==sha(source.ROOT/'feature_report.json')
    assert p['source_feature_verification_sha256']==sha(source.ROOT/'feature_verification.json')
    assert v['passed'] and v['feature_report_sha256']==p['source_feature_report_sha256']
    assert r['features_sha256']==sha(source.ROOT/'features.parquet')
    return r


def features():
    checked_source();assert not (ROOT/'feature_report.json').exists()
    f=pd.read_parquet(source.ROOT/'features.parquet')
    f['reliable_input_valid']=f.formula_input_valid
    f['research_population_eligible']=f.HG01.ge(1)
    f['formula_input_valid'] &= f.research_population_eligible
    ROOT.mkdir(parents=True,exist_ok=True);f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),source_feature_report_sha256=sha(source.ROOT/'feature_report.json'),
        features_sha256=sha(ROOT/'features.parquet'),rows=len(f),reliable_inputs=int(f.reliable_input_valid.sum()),
        valid=int(f.formula_input_valid.sum()),new_quality_invalid=0,
        excluded_by_research_population=int((f.reliable_input_valid & ~f.research_population_eligible).sum()),
        quality_and_research_population_separately_recorded=True,all_50_input_values_unchanged=True,
        expressions=EXPRESSIONS,native_header=HEADER,native_core_gate='HG01>=1',
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',r);return {k:v for k,v in r.items() if k not in ['expressions','native_header']}


def verify_features():
    checked_source();r=json.loads((ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['features_sha256']==sha(ROOT/'features.parquet')
    old=pd.read_parquet(source.ROOT/'features.parquet');actual=pd.read_parquet(ROOT/'features.parquet')
    pd.testing.assert_frame_equal(actual[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    c=base.conn();expected=c.sql(f'''SELECT date,code,formula_input_valid AS reliable_input_valid,
        coalesce(HG01>=1,false) AS research_population_eligible,
        formula_input_valid AND coalesce(HG01>=1,false) AS formula_input_valid
        FROM read_parquet('{source.ROOT}/features.parquet') ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(actual[expected.columns],expected,check_exact=True)
    assert r['valid']==int(actual.formula_input_valid.sum()) and r['reliable_inputs']==int(actual.reliable_input_valid.sum())
    assert r['excluded_by_research_population']==int((actual.reliable_input_valid&~actual.research_population_eligible).sum())
    selected=actual.loc[actual.formula_input_valid,list(EXPRESSIONS)]
    assert np.isfinite(selected).all().all() and selected.HG01.ge(1).all()
    proof=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=len(actual),valid=r['valid'],
        all_population_flags_independently_rebuilt=True,all_50_values_and_quality_flags_unchanged=True,
        numerical_validity_now_includes_explicit_research_population=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_verification.json',proof);return proof


def configure():
    checked_source();v=json.loads((ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256']==sha(ROOT/'feature_report.json')
    adapter.STEM=STEM;adapter.ROOT=ROOT;adapter.PROTOCOL=PROTOCOL;adapter.COMBINED_PROTOCOL=COMBINED_PROTOCOL
    adapter.CONTROL=Path('data/research')/(STEM+'_control');adapter.EXPRESSIONS=EXPRESSIONS;adapter.HEADER=HEADER
    base.native_core=native_core
    for fold in ['2024','recent','combined']:
        p=json.loads((Path('config')/(STEM+'_'+fold+'_protocol.json')).read_text())
        assert p['inputs_protocol_sha256']==sha(PROTOCOL)


def population(stage):
    configure();root=POPULATION
    c=base.conn();expected=c.sql(f'''SELECT date,code,half,board,decision_shares,
        date>='2025-01-01' AND date<'2026-01-01' AND formula_input_valid AS selected
        FROM read_parquet('{ROOT}/features.parquet') ORDER BY date,code''').df();c.close()
    if stage=='freeze':
        assert not (root/'selection_report.json').exists()
        assert not any((Path('data/research')/(STEM+'_'+fold)/'analysis_report.json').exists() for fold in ['2024','recent','2025','control','population'])
        root.mkdir(exist_ok=True);expected.to_parquet(root/'selection.parquet',index=False,compression='zstd')
        result=dict(protocol_sha256=sha(COMBINED_PROTOCOL),feature_report_sha256=sha(ROOT/'feature_report.json'),
            selection_sha256=sha(root/'selection.parquet'),selected=int(expected.selected.sum()),
            days=expected.loc[expected.selected,'date'].nunique(),arm='all_eligible_population',
            comparison_only_not_a_delivered_formula=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
        save_json(root/'selection_report.json',result);return result
    assert stage=='verify';r=json.loads((root/'selection_report.json').read_text())
    assert r['selection_sha256']==sha(root/'selection.parquet') and r['feature_report_sha256']==sha(ROOT/'feature_report.json')
    actual=pd.read_parquet(root/'selection.parquet')
    pd.testing.assert_frame_equal(actual,expected,check_exact=True)
    # Separate raw-column construction avoids checking only an SQL expression against itself.
    f=pd.read_parquet(ROOT/'features.parquet')
    np.testing.assert_array_equal(actual.selected,f.date.ge('2025-01-01')&f.date.lt('2026-01-01')&f.reliable_input_valid&f.HG01.ge(1))
    proof=dict(passed=True,selection_report_sha256=sha(root/'selection_report.json'),rows=len(expected),
        complete_population_flags_independently_rebuilt=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'selection_verification.json',proof);return proof


if __name__=='__main__':
    from . import tail_formula_context_2024 as linkage
    from . import tail_formula_recent as study
    from . import tail_formula_relative as relative
    from .tail_formula_offset_logit48 import verify_scores
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['features','verify_features','model','verify_model','scores','verify_scores','freeze','verify','analyze'])
    p.add_argument('--fold',choices=['2024','recent','combined','control','population'],default='2024');a=p.parse_args()
    if a.stage in ['features','verify_features']:
        result=globals()[a.stage]()
    else:
        configure()
        if a.stage=='analyze':
            assert all((Path('data/research')/(STEM+'_'+f)/'selection_verification.json').exists() for f in ['2024','recent','2025','control','population'])
        if a.fold in ['control','population']:
            assert a.stage in ['freeze','verify','analyze']
            root=adapter.CONTROL if a.fold=='control' else POPULATION
            result=linkage.common_analysis(root,COMBINED_PROTOCOL) if a.stage=='analyze' else (adapter.control(a.stage+'_control') if a.fold=='control' else population(a.stage))
        else:
            adapter.setup(a.fold)
            if a.fold=='combined':
                assert a.stage in ['freeze','verify','analyze']
                result=linkage.common_analysis(linkage.COMBINED,linkage.PROTOCOL) if a.stage=='analyze' else getattr(linkage,'combine' if a.stage=='freeze' else 'verify_combined')()
            elif a.stage in ['model','verify_model']:
                result=getattr(relative,a.stage)('relative')
            elif a.stage=='verify_scores':
                result=verify_scores()
            elif a.stage in ['freeze','verify']:
                result=getattr(study,a.stage)()
            else:
                result=getattr(base,a.stage)()
    print(json.dumps(result,ensure_ascii=False,indent=2))
