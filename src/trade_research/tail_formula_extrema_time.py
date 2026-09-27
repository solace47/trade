"""Timing of the most recent closing-price extremes in the fixed late window."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as previous
from . import tail_formula_path_variance as window
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json, sha

ROOT=Path('data/research/tail_formula_extrema_time')
PROTOCOL=Path('config/tail_formula_extrema_time_protocol.json')
COMBINED_PROTOCOL=Path('config/tail_formula_extrema_time_combined_protocol.json')
CONTROL=Path('data/research/tail_formula_extrema_time_control')
OLD_SELECTION=Path('data/research/tail_formula_float_2025')
PRICE_COLUMNS=[f'pv_c{n:02d}' for n in range(21,50)]


def age_expression(peak):
    result='28'
    for i in reversed(range(28)):
        result=f'IF(PE{i:02d}={peak},{i},{result})'
    return '100*'+result+'/28'


NEW_EXPRESSIONS={'E01':age_expression('PH29'),'E02':age_expression('PL29')}
EXPRESSIONS={**previous.EXPRESSIONS,**NEW_EXPRESSIONS}
HEADER=previous.HEADER+'PH29:=VALUEWHEN(TIME=1449,HHV(C,29));\nPL29:=VALUEWHEN(TIME=1449,LLV(C,29));\n'
HEADER+='\n'.join(f'PE{i:02d}:=VALUEWHEN(TIME=1449,'+('C' if i==0 else f'REF(C,{i})')+');' for i in range(29))+'\n'


def checked_source():
    p=json.loads(PROTOCOL.read_text())
    for key,path in [('previous_feature_report_sha256',previous.ROOT/'feature_report.json'),
        ('window_feature_report_sha256',window.ROOT/'feature_report.json'),
        ('window_feature_verification_sha256',window.ROOT/'feature_verification.json')]:assert p[key]==sha(path)
    for source in [previous.ROOT,window.ROOT]:
        r=json.loads((source/'feature_report.json').read_text());v=json.loads((source/'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256']==sha(source/'feature_report.json')
        assert r['features_sha256']==sha(source/'features.parquet')
    old=pd.read_parquet(previous.ROOT/'features.parquet')
    prices=pd.read_parquet(window.ROOT/'features.parquet',columns=['date','code','path_variance_valid']+PRICE_COLUMNS)
    pd.testing.assert_frame_equal(old[['date','code']],prices[['date','code']],check_exact=True)
    return p,old,prices


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace frozen extrema timing inputs')
    p,old,prices=checked_source();ROOT.mkdir(parents=True,exist_ok=True)
    f=old.copy();a=prices[PRICE_COLUMNS].to_numpy(dtype=float)[:,::-1]
    valid=prices.path_variance_valid&np.isfinite(a).all(axis=1)&(a>0).all(axis=1)
    high_age=np.argmax(a,axis=1);low_age=np.argmin(a,axis=1)
    f['extrema_valid']=valid
    f['latest_high_age']=pd.Series(high_age,index=f.index,dtype=float).where(valid)
    f['latest_low_age']=pd.Series(low_age,index=f.index,dtype=float).where(valid)
    f['E01']=100*f.latest_high_age/28;f['E02']=100*f.latest_low_age/28
    f['prior_formula_input_valid']=f.formula_input_valid
    f['formula_input_valid'] &= valid&np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),previous_feature_report_sha256=sha(previous.ROOT/'feature_report.json'),
        window_feature_report_sha256=sha(window.ROOT/'feature_report.json'),
        window_feature_verification_sha256=sha(window.ROOT/'feature_verification.json'),
        features_sha256=sha(ROOT/'features.parquet'),rows=len(f),valid=int(f.formula_input_valid.sum()),
        previous_valid=int(f.prior_formula_input_valid.sum()),newly_invalid=int((f.prior_formula_input_valid&~f.formula_input_valid).sum()),
        flat_valid_windows=int((f.formula_input_valid&(a.max(axis=1)==a.min(axis=1))).sum()),
        expressions=EXPRESSIONS,native_header=HEADER,all_previous_48_inputs_retained=True,
        minute_variance_Z_inputs_not_used=True,software_compilation_verified=False,native_source_parity_verified=False,
        outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',r);return {k:v for k,v in r.items() if k not in ['expressions','native_header']}


def control(stage):
    r=json.loads((ROOT/'feature_report.json').read_text())
    proof=json.loads((ROOT/'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256']==sha(ROOT/'feature_report.json')
    assert r['features_sha256']==sha(ROOT/'features.parquet')
    assert sha(OLD_SELECTION/'selection_report.json')=='07de34caa25a339fa8e7b8d3665c7d04c044792d01fa170d0fd48f6b77b8f5c5'
    s=json.loads((OLD_SELECTION/'selection_report.json').read_text())
    proof=json.loads((OLD_SELECTION/'selection_verification.json').read_text())
    assert proof['passed'] and proof['selection_report_sha256']==sha(OLD_SELECTION/'selection_report.json')
    assert s['selection_sha256']==sha(OLD_SELECTION/'selection.parquet')
    if stage=='freeze_control':
        if (CONTROL/'selection_report.json').exists():
            raise ValueError('Do not replace same-quality selection')
        assert not any((Path('data/research')/f'tail_formula_extrema_time_{fold}'/'analysis_report.json').exists()
                       for fold in ['2024','recent','2025'])
        f=pd.read_parquet(ROOT/'features.parquet',columns=['date','code','formula_input_valid'])
        old=pd.read_parquet(OLD_SELECTION/'selection.parquet')
        pd.testing.assert_frame_equal(old[['date','code']],f[['date','code']],check_exact=True)
        out=old.copy();out['selected'] &= f.formula_input_valid
        CONTROL.mkdir(exist_ok=True);out.to_parquet(CONTROL/'selection.parquet',index=False,compression='zstd')
        result=dict(protocol_sha256=sha(COMBINED_PROTOCOL),feature_report_sha256=sha(ROOT/'feature_report.json'),
            original_selection_report_sha256=sha(OLD_SELECTION/'selection_report.json'),
            selection_sha256=sha(CONTROL/'selection.parquet'),selected=int(out.selected.sum()),
            unchanged_original_selection=out.equals(old),model_refitted=False,new_group_outcomes_read=False,
            year_2025_is_exploratory=True,new_2026_prices_read=False,no_exit_rules=True)
        save_json(CONTROL/'selection_report.json',result);return result
    report=json.loads((CONTROL/'selection_report.json').read_text())
    for key,path in [('protocol_sha256',COMBINED_PROTOCOL),('feature_report_sha256',ROOT/'feature_report.json'),
        ('original_selection_report_sha256',OLD_SELECTION/'selection_report.json'),('selection_sha256',CONTROL/'selection.parquet')]:
        assert report[key]==sha(path)
    c=base.conn()
    expected=c.sql(f'''SELECT s.date,s.code,s.half,s.board,s.decision_shares,s.selected AND f.formula_input_valid AS selected
        FROM read_parquet('{OLD_SELECTION}/selection.parquet') s JOIN read_parquet('{ROOT}/features.parquet') f USING(date,code)
        ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(CONTROL/'selection.parquet'),expected,check_exact=True)
    assert report['unchanged_original_selection']==expected.equals(pd.read_parquet(OLD_SELECTION/'selection.parquet'))
    if stage=='verify_control':
        proof=dict(passed=True,selection_report_sha256=sha(CONTROL/'selection_report.json'),rows=len(expected),
            all_control_selection_flags_rebuilt=True,outcomes_read=False,new_2026_prices_read=False)
        save_json(CONTROL/'selection_verification.json',proof);return proof
    assert stage=='reuse_control' and report['unchanged_original_selection']
    proof=json.loads((OLD_SELECTION/'analysis_verification.json').read_text())
    assert proof['passed'] and proof['analysis_report_sha256']==sha(OLD_SELECTION/'analysis_report.json')
    result=dict(selection_report_sha256=sha(CONTROL/'selection_report.json'),
        reused_analysis_report_sha256=sha(OLD_SELECTION/'analysis_report.json'),
        reused_analysis_verification_sha256=sha(OLD_SELECTION/'analysis_verification.json'),
        all_selection_keys_and_flags_identical=True,old_model_and_statistics_not_recomputed=True,
        passed=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(CONTROL/'analysis_reuse.json',result);return result


def setup(fold):
    if fold=='combined':
        linkage.ROOT=Path('data/research/tail_formula_extrema_time_2024')
        linkage.H2=Path('data/research/tail_formula_extrema_time_recent')
        linkage.COMBINED=Path('data/research/tail_formula_extrema_time_2025')
        linkage.PROTOCOL=COMBINED_PROTOCOL
        linkage.H2_SELECTION_SHA=None;linkage.H2_MODEL_SHA=None;linkage.H2_OUTCOMES_PREVIOUSLY_SEEN=False
    else:
        previous.setup(fold)
        root=Path('data/research/tail_formula_extrema_time_'+fold)
        protocol=Path('config/tail_formula_extrema_time_'+fold+'_protocol.json')
        base.ROOT=root;base.PROTOCOL=protocol;relative.PROTOCOL=protocol;study.ROOT=root;study.PROTOCOL=protocol
        base.FEATURES=ROOT;base.EXPRESSIONS=EXPRESSIONS;base.HEADER=HEADER


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['features','model','verify_model','scores','freeze','verify','analyze',
        'freeze_control','verify_control','reuse_control'])
    p.add_argument('--fold',choices=['2024','recent','combined'],default='2024');a=p.parse_args()
    if a.stage=='features':
        r=globals()[a.stage]()
    elif a.stage.endswith('_control'):
        r=control(a.stage)
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
