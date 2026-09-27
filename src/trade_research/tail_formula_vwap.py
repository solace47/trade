"""Whole-day and late-vs-earlier transaction-average inputs for the frozen next-morning folds."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as previous
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json,sha

ROOT=Path('data/research/tail_formula_vwap')
PROTOCOL=Path('config/tail_formula_vwap_protocol.json')
CONTROL=Path('data/research/tail_formula_vwap_control')
COMBINED_PROTOCOL=Path('config/tail_formula_vwap_combined_protocol.json')
OLD_SELECTION=Path('data/research/tail_formula_float_2025')
NEW_EXPRESSIONS={'W01':'100*(Q/VW0-1)/V01','W02':'100*(VWT/VWE-1)/V01'}
EXPRESSIONS={**previous.EXPRESSIONS,**NEW_EXPRESSIONS}
HEADER=previous.HEADER+"""TA29:=VALUEWHEN(TIME=1449,SUM(AMOUNT,29));
TV29:=VALUEWHEN(TIME=1449,SUM(V,29));
VW0:=DA/(100*DV);
VWT:=TA29/(100*TV29);
VWE:=(DA-TA29)/(100*(DV-TV29));
"""
INTRADAY=Path('data/research/tail_formula_intraday')


def checked_source():
    config=json.loads(PROTOCOL.read_text())
    for key,path in [('previous_feature_report_sha256',previous.ROOT/'feature_report.json'),
        ('daily_feature_report_sha256',base.SOURCE/'feature_report.json'),
        ('intraday_feature_report_sha256',INTRADAY/'feature_report.json')]:
        assert config[key]==sha(path)
        proof=json.loads(path.with_name('feature_verification.json').read_text())
        assert proof['passed'] and proof['feature_report_sha256']==sha(path)
    r=json.loads((previous.ROOT/'feature_report.json').read_text())
    assert r['features_sha256']==sha(previous.ROOT/'features.parquet')
    return pd.read_parquet(previous.ROOT/'features.parquet')


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace frozen transaction-average inputs')
    f=checked_source()
    f['vwap_day']=f.amount_1449/f.volume_1449
    f['vwap_early']=(f.amount_1449-f.a29)/(f.volume_1449-f.v29)
    f['vwap_late']=f.a29/f.v29
    finite=np.isfinite(f[['price_1449','amount_1449','volume_1449','a29','v29','high_1449','low_1449',
        'V01','vwap_day','vwap_early','vwap_late']]).all(axis=1)
    f['vwap_source_valid']=(finite&f.amount_1449.gt(f.a29)&f.volume_1449.gt(f.v29)
        &f.a29.gt(0)&f.v29.gt(0)&f.V01.gt(0)&f.price_1449.gt(0)
        &f.vwap_day.ge(f.low_1449-.0101)&f.vwap_day.le(f.high_1449+.0101)
        &f.vwap_early.ge(f.low_1449-.0101)&f.vwap_early.le(f.high_1449+.0101))
    f['W01']=(100*(f.price_1449/f.vwap_day-1)/f.V01).where(f.vwap_source_valid)
    f['W02']=(100*(f.vwap_late/f.vwap_early-1)/f.V01).where(f.vwap_source_valid)
    f['prior_formula_input_valid']=f.formula_input_valid
    f['formula_input_valid'] &= f.vwap_source_valid&np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True,exist_ok=True);f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),previous_feature_report_sha256=sha(previous.ROOT/'feature_report.json'),
        daily_feature_report_sha256=sha(base.SOURCE/'feature_report.json'),
        intraday_feature_report_sha256=sha(INTRADAY/'feature_report.json'),features_sha256=sha(ROOT/'features.parquet'),
        rows=len(f),valid=int(f.formula_input_valid.sum()),previous_valid=int(f.prior_formula_input_valid.sum()),
        newly_invalid=int((f.prior_formula_input_valid&~f.formula_input_valid).sum()),
        invalid_vwap_sources=int((~f.vwap_source_valid).sum()),native_expressions=EXPRESSIONS,native_header=HEADER,
        last_input_clock='14:49',native_volume_unit='lots; multiply by 100 for shares',
        outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True,native_source_parity_verified=False)
    save_json(ROOT/'feature_report.json',r)
    return {k:v for k,v in r.items() if k not in ['native_expressions','native_header']}


def control(stage):
    check=json.loads((ROOT/'feature_verification.json').read_text())
    assert check['passed'] and check['feature_report_sha256']==sha(ROOT/'feature_report.json')
    report=json.loads((ROOT/'feature_report.json').read_text())
    assert report['features_sha256']==sha(ROOT/'features.parquet')
    assert sha(OLD_SELECTION/'selection_report.json')=='07de34caa25a339fa8e7b8d3665c7d04c044792d01fa170d0fd48f6b77b8f5c5'
    old_report=json.loads((OLD_SELECTION/'selection_report.json').read_text())
    assert old_report['selection_sha256']==sha(OLD_SELECTION/'selection.parquet')
    old_proof=json.loads((OLD_SELECTION/'selection_verification.json').read_text())
    assert old_proof['passed'] and old_proof['selection_report_sha256']==sha(OLD_SELECTION/'selection_report.json')
    if stage=='freeze_control':
        if (CONTROL/'selection_report.json').exists():
            raise ValueError('Do not replace the same-input-quality control')
        assert not any((Path('data/research')/name/'analysis_report.json').exists()
            for name in ['tail_formula_vwap_2024','tail_formula_vwap_recent','tail_formula_vwap_2025'])
        f=pd.read_parquet(ROOT/'features.parquet',columns=['date','code','formula_input_valid'])
        out=pd.read_parquet(OLD_SELECTION/'selection.parquet');pd.testing.assert_frame_equal(out[['date','code']],f[['date','code']])
        out['selected'] &= f.formula_input_valid
        CONTROL.mkdir(parents=True,exist_ok=True);out.to_parquet(CONTROL/'selection.parquet',index=False,compression='zstd')
        r=dict(protocol_sha256=sha(COMBINED_PROTOCOL),feature_report_sha256=sha(ROOT/'feature_report.json'),
            original_selection_report_sha256=sha(OLD_SELECTION/'selection_report.json'),
            selection_sha256=sha(CONTROL/'selection.parquet'),selected=int(out.selected.sum()),
            selection_is_original_48_and_new_input_valid=True,model_refitted=False,year_2025_is_exploratory=True,
            new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
        save_json(CONTROL/'selection_report.json',r);return r
    r=json.loads((CONTROL/'selection_report.json').read_text())
    assert r['protocol_sha256']==sha(COMBINED_PROTOCOL) and r['feature_report_sha256']==sha(ROOT/'feature_report.json')
    assert r['original_selection_report_sha256']==sha(OLD_SELECTION/'selection_report.json')
    assert r['selection_sha256']==sha(CONTROL/'selection.parquet')
    if stage=='verify_control':
        c=base.conn()
        expected=c.sql(f'''SELECT s.date,s.code,s.half,s.board,s.decision_shares,
            s.selected AND f.formula_input_valid AS selected
            FROM read_parquet('{OLD_SELECTION}/selection.parquet') s
            JOIN read_parquet('{ROOT}/features.parquet') f USING(date,code) ORDER BY date,code''').df()
        pd.testing.assert_frame_equal(pd.read_parquet(CONTROL/'selection.parquet'),expected,check_exact=True)
        assert int(expected.selected.sum())==r['selected']
        proof=dict(passed=True,selection_report_sha256=sha(CONTROL/'selection_report.json'),rows=len(expected),
            all_control_selection_flags_rebuilt=True,outcomes_read=False,new_2026_prices_read=False)
        save_json(CONTROL/'selection_verification.json',proof);return proof
    assert stage=='analyze_control'
    return linkage.common_analysis(CONTROL,COMBINED_PROTOCOL)


def setup(fold):
    if fold=='combined':
        linkage.ROOT=Path('data/research/tail_formula_vwap_2024')
        linkage.H2=Path('data/research/tail_formula_vwap_recent')
        linkage.COMBINED=Path('data/research/tail_formula_vwap_2025')
        linkage.PROTOCOL=COMBINED_PROTOCOL
        linkage.H2_SELECTION_SHA=None;linkage.H2_MODEL_SHA=None;linkage.H2_OUTCOMES_PREVIOUSLY_SEEN=False
    else:
        previous.setup(fold)
        root=Path('data/research/tail_formula_vwap_'+fold);protocol=Path('config/tail_formula_vwap_'+fold+'_protocol.json')
        base.ROOT=root;base.PROTOCOL=protocol;relative.PROTOCOL=protocol;study.ROOT=root;study.PROTOCOL=protocol
        base.FEATURES=ROOT;base.EXPRESSIONS=EXPRESSIONS;base.HEADER=HEADER


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['features','model','verify_model','scores','freeze','verify','analyze',
        'freeze_control','verify_control','analyze_control'])
    p.add_argument('--fold',choices=['2024','recent','combined'],default='2024');a=p.parse_args()
    if a.stage=='features':
        r=features()
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
