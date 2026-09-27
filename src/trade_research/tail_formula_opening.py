"""Add the observed opening half-hour to the frozen tail/float feature family."""
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

ROOT=Path('data/research/tail_formula_opening')
PROTOCOL=Path('config/tail_formula_opening_protocol.json')
OPENING=Path('data/research/opening_cash_history')
CONTROL=Path('data/research/tail_formula_opening_control')
COMBINED_PROTOCOL=Path('config/tail_formula_opening_combined_protocol.json')
OLD_SELECTION=Path('data/research/tail_formula_float_2025')
NEW_EXPRESSIONS={'M01':'100*(P10/DYNAINFO(4)-1)/V01','M02':'100*(Q/P10-1)/V01','M03':'100*OC30/DA'}
EXPRESSIONS={**previous.EXPRESSIONS,**NEW_EXPRESSIONS}
HEADER=previous.HEADER+'P10:=VALUEWHEN(TIME=1000,C);\nOC30:=VALUEWHEN(TIME=1000,SUM(INTPART(AMO*100+0.5)/100,30));\n'
WINDOW_NAMES={'price_1000':'opening_1000_price','volume_1000':'opening_1000_volume',
    'opening_amount_cents':'opening_cash_cents','window_valid':'opening_window_valid'}


def source_reports():
    config=json.loads(PROTOCOL.read_text())
    old=json.loads((previous.ROOT/'feature_report.json').read_text())
    proof=json.loads((previous.ROOT/'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256']==sha(previous.ROOT/'feature_report.json')
    assert sha(previous.ROOT/'feature_report.json')==config['previous_feature_report_sha256']
    assert old['features_sha256']==sha(previous.ROOT/'features.parquet')
    assert sha(OPENING/'raw_report.json')==config['opening_raw_report_sha256']
    assert sha(OPENING/'feature_verification.json')==config['opening_feature_verification_sha256']
    proof=json.loads((OPENING/'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256']==sha(OPENING/'feature_report.json')
    assert json.loads((OPENING/'feature_report.json').read_text())['raw_report_sha256']==sha(OPENING/'raw_report.json')
    raw=json.loads((OPENING/'raw_report.json').read_text())
    assert raw['manifest_sha256']==sha(OPENING/'manifest.json')
    for path,h in raw['parts_sha256'].items():
        assert sha(Path(path))==h
    return raw


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace frozen opening inputs')
    raw=source_reports();old=pd.read_parquet(previous.ROOT/'features.parquet')
    columns=['date','code',*WINDOW_NAMES]
    windows=pd.concat([pd.read_parquet(p,columns=columns) for p in raw['parts_sha256']],ignore_index=True)
    windows=windows.loc[windows.date.between('2024-01-01','2025-12-30')].rename(columns=WINDOW_NAMES)
    assert not windows.duplicated(['date','code']).any()
    f=old.merge(windows,on=['date','code'],how='left',validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
    cash=f.opening_cash_cents/100
    f['opening_source_valid']=(f.opening_window_valid.fillna(False).astype(bool)&f.opening_1000_price.gt(0)
        &f.opening_1000_volume.gt(0)&cash.gt(0)&cash.le(f.amount_1449+.151)
        &f.daily_open.gt(0)&f.V01.gt(0)&f.amount_1449.gt(0)
        &np.isfinite(f[['opening_1000_price','opening_1000_volume','opening_cash_cents','daily_open','V01','amount_1449']]).all(axis=1))
    f['M01']=(100*(f.opening_1000_price/f.daily_open-1)/f.V01).where(f.opening_source_valid)
    f['M02']=(100*(f.price_1449/f.opening_1000_price-1)/f.V01).where(f.opening_source_valid)
    f['M03']=(100*cash/f.amount_1449).where(f.opening_source_valid)
    f['prior_formula_input_valid']=f.formula_input_valid
    f['formula_input_valid'] &= f.opening_source_valid&np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True,exist_ok=True);f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),previous_feature_report_sha256=sha(previous.ROOT/'feature_report.json'),
        opening_raw_report_sha256=sha(OPENING/'raw_report.json'),opening_feature_verification_sha256=sha(OPENING/'feature_verification.json'),
        features_sha256=sha(ROOT/'features.parquet'),rows=len(f),valid=int(f.formula_input_valid.sum()),
        previous_valid=int(old.formula_input_valid.sum()),newly_invalid=int((f.prior_formula_input_valid&~f.formula_input_valid).sum()),
        missing_opening_window=int(f.opening_window_valid.isna().sum()),
        native_expressions=EXPRESSIONS,native_header=HEADER,source_start=f.date.min(),source_end=f.date.max(),
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
            for name in ['tail_formula_opening_2024','tail_formula_opening_recent','tail_formula_opening_2025'])
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
        linkage.ROOT=Path('data/research/tail_formula_opening_2024')
        linkage.H2=Path('data/research/tail_formula_opening_recent')
        linkage.COMBINED=Path('data/research/tail_formula_opening_2025')
        linkage.PROTOCOL=COMBINED_PROTOCOL
        linkage.H2_SELECTION_SHA=None;linkage.H2_MODEL_SHA=None;linkage.H2_OUTCOMES_PREVIOUSLY_SEEN=False
    else:
        previous.setup(fold)
        root=Path('data/research/tail_formula_opening_'+fold);protocol=Path('config/tail_formula_opening_'+fold+'_protocol.json')
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
