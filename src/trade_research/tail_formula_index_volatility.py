"""Index-relative historical volatility alongside the unchanged 48 inputs."""
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
from .corporate_cash import save_json, sha

ROOT=Path('data/research/tail_formula_index_volatility')
PROTOCOL=Path('config/tail_formula_index_volatility_protocol.json')
COMBINED_PROTOCOL=Path('config/tail_formula_index_volatility_combined_protocol.json')
INDEX=Path('data/research/tail_formula_market')
DAYS=Path('data/research/tail_formula_volume_history')
CONTROL=Path('data/research/tail_formula_index_volatility_control')
OLD_SELECTION=Path('data/research/tail_formula_float_2025')
EXPRESSIONS={**previous.EXPRESSIONS,'M01':'IVOL',**{f'M{i+1:02d}':f'J{i:02d}/IVOL' for i in range(1,5)}}
HEADER=previous.HEADER
for i in range(20):
    HEADER+=f'IR{i+1:02d}:=100*(REF(INDEXC,B{i})/REF(INDEXC,B{i+1})-1);\n'
HEADER+='IVOL:=SQRT(('+ '+'.join(f'IR{i:02d}*IR{i:02d}' for i in range(1,21))+')/20);\n'


def checked_source():
    p=json.loads(PROTOCOL.read_text())
    for key,path in [('previous_feature_report_sha256',previous.ROOT/'feature_report.json'),
        ('index_source_report_sha256',INDEX/'index_source_report.json'),
        ('stock_day_manifest_sha256',DAYS/'input_manifest.json'),
        ('stock_day_verification_sha256',DAYS/'feature_verification.json')]:
        assert p[key]==sha(path)
    for root in [previous.ROOT,DAYS]:
        r=json.loads((root/'feature_report.json').read_text())
        proof=json.loads((root/'feature_verification.json').read_text())
        assert proof['passed'] and proof['feature_report_sha256']==sha(root/'feature_report.json')
        assert r['features_sha256']==sha(root/'features.parquet')
    m=json.loads((DAYS/'input_manifest.json').read_text())
    assert m['stock_days_sha256']==sha(DAYS/'stock_days.parquet')
    s=json.loads((INDEX/'index_source_report.json').read_text())
    assert s['indices_sha256']==sha(INDEX/'indices.parquet')
    for path,digest in s['raw_files_sha256'].items():
        assert sha(Path(path))==digest
    assert p['history_observations']==20 and p['signal_last']<'2026-01-01'
    return p,pd.read_parquet(previous.ROOT/'features.parquet'),pd.read_parquet(DAYS/'stock_days.parquet'),pd.read_parquet(INDEX/'indices.parquet')


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace frozen index-volatility inputs')
    p,old,days,indices=checked_source(); ROOT.mkdir(parents=True,exist_ok=True)
    days['index_code']=np.where(days.code.str.startswith('sh.'),'sh.000001','sz.399001')
    indices=indices.rename(columns={'code':'index_code'})
    calendar={date:i for i,date in enumerate(sorted(indices.date.unique()))}
    joined=days.merge(indices[['date','index_code','close']],on=['date','index_code'],how='left',validate='many_to_one')
    history=[]
    for code,q in joined.groupby('code',sort=True):
        q=q.sort_values('date'); closes=q.close
        change=100*(closes/closes.shift(1)-1)
        valid=np.isfinite(closes)&closes.gt(0)
        change=change.where(valid&valid.shift(1,fill_value=False))
        ranks=q.date.map(calendar); gaps=ranks-ranks.shift(1)
        f=q[['date','code']].copy()
        f['index_vol_first_date']=q.date.shift(21)
        f['index_vol_last_date']=q.date.shift(1)
        f['index_vol_count']=change.rolling(20,min_periods=20).count().shift(1)
        f['index_rms20']=np.sqrt(change.pow(2).rolling(20,min_periods=20).mean().shift(1))
        f['index_gap_count20']=gaps.gt(1).astype(float).rolling(20,min_periods=20).sum().shift(1)
        f['index_max_gap20']=gaps.rolling(20,min_periods=20).max().shift(1)
        history.append(f)
    h=pd.concat(history,ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    h.to_parquet(ROOT/'history.parquet',index=False,compression='zstd')
    f=old.merge(h,on=['date','code'],how='left',validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
    f['index_volatility_valid']=(f.index_vol_count.eq(20)&f.index_vol_first_date.notna()&f.index_vol_last_date.lt(f.date)
        &np.isfinite(f.index_rms20)&f.index_rms20.gt(0))
    f['M01']=f.index_rms20.where(f.index_volatility_valid)
    for i in range(1,5):
        f[f'M{i+1:02d}']=(f[f'J{i:02d}']/f.index_rms20).where(f.index_volatility_valid)
    f['prior_formula_input_valid']=f.formula_input_valid
    f['formula_input_valid'] &= f.index_volatility_valid&np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),previous_feature_report_sha256=sha(previous.ROOT/'feature_report.json'),
        index_source_report_sha256=sha(INDEX/'index_source_report.json'),stock_day_manifest_sha256=sha(DAYS/'input_manifest.json'),
        stock_day_verification_sha256=sha(DAYS/'feature_verification.json'),history_sha256=sha(ROOT/'history.parquet'),
        features_sha256=sha(ROOT/'features.parquet'),rows=len(f),valid=int(f.formula_input_valid.sum()),
        previous_valid=int(f.prior_formula_input_valid.sum()),newly_invalid=int((f.prior_formula_input_valid&~f.formula_input_valid).sum()),
        valid_with_index_gap=int((f.formula_input_valid&f.index_gap_count20.gt(0)).sum()),
        maximum_market_gap_in_valid_history=float(f.loc[f.formula_input_valid,'index_max_gap20'].max()),
        expressions=EXPRESSIONS,native_header=HEADER,native_source_parity_verified=False,
        outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',r)
    return {k:v for k,v in r.items() if k not in ['expressions','native_header']}


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
        assert not any((Path('data/research')/f'tail_formula_index_volatility_{fold}'/'analysis_report.json').exists()
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
        linkage.ROOT=Path('data/research/tail_formula_index_volatility_2024')
        linkage.H2=Path('data/research/tail_formula_index_volatility_recent')
        linkage.COMBINED=Path('data/research/tail_formula_index_volatility_2025')
        linkage.PROTOCOL=COMBINED_PROTOCOL
        linkage.H2_SELECTION_SHA=None;linkage.H2_MODEL_SHA=None;linkage.H2_OUTCOMES_PREVIOUSLY_SEEN=False
    else:
        previous.setup(fold)
        root=Path('data/research/tail_formula_index_volatility_'+fold)
        protocol=Path('config/tail_formula_index_volatility_'+fold+'_protocol.json')
        base.ROOT=root;base.PROTOCOL=protocol;relative.PROTOCOL=protocol;study.ROOT=root;study.PROTOCOL=protocol
        base.FEATURES=ROOT;base.EXPRESSIONS=EXPRESSIONS;base.HEADER=HEADER


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['features','model','verify_model','scores','freeze','verify','analyze',
        'freeze_control','verify_control','reuse_control'])
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
