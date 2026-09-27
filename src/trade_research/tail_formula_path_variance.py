"""Late-minute variance intensity, downside share and change as native inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_context_2024 as linkage
from . import tail_formula_float as previous
from . import tail_formula_intraday as intraday
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import MINUTES, save_json, sha

ROOT=Path('data/research/tail_formula_path_variance')
PROTOCOL=Path('config/tail_formula_path_variance_protocol.json')
COMBINED_PROTOCOL=Path('config/tail_formula_path_variance_combined_protocol.json')
CONTROL=Path('data/research/tail_formula_path_variance_control')
OLD_SELECTION=Path('data/research/tail_formula_float_2025')
MANIFEST=Path('data/research/economic_winner/input_manifest.json')
PRICE_COLUMNS=[f'pv_c{n:02d}' for n in range(20,50)]
NEW_EXPRESSIONS={
    'Z01':'VALUEWHEN(TIME=1449,SQRT(PV29/29))/V01',
    'Z02':'VALUEWHEN(TIME=1449,PD29/MAX(PV29,0.000000000001))',
    'Z03':'VALUEWHEN(TIME=1449,(PV14-PV14P)/MAX(PV14+PV14P,0.000000000001))'}
EXPRESSIONS={**previous.EXPRESSIONS,**NEW_EXPRESSIONS}
HEADER=previous.HEADER+"PRT:=100*LN(C/REF(C,1));\nPV29:=SUM(PRT*PRT,29);\nPD29:=SUM(IF(PRT<0,PRT*PRT,0),29);\nPV14:=SUM(PRT*PRT,14);\nPV14P:=REF(PV14,14);\n"


def checked_source():
    p=json.loads(PROTOCOL.read_text())
    for key,path in [('previous_feature_report_sha256',previous.ROOT/'feature_report.json'),
        ('intraday_feature_report_sha256',intraday.ROOT/'feature_report.json'),('minute_manifest_sha256',MANIFEST)]:
        assert p[key]==sha(path)
    for path in [previous.ROOT,intraday.ROOT]:
        r=json.loads((path/'feature_report.json').read_text());proof=json.loads((path/'feature_verification.json').read_text())
        assert proof['passed'] and proof['feature_report_sha256']==sha(path/'feature_report.json')
        assert r['features_sha256']==sha(path/'features.parquet')
    assert p['signal_first']=='2024-01-01' and p['signal_last']=='2025-12-30'
    assert p['window_first']=='1420' and p['window_last']=='1449'
    return p,pd.read_parquet(previous.ROOT/'features.parquet'),json.loads(MANIFEST.read_text())['source_sha256']


def windows():
    if (ROOT/'window_report.json').exists():
        raise ValueError('Do not replace frozen minute-price windows')
    p,old,source_hashes=checked_source();ROOT.mkdir(parents=True,exist_ok=True)
    folder=ROOT/'parts';folder.mkdir(exist_ok=True);codes=sorted(old.code.unique());parts={};sources={}
    for offset in range(0,len(codes),64):
        subset=codes[offset:offset+64];path=folder/f'part_{offset//64:03d}.parquet';meta_path=path.with_suffix('.json')
        paths=[MINUTES/code[:2].upper()/(code[3:]+'.parquet') for code in subset]
        for source in paths:
            assert sha(source)==source_hashes[str(source)]
            sources[str(source)]=source_hashes[str(source)]
        if meta_path.exists():
            meta=json.loads(meta_path.read_text())
            assert meta['codes']==subset and meta['protocol_sha256']==sha(PROTOCOL)
            assert meta['extractor_sha256']==sha(Path(__file__)) and meta['sha256']==sha(path)
        else:
            c=base.conn();c.read_parquet([str(q) for q in paths]).create_view('raw')
            c.register('keys',old.loc[old.code.isin(subset),['date','code']])
            pivots=','.join(f"max(round(close,2)) FILTER(WHERE clock='14{n:02d}') AS pv_c{n:02d}" for n in range(20,50))
            d=c.sql(f"""WITH s AS(SELECT lower(exchange)||'.'||symbol AS code,
                strftime(timestamp,'%Y-%m-%d') AS date,strftime(timestamp,'%H%M') AS clock,
                timestamp,close::DOUBLE AS close FROM raw WHERE timestamp>=TIMESTAMP '2024-01-01'
                AND timestamp<TIMESTAMP '2026-01-01' AND strftime(timestamp,'%H%M') BETWEEN '1420' AND '1449'),
                selected AS(SELECT s.*,coalesce(timestamp=date_trunc('minute',timestamp) AND isfinite(close)
                AND close>0 AND abs(close-round(close,2))<=.0001,false) AS good FROM s JOIN keys USING(date,code))
                SELECT date,code,count(*) AS pv_bars,count(DISTINCT clock) AS pv_clocks,
                count(*) FILTER(WHERE good) AS pv_good_bars,{pivots}
                FROM selected GROUP BY date,code ORDER BY date,code""").df();c.close()
            d.to_parquet(path,index=False,compression='zstd')
            meta=dict(codes=subset,protocol_sha256=sha(PROTOCOL),extractor_sha256=sha(Path(__file__)),sha256=sha(path),rows=len(d))
            save_json(meta_path,meta)
        parts[str(path)]=meta['sha256']
        print(json.dumps(dict(codes=offset+len(subset),total_codes=len(codes))),flush=True)
    r=dict(protocol_sha256=sha(PROTOCOL),previous_feature_report_sha256=sha(previous.ROOT/'feature_report.json'),
        minute_manifest_sha256=sha(MANIFEST),extractor_sha256=sha(Path(__file__)),parts_sha256=parts,
        source_sha256=sources,window_first='1420',window_last='1449',outcomes_read=False,new_2026_prices_read=False)
    save_json(ROOT/'window_report.json',r);return {k:v for k,v in r.items() if k not in ['parts_sha256','source_sha256']}


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace frozen path-variance inputs')
    p,old,_=checked_source();r=json.loads((ROOT/'window_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['extractor_sha256']==sha(Path(__file__))
    for path,digest in r['parts_sha256'].items():assert sha(Path(path))==digest
    wide=pd.concat([pd.read_parquet(path) for path in r['parts_sha256']],ignore_index=True)
    f=old.merge(wide,on=['date','code'],how='left',validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
    price=f[PRICE_COLUMNS].to_numpy(dtype=float)
    valid=(f.pv_bars.eq(30)&f.pv_clocks.eq(30)&f.pv_good_bars.eq(30)
        &np.isfinite(price).all(axis=1)&(price>0).all(axis=1))
    with np.errstate(divide='ignore',invalid='ignore'):
        changes=100*np.log(price[:,1:]/price[:,:-1]);squares=changes*changes
    f['pv_sum29']=np.sum(squares,axis=1)
    f['pv_down29']=np.sum(np.where(changes<0,squares,0),axis=1)
    f['pv_tail14']=np.sum(squares[:,-14:],axis=1)
    f['pv_prior14']=np.sum(squares[:,-28:-14],axis=1)
    f['path_variance_valid']=valid
    f['Z01']=(np.sqrt(f.pv_sum29/29)/f.V01).where(valid)
    f['Z02']=(f.pv_down29/f.pv_sum29.clip(lower=1e-12)).where(valid)
    f['Z03']=((f.pv_tail14-f.pv_prior14)/(f.pv_tail14+f.pv_prior14).clip(lower=1e-12)).where(valid)
    f['prior_formula_input_valid']=f.formula_input_valid
    f['formula_input_valid'] &= valid&np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),previous_feature_report_sha256=sha(previous.ROOT/'feature_report.json'),
        window_report_sha256=sha(ROOT/'window_report.json'),features_sha256=sha(ROOT/'features.parquet'),
        rows=len(f),valid=int(f.formula_input_valid.sum()),previous_valid=int(f.prior_formula_input_valid.sum()),
        newly_invalid=int((f.prior_formula_input_valid&~f.formula_input_valid).sum()),
        flat_valid_windows=int((f.formula_input_valid&f.pv_sum29.eq(0)).sum()),
        expressions=EXPRESSIONS,native_header=HEADER,outcomes_read=False,new_2026_prices_read=False,
        no_exit_rules=True,native_source_parity_verified=False,software_compilation_verified=False)
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
        assert not any((Path('data/research')/f'tail_formula_path_variance_{fold}'/'analysis_report.json').exists()
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
        linkage.ROOT=Path('data/research/tail_formula_path_variance_2024')
        linkage.H2=Path('data/research/tail_formula_path_variance_recent')
        linkage.COMBINED=Path('data/research/tail_formula_path_variance_2025')
        linkage.PROTOCOL=COMBINED_PROTOCOL
        linkage.H2_SELECTION_SHA=None;linkage.H2_MODEL_SHA=None;linkage.H2_OUTCOMES_PREVIOUSLY_SEEN=False
    else:
        previous.setup(fold)
        root=Path('data/research/tail_formula_path_variance_'+fold)
        protocol=Path('config/tail_formula_path_variance_'+fold+'_protocol.json')
        base.ROOT=root;base.PROTOCOL=protocol;relative.PROTOCOL=protocol;study.ROOT=root;study.PROTOCOL=protocol
        base.FEATURES=ROOT;base.EXPRESSIONS=EXPRESSIONS;base.HEADER=HEADER


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['windows','features','model','verify_model','scores','freeze','verify','analyze',
        'freeze_control','verify_control','reuse_control'])
    p.add_argument('--fold',choices=['2024','recent','combined'],default='2024');a=p.parse_args()
    if a.stage in ['windows','features']:
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
