"""Two tail-volume ratios against strictly prior matching stock-day windows."""
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
from .corporate_cash import DAILY, MINUTES, save_json, sha

ROOT=Path('data/research/tail_formula_volume_history')
PROTOCOL=Path('config/tail_formula_volume_history_protocol.json')
COMBINED_PROTOCOL=Path('config/tail_formula_volume_history_combined_protocol.json')
CONTROL=Path('data/research/tail_formula_volume_history_control')
OLD_SELECTION=Path('data/research/tail_formula_float_2025')
MINUTE_MANIFEST=Path('data/research/economic_winner/input_manifest.json')
NEW_EXPRESSIONS={'HV01':'TV29/HV29','HV02':'TV4/HV4'}
EXPRESSIONS={**previous.EXPRESSIONS,**NEW_EXPRESSIONS}
HEADER=previous.HEADER+'TV29:=VALUEWHEN(TIME=1449,SUM(V,29));\nTV4:=VALUEWHEN(TIME=1449,SUM(V,4));\n'
for name,volume in [('HV29','TV29'),('HV4','TV4')]:
    HEADER+=name+':=('+ '+'.join(f'REF({volume},B{i})' for i in range(20))+')/20;\n'


def checked_source():
    p=json.loads(PROTOCOL.read_text())
    for key,root in [('previous_feature_report_sha256',previous.ROOT),('daily_feature_report_sha256',base.SOURCE)]:
        assert p[key]==sha(root/'feature_report.json')
        v=json.loads((root/'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256']==sha(root/'feature_report.json')
    r=json.loads((previous.ROOT/'feature_report.json').read_text())
    assert r['features_sha256']==sha(previous.ROOT/'features.parquet')
    assert p['minute_source_manifest_sha256']==sha(MINUTE_MANIFEST)
    assert p['history_days']==20 and p['signal_last']<'2026-01-01'
    return p,pd.read_parquet(previous.ROOT/'features.parquet')


def prepare():
    if (ROOT/'input_manifest.json').exists():
        raise ValueError('Do not replace the history-volume sources')
    p,old=checked_source(); ROOT.mkdir(parents=True,exist_ok=True)
    sources=json.loads((base.SOURCE/'feature_report.json').read_text())['source_sha256']
    minute_sources=json.loads(MINUTE_MANIFEST.read_text())['source_sha256']
    days=[]; daily_hashes={}; minute_hashes={}
    for code in sorted(old.code.unique()):
        path=DAILY/(code.replace('.','_')+'.parquet')
        assert sha(path)==sources[str(path)]; daily_hashes[str(path)]=sources[str(path)]
        d=pd.read_parquet(path,columns=['date','tradestatus'],filters=[('date','>=',p['history_first']),('date','<=',p['signal_last'])])
        d=d.loc[pd.to_numeric(d.tradestatus).eq(1),['date']].copy(); d['code']=code
        assert not d.date.duplicated().any()
        days.append(d)
        raw=MINUTES/code[:2].upper()/(code[3:]+'.parquet')
        minute_hashes[str(raw)]=minute_sources[str(raw)]
    schedule=pd.concat(days,ignore_index=True).sort_values(['code','date']).reset_index(drop=True)
    schedule.to_parquet(ROOT/'stock_days.parquet',index=False,compression='zstd')
    assert old[['date','code']].merge(schedule,on=['date','code'],how='inner').shape[0]==len(old)
    r=dict(protocol_sha256=sha(PROTOCOL),previous_feature_report_sha256=sha(previous.ROOT/'feature_report.json'),
        daily_feature_report_sha256=sha(base.SOURCE/'feature_report.json'),minute_source_manifest_sha256=sha(MINUTE_MANIFEST),
        stock_days_sha256=sha(ROOT/'stock_days.parquet'),daily_source_sha256=daily_hashes,minute_source_sha256=minute_hashes,
        history_first=p['history_first'],signal_last=p['signal_last'],stock_days=len(schedule),codes=len(days),
        no_outcomes_read=True,new_2026_prices_read=False)
    save_json(ROOT/'input_manifest.json',r)
    return {k:v for k,v in r.items() if k not in ['daily_source_sha256','minute_source_sha256']}


def manifest():
    r=json.loads((ROOT/'input_manifest.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['stock_days_sha256']==sha(ROOT/'stock_days.parquet')
    assert r['minute_source_manifest_sha256']==sha(MINUTE_MANIFEST)
    return r


def extract():
    if (ROOT/'window_report.json').exists():
        raise ValueError('Do not replace the volume windows')
    m=manifest(); schedule=pd.read_parquet(ROOT/'stock_days.parquet')
    codes=sorted(schedule.code.unique()); folder=ROOT/'parts'; folder.mkdir(exist_ok=True); parts={}
    for offset in range(0,len(codes),64):
        subset=codes[offset:offset+64]; path=folder/f'part_{offset//64:03d}.parquet'; meta=path.with_suffix('.json')
        if meta.exists():
            saved=json.loads(meta.read_text())
            assert saved['codes']==subset and saved['input_manifest_sha256']==sha(ROOT/'input_manifest.json')
            assert saved['extractor_sha256']==sha(Path(__file__)) and saved['sha256']==sha(path)
        else:
            sources=[MINUTES/code[:2].upper()/(code[3:]+'.parquet') for code in subset]
            for source in sources:
                assert sha(source)==m['minute_source_sha256'][str(source)]
            c=base.conn(); c.read_parquet([str(x) for x in sources]).create_view('raw')
            c.register('schedule',schedule.loc[schedule.code.isin(subset)])
            f=c.execute('''WITH raw_window AS(SELECT lower(exchange)||'.'||symbol AS code,
                strftime(timestamp,'%Y-%m-%d') AS date,strftime(timestamp,'%H%M') AS clock,timestamp,
                volume::DOUBLE AS volume FROM raw WHERE timestamp>=?::DATE AND timestamp<?::DATE+INTERVAL 1 DAY
                AND strftime(timestamp,'%H%M') BETWEEN '1421' AND '1449'),
                v AS(SELECT *,coalesce(timestamp=date_trunc('minute',timestamp) AND isfinite(volume)
                    AND volume>=0 AND volume=floor(volume),false) AS valid FROM raw_window JOIN schedule USING(date,code)),
                agg AS(SELECT date,code,count(*) AS bars,count(DISTINCT clock) AS clocks,
                    count(*) FILTER(WHERE valid) AS good_bars,count(*) FILTER(WHERE clock>='1446') AS bars4,
                    min(clock) AS first_clock,max(clock) AS last_clock,
                    sum(CASE WHEN valid THEN volume::BIGINT END)::DOUBLE AS tail_volume29,
                    sum(CASE WHEN valid AND clock>='1446' THEN volume::BIGINT END)::DOUBLE AS tail_volume4
                    FROM v GROUP BY date,code)
                SELECT *,bars=29 AND clocks=29 AND good_bars=29 AND bars4=4
                    AND first_clock='1421' AND last_clock='1449' AS volume_window_valid
                FROM agg ORDER BY code,date''',[m['history_first'],m['signal_last']]).df()
            assert not f.duplicated(['date','code']).any(); c.close()
            f.to_parquet(path,index=False,compression='zstd')
            saved=dict(codes=subset,input_manifest_sha256=sha(ROOT/'input_manifest.json'),
                extractor_sha256=sha(Path(__file__)),sha256=sha(path),rows=len(f),raw_minutes=int(f.bars.sum()))
            save_json(meta,saved)
        parts[str(path)]=saved['sha256']
        print(json.dumps(dict(codes=offset+len(subset),total=len(codes),windows=saved['rows'])),flush=True)
    windows=pd.concat([pd.read_parquet(x) for x in parts],ignore_index=True).sort_values(['code','date']).reset_index(drop=True)
    windows.to_parquet(ROOT/'windows.parquet',index=False,compression='zstd')
    r=dict(input_manifest_sha256=sha(ROOT/'input_manifest.json'),parts_sha256=parts,
        windows_sha256=sha(ROOT/'windows.parquet'),rows=len(windows),raw_minutes=int(windows.bars.sum()),
        no_price_fields_projected=True,new_2026_prices_read=False,no_outcomes_read=True)
    save_json(ROOT/'window_report.json',r)
    return {k:v for k,v in r.items() if k!='parts_sha256'}


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace the history-volume features')
    p,old=checked_source(); manifest()
    wr=json.loads((ROOT/'window_report.json').read_text())
    assert wr['input_manifest_sha256']==sha(ROOT/'input_manifest.json') and wr['windows_sha256']==sha(ROOT/'windows.parquet')
    schedule=pd.read_parquet(ROOT/'stock_days.parquet')
    windows=pd.read_parquet(ROOT/'windows.parquet')
    joined=schedule.merge(windows,on=['date','code'],how='left',validate='one_to_one').sort_values(['code','date'])
    joined['volume_window_valid']=joined.volume_window_valid.fillna(False).astype(bool)
    histories=[]
    for code,f in joined.groupby('code',sort=True):
        f=f.copy()
        f['history_first_date']=f.date.shift(20); f['history_last_date']=f.date.shift(1)
        f['history_valid_count']=f.volume_window_valid.astype(int).shift(1).rolling(20,min_periods=20).sum()
        for width in [29,4]:
            f[f'history_volume{width}']=f[f'tail_volume{width}'].where(f.volume_window_valid).shift(1).rolling(20,min_periods=20).mean()
        histories.append(f)
    h=pd.concat(histories,ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    h.to_parquet(ROOT/'history.parquet',index=False,compression='zstd')
    f=old.merge(h,on=['date','code'],how='left',validate='one_to_one',suffixes=('','_history')).sort_values(['date','code']).reset_index(drop=True)
    f['volume_history_valid']=(f.history_valid_count.eq(20)&f.history_first_date.notna()&f.history_last_date.lt(f.date)
        &np.isfinite(f[['history_volume29','history_volume4']]).all(axis=1)&f.history_volume29.gt(0)&f.history_volume4.gt(0))
    f['volume_current_valid']=(f.volume_window_valid.fillna(False)&f.v29.eq(f.tail_volume29)&f.v4.eq(f.tail_volume4))
    good=f.volume_history_valid&f.volume_current_valid
    f['HV01']=(f.v29/f.history_volume29).where(good)
    f['HV02']=(f.v4/f.history_volume4).where(good)
    f['prior_formula_input_valid']=f.formula_input_valid
    f['formula_input_valid'] &= good&np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    report=dict(protocol_sha256=sha(PROTOCOL),input_manifest_sha256=sha(ROOT/'input_manifest.json'),
        window_report_sha256=sha(ROOT/'window_report.json'),previous_feature_report_sha256=sha(previous.ROOT/'feature_report.json'),
        features_sha256=sha(ROOT/'features.parquet'),history_sha256=sha(ROOT/'history.parquet'),rows=len(f),valid=int(f.formula_input_valid.sum()),
        previous_valid=int(old.formula_input_valid.sum()),newly_invalid=int((old.formula_input_valid&~f.formula_input_valid).sum()),
        invalid_history=int((~f.volume_history_valid).sum()),invalid_current_volume=int((~f.volume_current_valid).sum()),
        expressions=EXPRESSIONS,native_header=HEADER,stock_day_history=20,
        native_source_parity_verified=False,outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',report)
    return {k:v for k,v in report.items() if k not in ['expressions','native_header']}


def control(stage):
    proof=json.loads((ROOT/'feature_verification.json').read_text())
    report=json.loads((ROOT/'feature_report.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256']==sha(ROOT/'feature_report.json')
    assert report['features_sha256']==sha(ROOT/'features.parquet')
    assert sha(OLD_SELECTION/'selection_report.json')=='07de34caa25a339fa8e7b8d3665c7d04c044792d01fa170d0fd48f6b77b8f5c5'
    prior=json.loads((OLD_SELECTION/'selection_report.json').read_text())
    assert prior['selection_sha256']==sha(OLD_SELECTION/'selection.parquet')
    proof=json.loads((OLD_SELECTION/'selection_verification.json').read_text())
    assert proof['passed'] and proof['selection_report_sha256']==sha(OLD_SELECTION/'selection_report.json')
    if stage=='freeze_control':
        if (CONTROL/'selection_report.json').exists():
            raise ValueError('Do not replace the same-quality original selection')
        assert not any((Path('data/research')/f'tail_formula_volume_history_{s}'/'analysis_report.json').exists()
                       for s in ['2024','recent','2025'])
        f=pd.read_parquet(ROOT/'features.parquet',columns=['date','code','formula_input_valid'])
        out=pd.read_parquet(OLD_SELECTION/'selection.parquet')
        pd.testing.assert_frame_equal(out[['date','code']],f[['date','code']],check_exact=True)
        out['selected'] &= f.formula_input_valid
        CONTROL.mkdir(exist_ok=True);out.to_parquet(CONTROL/'selection.parquet',index=False,compression='zstd')
        r=dict(protocol_sha256=sha(COMBINED_PROTOCOL),feature_report_sha256=sha(ROOT/'feature_report.json'),
            original_selection_report_sha256=sha(OLD_SELECTION/'selection_report.json'),
            selection_sha256=sha(CONTROL/'selection.parquet'),selected=int(out.selected.sum()),
            selection_is_original_48_and_new_input_valid=True,model_refitted=False,year_2025_is_exploratory=True,
            new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
        save_json(CONTROL/'selection_report.json',r);return r
    r=json.loads((CONTROL/'selection_report.json').read_text())
    for key,path in [('protocol_sha256',COMBINED_PROTOCOL),('feature_report_sha256',ROOT/'feature_report.json'),
        ('original_selection_report_sha256',OLD_SELECTION/'selection_report.json'),('selection_sha256',CONTROL/'selection.parquet')]:
        assert r[key]==sha(path)
    if stage=='verify_control':
        c=base.conn()
        expected=c.sql(f'''SELECT s.date,s.code,s.half,s.board,s.decision_shares,s.selected AND f.formula_input_valid AS selected
            FROM read_parquet('{OLD_SELECTION}/selection.parquet') s
            JOIN read_parquet('{ROOT}/features.parquet') f USING(date,code) ORDER BY date,code''').df();c.close()
        pd.testing.assert_frame_equal(pd.read_parquet(CONTROL/'selection.parquet'),expected,check_exact=True)
        assert int(expected.selected.sum())==r['selected']
        result=dict(passed=True,selection_report_sha256=sha(CONTROL/'selection_report.json'),rows=len(expected),
            all_control_selection_flags_rebuilt=True,outcomes_read=False,new_2026_prices_read=False)
        save_json(CONTROL/'selection_verification.json',result);return result
    assert stage=='analyze_control'
    return linkage.common_analysis(CONTROL,COMBINED_PROTOCOL)


def setup(fold):
    if fold=='combined':
        linkage.ROOT=Path('data/research/tail_formula_volume_history_2024')
        linkage.H2=Path('data/research/tail_formula_volume_history_recent')
        linkage.COMBINED=Path('data/research/tail_formula_volume_history_2025')
        linkage.PROTOCOL=COMBINED_PROTOCOL
        linkage.H2_SELECTION_SHA=None;linkage.H2_MODEL_SHA=None;linkage.H2_OUTCOMES_PREVIOUSLY_SEEN=False
    else:
        previous.setup(fold)
        root=Path('data/research/tail_formula_volume_history_'+fold)
        protocol=Path('config/tail_formula_volume_history_'+fold+'_protocol.json')
        base.ROOT=root;base.PROTOCOL=protocol;relative.PROTOCOL=protocol;study.ROOT=root;study.PROTOCOL=protocol
        base.FEATURES=ROOT;base.EXPRESSIONS=EXPRESSIONS;base.HEADER=HEADER


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','extract','features','model','verify_model','scores','freeze','verify','analyze',
        'freeze_control','verify_control','analyze_control'])
    p.add_argument('--fold',choices=['2024','recent','combined'],default='2024');a=p.parse_args()
    if a.stage in ['prepare','extract','features']:
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
