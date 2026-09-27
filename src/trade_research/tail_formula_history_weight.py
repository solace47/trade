"""Volume-weighted distribution of strictly prior daily closing prices."""
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
from .turnover_reference import CALENDAR

ROOT=Path('data/research/tail_formula_history_weight')
PROTOCOL=Path('config/tail_formula_history_weight_protocol.json')
COMBINED_PROTOCOL=Path('config/tail_formula_history_weight_combined_protocol.json')
CONTROL=Path('data/research/tail_formula_history_weight_control')
OLD_SELECTION=Path('data/research/tail_formula_float_2025')
PRICE_COLUMNS=[f'hc_c{i:02d}' for i in range(1,21)]
VOLUME_COLUMNS=[f'hc_v{i:02d}' for i in range(1,21)]
NEW_EXPRESSIONS={'K01':'100*(Q/HW20-1)/V01','K02':'100*SQRT(HS20)/HW20/V01'}
EXPRESSIONS={**previous.EXPRESSIONS,**NEW_EXPRESSIONS}
HEADER=previous.HEADER+''.join(f'HDV{i:02d}:=REF(SUM(V,B0),B{i-1});\n' for i in range(1,21))
HEADER+='HV20:='+ '+'.join(f'HDV{i:02d}' for i in range(1,21))+';\n'
HEADER+='HW20:=('+ '+'.join(f'DCP{i}*HDV{i:02d}' for i in range(1,21))+')/HV20;\n'
HEADER+='HS20:=('+ '+'.join(f'HDV{i:02d}*(DCP{i}-HW20)*(DCP{i}-HW20)' for i in range(1,21))+')/HV20;\n'


def source_files():
    p=json.loads(PROTOCOL.read_text())
    for key,path in [('previous_feature_report_sha256',previous.ROOT/'feature_report.json'),
                     ('daily_feature_report_sha256',base.SOURCE/'feature_report.json')]:assert p[key]==sha(path)
    report=json.loads((previous.ROOT/'feature_report.json').read_text())
    proof=json.loads((previous.ROOT/'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256']==sha(previous.ROOT/'feature_report.json')
    assert report['features_sha256']==sha(previous.ROOT/'features.parquet')
    assert p['history_stock_days']==20 and p['history_first']=='2023-06-01'
    return previous.context.market.stock.source_files()


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace frozen historical volume-weighted inputs')
    files=source_files();c=base.conn();c.read_parquet(files).create_view('daily')
    lagged=','.join(f'lag({field},{i}) OVER w AS hc_{key}{i:02d}'
                   for key,field in [('c','cl'),('v','v')] for i in range(1,21))
    history=c.sql(f"""WITH active AS(SELECT date,code,close::DOUBLE AS cl,volume::DOUBLE AS v,
        preclose::DOUBLE AS p,adjustflag::DOUBLE AS adj FROM daily
        WHERE date BETWEEN '2023-06-01' AND '2025-12-30' AND tradestatus=1),
        atoms AS(SELECT *,coalesce(isfinite(cl) AND cl>0 AND abs(cl-round(cl,2))<=.0001
          AND isfinite(v) AND v>0 AND v=floor(v) AND adj=3,false) AS good,
          CASE WHEN abs(p-lag(cl) OVER(PARTITION BY code ORDER BY date))>.005 THEN 1 ELSE 0 END AS ref_break
          FROM active),
        hist AS(SELECT date,code,{lagged},count(*) OVER h AS hc_rows,sum(good::INTEGER) OVER h AS hc_good,
          min(date) OVER h AS hc_first_date,max(date) OVER h AS hc_last_date,
          sum(ref_break) OVER h AS hc_reference_breaks FROM atoms
          WINDOW w AS(PARTITION BY code ORDER BY date),
          h AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING))
        SELECT * FROM hist WHERE date>='2024-01-01' ORDER BY date,code""").df();c.close()
    old=pd.read_parquet(previous.ROOT/'features.parquet')
    f=old.merge(history,on=['date','code'],how='left',validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
    price=f[PRICE_COLUMNS].to_numpy(dtype=float);volume=f[VOLUME_COLUMNS].to_numpy(dtype=float)
    valid=(f.hc_rows.eq(20)&f.hc_good.eq(20)&f.hc_last_date.lt(f.date)&(price>0).all(axis=1)
           &(volume>0).all(axis=1)&np.isfinite(price).all(axis=1)&np.isfinite(volume).all(axis=1)&f.V01.gt(0))
    f['hc_total_volume']=np.sum(volume,axis=1)
    f['hc_weighted_close']=np.sum(price*volume,axis=1)/f.hc_total_volume
    f['hc_weighted_variance']=np.sum(volume*(price-f.hc_weighted_close.to_numpy()[:,None])**2,axis=1)/f.hc_total_volume
    f['history_weight_valid']=valid
    f['K01']=(100*(f.price_1449/f.hc_weighted_close-1)/f.V01).where(valid)
    f['K02']=(100*np.sqrt(f.hc_weighted_variance)/f.hc_weighted_close/f.V01).where(valid)
    cal=pd.read_parquet(CALENDAR)
    days=sorted(cal.loc[cal.is_trading_day.eq('1')&cal.calendar_date.between('2023-06-01','2025-12-30'),'calendar_date'])
    ranks={d:i for i,d in enumerate(days)}
    f['hc_market_span']=f.hc_last_date.map(ranks)-f.hc_first_date.map(ranks)+1
    f['prior_formula_input_valid']=f.formula_input_valid
    f['formula_input_valid'] &= valid&np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True,exist_ok=True);f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),previous_feature_report_sha256=sha(previous.ROOT/'feature_report.json'),
        daily_feature_report_sha256=sha(base.SOURCE/'feature_report.json'),calendar_sha256=sha(CALENDAR),
        features_sha256=sha(ROOT/'features.parquet'),rows=len(f),valid=int(f.formula_input_valid.sum()),
        previous_valid=int(f.prior_formula_input_valid.sum()),newly_invalid=int((f.prior_formula_input_valid&~f.formula_input_valid).sum()),
        valid_with_reference_break=int((f.formula_input_valid&f.hc_reference_breaks.gt(0)).sum()),
        valid_with_historical_gaps=int((f.formula_input_valid&f.hc_market_span.gt(20)).sum()),
        maximum_valid_history_market_span=float(f.loc[f.formula_input_valid,'hc_market_span'].max()),
        first_history_date=f.hc_first_date.min(),last_history_date=f.hc_last_date.max(),
        expressions=EXPRESSIONS,native_header=HEADER,all_previous_48_inputs_retained=True,
        raw_unadjusted_statistic_not_holder_cost=True,software_compilation_verified=False,
        native_source_parity_verified=False,outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
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
        assert not any((Path('data/research')/f'tail_formula_history_weight_{fold}'/'analysis_report.json').exists()
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
        linkage.ROOT=Path('data/research/tail_formula_history_weight_2024')
        linkage.H2=Path('data/research/tail_formula_history_weight_recent')
        linkage.COMBINED=Path('data/research/tail_formula_history_weight_2025')
        linkage.PROTOCOL=COMBINED_PROTOCOL
        linkage.H2_SELECTION_SHA=None;linkage.H2_MODEL_SHA=None;linkage.H2_OUTCOMES_PREVIOUSLY_SEEN=False
    else:
        previous.setup(fold)
        root=Path('data/research/tail_formula_history_weight_'+fold)
        protocol=Path('config/tail_formula_history_weight_'+fold+'_protocol.json')
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
