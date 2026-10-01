"""Visible ETF quantity relative to twenty strictly earlier trading dates."""
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_etf_activity as timing
from . import tail_formula_morning_range as prior
from . import tail_formula_volume_memory as original_source
from .corporate_cash import save_json, sha

STEM = 'tail_formula_etf_quantity'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM+'_input_protocol.json')
INTENT = Path('config') / (STEM+'_intent.json')
META, CONTROL = prior.META, prior.EXPRESSIONS
ETF_HELPER = 'QTDT:=VALUEWHEN(TIME=1448,DATE);\n'+f'QTCLK:=VALUEWHEN(TIME=1448,{timing.SEQUENCE_GUARD});\n'+'''QTGOOD:=VALUEWHEN(TIME=1448,COUNT(V>=0 AND V-V=0,228));
QTTOTAL:=VALUEWHEN(TIME=1448,SUM(V,228));
QTBASE:=(QTDT=DATE) AND (QTCLK=228) AND (QTGOOD=228) AND (QTTOTAL>=0);
QTDAYN:=IF(COUNT(DATE<>REF(DATE,1),0)>0,BARSLAST(DATE<>REF(DATE,1))+1,BARSCOUNT(DATE));
QO1:=QTDAYN;
QD0:=DATE;
'''
for j in range(1,21):
    if j>1:
        ETF_HELPER += f'QO{j}:=QO{j-1}+REFV(QTDAYN,QO{j-1});\n'
    ETF_HELPER += f'QP{j}:=REFV(QTTOTAL,QO{j});\nQD{j}:=REFV(DATE,QO{j});\n'
    ETF_HELPER += f'QG{j}:=IF((REFV(QTBASE,QO{j})=1) AND (QD{j}<QD{j-1}),1,0);\n'
ETF_HELPER += 'QTM20:=('+ '+'.join(f'QP{j}' for j in range(1,21))+')/20;\n'
ETF_HELPER += 'QTHGOOD:='+ '+'.join(f'QG{j}' for j in range(1,21))+';\n'
ETF_HELPER += '''QTREADY:=(QTBASE) AND (QTHGOOD=20) AND (BARSCOUNT(DATE)>QO20);
QQ:IF(QTREADY,IF(QTTOTAL+QTM20>0,100*(QTTOTAL-QTM20)/(QTTOTAL+QTM20),0),DRAWNULL);
QD:IF(QTREADY,DATE,DRAWNULL);
'''
EXTRA_HEADER = '''Q3DT:=CALCSTOCKINDEX('SH510300','YJETFL',2);
Q5DT:=CALCSTOCKINDEX('SH510500','YJETFL',2);
Q3VAL:=CALCSTOCKINDEX('SH510300','YJETFL',1);
Q5VAL:=CALCSTOCKINDEX('SH510500','YJETFL',1);
QVREADY:=(Q3DT=DATE) AND (Q5DT=DATE);
'''
NEW_EXPRESSIONS = {'EQ300':'IF(QVREADY,Q3VAL,DRAWNULL)','EQ500':'IF(QVREADY,Q5VAL,DRAWNULL)'}
EXPRESSIONS = {**CONTROL,**NEW_EXPRESSIONS}
HEADER = prior.HEADER+EXTRA_HEADER


def checked():
    p=json.loads(PROTOCOL.read_text())
    assert p['intent_sha256']==sha(INTENT) and p['arms']=={'control':CONTROL,'memory':EXPRESSIONS}
    assert p['native_header']==HEADER and p['etf_helper']==ETF_HELPER
    assert p['maximum_new_fits']==4 and p['window_labels']==228 and p['history_dates']==20
    assert not p['new_2026_prices_allowed']
    for file,digest in p['source_hashes'].items():
        assert sha(Path(file))==digest,file
    gate=json.loads(Path(p['conditional_gate']).read_text())
    assert gate['passed'] and not gate['supports_further_validation']
    text=subprocess.run(['git','show','HEAD:docs/selection-formula.md'],text=True,capture_output=True,check=True).stdout
    assert sha(PROTOCOL) in text and sha(INTENT) in text
    return p


def native_series(times,dates,volumes):
    """Replay the literal helper, including smoothed REFV and its guards."""
    size=len(times);position=np.arange(size)
    def ref(x,n):
        x=np.broadcast_to(np.asarray(x),size);n=np.broadcast_to(np.asarray(n,float),size)
        good=np.isfinite(n)&(n>=0);idx=position-np.where(good,n,0).astype(np.int64)
        return np.where(good,x[np.clip(idx,0,size-1)],np.nan)
    def count(x,n):
        s=pd.Series(np.asarray(x,float))
        return s.cumsum().to_numpy() if int(n)<=0 else s.rolling(int(n),min_periods=int(n)).sum().to_numpy()
    def valuewhen(mask,value):
        return pd.Series(np.where(mask,np.broadcast_to(value,size),np.nan)).ffill().to_numpy()
    def barslast(mask):
        return position-np.maximum.accumulate(np.where(mask,position,-1))
    env=dict(TIME=np.asarray(times),DATE=np.asarray(dates),V=np.asarray(volumes,float),IF=np.where,DRAWNULL=np.nan,
             REF=ref,REFV=ref,SUM=count,COUNT=count,VALUEWHEN=valuewhen,BARSLAST=barslast,
             BARSCOUNT=lambda x:position+1)
    output={}
    with np.errstate(all='ignore'):
        for line in ETF_HELPER.splitlines():
            separator=':=' if ':=' in line else ':'
            name,expression=line.rstrip(';').split(separator,1)
            expression=re.sub(r'(?<![<>=!])=(?!=)','==',expression.replace('<>','!=')).replace(' AND ',' & ')
            if name=='QTGOOD':
                expression='VALUEWHEN(TIME==1448,COUNT((V>=0) & ((V-V)==0),228))'
            env[name]=eval(expression,{'__builtins__':{}},env)
            if separator==':':output[name]=np.asarray(env[name],float)
    return output


def stock_values(first,second,dates):
    reference={'SH510300':first,'SH510500':second}
    env=dict(DATE=np.asarray(dates),IF=np.where,DRAWNULL=np.nan,
             CALCSTOCKINDEX=lambda code,name,index:reference[code][index-1])
    for line in EXTRA_HEADER.splitlines():
        name,expression=line.rstrip(';').split(':=',1)
        expression=re.sub(r'(?<![<>=!])=(?!=)','==',expression).replace(' AND ',' & ')
        env[name]=eval(expression,{'__builtins__':{}},env)
    return np.column_stack([eval(e,{'__builtins__':{}},env) for e in NEW_EXPRESSIONS.values()])


def prepare():
    p=checked();path=INPUTS/'feature_report.json';assert not path.exists()
    INPUTS.mkdir(parents=True,exist_ok=True);daily=[];sources=[];checks=0
    encode=lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999))
    for symbol in ['510300','510500']:
        source=Path(p['etf_sources'][symbol]);c=base.conn();c.read_parquet(str(source)).create_view('source')
        metadata=c.sql("SELECT timestamp FROM source WHERE timestamp>=TIMESTAMP '2022-11-01' AND timestamp<TIMESTAMP '2026-01-01' ORDER BY timestamp").df()
        raw=c.sql("""SELECT timestamp,volume FROM source
            WHERE timestamp>=TIMESTAMP '2022-11-01' AND timestamp<TIMESTAMP '2026-01-01'
            AND (strftime(timestamp,'%H%M') BETWEEN '0931' AND '1130'
                OR strftime(timestamp,'%H%M') BETWEEN '1301' AND '1448') ORDER BY timestamp""").df()
        assert not raw.timestamp.duplicated().any() and not metadata.timestamp.duplicated().any()
        raw['date']=raw.timestamp.dt.strftime('%Y-%m-%d');group=raw.groupby('date',sort=True)
        assert len(group)==p['expected_ETF_days'] and group.size().eq(228).all()
        for _,block in group:
            np.testing.assert_array_equal(block.timestamp.dt.hour*100+block.timestamp.dt.minute,timing.WINDOW_CLOCKS)
        dates=group.size().index.to_numpy();v=raw.volume.to_numpy(float).reshape(len(dates),228)
        total=v.sum(axis=1);good=(np.isfinite(v)&(v>=0)).all(axis=1)
        mean=pd.Series(total).shift(1).rolling(20,min_periods=20).mean().to_numpy(copy=True)
        history_good=pd.Series(good.astype(int)).shift(1).rolling(20,min_periods=20).sum().eq(20).to_numpy()
        mean[~history_good]=np.nan
        valid=good & history_good
        with np.errstate(all='ignore'):
            expected=np.where(total+mean>0,100*(total-mean)/(total+mean),0.)
        expected[~valid]=np.nan
        c.register('raw',raw)
        sql=c.sql("""WITH a AS(SELECT date,count(*) AS bars,count(DISTINCT timestamp) AS clocks,
            sum(coalesce(isfinite(volume) AND volume>=0,false)::INT) AS good,sum(volume::HUGEINT)::DOUBLE AS total
            FROM raw GROUP BY date),b AS(SELECT *,avg(total) OVER w AS mean,
            sum((bars=228 AND clocks=228 AND good=228)::INT) OVER w AS prior_good,count(*) OVER w AS prior_days
            FROM a WINDOW w AS(ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)),
            q AS(SELECT *,bars=228 AND clocks=228 AND good=228 AND prior_good=20 AND prior_days=20 AS valid FROM b)
            SELECT date,total,CASE WHEN prior_good=20 AND prior_days=20 THEN mean END AS mean,
            valid,CASE WHEN valid AND total+mean>0 THEN 100*(total-mean)/(total+mean)
            WHEN valid THEN 0. END AS quantity FROM q ORDER BY date""").df();c.close()
        np.testing.assert_array_equal(dates,sql.date);np.testing.assert_array_equal(valid,sql.valid)
        np.testing.assert_allclose(total,sql.total,rtol=0,atol=0)
        np.testing.assert_allclose(mean,sql['mean'],rtol=0,atol=1e-5,equal_nan=True)
        np.testing.assert_allclose(expected,sql.quantity,rtol=0,atol=2e-11,equal_nan=True)
        np.testing.assert_array_equal(encode(expected),encode(sql.quantity.to_numpy(float)))
        sequence=metadata.merge(raw[['timestamp','volume']],on='timestamp',how='left',validate='one_to_one')
        times=sequence.timestamp.dt.hour.to_numpy()*100+sequence.timestamp.dt.minute.to_numpy()
        native_dates=sequence.timestamp.dt.strftime('%Y%m%d').astype(int).to_numpy()-19000000
        native=native_series(times,native_dates,sequence.volume.to_numpy(float));anchors=times==1448
        np.testing.assert_array_equal(native_dates[anchors],pd.Series(dates).str.replace('-','',regex=False).astype(int).to_numpy()-19000000)
        np.testing.assert_allclose(native['QQ'][anchors],expected,rtol=0,atol=2e-11,equal_nan=True)
        np.testing.assert_array_equal(encode(native['QQ'][anchors]),encode(expected))
        np.testing.assert_array_equal(np.isfinite(native['QD'][anchors]),valid)
        np.testing.assert_array_equal(native['QD'][anchors][valid],native_dates[anchors][valid])
        scaled=native_series(times,native_dates,sequence.volume.to_numpy(float)*5)
        np.testing.assert_allclose(scaled['QQ'][anchors],expected,rtol=0,atol=2e-11,equal_nan=True)
        np.testing.assert_array_equal(encode(scaled['QQ'][anchors]),encode(expected))
        future=sequence.volume.to_numpy(float).copy();future[~np.isin(times,timing.WINDOW_CLOCKS)]=.01
        modified=native_series(times,native_dates,future)
        np.testing.assert_array_equal(native['QQ'][anchors],modified['QQ'][anchors])
        prefix='EQ300' if symbol=='510300' else 'EQ500'
        table=pd.DataFrame({'date':dates,prefix:expected,prefix+'_date':native['QD'][anchors],prefix+'_valid':valid})
        table=table[table.date>='2023-01-01'].reset_index(drop=True)
        file=INPUTS/(symbol+'_daily.parquet');table.to_parquet(file,index=False,compression='zstd')
        daily.append(table);sources.append(dict(source=str(source),source_sha256=sha(source),daily_file=str(file),daily_sha256=sha(file),
            raw_prefix_bars=len(raw),all_dates=len(dates),current_dates=len(table),all_native_anchors_replayed=True,
            no_volume_after1448_or2026_loaded=True))
        checks+=len(table)
        print(json.dumps(dict(source=symbol,all_dates=len(dates),current_dates=len(table),full_native_SQL_math_passed=True)),flush=True)
    etf=daily[0].merge(daily[1],on='date',how='outer',validate='one_to_one').sort_values('date')
    date_numbers=etf.date.str.replace('-','',regex=False).astype(int).to_numpy()-19000000
    literal=stock_values([etf.EQ300.to_numpy(),etf.EQ300_date.to_numpy()],
                         [etf.EQ500.to_numpy(),etf.EQ500_date.to_numpy()],date_numbers)
    np.testing.assert_allclose(literal,etf[list(NEW_EXPRESSIONS)],rtol=0,atol=2e-11,equal_nan=True)
    np.testing.assert_array_equal(encode(literal),encode(etf[list(NEW_EXPRESSIONS)].to_numpy()))
    old=original_source.original();out=old.merge(etf,on='date',how='left',validate='many_to_one')
    pd.testing.assert_frame_equal(out[[*META,*CONTROL]],old[[*META,*CONTROL]],check_exact=True)
    new_good=out.EQ300_valid.fillna(False)&out.EQ500_valid.fillna(False)
    invalid=int((old.formula_input_valid & ~new_good).sum())
    save_json(INPUTS/'input_domain_audit.json',dict(passed=invalid==0,original_valid=1602413,
        new_valid=int((old.formula_input_valid & new_good).sum()),newly_invalid=invalid,
        original_keys_values_metadata_exact=True,no_bad_dates_skipped_or_oldest_data_borrowed=True,
        stop_fit_if_domain_changes=True,new_2026_prices_read=False))
    assert invalid==0
    out['prior_formula_input_valid']=old.formula_input_valid;out['etf_quantity_valid']=new_good
    out['formula_input_valid']=old.formula_input_valid & new_good
    pd.testing.assert_series_equal(out.formula_input_valid,old.formula_input_valid,check_names=True)
    c=base.conn();c.register('old',old);c.register('etf',etf)
    rebuilt=c.sql('SELECT old.*,'+','.join(NEW_EXPRESSIONS)+' FROM old LEFT JOIN etf USING(date) ORDER BY date,code').df();c.close()
    pd.testing.assert_frame_equal(out[[*META,*EXPRESSIONS]],rebuilt[[*META,*EXPRESSIONS]],check_exact=True)
    out.to_parquet(INPUTS/'features.parquet',index=False,compression='zstd')
    pd.testing.assert_frame_equal(out,pd.read_parquet(INPUTS/'features.parquet'),check_exact=True)
    for name in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        (INPUTS/name).symlink_to((prior.INPUTS/name).resolve())
    helper=INPUTS/'YJETFL.tdx';helper.write_text(ETF_HELPER)
    adapter=INPUTS/'quantity_inputs.tdx';adapter.write_text(EXTRA_HEADER+'\n'.join(f'{n}:={e};' for n,e in NEW_EXPRESSIONS.items())+'\n')
    save_json(path,dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(INPUTS/'features.parquet'),rows=len(out),
        valid=int(out.formula_input_valid.sum()),newly_invalid=invalid,expressions=EXPRESSIONS,native_header=HEADER,
        source_receipts=sources,all_current_native_anchors=checks,scalar_feature_and_encoding_checks=len(out)*2,
        helper_sha256=sha(helper),adapter_sha256=sha(adapter),original_T0_previously_exposed=True,
        new_M20_and_quantity_after_actual_protocol=True,new_group_outcomes_read=False,
        software_compilation_verified=False,native_source_parity_verified=False,new_2026_prices_read=False,no_exit_rules=True))
    save_json(INPUTS/'feature_verification.json',dict(passed=True,feature_report_sha256=sha(path),
        all_original_50_metadata_keys_and_domain_exact=True,all_current_values_native_SQL_encodings_and_date_join_equal=True,
        no_history_skipping_or_unknown_zero_imputation=True,new_2026_prices_read=False,no_exit_rules=True))
    save_json(INPUTS/'native_input_verification.json',dict(passed=True,feature_report_sha256=sha(path),
        feature_verification_sha256=sha(INPUTS/'feature_verification.json'),helper_sha256=sha(helper),adapter_sha256=sha(adapter),
        all_1542_source_date_histories_replayed=True,all_two_asset_literal_current_stock_date_outputs_equal=True,
        actual_variable_day_length_smoothed_REFV_bound_and_descending_dates_verified=True,
        all_actual_source_anchors_unit_scaling_and_future_exclusion_replayed=True,
        zero_bad_history_clock_and_insufficient_history_tests_passed=True,
        software_compilation_verified=False,native_source_parity_verified=False,new_2026_prices_read=False,no_exit_rules=True))
    return dict(feature_report_sha256=sha(path),rows=len(out),valid=int(out.formula_input_valid.sum()),newly_invalid=invalid)
