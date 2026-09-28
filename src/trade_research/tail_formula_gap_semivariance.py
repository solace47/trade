"""Prior positive and negative overnight jump second moments, with fixed scope."""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_overnight as parent
from .corporate_cash import save_json,sha

STEM='tail_formula_gap_semivariance'
ROOT=Path('data/research')/STEM
PROTOCOL=Path('config')/(STEM+'_protocol.json')
DAILY_REPORT=Path('data/research/tail_formula_1000/feature_report.json')
HEADER=previous.HEADER+''.join(f'OG{i:02d}:=INTPART(REF(O,B{i}-1)*100+0.5)/MAX(INTPART(DCP{i+1}*100+0.5),1)-1;\n' for i in range(1,21))
NEW_EXPRESSIONS={name:'100*SQRT(('+ '+'.join(f'{fn}(OG{i:02d},0)*{fn}(OG{i:02d},0)' for i in range(1,21))+')/20)/V01'
    for name,fn in [('GD01','MIN'),('GU01','MAX')]}
EXPRESSIONS={**previous.EXPRESSIONS,**NEW_EXPRESSIONS}


def checked_sources():
    p=json.loads(PROTOCOL.read_text())
    for file,digest in p['source_hashes'].items():assert sha(Path(file))==digest
    for module in [previous,parent]:
        r=json.loads((module.ROOT/'feature_report.json').read_text());v=json.loads((module.ROOT/'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256']==sha(module.ROOT/'feature_report.json')
        assert r['features_sha256']==sha(module.ROOT/'features.parquet')
    for file,digest in json.loads(DAILY_REPORT.read_text())['source_sha256'].items():assert sha(Path(file))==digest
    assert p['history_days']==20 and p['new_fields']==list(NEW_EXPRESSIONS)
    assert (p['history_first'],p['history_last'])==('2023-06-01','2025-12-30') and not p['new_2026_prices_allowed']
    return p


def components(opens,previous_closes,scale):
    oc=np.floor(np.asarray(opens)*100+.5);pc=np.floor(np.asarray(previous_closes)*100+.5)
    assert oc.shape==pc.shape and oc.shape[1]==20
    with np.errstate(divide='ignore',invalid='ignore'):
        g=(oc-pc)/pc
        return np.column_stack([100*np.sqrt(np.mean(np.minimum(g,0)**2,axis=1))/scale,
                                100*np.sqrt(np.mean(np.maximum(g,0)**2,axis=1))/scale])


def features():
    p=checked_sources();assert not (ROOT/'feature_report.json').exists()
    c=base.conn();c.read_parquet(list(json.loads(DAILY_REPORT.read_text())['source_sha256'])).create_view('daily')
    hist=c.sql('''WITH a AS(SELECT date,code,open::DOUBLE AS o,close::DOUBLE AS cl,preclose::DOUBLE AS pre,
        adjustflag::DOUBLE AS adj FROM daily WHERE date BETWEEN '2023-06-01' AND '2025-12-30' AND tradestatus=1),
        l AS(SELECT *,lag(cl) OVER w AS pc,lag(adj) OVER w AS padj,lag(date) OVER w AS pdate
            FROM a WINDOW w AS(PARTITION BY code ORDER BY date)),
        atoms AS(SELECT *,coalesce(isfinite(o) AND isfinite(cl) AND isfinite(pc) AND least(o,cl,pc)>0
            AND adj=3 AND padj=3 AND abs(o-round(o,2))<=.0001 AND abs(cl-round(cl,2))<=.0001
            AND abs(pc-round(pc,2))<=.0001,false) AS good,
            (round(o*100)-round(pc*100))/round(pc*100) AS gap FROM l),
        hist AS(SELECT date,code,count(*) OVER w AS gs_rows,sum(good::INT) OVER w AS gs_good,
            min(date) OVER w AS gs_start,max(date) OVER w AS gs_end,min(pdate) OVER w AS gs_reference_start,
            sum(CASE WHEN abs(pre-pc)>.005 THEN 1 ELSE 0 END) OVER w AS gs_reference_breaks,
            sum(CASE WHEN good AND gap<0 THEN 1 ELSE 0 END) OVER w AS gs_down_count,
            sum(CASE WHEN good AND gap>0 THEN 1 ELSE 0 END) OVER w AS gs_up_count,
            avg(CASE WHEN good THEN power(least(gap,0),2) END) OVER w AS neg,
            avg(CASE WHEN good THEN power(greatest(gap,0),2) END) OVER w AS pos
            FROM atoms WINDOW w AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING))
        SELECT * EXCLUDE(neg,pos),CASE WHEN gs_rows=0 THEN NULL WHEN gs_down_count=0 THEN 0 ELSE greatest(neg,0) END AS gs_neg_sq,
            CASE WHEN gs_rows=0 THEN NULL WHEN gs_up_count=0 THEN 0 ELSE greatest(pos,0) END AS gs_pos_sq
        FROM hist WHERE date>='2024-01-01' ORDER BY date,code''').df();c.close()
    old=pd.read_parquet(previous.ROOT/'features.parquet')
    f=old.merge(hist,on=['date','code'],how='left',validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
    valid=(f.gs_rows.eq(20)&f.gs_good.eq(20)&f.gs_end.lt(f.date)&f.gs_reference_start.lt(f.gs_start)
        &np.isfinite(f[['gs_neg_sq','gs_pos_sq','V01']]).all(axis=1)&f.V01.gt(0))
    f['gap_source_valid']=valid;f['prior_formula_input_valid']=f.formula_input_valid
    f['GD01']=(100*np.sqrt(f.gs_neg_sq)/f.V01).where(valid)
    f['GU01']=(100*np.sqrt(f.gs_pos_sq)/f.V01).where(valid)
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True,exist_ok=True);hist.to_parquet(ROOT/'history.parquet',index=False,compression='zstd')
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),source_hashes=p['source_hashes'],history_sha256=sha(ROOT/'history.parquet'),
        features_sha256=sha(ROOT/'features.parquet'),rows=len(f),valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid&~f.formula_input_valid).sum()),
        valid_with_reference_breaks=int((f.formula_input_valid&f.gs_reference_breaks.gt(0)).sum()),
        expressions=EXPRESSIONS,native_header=HEADER,native_source_parity_verified=False,software_compilation_verified=False,
        all_original_keys_preserved=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',r);return {k:v for k,v in r.items() if k not in ['source_hashes','expressions','native_header']}


def verify_features():
    checked_sources();r=json.loads((ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['features_sha256']==sha(ROOT/'features.parquet')
    assert r['history_sha256']==sha(ROOT/'history.parquet')
    pieces=[]
    for file in json.loads(DAILY_REPORT.read_text())['source_sha256']:
        d=pd.read_parquet(file,columns=['date','code','open','close','preclose','adjustflag','tradestatus'],
            filters=[('date','>=','2023-06-01'),('date','<=','2025-12-30')])
        d=d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True)
        if d.empty:continue
        assert not d.date.duplicated().any()
        o=d.open.astype(float);cl=d.close.astype(float);pc=cl.shift()
        oc=np.floor(o*100+.5);cc=np.floor(cl*100+.5);pcc=cc.shift()
        good=(np.isfinite(o)&np.isfinite(cl)&np.isfinite(pc)&o.gt(0)&cl.gt(0)&pc.gt(0)
            &d.adjustflag.eq(3)&d.adjustflag.shift().eq(3)&o.sub(oc/100).abs().le(.0001)
            &cl.sub(cc/100).abs().le(.0001)&pc.sub(pcc/100).abs().le(.0001))
        g=(oc-pcc)/pcc;q=d[['date','code']].copy()
        q['gs_rows']=np.minimum(np.arange(len(d)),20);q['gs_good']=good.astype(int).rolling(20,min_periods=1).sum().shift()
        q['gs_start']=d.date.shift(20).fillna(d.date.iloc[0]);q.loc[0,'gs_start']=None
        q['gs_end']=d.date.shift();q['gs_reference_start']=d.date.shift(21).fillna(d.date.iloc[0]);q.loc[:1,'gs_reference_start']=None
        q['gs_reference_breaks']=d.preclose.sub(pc).abs().gt(.005).astype(int).rolling(20,min_periods=1).sum().shift()
        for side,condition in [('down',g.lt(0)),('up',g.gt(0))]:
            q['gs_'+side+'_count']=(good&condition).astype(int).rolling(20,min_periods=1).sum().shift()
        # Explicit windows avoid subtracting old squared values from rolling sums.
        for side,part,count in [('neg',g.clip(upper=0),'down'),('pos',g.clip(lower=0),'up')]:
            values=part.pow(2).where(good).to_numpy();means=np.full(len(d),np.nan)
            for i in range(1,len(d)):
                window=values[max(0,i-20):i];known=window[np.isfinite(window)]
                means[i]=float(known.sum()/len(known)) if len(known) else np.nan
            q['gs_'+side+'_sq']=pd.Series(means).mask(q['gs_'+count+'_count'].eq(0),0)
        pieces.append(q.loc[q.date.ge('2024-01-01')])
    expected=pd.concat(pieces,ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    actual=pd.read_parquet(ROOT/'history.parquet')
    pd.testing.assert_frame_equal(actual,expected[actual.columns],check_dtype=False,rtol=0,atol=3e-14)
    old=pd.read_parquet(previous.ROOT/'features.parquet');f=pd.read_parquet(ROOT/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    e=old[['date','code']].merge(expected,on=['date','code'],how='left',validate='one_to_one')
    valid=(e.gs_rows.eq(20)&e.gs_good.eq(20)&e.gs_end.lt(e.date)&e.gs_reference_start.lt(e.gs_start)
        &np.isfinite(e[['gs_neg_sq','gs_pos_sq']]).all(axis=1)&np.isfinite(old.V01)&old.V01.gt(0))
    values=np.column_stack([100*np.sqrt(e.gs_neg_sq)/old.V01,100*np.sqrt(e.gs_pos_sq)/old.V01])
    values[~valid]=np.nan
    np.testing.assert_allclose(f[list(NEW_EXPRESSIONS)],values,rtol=0,atol=2e-10,equal_nan=True)
    final=old.formula_input_valid&valid&np.isfinite(values).all(axis=1)
    np.testing.assert_array_equal(f.formula_input_valid,final)
    np.testing.assert_array_equal(np.floor(np.clip(values[final]*100+10000+.000001,0,999999)).astype('int32'),
        np.floor(np.clip(f.loc[final,list(NEW_EXPRESSIONS)].to_numpy()*100+10000+.000001,0,999999)).astype('int32'))
    par=pd.read_parquet(parent.ROOT/'features.parquet',columns=['date','code','history_source_valid'])
    pd.testing.assert_frame_equal(f[['date','code']],par[['date','code']],check_exact=True)
    assert not (valid&~par.history_source_valid).any()
    names=re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS)
    assert len(names)==len(set(names)) and r['expressions']==EXPRESSIONS and r['native_header']==HEADER
    proof=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=len(f),valid=int(final.sum()),
        all_previous_48_values_unchanged=True,all_prior_dates_and_two_sided_moments_independently_rebuilt=True,
        all_new_integer_encodings_rebuilt=True,effective_input_intersection_unchanged=bool(np.array_equal(final,old.formula_input_valid)),
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_verification.json',proof);return proof


def native():
    checked_sources();v=json.loads((ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256']==sha(ROOT/'feature_report.json')
    prior=json.loads((parent.ROOT/'feature_verification.json').read_text());cases=prior['original_cases']
    assert prior['raw_open_mismatches']==prior['raw_close_mismatches']==0 and len(cases)==32
    f=pd.read_parquet(ROOT/'features.parquet',columns=['date','code','formula_input_valid','V01','GD01','GU01']).set_index(['date','code'])
    c=base.conn();c.read_parquet(list(json.loads(DAILY_REPORT.read_text())['source_sha256'])).create_view('daily')
    records=[]
    for case in cases:
        row=f.loc[(case['date'],case['code'])];assert row.formula_input_valid
        d=c.execute('''SELECT date,open::DOUBLE AS o,close::DOUBLE AS cl FROM daily
            WHERE code=? AND date>=? AND date<? AND tradestatus=1 ORDER BY date''',[case['code'],case['source_start'],case['date']]).df()
        assert len(d)==21 and d.date.iloc[-1]==case['source_end']
        gaps=[math.floor(o*100+.5)/max(math.floor(cl*100+.5),1)-1 for o,cl in zip(d.o.iloc[1:],d.cl.iloc[:-1])]
        values=[100*math.sqrt(math.fsum(fn(g,0)**2 for g in gaps)/20)/row.V01 for fn in [min,max]]
        np.testing.assert_allclose(values,row[['GD01','GU01']].to_numpy(float),rtol=0,atol=2e-10)
        np.testing.assert_array_equal(np.floor(np.array(values)*100+10000+.000001),np.floor(row[['GD01','GU01']].to_numpy(float)*100+10000+.000001))
        records.append(dict(date=case['date'],code=case['code'],source_days=21,GD01=values[0],GU01=values[1]))
    c.close()
    proof=dict(passed=True,protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(ROOT/'feature_report.json'),
        feature_verification_sha256=sha(ROOT/'feature_verification.json'),reused_raw_price_proof_sha256=sha(parent.ROOT/'feature_verification.json'),
        cases=records,original_raw_windows_not_reextracted=True,both_native_formulas_and_encodings_rebuilt=True,
        software_compilation_verified=False,native_source_parity_verified=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'native_input_verification.json',proof);return {k:v for k,v in proof.items() if k!='cases'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['features','verify_features','native'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
