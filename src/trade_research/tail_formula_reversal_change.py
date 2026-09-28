"""Change in completed-day reversal frequency versus disjoint older stock days."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from .corporate_cash import DAILY, MINUTES, save_json, sha
from .tail_formula_quarter_position import trading_minute_projection

STEM = 'tail_formula_reversal_change'
ROOT = Path('data/research') / STEM
CONTROL_INPUTS = Path('data/research') / (STEM+'_control_inputs')
PROTOCOL = Path('config') / (STEM+'_protocol.json')
DAILY_REPORT = Path('data/research/tail_formula_1000/feature_report.json')
MINUTE_MANIFEST = Path('data/research/economic_winner/input_manifest.json')
EXTRA_HEADER = ''.join(f'B{i}:=B{i-1}+REF(BARSLAST(DD)+1,B{i-1});\n' for i in range(21,260))
EVENT_HEADER = 'RAO:=REF(O,B0-1);\nRAN:=IF(RAO>REF(C,B0) AND C<RAO,1,0);\nRAP:=IF(RAO<REF(C,B0) AND C>RAO,1,0);\n'
EXTRA_HEADER += EVENT_HEADER
NEW_EXPRESSIONS = {name:'100*(('+'+'.join(f'REF({event},B{i})' for i in range(20))+')/20-('
    +'+'.join(f'REF({event},B{i})' for i in range(20,260))+')/240)'
    for name,event in [('RA01','RAN'),('RA02','RAP')]}
EXPRESSIONS = {**previous.EXPRESSIONS,**NEW_EXPRESSIONS}
HEADER = previous.HEADER+EXTRA_HEADER
native_core = base.native_core


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['history_first'] == '2022-01-01' and p['recent_stock_days'] == 20 and p['baseline_stock_days'] == 240
    assert p['expressions'] == NEW_EXPRESSIONS and p['native_header'] == EXTRA_HEADER
    assert not p['new_2026_prices_allowed']
    for file,digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    r = json.loads((previous.ROOT/'feature_report.json').read_text())
    v = json.loads((previous.ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(previous.ROOT/'feature_report.json')
    assert r['features_sha256'] == sha(previous.ROOT/'features.parquet')
    sources = json.loads(DAILY_REPORT.read_text())['source_sha256']
    for file,digest in sources.items():
        assert sha(Path(file)) == digest
    return p,sources


def features():
    p,sources = checked_sources()
    assert not (ROOT/'feature_report.json').exists()
    c = base.conn();c.read_parquet(list(sources)).create_view('daily')
    hist = c.sql('''WITH active AS(SELECT date,code,open::DOUBLE AS raw_o,close::DOUBLE AS raw_c,
        round(open::DOUBLE,2) AS o,round(close::DOUBLE,2) AS cl,preclose::DOUBLE AS preclose,
        adjustflag::DOUBLE AS adj FROM daily
        WHERE tradestatus=1 AND date BETWEEN '2022-01-01' AND '2025-12-30'),
        lagged AS(SELECT *,lag(cl) OVER w AS pc,lag(raw_c) OVER w AS raw_pc,lag(adj) OVER w AS padj,
            lag(date) OVER w AS pdate FROM active WINDOW w AS(PARTITION BY code ORDER BY date)),
        atoms AS(SELECT *,coalesce(isfinite(raw_o) AND raw_o>0 AND abs(raw_o-o)<=.0001
            AND isfinite(raw_c) AND raw_c>0 AND abs(raw_c-cl)<=.0001
            AND isfinite(raw_pc) AND raw_pc>0 AND abs(raw_pc-pc)<=.0001 AND adj=3 AND padj=3,false) AS good,
            (o>pc AND cl<o)::INT AS nr,(o<pc AND cl>o)::INT AS pr,
            coalesce(abs(preclose-pc)>.005,false)::INT AS reference_break FROM lagged),
        h AS(SELECT date,code,count(*) OVER w260 AS rc_rows,sum(good::INT) OVER w260 AS rc_good,
            min(date) OVER w260 AS rc_first_date,max(date) OVER w260 AS rc_last_date,
            first_value(pdate) OVER w260 AS rc_reference_date,
            sum(nr) OVER w20 AS rc_nr20,sum(pr) OVER w20 AS rc_pr20,
            sum(nr) OVER w240 AS rc_nr240,sum(pr) OVER w240 AS rc_pr240,
            sum(reference_break) OVER w260 AS rc_reference_breaks
            FROM atoms WINDOW w260 AS(PARTITION BY code ORDER BY date ROWS BETWEEN 260 PRECEDING AND 1 PRECEDING),
            w20 AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
            w240 AS(PARTITION BY code ORDER BY date ROWS BETWEEN 260 PRECEDING AND 21 PRECEDING))
        SELECT * FROM h WHERE date>='2024-01-01' ORDER BY date,code''').df();c.close()
    old = pd.read_parquet(previous.ROOT/'features.parquet')
    f = old.merge(hist,on=['date','code'],how='left',validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
    valid = f.rc_rows.eq(260) & f.rc_good.eq(260) & f.rc_reference_date.lt(f.rc_first_date) & f.rc_last_date.lt(f.date)
    f['reversal_change_valid'] = valid
    f['RA01'] = (100*(f.rc_nr20/20-f.rc_nr240/240)).where(valid)
    f['RA02'] = (100*(f.rc_pr20/20-f.rc_pr240/240)).where(valid)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(NEW_EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True,exist_ok=True)
    hist.to_parquet(ROOT/'history.parquet',index=False,compression='zstd')
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(ROOT/'features.parquet'),
        history_sha256=sha(ROOT/'history.parquet'),rows=len(f),valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        valid_with_reference_breaks=int((f.formula_input_valid & f.rc_reference_breaks.gt(0)).sum()),
        first_reference_date=f.rc_reference_date.dropna().min(),last_history_date=f.rc_last_date.dropna().max(),
        expressions=EXPRESSIONS,native_header=HEADER,raw_unadjusted_price_path=True,
        software_history_depth_verified=False,software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',r)
    return {k:v for k,v in r.items() if k not in ['expressions','native_header']}


def rebuild_history(d):
    d = d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True)
    assert not d.date.duplicated().any()
    if d.empty:
        return None
    o = np.floor(d.open.astype(float)*100+.5)/100
    cl = np.floor(d.close.astype(float)*100+.5)/100
    good_price = lambda raw,rounded:np.isfinite(raw) & raw.gt(0) & raw.sub(rounded).abs().le(.0001)
    good_close = good_price(d.close,cl) & d.adjustflag.eq(3)
    good = good_price(d.open,o) & good_close & good_close.shift(fill_value=False)
    nr = (o.gt(cl.shift()) & cl.lt(o)).astype(float)
    pr = (o.lt(cl.shift()) & cl.gt(o)).astype(float)
    # SQL comparisons to the missing previous close are unknown unless the
    # second conjunct is false. These atoms are never valid training inputs.
    nr.loc[0] = np.nan if cl.iloc[0]<o.iloc[0] else 0.
    pr.loc[0] = np.nan if cl.iloc[0]>o.iloc[0] else 0.
    out = d[['date','code']].copy()
    out['rc_rows'] = np.minimum(np.arange(len(d)),260)
    out['rc_good'] = good.astype(int).rolling(260,min_periods=1).sum().shift()
    out['rc_first_date'] = d.date.shift(260).fillna(d.date.iloc[0]);out.loc[0,'rc_first_date'] = None
    out['rc_last_date'] = d.date.shift()
    out['rc_reference_date'] = d.date.shift(261)
    for name,event in [('nr',nr),('pr',pr)]:
        out['rc_'+name+'20'] = event.rolling(20,min_periods=1).sum().shift()
        out['rc_'+name+'240'] = event.rolling(240,min_periods=1).sum().shift(21)
    breaks = d.preclose.sub(cl.shift()).abs().gt(.005).astype(int)
    out['rc_reference_breaks'] = breaks.rolling(260,min_periods=1).sum().shift()
    return out


def verify_features():
    p,sources = checked_sources()
    r = json.loads((ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL)
    for key in ['history','features']:
        assert r[key+'_sha256'] == sha(ROOT/(key+'.parquet'))
    chunks = []
    for file in sources:
        d = pd.read_parquet(file,columns=['date','code','open','close','preclose','tradestatus','adjustflag'],
            filters=[('date','>=','2022-01-01'),('date','<=','2025-12-30')])
        rebuilt = rebuild_history(d)
        if rebuilt is not None:
            chunks.append(rebuilt.loc[rebuilt.date.ge('2024-01-01')])
    hist = pd.concat(chunks,ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    actual = pd.read_parquet(ROOT/'history.parquet')
    pd.testing.assert_frame_equal(actual,hist[actual.columns],check_dtype=False,check_exact=True)
    old = pd.read_parquet(previous.ROOT/'features.parquet');got = pd.read_parquet(ROOT/'features.parquet')
    ex = old[['date','code']].merge(hist,on=['date','code'],how='left',validate='one_to_one')
    pd.testing.assert_frame_equal(got[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    pd.testing.assert_series_equal(got.prior_formula_input_valid,old.formula_input_valid,check_names=False,check_exact=True)
    pd.testing.assert_frame_equal(got[ex.columns],ex,check_dtype=False,check_exact=True)
    valid = ex.rc_rows.eq(260) & ex.rc_good.eq(260) & ex.rc_reference_date.lt(ex.rc_first_date) & ex.rc_last_date.lt(ex.date)
    np.testing.assert_array_equal(got.reversal_change_valid,valid)
    final = valid & old.formula_input_valid
    np.testing.assert_array_equal(got.formula_input_valid,final)
    for name,event in [('RA01','nr'),('RA02','pr')]:
        values = (100*(ex['rc_'+event+'20']/20-ex['rc_'+event+'240']/240)).where(valid)
        np.testing.assert_allclose(got[name],values,rtol=0,atol=2e-12,equal_nan=True)
        encode = lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999)).astype('int32')
        np.testing.assert_array_equal(encode(got.loc[final,name]),encode(values.loc[final]))
        assert got.loc[final,name].between(-100,100).all()
    assert r['rows'] == len(got) == p['expected_keys'] and r['valid'] == int(final.sum())
    assert r['newly_invalid'] == int((old.formula_input_valid & ~final).sum())
    assert r['valid_with_reference_breaks'] == int((final & ex.rc_reference_breaks.gt(0)).sum())
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names})
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    proof = dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=len(got),history_rows=len(hist),
        all_260_prior_events_and_disjoint_20_240_windows_independently_rebuilt=True,
        all_old_values_keys_unknowns_and_integer_encodings_verified=True,
        current_signal_day_not_in_history=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_verification.json',proof)
    return proof


def native_values(opens,closes,dates):
    opens,closes,dates = np.asarray(opens,float),np.asarray(closes,float),np.asarray(dates)
    assert len(opens) == len(closes) == len(dates)
    dd = np.r_[True,dates[1:]!=dates[:-1]]
    bars = np.arange(len(dates))-np.maximum.accumulate(np.where(dd,np.arange(len(dates)),0))+1
    def series_ref(values,lag):
        lag = np.broadcast_to(np.asarray(lag,float),len(values))
        valid = np.isfinite(lag) & (lag>=0) & (lag==np.floor(lag))
        offsets = np.where(valid,lag,0).astype(int)
        index = np.arange(len(values))-offsets
        valid &= index>=0
        out = np.full(len(values),np.nan)
        out[valid] = np.asarray(values)[index[valid]]
        return out
    env = dict(O=opens,C=closes,B0=bars,REF=series_ref,IF=np.where)
    for statement in EVENT_HEADER.strip().split(';'):
        if not statement:continue
        name,expr = statement.strip().split(':=')
        if ' AND ' in expr:
            expr = re.sub(r'IF\((.+) AND (.+),1,0\)',r'IF((\1)&(\2),1,0)',expr)
        env[name] = eval(expr,{'__builtins__':{}},env)
    def scalar_ref(values,lag):
        assert np.isfinite(lag) and lag>=0 and lag==int(lag)
        index = len(values)-1-int(lag)
        assert index>=0
        return np.asarray(values)[index]
    env.update(B0=int(bars[-1]),DD=dd,BARSLAST=lambda _:bars-1,REF=scalar_ref)
    for line in HEADER.splitlines():
        if re.match(r'B[1-9][0-9]*:=',line):
            name,expr = line.rstrip(';').split(':=')
            env[name] = eval(expr,{'__builtins__':{}},env)
    values = [float(eval(expr,{'__builtins__':{}},env)) for expr in NEW_EXPRESSIONS.values()]
    return values,[int(env[f'B{i}']) for i in range(260)]


def native():
    _,daily_hashes = checked_sources()
    v = json.loads((ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT/'feature_report.json')
    f = pd.read_parquet(ROOT/'features.parquet');f = f.loc[f.formula_input_valid].copy()
    f['identity'] = [hashlib.sha256((d+'|'+s+'|reversal-change-native-v1').encode()).hexdigest() for d,s in zip(f.date,f.code)]
    selected = f.sort_values('identity').groupby('half',sort=True).head(8).sort_values(['date','code'])
    assert len(selected) == 32
    hashes = json.loads(MINUTE_MANIFEST.read_text())['source_sha256'];receipts = []
    for row in selected.itertuples():
        daily_path = DAILY/(row.code.replace('.','_')+'.parquet')
        assert sha(daily_path) == daily_hashes[str(daily_path)]
        daily_all = pd.read_parquet(daily_path,columns=['date','open','close','tradestatus'],
            filters=[('date','>=',row.rc_reference_date),('date','<',row.date)])
        daily = daily_all.loc[daily_all.tradestatus.eq(1)].sort_values('date')
        assert len(daily) == 261 and daily.date.iloc[-1] == row.rc_last_date
        path = MINUTES/row.code[:2].upper()/(row.code[3:]+'.parquet');assert sha(path) == hashes[str(path)]
        raw = pd.read_parquet(path,columns=['timestamp','open','close','volume','turnover'],filters=[
            ('timestamp','>=',pd.Timestamp(row.rc_reference_date)),('timestamp','<=',pd.Timestamp(row.date+' 14:49'))]).sort_values('timestamp').reset_index(drop=True)
        vendor_rows = len(raw);raw,removed = trading_minute_projection(raw,daily_all,row.date)
        dates = raw.timestamp.dt.strftime('%Y-%m-%d')
        assert dates.drop_duplicates().tolist() == daily.date.tolist()+[row.date]
        assert not raw.timestamp.duplicated().any()
        first,last = raw.groupby(dates,sort=True).head(1),raw.groupby(dates,sort=True).tail(1)
        assert last.timestamp.dt.strftime('%H:%M').tolist() == ['15:00']*261+['14:49']
        close = np.floor(raw.close.to_numpy(float)*100+.5)/100
        opened = np.floor(raw.open.to_numpy(float)*100+.5)/100
        np.testing.assert_allclose(close[last.index[:-1]],daily.close,rtol=0,atol=2e-12)
        np.testing.assert_allclose(opened[first.index[:-1]],daily.open,rtol=0,atol=2e-12)
        assert close[-1] == row.price_1449
        values,offsets = native_values(opened,close,dates)
        np.testing.assert_array_equal(close[len(close)-1-np.asarray(offsets)],daily.close.to_numpy()[::-1][:260])
        np.testing.assert_allclose(values,[row.RA01,row.RA02],rtol=0,atol=2e-10)
        enc = lambda a:np.floor(np.clip(100*np.asarray(a)+10000+.000001,0,999999))
        np.testing.assert_array_equal(enc(values),enc([row.RA01,row.RA02]))
        receipts.append(dict(date=row.date,code=row.code,first_history_date=row.rc_reference_date,
            raw_minutes=len(raw),vendor_rows=vendor_rows,removed_halt_padding=removed,source_sha256=hashes[str(path)],daily_source_sha256=daily_hashes[str(daily_path)]))
    proof = dict(passed=True,protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(ROOT/'feature_report.json'),
        feature_verification_sha256=sha(ROOT/'feature_verification.json'),samples=receipts,
        raw_minutes=sum(x['raw_minutes'] for x in receipts),all_generated_260_offsets_events_and_sums_replayed=True,
        conditional_active_day_projection=True,software_history_depth_verified=False,software_compilation_verified=False,
        native_source_parity_verified=False,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'native_input_verification.json',proof)
    return {k:v for k,v in proof.items() if k!='samples'}


def control():
    v = json.loads((ROOT/'feature_verification.json').read_text())
    n = json.loads((ROOT/'native_input_verification.json').read_text())
    assert v['passed'] and n['passed'] and v['feature_report_sha256'] == n['feature_report_sha256'] == sha(ROOT/'feature_report.json')
    r = json.loads((ROOT/'feature_report.json').read_text())
    assert r['features_sha256'] == sha(ROOT/'features.parquet')
    assert not (CONTROL_INPUTS/'feature_report.json').exists()
    CONTROL_INPUTS.mkdir(parents=True,exist_ok=True)
    os.link(ROOT/'features.parquet',CONTROL_INPUTS/'features.parquet')
    r.update(expressions=previous.EXPRESSIONS,native_header=previous.HEADER,
        source_feature_report_sha256=sha(ROOT/'feature_report.json'),same_exact_table_and_validity=True,
        all_extra_columns_are_unused_metadata=True)
    save_json(CONTROL_INPUTS/'feature_report.json',r)
    proof = dict(passed=True,feature_report_sha256=sha(CONTROL_INPUTS/'feature_report.json'),
        source_feature_verification_sha256=sha(ROOT/'feature_verification.json'),
        same_exact_table_bytes=True,original_48_expressions_unchanged=True,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(CONTROL_INPUTS/'feature_verification.json',proof)
    return proof


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['features','verify_features','native','control'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
