"""Independently rebuild prior-day direction volumes and native minute histories."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from trade_research import tail_formula_history_direction as study
from trade_research.corporate_cash import DAILY, MINUTES, save_json, sha

ROOT = study.ROOT


def features():
    files = study.checked_sources(); rebuilt = []
    r = json.loads((ROOT / 'feature_report.json').read_text())
    for key, path in [('protocol_sha256', study.PROTOCOL), ('history_sha256', ROOT / 'history.parquet'),
                      ('features_sha256', ROOT / 'features.parquet'), ('calendar_sha256', study.CALENDAR)]:
        assert r[key] == sha(path)
    for file in files:
        d = pd.read_parquet(file, columns=['date','code','close','preclose','volume','tradestatus','adjustflag'],
            filters=[('date','>=','2023-06-01'),('date','<=','2025-12-30')])
        d = d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True)
        if d.empty:
            continue
        assert not d.date.duplicated().any()
        q = np.floor(d.close.astype(float)*100+.5)/100; prior = q.shift()
        good = (np.isfinite(d.close) & d.close.gt(0) & d.close.sub(q).abs().le(.0001)
            & np.isfinite(d.close.shift()) & d.close.shift().gt(0) & d.close.shift().sub(prior).abs().le(.0001)
            & d.adjustflag.eq(3) & d.adjustflag.shift().eq(3) & np.isfinite(d.volume)
            & d.volume.ge(0) & d.volume.eq(np.floor(d.volume)))
        signed = d.volume*np.select([q.gt(prior),q.lt(prior)],[1.,-1.],default=0.)
        out = d[['date','code']].copy()
        out['hd_rows20'] = np.minimum(np.arange(len(d)),20)
        out['hd_rows5'] = np.minimum(np.arange(len(d)),5)
        out['hd_good20'] = good.astype(int).rolling(20,min_periods=1).sum().shift()
        out['hd_first_date'] = d.date.shift(20).fillna(d.date.iloc[0]); out.loc[0,'hd_first_date'] = None
        out['hd_last_date'] = d.date.shift(); out['hd_reference_date'] = d.date.shift(21)
        for length in [20,5]:
            out[f'hd_volume{length}'] = d.volume.rolling(length,min_periods=1).sum().shift()
            out[f'hd_signed{length}'] = signed.rolling(length,min_periods=1).sum().shift()
        breaks = (np.floor(d.preclose*100+.5)/100-prior).abs().gt(.005)
        out['hd_reference_breaks'] = breaks.astype(int).rolling(20,min_periods=1).sum().shift()
        rebuilt.append(out)
    hist = pd.concat(rebuilt,ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    actual_hist = pd.read_parquet(ROOT / 'history.parquet')
    pd.testing.assert_frame_equal(actual_hist,hist[actual_hist.columns],check_dtype=False,rtol=0,atol=1e-7)
    old = pd.read_parquet(study.previous.ROOT / 'features.parquet')
    f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    assert f.prior_formula_input_valid.equals(old.formula_input_valid)
    e = old[['date','code','formula_input_valid']].merge(hist,on=['date','code'],how='left',validate='one_to_one')
    pd.testing.assert_frame_equal(f[hist.columns],e[hist.columns],check_dtype=False,rtol=0,atol=1e-7)
    valid = (e.hd_rows20.eq(20)&e.hd_rows5.eq(5)&e.hd_good20.eq(20)&e.hd_volume20.gt(0)&e.hd_volume5.gt(0)
             &e.hd_reference_date.lt(e.hd_first_date)&e.hd_last_date.lt(e.date))
    expected = pd.DataFrame(dict(DN01=(e.hd_signed20/e.hd_volume20*100).where(valid),
        DN02=((e.hd_signed5*e.hd_volume20-e.hd_signed20*e.hd_volume5)/(e.hd_volume5*e.hd_volume20)*50).where(valid)))
    np.testing.assert_allclose(f[list(expected)],expected,rtol=0,atol=2e-12,equal_nan=True)
    assert valid.equals(f.history_direction_valid)
    final_valid = old.formula_input_valid & valid & np.isfinite(expected).all(axis=1)
    assert final_valid.equals(f.formula_input_valid) and int(final_valid.sum()) == r['valid']
    assert r['newly_invalid'] == int((old.formula_input_valid & ~final_valid).sum())
    a,b = f.loc[final_valid,list(expected)].to_numpy(),expected.loc[final_valid].to_numpy()
    assert np.abs(a).max() <= 100+1e-12
    np.testing.assert_array_equal(np.floor(np.clip(a*100+10000+.000001,0,999999)),
                                  np.floor(np.clip(b*100+10000+.000001,0,999999)))
    cal = pd.read_parquet(study.CALENDAR)
    dates = sorted(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between('2023-06-01','2025-12-30'),'calendar_date'])
    ranks = {d:i for i,d in enumerate(dates)}
    np.testing.assert_allclose(f.hd_market_span,e.hd_last_date.map(ranks)-e.hd_first_date.map(ranks)+1,rtol=0,atol=0,equal_nan=True)
    np.testing.assert_allclose(f.hd_last_gap,e.date.map(ranks)-e.hd_last_date.map(ranks),rtol=0,atol=0,equal_nan=True)
    assert r['valid_with_reference_breaks'] == int((final_valid & e.hd_reference_breaks.gt(0)).sum())
    assert r['valid_with_market_gaps'] == int((final_valid & (f.hd_market_span.gt(20)|f.hd_last_gap.gt(1))).sum())
    assert r['expressions'] == study.EXPRESSIONS and r['native_header'] == study.HEADER
    names = re.findall(r'(?m)^([A-Za-z][A-Za-z0-9]*):=',study.HEADER) + list(study.EXPRESSIONS)
    assert len(names) == len(set(names))
    proof = dict(passed=True,feature_report_sha256=sha(ROOT / 'feature_report.json'),rows=len(f),history_rows=len(hist),
        valid=int(final_valid.sum()),all_stock_day_lags_direction_volumes_and_windows_independently_rebuilt=True,
        all_values_encodings_and_validity_verified=True,original_48_inputs_unchanged=True,
        new_selection_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json',proof); return proof


def native():
    study.checked_sources()
    proof = json.loads((ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    f = pd.read_parquet(ROOT / 'features.parquet',columns=['date','code','half','formula_input_valid','history_direction_valid','DN01','DN02'])
    f['identity'] = [hashlib.sha256(f'history_direction_native|{d}|{c}'.encode()).hexdigest() for d,c in zip(f.date,f.code)]
    sample = f.sort_values('identity').groupby(['half','formula_input_valid'],sort=True).head(4)
    source_manifest = Path('data/research/economic_winner/input_manifest.json')
    hashes = json.loads(source_manifest.read_text())['source_sha256']
    cases=[]; sources={}; total=0; maximum_volume_difference=0.; maximum_scalar_difference=0.
    for row in sample.itertuples(index=False):
        d = pd.read_parquet(DAILY / (row.code.replace('.','_')+'.parquet'),
            columns=['date','close','volume','tradestatus'],filters=[('date','>=','2023-06-01'),('date','<=',row.date)])
        d = d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True)
        wanted = d.loc[d.date.lt(row.date),'date'].tail(21).tolist(); assert len(wanted)==21
        path = MINUTES / row.code[:2].upper() / (row.code[3:]+'.parquet')
        if str(path) not in sources:
            assert sha(path)==hashes[str(path)]; sources[str(path)]=hashes[str(path)]
        q = pd.read_parquet(path,columns=['timestamp','close','volume'],filters=[
            ('timestamp','>=',pd.Timestamp(wanted[0])),('timestamp','<=',pd.Timestamp(row.date+' 14:49:00'))])
        q['date']=q.timestamp.dt.strftime('%Y-%m-%d');q=q.loc[q.date.isin(wanted+[row.date])].sort_values('timestamp').reset_index(drop=True)
        assert q.timestamp.max()<=pd.Timestamp(row.date+' 14:49:00')
        total+=len(q)
        q['price']=np.floor(q.close.astype(float)*100+.5)/100
        # Literal minute-series replay: B0 counts current stock-day bars,
        # REF(C,B0) always refers to the previous stock-day's final bar.
        b0=q.groupby('date').cumcount().to_numpy()+1
        offsets=np.arange(len(q))-b0; valid_offset=offsets>=0
        prior=np.full(len(q),np.nan);prior[valid_offset]=q.price.to_numpy()[offsets[valid_offset]]
        q['HDV']=q.groupby('date').volume.cumsum()
        q['HDS']=q.HDV*np.select([q.price.gt(prior),q.price.lt(prior)],[1.,-1.],default=0.)
        ends=q.groupby('date',sort=True).tail(1).set_index('date')
        hist=d.set_index('date').loc[wanted];actual=ends.loc[wanted]
        if row.history_direction_valid:
            assert q.loc[q.date.isin(wanted)].groupby('date').size().eq(241).all()
            assert not q.timestamp.duplicated().any()
            np.testing.assert_allclose(actual.price,hist.close,rtol=0,atol=.0001)
            differences=(actual.HDV-hist.volume).abs()
            assert differences.le(100).all()
            maximum_volume_difference=max(maximum_volume_difference,float(differences.max()))
            h=ends.loc[wanted[-20:]]
            v20=float(h.HDV.sum());v5=float(h.HDV.tail(5).sum())
            dn1=100*float(h.HDS.sum())/v20
            dn2=50*(float(h.HDS.tail(5).sum())/v5-float(h.HDS.sum())/v20)
            native=np.array([dn1,dn2]);observed=np.array([row.DN01,row.DN02])
            maximum_scalar_difference=max(maximum_scalar_difference,float(np.abs(native-observed).max()))
            np.testing.assert_array_equal(np.floor(np.clip(native*100+10000+.000001,0,999999)),
                                          np.floor(np.clip(observed*100+10000+.000001,0,999999)))
        else:
            assert pd.isna(row.DN01) and pd.isna(row.DN02)
        cases.append(dict(date=row.date,code=row.code,formula_input_valid=bool(row.formula_input_valid),
            history_direction_valid=bool(row.history_direction_valid),raw_minutes=len(q),previous_stock_days=21))
    assert len(cases)==32
    r=dict(passed=True,feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'),source_manifest_sha256=sha(source_manifest),
        samples=len(cases),raw_minutes=total,cases=cases,source_sha256=sources,
        maximum_daily_volume_difference_shares=maximum_volume_difference,maximum_native_scalar_difference=maximum_scalar_difference,
        all_sampled_native_day_end_references_and_final_integer_encodings_verified=True,
        raw_close_and_volume_checked_against_daily_source=True,
        software_compilation_verified=False,native_client_data_parity_verified=False,
        new_selection_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json',r)
    return {k:v for k,v in r.items() if k not in ['cases','source_sha256']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['features','native'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
