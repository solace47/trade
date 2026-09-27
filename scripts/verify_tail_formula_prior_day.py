"""Independent original-file lag and native aggregation checks for yesterday."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_prior_day as study
from trade_research.corporate_cash import MINUTES, save_json, sha


def main():
    root = study.ROOT; r = json.loads((root/'feature_report.json').read_text())
    for key,path in [('protocol_sha256',study.PROTOCOL),('previous_feature_report_sha256',study.previous.ROOT/'feature_report.json'),
                     ('daily_feature_report_sha256',study.base.SOURCE/'feature_report.json'),('features_sha256',root/'features.parquet'),
                     ('calendar_sha256',study.CALENDAR)]:
        assert r[key] == sha(path), key
    paths = study.source_files()
    f = pd.read_parquet(root/'features.parquet'); old = pd.read_parquet(study.previous.ROOT/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    assert f.prior_formula_input_valid.equals(old.formula_input_valid)
    groups = dict(tuple(f.groupby('code',sort=False)))
    cal = pd.read_parquet(study.CALENDAR)
    days = sorted(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between('2023-06-01','2025-12-30'),'calendar_date'])
    ranks = {d:i for i,d in enumerate(days)}
    cache = {}; checked = 0
    for path in paths:
        code = Path(path).stem.replace('_','.')
        if code not in groups:
            continue
        q = groups[code]
        d = pd.read_parquet(path,columns=['date','code','open','high','low','close','volume','preclose','adjustflag','tradestatus'],
            filters=[('date','>=','2023-06-01'),('date','<=','2025-12-30')])
        d = d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True)
        assert d.code.eq(code).all() and not d.date.duplicated().any()
        ix = pd.Index(d.date).get_indexer(q.date)
        assert (ix >= 20).all()
        h = d[['date','code']].copy()
        for name in ['open','high','low','close','volume','preclose','adjustflag']:
            h['yd_'+name] = d[name].shift()
        h['yd_source_date'] = d.date.shift(); h['yd_reference_date'] = d.date.shift(2)
        h['yd_previous_close'] = d.close.shift(2); h['yd_reference_adjustflag'] = d.adjustflag.shift(2)
        vgood = np.isfinite(d.volume) & d.volume.gt(0) & d.volume.eq(np.floor(d.volume)) & d.adjustflag.eq(3)
        h['yd_volume_mean'] = d.volume.astype(float).rolling(20,min_periods=1).mean().shift()
        h['yd_volume_rows'] = np.minimum(np.arange(len(d)),20)
        h['yd_volume_good'] = vgood.astype(int).rolling(20,min_periods=1).sum().shift()
        h['yd_volume_first_date'] = d.date.shift(20)
        e = h.iloc[ix].reset_index(drop=True)
        pd.testing.assert_frame_equal(q[e.columns].reset_index(drop=True),e,check_dtype=False,atol=2e-10,rtol=1e-13)
        o,hi,lo,cl,pc = (e[name].to_numpy(dtype=float) for name in ['yd_open','yd_high','yd_low','yd_close','yd_previous_close'])
        p = np.column_stack([o,hi,lo,cl,pc])
        valid = (np.isfinite(p).all(axis=1) & (p>0).all(axis=1) & (np.abs(p-np.rint(p*100)/100)<=.0001).all(axis=1)
            & (hi+.0001>=np.maximum.reduce([o,lo,cl])) & (lo-.0001<=np.minimum(o,cl))
            & e.yd_adjustflag.eq(3).to_numpy() & e.yd_reference_adjustflag.eq(3).to_numpy()
            & e.yd_volume_rows.eq(20).to_numpy() & e.yd_volume_good.eq(20).to_numpy() & e.yd_volume_mean.gt(0).to_numpy()
            & e.yd_source_date.lt(e.date).to_numpy() & e.yd_reference_date.lt(e.yd_source_date).to_numpy() & q.V01.gt(0).to_numpy())
        np.testing.assert_array_equal(q.prior_day_valid,valid)
        values = np.column_stack([100*(cl/pc-1)/q.V01.to_numpy(),100*(cl-lo)/np.maximum(hi-lo,.01),
            100*(hi-lo)/pc/q.V01.to_numpy(),e.yd_volume.to_numpy()/e.yd_volume_mean.to_numpy()])
        values[~valid] = np.nan
        np.testing.assert_allclose(q[list(study.NEW_EXPRESSIONS)],values,atol=2e-10,rtol=0,equal_nan=True)
        final = valid & q.prior_formula_input_valid.to_numpy() & np.isfinite(q[list(study.EXPRESSIONS)]).all(axis=1).to_numpy()
        np.testing.assert_array_equal(q.formula_input_valid,final)
        encode = lambda x: np.floor(np.clip(100*x+10000+.000001,0,999999))
        np.testing.assert_array_equal(encode(q.loc[final,list(study.NEW_EXPRESSIONS)].to_numpy()),encode(values[final]))
        for name,value in [('yd_source_gap',e.date.map(ranks)-e.yd_source_date.map(ranks)),
                           ('yd_history_span',e.yd_source_date.map(ranks)-e.yd_volume_first_date.map(ranks)+1),
                           ('yd_reference_break',(e.yd_preclose-e.yd_previous_close).abs().gt(.005))]:
            np.testing.assert_array_equal(q[name],value)
        checked += len(q); cache[code] = d
    assert checked == len(f) == r['rows']
    valid = f.formula_input_valid
    for name,value in [('valid',valid.sum()),('previous_valid',f.prior_formula_input_valid.sum()),
                       ('newly_invalid',(f.prior_formula_input_valid&~valid).sum()),
                       ('flat_prior_days',(valid&f.yd_high.eq(f.yd_low)).sum()),
                       ('prior_day_reference_breaks',(valid&f.yd_reference_break).sum()),
                       ('prior_stock_day_gaps',(valid&f.yd_source_gap.gt(1)).sum()),
                       ('volume_history_gaps',(valid&f.yd_history_span.gt(20)).sum())]:
        assert int(value) == r[name], name
    assert r['first_history_date'] == f.yd_volume_first_date.min() and r['last_history_date'] == f.yd_source_date.max()
    assert f.loc[valid&f.yd_high.eq(f.yd_low),'Y02'].eq(0).all()
    assert r['expressions'] == study.EXPRESSIONS and r['native_header'] == study.HEADER
    symbols = re.findall(r'(?m)^([A-Za-z][A-Za-z0-9]*):=',study.HEADER)+list(study.EXPRESSIONS)
    symbols += [f'T{i:02d}' for i in range(1,65)]+[f'X{i:02d}' for i in range(1,53)]
    assert len(symbols) == len(set(symbols))
    assert 'YDH:=REF(HHV(H,B0),B0);' in study.HEADER and 'YDL:=REF(LLV(L,B0),B0);' in study.HEADER
    for i in range(1,21):
        assert f'YDV{i:02d}:=REF(SUM(V,B0),B{i-1});' in study.HEADER
    assert study.NEW_EXPRESSIONS == {'Y01':'100*(DCP1/DCP2-1)/V01','Y02':'100*(DCP1-YDL)/MAX(YDH-YDL,0.01)',
        'Y03':'100*(YDH-YDL)/DCP2/V01','Y04':'20*YDV01/('+ '+'.join(f'YDV{i:02d}' for i in range(1,21))+')'}
    sample = f.loc[valid].copy()
    sample['hash'] = [hashlib.sha256(('prior-day-v1|'+d+'|'+c).encode()).hexdigest() for d,c in zip(sample.date,sample.code)]
    sample = sample.sort_values('hash').groupby('half',sort=True).head(8)
    hashes = json.loads(Path('data/research/economic_winner/input_manifest.json').read_text())['source_sha256']
    cases = []; source_minutes = 0
    for row in sample.itertuples():
        d = cache[row.code]; past = d.loc[d.date.lt(row.date)].tail(20)
        assert len(past) == 20 and past.date.iloc[-1] == row.yd_source_date and past.date.iloc[-2] == row.yd_reference_date
        corrupted = d.copy(); corrupted.loc[corrupted.date.ge(row.date),['open','high','low','close','volume']] = 999999.
        pd.testing.assert_frame_equal(past,corrupted.loc[corrupted.date.lt(row.date)].tail(20),check_exact=True)
        path = MINUTES/row.code[:2].upper()/(row.code[3:]+'.parquet'); assert sha(path) == hashes[str(path)]
        raw = pd.read_parquet(path,columns=['timestamp','open','high','low','close','volume'],
            filters=[('timestamp','>=',pd.Timestamp(past.date.iloc[0])),('timestamp','<',pd.Timestamp(row.date))]).sort_values('timestamp')
        source_minutes += len(raw); raw['date'] = raw.timestamp.dt.strftime('%Y-%m-%d')
        agg = raw.groupby('date').agg(o=('open','first'),h=('high','max'),l=('low','min'),c=('close','last'),v=('volume','sum')).tail(20)
        aligned = agg.index.tolist() == past.date.tolist()
        details = dict(date=row.date,code=row.code,prior_dates_aligned=aligned,prior_days=len(agg),field_differences={})
        if aligned:
            for name,a,b in [('volume',agg.v.to_numpy(),past.volume.to_numpy()),
                             ('previous_close',agg.c.iloc[-2:].round(2).to_numpy(),past.close.iloc[-2:].to_numpy()),
                             ('prior_ohlc',agg[['o','h','l','c']].iloc[-1:].round(2).to_numpy(),past[['open','high','low','close']].iloc[-1:].to_numpy())]:
                if not np.array_equal(a,b):
                    details['field_differences'][name] = float(np.max(np.abs(a-b)))
            # Explicit scalar evaluation of the four native expressions after legal cent restoration.
            p1,p2 = agg.c.iloc[-1:].round(2).iloc[0],agg.c.iloc[-2:].round(2).iloc[0]
            high,low = round(float(agg.h.iloc[-1]),2),round(float(agg.l.iloc[-1]),2)
            values = [100*(p1/p2-1)/row.V01,100*(p1-low)/max(high-low,.01),100*(high-low)/p2/row.V01,20*agg.v.iloc[-1]/sum(agg.v)]
            if not details['field_differences']:
                np.testing.assert_allclose(values,[row.Y01,row.Y02,row.Y03,row.Y04],atol=2e-10,rtol=0)
        cases.append(details)
    assert len(cases) == 32
    proof = dict(passed=True,feature_report_sha256=sha(root/'feature_report.json'),rows=len(f),valid=int(valid.sum()),
        all_previous_day_values_and_volume_histories_rebuilt=True,all_integer_encodings_rebuilt=True,
        original_48_values_and_keys_unchanged=True,all_native_variable_names_unique=True,
        fixed_native_cases=cases,native_case_count=len(cases),prior_daily_points=len(cases)*20,raw_minutes_rechecked=source_minutes,
        native_probe_date_mismatches=sum(not x['prior_dates_aligned'] for x in cases),
        native_probe_field_mismatches=sum(bool(x['field_differences']) for x in cases),
        current_and_future_contamination_does_not_change_prior_inputs=True,
        native_source_parity_verified=False,software_compilation_verified=False,
        outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'feature_verification.json',proof)
    return {k:v for k,v in proof.items() if k != 'fixed_native_cases'}


if __name__ == '__main__':
    print(json.dumps(main(),ensure_ascii=False,indent=2))
