"""Independently rebuild past-volume means and replay fixed raw minute cases."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from trade_research import tail_formula_volume_history as study
from trade_research.corporate_cash import MINUTES, save_json, sha


def main():
    p,old=study.checked_source(); m=study.manifest(); root=study.ROOT
    report=json.loads((root/'feature_report.json').read_text())
    for key,path in [('protocol_sha256',study.PROTOCOL),('input_manifest_sha256',root/'input_manifest.json'),
        ('window_report_sha256',root/'window_report.json'),('previous_feature_report_sha256',study.previous.ROOT/'feature_report.json'),
        ('features_sha256',root/'features.parquet'),('history_sha256',root/'history.parquet')]:
        assert report[key]==sha(path)
    wr=json.loads((root/'window_report.json').read_text())
    assert wr['windows_sha256']==sha(root/'windows.parquet')
    for path,digest in wr['parts_sha256'].items():
        assert sha(Path(path))==digest
        meta=json.loads(Path(path).with_suffix('.json').read_text())
        assert meta['sha256']==digest and meta['input_manifest_sha256']==sha(root/'input_manifest.json')
        assert meta['extractor_sha256']==sha(Path(study.__file__))
    for path,digest in m['daily_source_sha256'].items():
        assert sha(Path(path))==digest
    c=study.base.conn(); c.read_parquet(list(m['daily_source_sha256'])).create_view('daily_source')
    expected_days=c.execute('''SELECT date,code FROM daily_source
        WHERE date>=? AND date<=? AND CAST(tradestatus AS DOUBLE)=1 ORDER BY code,date''',
        [p['history_first'],p['signal_last']]).df()
    got_days=pd.read_parquet(root/'stock_days.parquet')
    pd.testing.assert_frame_equal(got_days,expected_days,check_exact=True)
    c.register('stock_days',expected_days)
    windows=pd.read_parquet(root/'windows.parquet'); c.register('windows',windows)
    assert not windows.duplicated(['code','date']).any() and len(windows)==wr['rows']
    assert int(windows.bars.sum())==wr['raw_minutes']
    concat=pd.concat([pd.read_parquet(x) for x in wr['parts_sha256']],ignore_index=True).sort_values(['code','date']).reset_index(drop=True)
    pd.testing.assert_frame_equal(windows,concat,check_exact=True)
    c.execute('''CREATE VIEW hist AS WITH j AS(SELECT s.date,s.code,w.tail_volume29,w.tail_volume4,
        coalesce(w.volume_window_valid,false) AS volume_window_valid FROM stock_days s LEFT JOIN windows w USING(date,code)),
        h AS(SELECT *,lag(date,20) OVER w AS history_first_date,lag(date) OVER w AS history_last_date,
        CASE WHEN count(*) OVER prior=20 THEN sum(CAST(volume_window_valid AS INTEGER)) OVER prior END AS history_valid_count,
        CASE WHEN count(CASE WHEN volume_window_valid THEN tail_volume29 END) OVER prior=20
            THEN avg(CASE WHEN volume_window_valid THEN tail_volume29 END) OVER prior END AS history_volume29,
        CASE WHEN count(CASE WHEN volume_window_valid THEN tail_volume4 END) OVER prior=20
            THEN avg(CASE WHEN volume_window_valid THEN tail_volume4 END) OVER prior END AS history_volume4
        FROM j WINDOW w AS(PARTITION BY code ORDER BY date),
        prior AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)) SELECT * FROM h''')
    h=c.sql('SELECT * FROM hist ORDER BY date,code').df()
    saved=pd.read_parquet(root/'history.parquet')
    pd.testing.assert_frame_equal(saved[h.columns],h,check_dtype=False,rtol=0,atol=1e-8)
    f=pd.read_parquet(root/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    assert f.prior_formula_input_valid.equals(old.formula_input_valid)
    c.register('original',old)
    new=c.sql('''WITH q AS(SELECT o.date,o.code,o.v29,o.v4,h.* EXCLUDE(date,code),
        coalesce(history_valid_count=20 AND history_first_date IS NOT NULL AND history_last_date<o.date
            AND isfinite(history_volume29) AND isfinite(history_volume4) AND history_volume29>0 AND history_volume4>0,false)
            AS volume_history_valid,
        coalesce(volume_window_valid AND o.v29=tail_volume29 AND o.v4=tail_volume4,false) AS volume_current_valid
        FROM original o LEFT JOIN hist h USING(date,code))
        SELECT date,code,history_first_date,history_last_date,history_valid_count,history_volume29,history_volume4,
            tail_volume29,tail_volume4,volume_window_valid,volume_history_valid,volume_current_valid,
            CASE WHEN volume_history_valid AND volume_current_valid THEN v29/history_volume29 END AS HV01,
            CASE WHEN volume_history_valid AND volume_current_valid THEN v4/history_volume4 END AS HV02
        FROM q ORDER BY date,code''').df()
    pd.testing.assert_frame_equal(f[new.columns],new,check_dtype=False,rtol=0,atol=1e-8)
    expected_valid=old.formula_input_valid & new.volume_history_valid & new.volume_current_valid & np.isfinite(new[['HV01','HV02']]).all(axis=1)
    assert expected_valid.equals(f.formula_input_valid)
    def encode(values):
        return np.floor(np.clip(100*values+10000+.000001,0,999999)).astype('int32')
    np.testing.assert_array_equal(encode(f.loc[expected_valid,['HV01','HV02']].to_numpy()),encode(new.loc[expected_valid,['HV01','HV02']].to_numpy()))
    assert len(f)==report['rows'] and int(expected_valid.sum())==report['valid']
    assert int((old.formula_input_valid&~expected_valid).sum())==report['newly_invalid']
    assert int((~new.volume_history_valid).sum())==report['invalid_history']
    assert int((~new.volume_current_valid).sum())==report['invalid_current_volume']
    assert f.date.between(p['signal_first'],p['signal_last']).all()
    assert list(report['expressions'].items())==list(study.EXPRESSIONS.items()) and report['native_header']==study.HEADER
    names=re.findall(r'\b([A-Z][A-Z0-9]*):=',study.HEADER)
    assert len(names)==len(set(names)) and len(study.EXPRESSIONS)==50
    for label,source in [('HV29','TV29'),('HV4','TV4')]:
        line=next(s for s in study.HEADER.splitlines() if s.startswith(label+':='))
        assert re.findall(r'REF\((TV29|TV4),B(\d+)\)',line)==[(source,str(i)) for i in range(20)]

    sample=f[['date','code','half','volume_history_valid','HV01','HV02']].copy()
    sample['key_hash']=[hashlib.sha256(('tail-volume-history/'+d+'/'+k).encode()).hexdigest() for d,k in zip(sample.date,sample.code)]
    sample=sample.sort_values('key_hash').groupby(['half','volume_history_valid'],sort=True).head(4)
    window_index=windows.set_index(['code','date']); checked=set(); cases=[]; checked_minutes=0
    for row in sample.sort_values(['code','date']).itertuples():
        path=MINUTES/row.code[:2].upper()/(row.code[3:]+'.parquet')
        if path not in checked:
            assert sha(path)==m['minute_source_sha256'][str(path)];checked.add(path)
        dates=expected_days.loc[expected_days.code.eq(row.code)&expected_days.date.le(row.date),'date'].tail(21).tolist()
        assert dates[-1]==row.date
        raw=c.execute('''SELECT timestamp,volume::DOUBLE AS volume FROM read_parquet(?)
            WHERE timestamp>=?::DATE AND timestamp<=?::TIMESTAMP
            AND strftime(timestamp,'%H%M') BETWEEN '1421' AND '1449' ORDER BY timestamp''',
            [str(path),dates[0],row.date+' 14:49:00']).df()
        raw['date']=raw.timestamp.dt.strftime('%Y-%m-%d'); raw['clock']=raw.timestamp.dt.strftime('%H%M')
        values=[]; minutes=0
        for day in dates:
            q=raw.loc[raw.date.eq(day)]
            valid=(q.timestamp.eq(q.timestamp.dt.floor('min'))&np.isfinite(q.volume)&q.volume.ge(0)&q.volume.eq(np.floor(q.volume)))
            expected_times=list(pd.date_range(day+' 14:21:00',day+' 14:49:00',freq='min'))
            good=q.timestamp.tolist()==expected_times and bool(valid.all())
            minutes+=len(q)
            if q.empty:
                assert (row.code,day) not in window_index.index
                sums=[np.nan,np.nan]
            else:
                saved_window=window_index.loc[(row.code,day)]
                assert saved_window.volume_window_valid==good
                assert [saved_window.bars,saved_window.clocks,saved_window.good_bars,saved_window.bars4]==[
                    len(q),q.clock.nunique(),int(valid.sum()),int(q.clock.ge('1446').sum())]
                sums=[float(q.loc[valid,'volume'].sum()) if valid.any() else np.nan,
                      float(q.loc[valid&q.clock.ge('1446'),'volume'].sum()) if (valid&q.clock.ge('1446')).any() else np.nan]
                np.testing.assert_allclose(sums,[saved_window.tail_volume29,saved_window.tail_volume4],rtol=0,atol=1e-8,equal_nan=True)
            values.append((good,*sums))
        history_good=len(values)==21 and all(v[0] for v in values[:-1])
        means=np.mean(np.array([v[1:] for v in values[:-1]],dtype=float),axis=0) if history_good else np.array([np.nan,np.nan])
        history_good=bool(history_good and np.isfinite(means).all() and (means>0).all())
        assert history_good==row.volume_history_valid
        if history_good and values[-1][0]:
            # Native V uses lots; both the current and past numerator change by
            # exactly 100, so the two ratios must retain their research values.
            prior=np.array([v[1:] for v in values[:-1]],dtype=float)/100
            current=np.array(values[-1][1:],dtype=float)/100
            ratios=current/prior.mean(axis=0)
            target=f.loc[f.date.eq(row.date)&f.code.eq(row.code)].iloc[0]
            if target.volume_current_valid:
                np.testing.assert_allclose(ratios,[row.HV01,row.HV02],rtol=0,atol=2e-11)
                np.testing.assert_array_equal(encode(ratios),encode(np.array([row.HV01,row.HV02])))
        checked_minutes+=minutes
        cases.append(dict(date=row.date,code=row.code,history_valid=history_good,stock_days=len(dates),
            first_window=dates[0],last_window=dates[-1],raw_minutes=minutes))
    c.close()
    proof=dict(passed=True,feature_report_sha256=sha(root/'feature_report.json'),rows=len(f),valid=int(expected_valid.sum()),
        all_stock_day_dates_prior_windows_rolling_means_validity_and_integer_inputs_rebuilt=True,
        original_48_values_and_keys_unchanged=True,fixed_raw_cases=cases,raw_cases=len(cases),
        raw_minutes_rechecked=checked_minutes,native_lot_scaling_rebuilt=True,
        native_source_parity_verified=False,outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'feature_verification.json',proof)
    return {k:v for k,v in proof.items() if k!='fixed_raw_cases'}


if __name__=='__main__':
    print(json.dumps(main(),ensure_ascii=False,indent=2))
