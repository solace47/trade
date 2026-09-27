"""Rebuild the opening join and three inputs; recheck fixed original minute windows."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_opening as study
from trade_research.corporate_cash import MINUTES,save_json,sha
from verify_opening_cash_history import source_window


def main():
    root=study.ROOT;raw=study.source_reports();r=json.loads((root/'feature_report.json').read_text())
    for key,path in [('protocol_sha256',study.PROTOCOL),('previous_feature_report_sha256',study.previous.ROOT/'feature_report.json'),
        ('opening_raw_report_sha256',study.OPENING/'raw_report.json'),
        ('opening_feature_verification_sha256',study.OPENING/'feature_verification.json'),('features_sha256',root/'features.parquet')]:
        assert r[key]==sha(path)
    assert list(r['native_expressions'].items())==list(study.EXPRESSIONS.items())
    assert r['native_header']==study.HEADER
    old=pd.read_parquet(study.previous.ROOT/'features.parquet');f=pd.read_parquet(root/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    assert f.prior_formula_input_valid.equals(old.formula_input_valid)
    c=study.base.conn();c.register('old',old);c.read_parquet(list(raw['parts_sha256'])).create_view('windows')
    a=c.sql('''WITH joined AS(SELECT o.date,o.code,o.daily_open,o.price_1449,o.V01,o.amount_1449,
        w.price_1000 AS opening_1000_price,w.volume_1000 AS opening_1000_volume,
        w.opening_amount_cents AS opening_cash_cents,w.window_valid AS opening_window_valid,
        coalesce(w.window_valid AND w.price_1000>0 AND w.volume_1000>0 AND w.opening_amount_cents>0
            AND w.opening_amount_cents/100.<=o.amount_1449+.151 AND o.daily_open>0 AND o.V01>0 AND o.amount_1449>0
            AND isfinite(w.price_1000) AND isfinite(w.volume_1000) AND isfinite(w.opening_amount_cents)
            AND isfinite(o.daily_open) AND isfinite(o.V01) AND isfinite(o.amount_1449),false) AS opening_source_valid
        FROM old o LEFT JOIN windows w USING(date,code))
        SELECT date,code,opening_1000_price,opening_1000_volume,opening_cash_cents,opening_window_valid,opening_source_valid,
        CASE WHEN opening_source_valid THEN (opening_1000_price-daily_open)*100/(daily_open*V01) END AS M01,
        CASE WHEN opening_source_valid THEN (price_1449-opening_1000_price)*100/(opening_1000_price*V01) END AS M02,
        CASE WHEN opening_source_valid THEN opening_cash_cents/amount_1449 END AS M03 FROM joined ORDER BY date,code''').df()
    pd.testing.assert_frame_equal(f[a.columns.drop(['M01','M02','M03'])],a.drop(columns=['M01','M02','M03']),check_exact=True,check_dtype=False)
    names=list(study.NEW_EXPRESSIONS)
    np.testing.assert_allclose(f[names],a[names],rtol=2e-14,atol=1e-10,equal_nan=True)
    good=old.formula_input_valid&a.opening_source_valid&np.isfinite(f[list(study.EXPRESSIONS)]).all(axis=1)
    assert good.equals(f.formula_input_valid) and int(good.sum())==r['valid']
    expected=np.floor(np.clip(a.loc[good,names].to_numpy()*100+10000+.000001,0,999999)).astype('int32')
    actual=np.floor(np.clip(f.loc[good,names].to_numpy()*100+10000+.000001,0,999999)).astype('int32')
    np.testing.assert_array_equal(actual,expected)
    assert len(f)==r['rows'] and int((old.formula_input_valid&~good).sum())==r['newly_invalid']
    assert int(f.opening_window_valid.isna().sum())==r['missing_opening_window']
    assert f.date.between('2024-01-01','2025-12-30').all()
    # Fixed date/key hashes select both valid and invalid opening windows, without outcomes.
    candidates=f.loc[f.prior_formula_input_valid,['date','code','half','opening_source_valid']].copy()
    candidates['key_hash']=[hashlib.sha256((d+'/'+k).encode()).hexdigest() for d,k in zip(candidates.date,candidates.code)]
    sample=candidates.sort_values('key_hash').groupby(['half','opening_source_valid'],sort=True).head(8)
    source_manifest=Path('data/research/economic_winner/input_manifest.json')
    manifest=json.loads((study.OPENING/'manifest.json').read_text())
    assert sha(source_manifest)==manifest['source_manifest_sha256']
    hashes=json.loads(source_manifest.read_text())['source_sha256'];checked=set();cases=[];total_bars=0
    for row in sample.sort_values(['date','code']).itertuples():
        path=MINUTES/row.code[:2].upper()/(row.code[3:]+'.parquet')
        if path not in checked:
            assert sha(path)==hashes[str(path)];checked.add(path)
        bars=c.execute('''SELECT timestamp,open,high,low,close,volume,turnover FROM read_parquet(?)
            WHERE timestamp>=?::TIMESTAMP AND timestamp<=?::TIMESTAMP ORDER BY timestamp''',
            [str(path),row.date+' 09:31:00',row.date+' 10:00:00']).df()
        rebuilt=source_window(bars)
        cache=c.execute('SELECT * FROM windows WHERE date=? AND code=?',[row.date,row.code]).df()
        assert rebuilt is not None and len(cache)==1
        for name,value in rebuilt.items():
            observed=cache.iloc[0][name]
            if isinstance(value,str) or isinstance(value,(bool,np.bool_)):
                assert observed==value
            else:
                np.testing.assert_allclose(observed,value,rtol=0,atol=1e-8,equal_nan=True)
        amounts=bars.turnover.to_numpy(dtype=float);amounts=amounts[np.isfinite(amounts)&(amounts>=0)]
        native_cents=int(np.floor(amounts*100+.5).sum())
        cash_difference=native_cents-rebuilt['opening_amount_cents']
        cases.append(dict(date=row.date,code=row.code,source_valid=bool(row.opening_source_valid),
            bars=len(bars),cached_window_rebuilt=True,mathematical_native_cash_difference_cents=cash_difference))
        total_bars+=len(bars)
    proof=dict(passed=True,feature_report_sha256=sha(root/'feature_report.json'),rows=len(f),valid=int(good.sum()),
        all_previous_48_values_unchanged=True,all_same_date_joins_three_inputs_and_new_integer_encodings_rebuilt=True,
        original_windows_checked=len(cases),original_bars=total_bars,original_cases=cases,
        maximum_mathematical_native_cash_difference_cents=max(abs(x['mathematical_native_cash_difference_cents']) for x in cases),
        old_independent_raw_window_verification_sha256=sha(study.OPENING/'feature_verification.json'),
        native_client_values_verified=False,outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'feature_verification.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k!='original_cases'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
