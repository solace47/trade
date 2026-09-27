"""Independently verify expanded eligible keys and the unchanged 48 inputs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from verify_tail_formula_intraday_inputs import rebuild
from trade_research import tail_formula_additive as base
from trade_research import tail_formula_float as original
from trade_research import tail_formula_replay48 as adapter
from trade_research.corporate_cash import DAILY, MINUTES, save_json, sha
from trade_research import tail_formula_liquidity_inputs as study
from trade_research.tail_formula_liquidity_inputs import OUT, PROTOCOL, VISIBLE, LEGACY_PROOF
from trade_research.tail_formula_volatility import NORMALIZED


def main():
    p, daily_hashes, minute_hashes = study.checked_sources()
    for fold in ['2024','recent']:
        study.source(fold)
    r = json.loads((OUT/'feature_report.json').read_text())
    for key,path in [('protocol_sha256',PROTOCOL),('source_manifest_sha256',OUT/'source_manifest.json'),
        ('base_report_sha256',OUT/'base_report.json'),('features_sha256',OUT/'features.parquet'),
        ('legacy_replay_verification_sha256',LEGACY_PROOF),('adapter_sha256',Path(adapter.__file__)),
        ('extractor_sha256',Path(study.__file__)),('indices_sha256',OUT/'indices.parquet'),
        ('index_points_sha256',OUT/'index_points.parquet')]:assert r[key]==sha(path)
    for path,digest in r['parts_sha256'].items():assert sha(Path(path))==digest
    sources=json.loads((OUT/'source_manifest.json').read_text())
    for key,expected in [('daily_sha256',daily_hashes),('minute_sha256',minute_hashes)]:
        for path,digest in sources[key].items():assert digest==expected[path] and sha(Path(path))==digest
    f=pd.read_parquet(OUT/'features.parquet');universe=pd.read_parquet(OUT/'universe.parquet')
    pd.testing.assert_frame_equal(f[universe.columns],universe,check_exact=True)
    assert f.isST.eq(0).all() and f.tradestatus.eq(1).all() and f.board.eq('main').all()
    assert not f.code.str[3:].str.startswith(('92','688','300','301')).any()
    assert f.date.between(p['first'],p['last']).all() and f.amount_1449.gt(0).all() and f.amount_1449.lt(3e7).all()
    indices,points=study.index_inputs()
    pd.testing.assert_frame_equal(indices,pd.read_parquet(OUT/'indices.parquet'),check_exact=True)
    pd.testing.assert_frame_equal(points,pd.read_parquet(OUT/'index_points.parquet'),check_exact=True)
    c=base.conn();c.register('indices',indices);c.register('points',points);c.register('universe',universe)
    c.read_parquet(str(VISIBLE)).create_view('visible')
    expected_keys=c.sql("""WITH b AS(SELECT *,
        floor(floor(2000000/round(price_1449*100))/100)*100 AS shares,
        floor((round(preclose*100)*110+50)/100)/100 AS cap FROM visible
        WHERE date BETWEEN '2025-01-01' AND '2025-12-30' AND (code LIKE 'sh.60%' OR code LIKE 'sz.00%')
        AND board='main' AND isST=0 AND tradestatus=1 AND listing_age_sessions>=60
        AND price_1449>0 AND price_1449<=200 AND amount_1449>0 AND amount_1449<30000000
        AND volume_1449>0 AND NOT reference_gap AND NOT known_delisting)
        SELECT date,code,shares::BIGINT AS decision_shares,cap AS upper_limit FROM b
        WHERE shares>0 AND price_1449+greatest(price_1449*.0015,.005)<cap-.005 ORDER BY date,code""").df()
    pd.testing.assert_frame_equal(universe[expected_keys.columns],expected_keys,check_exact=True)
    paths = [DAILY/(code.replace('.','_')+'.parquet') for code in sorted(f.code.unique())]
    for path in paths:
        assert sha(path) == sources['daily_sha256'][str(path)]
    c.read_parquet([str(x) for x in paths]).create_view('raw_daily')
    first,last = p['warmup_first'],p['last']
    history = c.sql(f'''WITH d AS(SELECT d.*,i.close AS ic,
        lag(d.close) OVER w AS prior_close,
        coalesce(isfinite(d.open) AND isfinite(d.high) AND isfinite(d.low) AND isfinite(d.close)
            AND least(d.open,d.high,d.low,d.close)>0 AND d.adjustflag=3 AND d.volume>0
            AND d.high>=greatest(d.open,d.high,d.low,d.close)-.0001
            AND d.low<=least(d.open,d.high,d.low,d.close)+.0001
            AND abs(d.open-round(d.open,2))<=.0001 AND abs(d.high-round(d.high,2))<=.0001
            AND abs(d.low-round(d.low,2))<=.0001 AND abs(d.close-round(d.close,2))<=.0001,false) AS good
        FROM raw_daily d LEFT JOIN indices i ON i.date=d.date AND i.code=
            CASE WHEN d.code LIKE 'sh.%' THEN 'sh.000001' ELSE 'sz.399001' END
        WHERE d.tradestatus=1 AND d.date BETWEEN '{first}' AND '{last}'
        WINDOW w AS(PARTITION BY d.code ORDER BY d.date)),
        t AS(SELECT *,greatest(high-low,abs(high-prior_close),abs(low-prior_close)) AS tr FROM d),
        h AS(SELECT date,code,lag(close) OVER w AS p1,lag(close,6) OVER w AS p6,lag(close,21) OVER w AS p21,
        CASE WHEN count(close) OVER v5=5 THEN avg(close) OVER v5 END AS ma5,
        CASE WHEN count(close) OVER v20=20 THEN avg(close) OVER v20 END AS ma20,
        CASE WHEN count(volume) OVER v5=5 THEN avg(volume) OVER v5 END AS v5,
        CASE WHEN count(high) OVER v20=20 THEN max(high) OVER v20 END AS h20,
        CASE WHEN count(low) OVER v20=20 THEN min(low) OVER v20 END AS l20,
        sum(good::INT) OVER v21=21 AS valid_history,
        CASE WHEN count(tr) OVER v20=20 THEN avg(tr) OVER v20 END AS atr20,
        lag(ic) OVER w AS index_prior_close,
        100*(lag(ic) OVER w/lag(ic,2) OVER w-1) AS I01,
        100*(lag(ic) OVER w/lag(ic,6) OVER w-1) AS I02,
        100*(lag(ic) OVER w/lag(ic,21) OVER w-1) AS I03,
        100*(lag(ic) OVER w/avg(ic) OVER v20-1) AS I04,
        count(ic) OVER v20=20 AND lag(ic,21) OVER w IS NOT NULL AS index_history_valid,
        lag(date) OVER w AS float_source_date,lag(volume::DOUBLE) OVER w AS float_prior_volume,
        lag(turn::DOUBLE) OVER w AS float_prior_turn,lag(adjustflag::DOUBLE) OVER w AS float_prior_adjustflag
        FROM t WINDOW w AS(PARTITION BY code ORDER BY date),
        v5 AS(PARTITION BY code ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
        v20 AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
        v21 AS(PARTITION BY code ORDER BY date ROWS BETWEEN 21 PRECEDING AND 1 PRECEDING))
        SELECT h.* FROM h JOIN universe USING(date,code) ORDER BY date,code''').df()
    history['valid_history'] = history.valid_history.fillna(False)
    pd.testing.assert_frame_equal(f[history.columns],history,check_dtype=False,rtol=3e-13,atol=3e-9)
    aggregate_names = ['bars','clocks','good_bars','window_valid','p49','p35','p20','p01','v29','a29',
        'vprev29','v14','vprev14','v4','a4','lo29','hi29','hi109','lo109','vmax29','signed_v29','up29','path29']
    c.register('aggregates',f[['date','code',*aggregate_names]]); c.register('history',history)
    c.execute('''CREATE VIEW inputs AS SELECT u.*,a.* EXCLUDE(date,code),h.* EXCLUDE(date,code),
        p.* EXCLUDE(date,index_code) FROM universe u JOIN aggregates a USING(date,code)
        JOIN history h USING(date,code) LEFT JOIN points p ON p.date=u.date AND p.index_code=
        CASE WHEN u.code LIKE 'sh.%' THEN 'sh.000001' ELSE 'sz.399001' END''')
    c.execute('''CREATE VIEW scalar AS SELECT *,100*(p49/preclose-1) AS A01,100*(daily_open/preclose-1) AS A02,
        100*(p49/daily_open-1) AS A03,p49 AS A04,100*(p49/p20-1) AS A05,100*(p49/p35-1) AS A06,
        100*(p20/p01-1) AS A07,v29/vprev29 AS A08,v14/vprev14 AS A09,(v4/4)/((v29-v4)/25) AS A10,
        100*(p49-lo29)/greatest(hi29-lo29,.01) AS A11,a29/1e8 AS A12,100*(p49/(a29/v29)-1) AS A13,
        100*(p49/(a4/v4)-1) AS A14,100*signed_v29/v29 AS A15,100*up29/29 AS A16,
        100*vmax29/v29 AS A17,100*(p49/hi109-1) AS A18,100*(hi109-lo109)/preclose AS A19,
        100*abs(ln(p49/p20))/greatest(path29,.000001) AS A20,
        100*(p1/p6-1) AS D01,100*(p1/p21-1) AS D02,100*(price_1449/ma5-1) AS D03,
        100*(price_1449/ma20-1) AS D04,100*(price_1449-low_1449)/greatest(high_1449-low_1449,.01) AS C01,
        100*(high_1449-low_1449)/p1 AS C02,100*(high_1449-price_1449)/p1 AS C03,
        100*(least(daily_open,price_1449)-low_1449)/p1 AS C04,volume_1449/v5 AS C05,
        amount_1449/1e8 AS C06,100*(price_1449/h20-1) AS C07,100*(price_1449/l20-1) AS C08,
        100*atr20/preclose AS V01,100*(ip48/index_prior_close-1) AS J01,100*(ip48/ip20-1) AS J02,
        100*(ip48/ip35-1) AS J03,100*(ip20/ip01-1) AS J04,
        float_source_date<date AND float_prior_adjustflag=3 AND isfinite(float_prior_volume)
        AND float_prior_volume>0 AND isfinite(float_prior_turn) AND float_prior_turn>0 AS float_source_valid
        FROM inputs''')
    normalized = ','.join(f'{name}/V01 AS N{name}' for name in NORMALIZED)
    c.execute('CREATE VIEW normalized AS SELECT *,'+normalized+''',
        CASE WHEN float_source_valid THEN 100*float_prior_volume/float_prior_turn END AS float_shares_proxy,
        (A01-J01)/V01 AS R01,(A05-J02)/V01 AS R02,(A06-J03)/V01 AS R03,(A07-J04)/V01 AS R04 FROM scalar''')
    c.execute('''CREATE VIEW finished AS SELECT *,ln(1+float_shares_proxy*price_1449/100000000) AS S01,
        100*volume_1449/float_shares_proxy AS S02,100*v29/float_shares_proxy AS S03 FROM normalized''')
    names = list(original.EXPRESSIONS)
    expected = c.sql('SELECT date,code,'+','.join(names)+' FROM finished ORDER BY date,code').df()
    np.testing.assert_allclose(f[names],expected[names],atol=3e-9,rtol=3e-13,equal_nan=True)
    a_names = [f'A{i:02d}' for i in range(1,21)]
    daily_names = [f'D{i:02d}' for i in range(1,5)]+[f'C{i:02d}' for i in range(1,9)]
    finite = ' AND '.join('isfinite('+name+')' for name in list(dict.fromkeys(names+a_names+daily_names)))
    valid = c.sql('''SELECT coalesce(window_valid AND p49=price_1449 AND valid_history AND abs(p1-preclose)<=.005
        AND isfinite(100*(price_1449/p1-1)) AND isfinite(100*(daily_open/p1-1))
        AND isfinite(100*(price_1449/daily_open-1)) AND atr20>0 AND index_history_valid
        AND prefix_valid AND float_source_valid AND '''+finite+''' ,false) AS valid FROM finished ORDER BY date,code''').df().valid
    assert np.array_equal(f.formula_input_valid,valid)
    base.EXPRESSIONS = original.EXPRESSIONS
    np.testing.assert_array_equal(base.encode(f.loc[valid]),base.encode(expected.loc[valid]))
    c.close()
    sample=f[['date','code','formula_input_valid']].copy();sample['month']=sample.date.str[:7]
    sample['hash']=[hashlib.sha256(('liquidity48-input-v1|'+d+'|'+code).encode()).hexdigest() for d,code in zip(sample.date,sample.code)]
    sample=sample.sort_values('hash').groupby(['month','formula_input_valid']).head(4)
    actual=f.set_index(['date','code']);checks=0;minute_rows=0;prefix_sizes=[]
    for code,group in sample.groupby('code'):
        path=MINUTES/code[:2].upper()/(code[3:]+'.parquet')
        assert sha(path)==sources['minute_sha256'][str(path)]
        filters=[[('timestamp','>=',pd.Timestamp(day+' 09:30').to_pydatetime()),
                  ('timestamp','<',pd.Timestamp(day+' 14:50').to_pydatetime())] for day in group.date]
        raw=pq.read_table(path,filters=filters).to_pandas();raw['date']=raw.timestamp.dt.strftime('%Y-%m-%d')
        raw['clock']=raw.timestamp.dt.strftime('%H%M')
        raw=raw.loc[raw.clock.between('0930','1130')|raw.clock.between('1301','1449')]
        for day in group.date:
            q=raw.loc[raw.date.eq(day)];a=actual.loc[(day,code)];minute_rows+=len(q);prefix_sizes.append(len(q))
            values=dict(price_1449=q.loc[q.clock.eq('1449'),'close'].max().round(2),
                volume_1449=q.volume.sum(),amount_1449=q.turnover.sum(),
                high_1449=q.loc[q.volume.gt(0),'high'].max().round(2),low_1449=q.loc[q.volume.gt(0),'low'].min().round(2))
            values.update(rebuild(q.loc[q.clock.between('1301','1449')]))
            for key,value in values.items():
                np.testing.assert_allclose(value,a[key],rtol=1e-12,atol=2e-8,equal_nan=True,err_msg=f'{day} {code} {key}')
                checks+=1
    assert len(f)==r['rows'] and int(valid.sum())==r['valid']
    expected_counts=f.groupby('half').agg(rows=('code','size'),valid=('formula_input_valid','sum')).reset_index().to_dict('records')
    assert r['by_half']==expected_counts
    proof=dict(passed=True,feature_report_sha256=sha(OUT/'feature_report.json'),rows=len(f),valid=int(valid.sum()),
        all_eligible_keys_and_hard_filters_rebuilt=True,all_daily_histories_independently_rebuilt=True,
        scalar_fields=48,all_scalar_values_and_integer_inputs_rebuilt=True,all_validity_flags_rebuilt=True,
        fixed_raw_prefixes=len(sample),raw_minute_rows=minute_rows,raw_aggregate_checks=checks,
        sampled_prefix_size_range=[min(prefix_sizes),max(prefix_sizes)],sample_keys=sample[['date','code']].to_dict('records'),
        original_models_and_inputs_unchanged=True,low_amount_model_extrapolation_unproven=True,
        new_2026_prices_read=False,next_morning_stock_outcomes_read=False,native_source_parity_verified=False,no_exit_rules=True)
    save_json(OUT/'feature_verification.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k!='sample_keys'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
