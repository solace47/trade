"""Replay bounded listing-scope inputs, without scores, labels or new fits."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from audit_tail_formula_listing_scope import minute_sources
from trade_research.research_io import check_runtime, check_sources, sha, save_json
from trade_research.tail_formula_additive import conn, encode
from trade_research.tail_formula_baseline import EXPRESSIONS, META
from trade_research import tail_formula_visible as replay

PROTOCOL = Path('config/tail_formula_listing_inputs.json')
VISIBLE_COLUMNS = ['date','code','daily_open','high_1449','low_1449']


def checked():
    check_runtime()
    p = json.loads(PROTOCOL.read_text()); check_sources(p['source_hashes'])
    assert subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    assert sha(PROTOCOL) in subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    gate = json.loads(Path(p['completed_receipt']).read_text())
    assert gate['passed']; check_sources(gate['source_hashes'])
    assert p['expressions'] == EXPRESSIONS and p['new_fits'] == 0
    return p, Path(p['output_root'])


def inputs(p, root):
    root.mkdir(exist_ok=False)
    extra = pd.read_parquet(p['extra_inputs'])
    parent = pd.read_parquet(p['original_features'], filters=[('date','>=',p['extract_first']),('date','<=',p['extract_last'])])
    codes = sorted(extra.code.unique())
    controls = parent.loc[parent.code.isin(codes)].reset_index(drop=True)
    assert len(extra) == p['extra_rows'] and len(controls) == p['control_rows']
    pool = pd.read_parquet(p['visible_pool'])
    u = pool.loc[pool.code.isin(codes)].sort_values(['date','code']).reset_index(drop=True)
    details = pd.read_parquet(p['visible_base'], columns=VISIBLE_COLUMNS,
        filters=[('date','>=',p['extract_first']),('date','<=',p['extract_last'])])
    u = u.merge(details, on=['date','code'], how='left', validate='one_to_one')
    assert len(u) == len(extra)+len(controls) and not u.duplicated(['date','code']).any()
    assert u.board.eq('main').all() and u.isST.eq(0).all() and u.tradestatus.eq(1).all()
    assert u.code.str.startswith(('sh.60','sz.00')).all() and u.listing_age_sessions.ge(20).all()
    pd.testing.assert_frame_equal(u.loc[u.extra_pool,extra.columns.intersection(u.columns)].reset_index(drop=True),
        extra[extra.columns.intersection(u.columns)].reset_index(drop=True),check_exact=True)
    pd.testing.assert_frame_equal(u.loc[u.original_pool, META[:-1]].reset_index(drop=True), controls[META[:-1]],check_exact=True)
    mapping = minute_sources(json.loads(Path(p['minute_manifest']).read_text())['source_sha256'])
    assert {c:mapping[c] for c in codes} == p['minute_files']
    sources = dict(p['source_hashes'])
    for file,digest in p['raw_source_hashes'].items():
        assert sha(Path(file)) == digest, file
        sources[file] = digest
    c = conn(); c.register('keys', u[['date','code']])
    c.read_parquet(list(p['minute_files'].values())).create_view('original')
    raw = c.execute('''WITH b AS(SELECT lower(exchange)||'.'||symbol AS code,
        strftime(timestamp,'%Y-%m-%d') AS date,exchange,symbol,timestamp,
        open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,close::DOUBLE AS close,
        volume::DOUBLE AS volume,turnover::DOUBLE AS turnover FROM original
        WHERE timestamp>=CAST(? AS TIMESTAMP) AND timestamp<CAST(? AS DATE)+INTERVAL 1 DAY
        AND (strftime(timestamp,'%H%M') BETWEEN '0930' AND '1130'
          OR strftime(timestamp,'%H%M') BETWEEN '1301' AND '1449'))
        SELECT b.* FROM b JOIN keys USING(date,code) ORDER BY date,code,timestamp''',
        [p['extract_first'],p['extract_last']]).df(); c.close()
    assert raw.date.between(p['extract_first'],p['extract_last']).all()
    assert raw.timestamp.dt.year.lt(2026).all() and raw.timestamp.dt.hour.le(14).all()
    assert not raw.duplicated(['date','code','timestamp']).any()
    raw.to_parquet(root/'raw_prefixes.parquet',index=False,compression='zstd')
    af = replay.afternoon_aggregates(u,[root/'raw_prefixes.parquet'],p['extract_first'],p['extract_last'])
    morning_raw = raw.loc[raw.timestamp.dt.strftime('%H%M').between('0930','1130')]
    am = morning_pandas(morning_raw)
    daily = replay.read_daily(list(p['daily_files'].values()),p['warmup_first'],p['extract_last'])
    indices = pd.read_parquet(p['indices'],filters=[('date','>=',p['warmup_first']),('date','<=',p['extract_last'])])
    points = pd.read_parquet(p['index_points'],filters=[('date','>=',p['extract_first']),('date','<=',p['extract_last'])])
    for frame in [daily,indices,points]:
        assert frame.date.lt('2026-01-01').all()
    f = replay.combine(u,af,daily,indices,points)
    f['replay48_valid'] = f.formula_input_valid
    f = replay.add_morning(f,am)
    assert np.isfinite(f.loc[f.formula_input_valid,list(EXPRESSIONS)]).all().all()
    old = f.loc[f.original_pool].reset_index(drop=True)
    pd.testing.assert_frame_equal(old[META],controls[META],check_exact=True)
    valid = controls.formula_input_valid
    np.testing.assert_allclose(old.loc[valid,list(EXPRESSIONS)],controls.loc[valid,list(EXPRESSIONS)],rtol=3e-13,atol=3e-9)
    np.testing.assert_array_equal(encode(old.loc[valid]),encode(controls.loc[valid]))
    new = f.loc[f.extra_pool].reset_index(drop=True)
    expanded = pd.concat([parent,new[[*META,*EXPRESSIONS]]],ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(expanded.merge(pool.loc[pool.original_pool,['date','code']],on=['date','code'],validate='one_to_one'),parent,check_exact=True)
    for name,frame in [('universe',u),('replayed',f),('extra_features',new[[*META,*EXPRESSIONS]]),('features',expanded)]:
        file=root/(name+'.parquet'); frame.to_parquet(file,index=False,compression='zstd'); sources[str(file)]=sha(file)
    sources[str(root/'raw_prefixes.parquet')] = sha(root/'raw_prefixes.parquet')
    groups = new.groupby('half').agg(rows=('code','size'),valid=('formula_input_valid','sum'),
        valid_history=('valid_history','sum'),afternoon_valid=('window_valid','sum'),morning_valid=('morning_input_valid','sum')).reset_index().to_dict('records')
    save_json(root/'input_report.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),source_hashes=sources,
        replay_rows=len(f),control_rows=len(old),extra_rows=len(new),extra_valid=int(new.formula_input_valid.sum()),
        expanded_rows=len(expanded),expanded_valid=int(expanded.formula_input_valid.sum()),extra_by_half=groups,
        original_all_values_and_validity_exactly_copied=True,all_control_valid_scalars_and_encodings_match=True,
        raw_prefix_rows=len(raw),new_fits=0,new_outcomes_or_scores_read=False,new_2026_prices_read=False,
        client_source_parity_verified=False,no_exit_rules=True))
    print(json.dumps(dict(report_sha256=sha(root/'input_report.json'),extra_by_half=groups)),flush=True)



def morning_pandas(raw):
    d = raw.copy(); a = d[['open', 'high', 'low', 'close', 'volume']].to_numpy(float)
    o, h, l, cl, v = a.T
    good = (np.isfinite(a).all(axis=1) & (a[:,:4].min(axis=1)>0) & (v>=0)
        & (h+.0001>=np.maximum.reduce([o,cl,l])) & (l-.0001<=np.minimum(o,cl))
        & (np.abs(a[:,:4] - np.rint(a[:,:4]*100)/100)<=.0001).all(axis=1)
        & d.timestamp.eq(d.timestamp.dt.floor('min')))
    d['good'] = good; d['active'] = v>0; d['clock'] = d.timestamp.dt.strftime('%H%M')
    d['hc'] = np.floor(h*100+.5); d['lc'] = np.floor(l*100+.5)
    d.loc[~d.active, ['hc', 'lc']] = np.nan
    return d.groupby(['date','code']).agg(bars=('timestamp','size'), clocks=('clock','nunique'),
        good_bars=('good','sum'), active=('active','sum'), high_cents=('hc','max'), low_cents=('lc','min')).reset_index()

def rebuild_afternoon(raw):
    r=raw.sort_values('timestamp').copy()
    prices=r[['open','high','low','close']]
    finite=np.isfinite(r[['open','high','low','close','volume','turnover']]).all(axis=1)
    good=(finite & prices.gt(0).all(axis=1) & (prices-prices.round(2)).abs().le(.0001).all(axis=1)
        & r.timestamp.eq(r.timestamp.dt.floor('min'))
        & r.high.add(.0001).ge(prices[['open','low','close']].max(axis=1))
        & r.low.sub(.0001).le(prices[['open','close']].min(axis=1))
        & r.volume.ge(0) & r.turnover.ge(0) & r.volume.eq(0).eq(r.turnover.eq(0))
        & (r.volume.eq(0) | (r.turnover/r.volume).between(r.low-.0101,r.high+.0101)))
    r[['open','high','low','close']]=prices.round(2)
    r['clock']=r.timestamp.dt.strftime('%H%M')
    prior=r.close.shift()
    r['signed']=np.sign(r.close-prior).fillna(0)*r.volume
    r['up']=r.close.gt(prior)
    r['path']=np.abs(np.log(r.close/prior))
    last29=r.loc[r.clock.ge('1421')]
    last4=r.loc[r.clock.ge('1446')]
    at=lambda clock:r.loc[r.clock.eq(clock),'close'].max()
    return dict(bars=len(r),clocks=r.clock.nunique(),good_bars=int(good.sum()),
        window_valid=len(r)==109 and r.clock.nunique()==109 and bool(good.all()),
        p49=at('1449'),p35=at('1435'),p20=at('1420'),p01=at('1301'),
        v29=last29.volume.sum(min_count=1),a29=last29.turnover.sum(min_count=1),
        vprev29=r.loc[r.clock.between('1352','1420'),'volume'].sum(min_count=1),
        v14=r.loc[r.clock.ge('1436'),'volume'].sum(min_count=1),
        vprev14=r.loc[r.clock.between('1422','1435'),'volume'].sum(min_count=1),
        v4=last4.volume.sum(min_count=1),a4=last4.turnover.sum(min_count=1),
        lo29=last29.low.min(),hi29=last29.high.max(),hi109=r.high.max(),lo109=r.low.min(),
        vmax29=last29.volume.max(),signed_v29=last29.signed.sum(min_count=1),
        up29=int(last29.up.sum()),path29=last29.path.sum(min_count=1))


def encode48(data):
    return np.floor(np.clip(100*data.to_numpy(float)+10000+.000001,0,999999)).astype('int32')


def verify(p, root):
    report = json.loads((root/'input_report.json').read_text())
    assert report['passed'] and report['protocol_sha256'] == sha(PROTOCOL)
    check_sources(report['source_hashes'])
    full = pd.read_parquet(root/'replayed.parquet')
    f = full.drop(columns=['bars','clocks','good_bars']).rename(columns={
        'af_bars':'bars','af_clocks':'clocks','af_good_bars':'good_bars'})
    universe = pd.read_parquet(root/'universe.parquet')
    indices = pd.read_parquet(p['indices'],filters=[('date','>=',p['warmup_first']),('date','<=',p['extract_last'])])
    points = pd.read_parquet(p['index_points'],filters=[('date','>=',p['extract_first']),('date','<=',p['extract_last'])])
    c=conn(); c.register('indices',indices); c.register('points',points); c.register('universe',universe)
    c.read_parquet(list(p['daily_files'].values())).create_view('raw_daily')
    first,last=p['warmup_first'],p['extract_last']
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
    normalized = ','.join(f'{name}/V01 AS N{name}' for name in replay.NORMALIZED)
    c.execute('CREATE VIEW normalized AS SELECT *,'+normalized+''',
        CASE WHEN float_source_valid THEN 100*float_prior_volume/float_prior_turn END AS float_shares_proxy,
        (A01-J01)/V01 AS R01,(A05-J02)/V01 AS R02,(A06-J03)/V01 AS R03,(A07-J04)/V01 AS R04 FROM scalar''')
    c.execute('''CREATE VIEW finished AS SELECT *,ln(1+float_shares_proxy*price_1449/100000000) AS S01,
        100*volume_1449/float_shares_proxy AS S02,100*v29/float_shares_proxy AS S03 FROM normalized''')
    names = list(replay.REPLAY_EXPRESSIONS)
    expected = c.sql('SELECT date,code,'+','.join(names)+' FROM finished ORDER BY date,code').df()
    np.testing.assert_allclose(f[names],expected[names],atol=3e-9,rtol=3e-13,equal_nan=True)
    a_names = [f'A{i:02d}' for i in range(1,21)]
    daily_names = [f'D{i:02d}' for i in range(1,5)]+[f'C{i:02d}' for i in range(1,9)]
    finite = ' AND '.join('isfinite('+name+')' for name in list(dict.fromkeys(names+a_names+daily_names)))
    valid = c.sql('''SELECT coalesce(window_valid AND p49=price_1449 AND valid_history AND abs(p1-preclose)<=.005
        AND isfinite(100*(price_1449/p1-1)) AND isfinite(100*(daily_open/p1-1))
        AND isfinite(100*(price_1449/daily_open-1)) AND atr20>0 AND index_history_valid
        AND prefix_valid AND float_source_valid AND '''+finite+''' ,false) AS valid FROM finished ORDER BY date,code''').df().valid
    assert np.array_equal(f.replay48_valid,valid)
    cols = list(replay.REPLAY_EXPRESSIONS)
    for data in [f.loc[valid,cols],expected.loc[valid,cols]]:
        assert np.isfinite(data).all().all()
    np.testing.assert_array_equal(encode48(f.loc[valid,cols]),encode48(expected.loc[valid,cols]))
    c.close()

    c=conn();c.read_parquet(str(root/'raw_prefixes.parquet')).create_view('prefix')
    morning = c.sql("""WITH b AS(SELECT *,coalesce(timestamp=date_trunc('minute',timestamp)
        AND isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close) AND isfinite(volume)
        AND least(open,high,low,close)>0 AND volume>=0 AND high+.0001>=greatest(open,close,low)
        AND low-.0001<=least(open,close) AND abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
        AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001,false) AS good
        FROM prefix WHERE strftime(timestamp,'%H%M') BETWEEN '0930' AND '1130')
        SELECT date,code,count(*) AS bars,count(DISTINCT strftime(timestamp,'%H%M')) AS clocks,
        count(*) FILTER(WHERE good) AS good_bars,count(*) FILTER(WHERE volume>0) AS active,
        max(floor(high*100+.5)) FILTER(WHERE volume>0) AS high_cents,
        min(floor(low*100+.5)) FILTER(WHERE volume>0) AS low_cents
        FROM b GROUP BY date,code ORDER BY date,code""").df()
    pd.testing.assert_frame_equal(full[morning.columns],morning,check_dtype=False,check_exact=True)
    c.register('morning',morning);c.register('f',f)
    am=c.sql("""WITH b AS(SELECT f.date,f.code,f.A04,f.V01,f.replay48_valid,
        coalesce(m.bars=121 AND m.clocks=121 AND m.good_bars=121 AND m.active>0
        AND m.high_cents>0 AND m.low_cents>0 AND f.V01>0 AND isfinite(f.V01) AND f.A04>0,false) AS valid,
        m.high_cents,m.low_cents FROM f LEFT JOIN morning m USING(date,code)),
        a AS(SELECT *,CASE WHEN valid THEN 100*(floor(A04*100+.5)/high_cents-1)/V01 END AS AMHD,
        CASE WHEN valid THEN 100*(floor(A04*100+.5)/low_cents-1)/V01 END AS AMLD FROM b)
        SELECT date,code,AMHD,AMLD,valid AND coalesce(isfinite(AMHD) AND isfinite(AMLD),false) AS morning_input_valid,
        replay48_valid AND valid AND coalesce(isfinite(AMHD) AND isfinite(AMLD),false) AS formula_input_valid
        FROM a ORDER BY date,code""").df(); c.close()
    pd.testing.assert_frame_equal(full[am.columns],am,check_dtype=False,rtol=3e-13,atol=3e-9)
    expected50 = expected.merge(am[['date','code','AMHD','AMLD']],on=['date','code'],validate='one_to_one')
    final_valid = full.formula_input_valid
    np.testing.assert_array_equal(encode(full.loc[final_valid]),encode(expected50.loc[final_valid]))
    # All new prefixes, plus input-only hash-selected controls in each half.
    control=f.loc[f.original_pool,['date','code','half']].copy()
    control['hash']=[hashlib.sha256(('listing-input-control-v1|'+d+'|'+code).encode()).hexdigest() for d,code in zip(control.date,control.code)]
    control=control.sort_values('hash').groupby('half').head(p['control_prefixes_per_half'])
    chosen=pd.concat([f.loc[f.extra_pool,['date','code']],control[['date','code']]],ignore_index=True)
    c=conn();c.register('chosen',chosen)
    c.read_parquet(str(root/'raw_prefixes.parquet')).create_view('prefix')
    raw=c.sql('SELECT p.* FROM prefix p JOIN chosen USING(date,code) ORDER BY date,code,timestamp').df();c.close()
    pd.testing.assert_frame_equal(raw[['date','code']].drop_duplicates().sort_values(['date','code']).reset_index(drop=True),
        chosen.sort_values(['date','code']).reset_index(drop=True),check_exact=True)
    actual=f.set_index(['date','code']);checks=0
    for (day,code),q in raw.groupby(['date','code'],sort=True):
        clocks=q.timestamp.dt.strftime('%H%M'); a=actual.loc[(day,code)]
        values=dict(price_1449=q.loc[clocks.eq('1449'),'close'].max().round(2),
            volume_1449=q.volume.sum(),amount_1449=q.turnover.sum(),
            high_1449=q.loc[q.volume.gt(0),'high'].max().round(2),low_1449=q.loc[q.volume.gt(0),'low'].min().round(2))
        values.update(rebuild_afternoon(q.loc[clocks.between('1301','1449')]))
        for key,value in values.items():
            np.testing.assert_allclose(value,a[key],rtol=1e-12,atol=2e-8,equal_nan=True,err_msg=f'{day} {code} {key}')
            checks+=1
    parent=pd.read_parquet(p['original_features'],filters=[('date','>=',p['extract_first']),('date','<=',p['extract_last'])])
    new=pd.read_parquet(root/'extra_features.parquet'); expanded=pd.read_parquet(root/'features.parquet')
    expected_all=pd.concat([parent,new],ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(expanded,expected_all,check_exact=True)
    assert len(expanded)==1260576 and not expanded.duplicated(['date','code']).any()
    c=conn();c.register('parent',parent);c.register('new',new)
    rebuilt=c.sql('SELECT * FROM (SELECT * FROM parent UNION ALL SELECT * FROM new) ORDER BY date,code').df();c.close()
    pd.testing.assert_frame_equal(expanded,rebuilt,check_exact=True)
    groups=full.loc[full.extra_pool].groupby('half').agg(rows=('code','size'),valid=('formula_input_valid','sum'),
        valid_history=('valid_history','sum'),afternoon_valid=('window_valid','sum'),morning_valid=('morning_input_valid','sum')).reset_index().to_dict('records')
    assert report['extra_by_half']==groups and report['expanded_valid']==int(expanded.formula_input_valid.sum())
    sources={**report['source_hashes'],str(root/'input_report.json'):sha(root/'input_report.json')}
    save_json(root/'complete_receipt.json',dict(passed=True,source_hashes=sources,protocol_sha256=sha(PROTOCOL),
        original_1258085_full_values_and_validity_exact=True,all_16335_histories_scalars_validity_and_encodings_verified=True,
        all_new_prefixes_independently_pandas_rebuilt=True,controls_prefixes=len(control),raw_prefixes_verified=len(chosen),
        raw_aggregate_checks=checks,extra_by_half=groups,expanded_rows=len(expanded),expanded_valid=report['expanded_valid'],
        only_prior_21_complete_stock_days_required=True,new_fits=0,new_outcomes_or_scores_read=False,
        client_source_parity_verified=False,new_2026_prices_read=False,no_exit_rules=True))
    print(json.dumps(dict(complete_sha256=sha(root/'complete_receipt.json'),groups=groups,raw_aggregate_checks=checks)),flush=True)

def main():
    a=argparse.ArgumentParser(); a.add_argument('stage',choices=['inputs','verify']); args=a.parse_args()
    p,root=checked()
    if args.stage=='inputs': inputs(p,root)
    else: verify(p,root)


if __name__ == '__main__':
    main()
