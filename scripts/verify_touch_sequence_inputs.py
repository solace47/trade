"""Independent clock-string prefix reconstruction and frozen nearest controls."""
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import MINUTES, save_json, sha

ROOT = Path('data/research/touch_sequence')
BASE = Path('data/research/next_day_winner/visible_base.parquet')


def check():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    raw_report = json.loads((ROOT / 'raw_report.json').read_text())
    assert manifest['protocol_sha256'] == sha(Path('config/touch_sequence_protocol.json'))
    assert manifest['base_sha256'] == sha(BASE)
    assert raw_report['manifest_sha256'] == sha(ROOT / 'manifest.json')
    assert raw_report['features_sha256'] == sha(ROOT / 'features.parquet')
    for name, digest in manifest['output_sha256'].items():
        assert sha(ROOT / name) == digest
    for path, digest in raw_report['parts_sha256'].items():
        assert sha(Path(path)) == digest
    f = pd.read_parquet(ROOT / 'features.parquet').sort_values(['date','code']).reset_index(drop=True)
    frozen = pd.read_parquet(ROOT / 'frozen_cohort.parquet')
    pd.testing.assert_frame_equal(f[frozen.columns], frozen, check_exact=True)
    c = duckdb.connect()
    c.read_parquet(str(BASE)).create_view('base')
    c.execute("""CREATE VIEW eligible AS SELECT *,round(high_1449*100)=round(upper_limit*100) AS touched
        FROM base WHERE board='main' AND necessary_tradeable""")
    c.execute("""CREATE VIEW distances AS SELECT a.date,a.code,a.half,b.code AS control_code,
        abs(a.return_1449-b.return_1449)/.01+abs(a.return20_prior_adjusted-b.return20_prior_adjusted)/.05
        +abs(log2(b.price_1449/a.price_1449))+abs(log2(b.amount_1449/a.amount_1449)) AS distance,
        a.return_1449-b.return_1449 AS return_1449_difference,
        a.return20_prior_adjusted-b.return20_prior_adjusted AS return20_prior_adjusted_difference,
        a.price_1449-b.price_1449 AS price_1449_difference,a.amount_1449-b.amount_1449 AS amount_1449_difference
        FROM eligible a JOIN eligible b ON a.date=b.date AND substr(a.code,1,2)=substr(b.code,1,2)
        AND NOT b.touched AND b.high_1449<b.upper_limit-.005
        AND abs(a.return_1449-b.return_1449)<=.01
        AND abs(a.return20_prior_adjusted-b.return20_prior_adjusted)<=.05
        AND b.price_1449/a.price_1449 BETWEEN .5 AND 2
        AND b.amount_1449/a.amount_1449 BETWEEN .5 AND 2 WHERE a.touched""")
    pairs = c.sql("""SELECT * FROM distances QUALIFY row_number() OVER(PARTITION BY date,code ORDER BY distance,control_code)=1
        ORDER BY date,code""").df()
    saved = pd.read_parquet(ROOT / 'frozen_pairs.parquet')
    pd.testing.assert_frame_equal(saved[pairs.columns], pairs, check_dtype=False, atol=2e-12, rtol=0)
    touched = c.sql('SELECT date,code FROM eligible WHERE touched ORDER BY date,code').df()
    pd.testing.assert_frame_equal(f.loc[f.touched, ['date','code']].reset_index(drop=True), touched, check_dtype=False)
    key_index = f.set_index(['date','code'])
    expected_times = set(pd.date_range('2024-01-02 09:30', '2024-01-02 11:30', freq='min').strftime('%H:%M'))
    expected_times |= set(pd.date_range('2024-01-02 13:01', '2024-01-02 14:49', freq='min').strftime('%H:%M'))
    assert len(expected_times) == 230
    expected_parts = []
    total_bars = 0
    for path in raw_report['parts_sha256']:
        p = pd.read_parquet(path)
        total_bars += len(p)
        assert p.date.between('2024-01-01','2025-12-30').all()
        p['clock'] = p.timestamp.dt.strftime('%H:%M')
        assert p.clock.isin(expected_times).all()
        assert p.timestamp.dt.strftime('%Y-%m-%d').eq(p.date).all()
        assert not p.duplicated(['date','code','timestamp']).any()
        p = p.merge(frozen[['date','code','upper_limit','preclose']], on=['date','code'], validate='many_to_one')
        ohlc = p[['open','high','low','close']]
        finite = np.isfinite(p[['open','high','low','close','volume','amount']]).all(axis=1)
        cent = (ohlc-ohlc.round(2)).abs().le(.0001).all(axis=1)
        positive = p.volume.gt(0)
        valid = (finite & cent & ohlc.gt(0).all(axis=1)
            & p.high.ge(ohlc.max(axis=1)-.0001) & p.low.le(ohlc.min(axis=1)+.0001)
            & p.volume.ge(0) & p.amount.ge(0) & p.volume.eq(0).eq(p.amount.eq(0))
            & (p.volume.eq(0) | (p.amount/p.volume).between(p.low-.0101,p.high+.0101))
            & p.timestamp.dt.second.eq(0) & p.timestamp.dt.microsecond.eq(0))
        lower_by_key = {key: float((Decimal(str(r.preclose)) * Decimal('.90')).quantize(Decimal('.01'),rounding=ROUND_HALF_UP))
            for key,r in key_index.loc[pd.MultiIndex.from_frame(p[['date','code']].drop_duplicates())].iterrows()}
        p['lower'] = [lower_by_key[(d,code)] for d,code in zip(p.date,p.code)]
        p['valid'] = valid
        p['bounds'] = ~positive | (p.high.le(p.upper_limit+.0001) & p.low.ge(p.lower-.0001))
        hit = positive & p.high.round(2).eq(p.upper_limit)
        p['hit_clock'] = p.clock.where(hit)
        p['positive_high'] = p.high.where(positive)
        g = p.groupby(['date','code'])
        a = g.agg(bars=('clock','size'), labels=('timestamp','nunique'), valid_bars=('valid','all'),
            within_limits=('bounds','all'), raw_high=('positive_high','max'))
        times = p.loc[hit].groupby(['date','code']).clock.agg(['min','max']).rename(columns={'min':'first_clock','max':'last_clock'})
        a = a.join(times)
        for time,name in [('14:20','1420'),('14:49','1449')]:
            v = p.loc[p.clock.eq(time)].set_index(['date','code'])
            a['raw_'+name] = v.close
            a['raw_volume_'+name] = v.volume
        # Every expected time must occur once, not merely 230 arbitrary labels.
        complete = g.clock.agg(lambda x: set(x) == expected_times)
        src = key_index.reindex(a.index)
        valid_sources = (complete & a.bars.eq(230) & a.labels.eq(230) & a.valid_bars & a.within_limits
            & a.raw_volume_1420.gt(0) & a.raw_volume_1449.gt(0)
            & (a.raw_1420-src.price_1420).abs().le(.0001)
            & (a.raw_1449-src.price_1449).abs().le(.0001)
            & (a.raw_high-src.high_1449).abs().le(.0001)
            & src.touched.eq(a.first_clock.notna()))
        a['source_valid'] = valid_sources
        for stem in ['first','last']:
            a[stem+'_touch'] = pd.to_datetime(a[stem+'_clock'],format='%H:%M',errors='coerce').dt.hour*60 \
                +pd.to_datetime(a[stem+'_clock'],format='%H:%M',errors='coerce').dt.minute
        a['group'] = [
            'unknown_source' if not r.source_valid else
            'untouched_control' if not key_index.loc[key,'touched'] else
            'late_first_touch_open' if r.first_clock>='14:20' else
            'tail_retouch_open' if r.last_clock>='14:20' else
            'early_open_rising' if r.raw_1449>r.raw_1420 else 'early_open_not_rising'
            for key,r in a.iterrows()]
        a['primary'] = a.group.eq('early_open_rising') & src.touched
        expected_parts.append(a.drop(columns=['first_clock','last_clock']).reset_index())
    expected = pd.concat(expected_parts).sort_values(['date','code']).reset_index(drop=True)
    assert len(expected) == len(f)
    for name in ['first_touch','last_touch']:
        expected[name] = expected[name].astype('Int64')
    pd.testing.assert_frame_equal(f[expected.columns],expected,check_dtype=False,atol=1e-12,rtol=0)
    sample = f[['date','code','half']].copy()
    sample['hash'] = [hashlib.sha256(('touch-sequence|'+d+'|'+code).encode()).hexdigest() for d,code in zip(sample.date,sample.code)]
    sample = sample.sort_values(['half','hash']).groupby('half').head(24)
    cache = {}
    for path in raw_report['parts_sha256']:
        p = pd.read_parquet(path)
        for key, values in p.groupby(['date','code']):
            if ((sample.date==key[0]) & (sample.code==key[1])).any():
                cache[key] = values
    columns = ['timestamp','open','high','low','close','volume','amount']
    for r in sample.itertuples():
        file = MINUTES / r.code[:2].upper() / (r.code[3:]+'.parquet')
        p = pd.read_parquet(file,columns=['timestamp','open','high','low','close','volume','turnover'],
            filters=[('timestamp','>=',pd.Timestamp(r.date+' 09:30:00')),('timestamp','<=',pd.Timestamp(r.date+' 14:49:00'))])
        p = p.rename(columns={'turnover':'amount'})
        p = p.loc[p.timestamp.dt.strftime('%H:%M').isin(expected_times),columns].sort_values('timestamp').reset_index(drop=True)
        want = cache[(r.date,r.code)][columns].sort_values('timestamp').reset_index(drop=True)
        pd.testing.assert_frame_equal(p,want,check_dtype=False,check_exact=True)
    report = dict(passed=True,raw_report_sha256=sha(ROOT/'raw_report.json'),features_sha256=sha(ROOT/'features.parquet'),
        rows=len(f),raw_rows=total_bars,independent_original_prefixes=len(sample),nearest_pairs=len(pairs),
        primary=int(f.primary.sum()),source_unknown=int((~f.source_valid).sum()),
        counts=f.groupby(['half','touched','group']).size().rename('rows').reset_index().to_dict('records'),
        outcomes_read=False,new_2026_prices_read=False)
    save_json(ROOT/'prefix_verification.json',report)
    return report


if __name__=='__main__':
    print(json.dumps(check(),ensure_ascii=False,indent=2))
