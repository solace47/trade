"""Replay the established 48 numeric inputs on an explicitly bounded universe."""
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_intraday as afternoon
from . import tail_formula_float as original
from .tail_formula_volatility import NORMALIZED


def afternoon_aggregates(keys, files, first, last):
    c = afternoon.conn()
    c.read_parquet([str(p) for p in files]).create_view('raw')
    c.register('keys', keys[['date', 'code']])
    result = c.execute('''WITH s AS(
        SELECT lower(exchange)||'.'||symbol AS code,strftime(timestamp,'%Y-%m-%d') AS date,
            strftime(timestamp,'%H%M') AS clock,timestamp,open::DOUBLE AS open,high::DOUBLE AS high,
            low::DOUBLE AS low,close::DOUBLE AS close,volume::DOUBLE AS volume,turnover::DOUBLE AS amount
        FROM raw WHERE timestamp>=CAST(? AS TIMESTAMP) AND timestamp<CAST(? AS DATE)+INTERVAL 1 DAY
            AND strftime(timestamp,'%H%M') BETWEEN '1301' AND '1449'),
        b AS(SELECT s.* EXCLUDE(open,high,low,close),round(open,2) AS open,round(high,2) AS high,
            round(low,2) AS low,round(close,2) AS close,
            coalesce(timestamp=date_trunc('minute',timestamp) AND isfinite(open) AND isfinite(high)
            AND isfinite(low) AND isfinite(close) AND isfinite(volume) AND isfinite(amount)
            AND least(open,high,low,close)>0 AND high+.0001>=greatest(open,close,low)
            AND low-.0001<=least(open,close) AND abs(open-round(open,2))<=.0001
            AND abs(high-round(high,2))<=.0001 AND abs(low-round(low,2))<=.0001
            AND abs(close-round(close,2))<=.0001 AND volume>=0 AND amount>=0
            AND (volume=0)=(amount=0) AND (volume=0 OR amount/volume BETWEEN low-.0101 AND high+.0101),false) AS good
            FROM s JOIN keys USING(date,code)),
        p AS(SELECT *,lag(close) OVER(PARTITION BY date,code ORDER BY timestamp) AS prior FROM b)
        SELECT date,code,count(*) AS bars,count(DISTINCT clock) AS clocks,count(*) FILTER(WHERE good) AS good_bars,
            bars=109 AND clocks=109 AND good_bars=109 AS window_valid,
            max(close) FILTER(WHERE clock='1449') AS p49,max(close) FILTER(WHERE clock='1435') AS p35,
            max(close) FILTER(WHERE clock='1420') AS p20,max(close) FILTER(WHERE clock='1301') AS p01,
            sum(volume) FILTER(WHERE clock>='1421') AS v29,sum(amount) FILTER(WHERE clock>='1421') AS a29,
            sum(volume) FILTER(WHERE clock BETWEEN '1352' AND '1420') AS vprev29,
            sum(volume) FILTER(WHERE clock>='1436') AS v14,
            sum(volume) FILTER(WHERE clock BETWEEN '1422' AND '1435') AS vprev14,
            sum(volume) FILTER(WHERE clock>='1446') AS v4,sum(amount) FILTER(WHERE clock>='1446') AS a4,
            min(low) FILTER(WHERE clock>='1421') AS lo29,max(high) FILTER(WHERE clock>='1421') AS hi29,
            max(high) AS hi109,min(low) AS lo109,max(volume) FILTER(WHERE clock>='1421') AS vmax29,
            sum(CASE WHEN close>prior THEN volume WHEN close<prior THEN -volume ELSE 0 END)
                FILTER(WHERE clock>='1421') AS signed_v29,
            count(*) FILTER(WHERE clock>='1421' AND close>prior) AS up29,
            sum(abs(ln(close/nullif(prior,0)))) FILTER(WHERE clock>='1421') AS path29
        FROM p GROUP BY date,code ORDER BY date,code''', [first, last]).df()
    c.close()
    return result


def daily_components(raw, index):
    d = raw.loc[raw.tradestatus.eq(1)].sort_values('date').reset_index(drop=True)
    assert not d.date.duplicated().any()
    prices = d[['open', 'high', 'low', 'close']]
    good = (np.isfinite(prices).all(axis=1) & prices.gt(0).all(axis=1) & d.adjustflag.eq(3)
        & d.high.ge(prices.max(axis=1)-.0001) & d.low.le(prices.min(axis=1)+.0001)
        & (prices-prices.round(2)).abs().le(.0001).all(axis=1) & d.volume.gt(0))
    p = pd.DataFrame(dict(date=d.date, p1=d.close.shift(), p6=d.close.shift(6), p21=d.close.shift(21),
        ma5=d.close.rolling(5, min_periods=5).mean().shift(),
        ma20=d.close.rolling(20, min_periods=20).mean().shift(),
        v5=d.volume.astype(float).rolling(5, min_periods=5).mean().shift(),
        h20=d.high.rolling(20, min_periods=20).max().shift(),
        l20=d.low.rolling(20, min_periods=20).min().shift(),
        valid_history=good.astype(int).rolling(21, min_periods=21).sum().shift().eq(21)))
    tr = pd.concat([d.high-d.low, (d.high-d.close.shift()).abs(), (d.low-d.close.shift()).abs()], axis=1).max(axis=1)
    p['atr20'] = tr.rolling(20, min_periods=20).mean().shift()
    ic = d.date.map(index)
    p['index_prior_close'] = ic.shift()
    mean20 = ic.rolling(20, min_periods=1).mean().shift()
    for name, value in [('I01', 100*(ic.shift()/ic.shift(2)-1)), ('I02', 100*(ic.shift()/ic.shift(6)-1)),
                        ('I03', 100*(ic.shift()/ic.shift(21)-1)), ('I04', 100*(ic.shift()/mean20-1))]:
        p[name] = value
    p['index_history_valid'] = ic.rolling(20, min_periods=1).count().shift().eq(20) & ic.shift(21).notna()
    p['float_source_date'] = d.date.shift()
    p['float_prior_volume'] = d.volume.astype(float).shift()
    p['float_prior_turn'] = d.turn.astype(float).shift()
    p['float_prior_adjustflag'] = d.adjustflag.astype(float).shift()
    return p


def combine(universe, aggregates, daily, indices, points):
    """All source dates must already be bounded by the caller; no I/O here."""
    f = afternoon.compute(universe.merge(aggregates, on=['date', 'code'], how='left', validate='one_to_one'))
    history = []
    for code, raw in daily.groupby('code', sort=True):
        index_code = 'sh.000001' if code.startswith('sh.') else 'sz.399001'
        index = indices.loc[indices.code.eq(index_code)].set_index('date').close
        h = daily_components(raw, index); h['code'] = code
        history.append(h)
    f = f.merge(pd.concat(history, ignore_index=True), on=['date', 'code'], how='left', validate='one_to_one')
    f['D01'] = 100*(f.p1/f.p6-1); f['D02'] = 100*(f.p1/f.p21-1)
    f['D03'] = 100*(f.price_1449/f.ma5-1); f['D04'] = 100*(f.price_1449/f.ma20-1)
    price, op, hi, lo = f.price_1449, f.daily_open, f.high_1449, f.low_1449
    values = [100*(price-lo)/(hi-lo).clip(lower=.01), 100*(hi-lo)/f.p1,
        100*(hi-price)/f.p1, 100*(np.minimum(op, price)-lo)/f.p1, f.volume_1449/f.v5,
        f.amount_1449/1e8, 100*(price/f.h20-1), 100*(price/f.l20-1)]
    for i, value in enumerate(values, 1):
        f[f'C{i:02d}'] = value
    # Preserve the complete original daily-input validity, including fields that
    # are subsequently expressed in normalized form or replaced by intraday A.
    daily_values = pd.DataFrame({'F01': 100*(price/f.p1-1), 'F02': 100*(op/f.p1-1),
        'F03': 100*(price/op-1), 'F10': price, **{n:f[n] for n in ['D01', 'D02', 'D03', 'D04']},
        **{f'C{i:02d}':f[f'C{i:02d}'] for i in range(1, 9)}})
    f['daily_input_valid'] = (f.valid_history.fillna(False) & (f.p1-f.preclose).abs().le(.005)
        & np.isfinite(daily_values).all(axis=1))
    f['V01'] = 100*f.atr20/f.preclose
    for name in NORMALIZED:
        f['N'+name] = f[name]/f.V01
    f['index_code'] = np.where(f.code.str.startswith('sh.'), 'sh.000001', 'sz.399001')
    f = f.merge(points, on=['date', 'index_code'], how='left', validate='many_to_one')
    f['J01'] = 100*(f.ip48/f.index_prior_close-1); f['J02'] = 100*(f.ip48/f.ip20-1)
    f['J03'] = 100*(f.ip48/f.ip35-1); f['J04'] = 100*(f.ip20/f.ip01-1)
    for output, stock, idx in [('R01','A01','J01'), ('R02','A05','J02'), ('R03','A06','J03'), ('R04','A07','J04')]:
        f[output] = (f[stock]-f[idx])/f.V01
    f['float_source_valid'] = (f.float_source_date.lt(f.date) & f.float_prior_adjustflag.eq(3)
        & np.isfinite(f.float_prior_volume) & f.float_prior_volume.gt(0)
        & np.isfinite(f.float_prior_turn) & f.float_prior_turn.gt(0))
    f['float_shares_proxy'] = (100*f.float_prior_volume/f.float_prior_turn).where(f.float_source_valid)
    f['S01'] = np.log1p(f.float_shares_proxy*f.price_1449/1e8)
    f['S02'] = 100*f.volume_1449/f.float_shares_proxy; f['S03'] = 100*f.v29/f.float_shares_proxy
    f['formula_input_valid'] &= (f.daily_input_valid & f.atr20.gt(0) & f.index_history_valid.fillna(False)
        & f.prefix_valid.fillna(False) & f.float_source_valid & np.isfinite(f[list(original.EXPRESSIONS)]).all(axis=1))
    return f.sort_values(['date', 'code']).reset_index(drop=True)


def read_daily(files, first, last):
    columns = ['date', 'code', 'open', 'high', 'low', 'close', 'volume', 'turn', 'tradestatus', 'adjustflag']
    return pd.concat([pd.read_parquet(Path(p), columns=columns,
        filters=[('date', '>=', first), ('date', '<=', last)]) for p in files], ignore_index=True)
