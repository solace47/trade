"""Complete the frozen listing-scope comparison with the unchanged buy simulator."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research.research_io import check_runtime, check_sources, sha, save_json
from trade_research.tail_formula_additive import conn
from tail_formula_reports import checked_selection
from tail_formula_order_statistics import KEYS, COUNTS, METRICS, aggregate, summarize, paired
from tail_formula_statistics import eq, interval
from trade_research.tail_formula_boundary_evaluation import period

PROTOCOL=Path('config/tail_formula_listing_evaluation.json')
BUY_COLUMNS=['date','code','half','board','necessary_tradeable','decision_shares','price_1449',
    'preclose','upper_limit','next_date','day_close','next_preclose','next_trade_status','next_isST','next_adjustflag',
    'entry_bars','entry_labels','entry_source_valid','entry_bounds_valid','entry_volume','entry_vwap','entry_low','entry_high',
    'entry_fill_status','entry_recorded','entry_queue_unknown','catalog_covered','action_exposure',
    'period_entry_bad_day','period_exit_bad_day','period_bad_symbol']
LABEL_KEYS=['date','code','next_date']
NAMES=['first_positive_end','first_space_end','first_bad3','positive_before_bad3','space_before_bad3',
    'positive_then_bad3','bad3_before_positive','adverse_until_positive','adverse_until_space','risk_observed']


def checked():
    check_runtime();p=json.loads(PROTOCOL.read_text());check_sources(p['source_hashes'])
    assert subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}'])==PROTOCOL.read_bytes()
    committed=subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    assert sha(PROTOCOL) in committed and sha(Path(p['joint_selection'])) in committed
    joint=json.loads(Path(p['joint_selection']).read_text());assert joint['passed'];check_sources(joint['source_hashes'])
    assert p['economic_window']==['09:31','09:59'] and p['new_fits']==0
    return p,Path(p['output_root']),joint


def equal(a,b,exact=False):
    pd.testing.assert_frame_equal(a.reset_index(drop=True),b.reset_index(drop=True),
        check_dtype=False,check_exact=exact,rtol=0,atol=2e-10)



def observations_pandas(raw):
    """Pandas implementation; producer uses SQL windows and filtered aggregates."""
    raw = raw.sort_values(['date', 'code', 'timestamp']).reset_index(drop=True).copy()
    assert not raw.duplicated(['date', 'code', 'timestamp']).any()
    assert raw.clock.eq(raw.timestamp.dt.strftime('%H:%M')).all()
    assert raw.next_date.eq(raw.timestamp.dt.strftime('%Y-%m-%d')).all()
    assert raw.clock.between('09:31', '10:00').all()
    p = raw[['open', 'high', 'low', 'close']].astype(float)
    v, a, t = raw.volume, raw.amount, raw.timestamp
    good = (np.isfinite(p).all(axis=1) & p.gt(0).all(axis=1) & np.isfinite(v) & np.isfinite(a)
        & p.high.add(.0001).ge(p.max(axis=1)) & p.low.sub(.0001).le(p.min(axis=1))
        & (p-p.round(2)).abs().le(.0001).all(axis=1) & v.ge(0) & a.ge(0)
        & v.eq(0).eq(a.eq(0)) & (v.eq(0) | (a/v).between(p.low-.0101, p.high+.0101))
        & t.eq(t.dt.floor('min')))
    raw['good'] = good; raw['active'] = good & v.gt(0)
    raw['before'] = raw.clock.le('09:59'); raw['before_good'] = good & raw.before
    raw['before_active'] = raw.active & raw.before
    same = raw.date.eq(raw.date.shift(2)) & raw.code.eq(raw.code.shift(2))
    three = (same & raw.active.astype(int).rolling(3, min_periods=3).sum().eq(3)
        & t.sub(t.shift(2)).eq(pd.Timedelta(minutes=2)) & raw.before)
    raw['sustained'] = raw.close.rolling(3, min_periods=3).min().where(three)
    raw['active_close'] = raw.close.where(raw.before_active)
    raw['active_low'] = raw.low.where(raw.before_active)
    raw['at0959'] = raw.close.where(raw.before_active & raw.clock.eq('09:59'))
    raw['clock29'] = raw.clock.where(raw.before)
    return raw.groupby(LABEL_KEYS).agg(bars30=('timestamp', 'size'), labels30=('clock', 'nunique'),
        good30=('good', 'sum'), bars29=('before', 'sum'), labels29=('clock29', 'nunique'),
        good29=('before_good', 'sum'), active_minutes=('before_active', 'sum'),
        max_close=('active_close', 'max'), sustained_close=('sustained', 'max'),
        min_low=('active_low', 'min'), price_0959=('at0959', 'max')).reset_index()

def classify(entry, obs):
    e = entry.copy(); e['necessary_tradeable'] = True
    liquid = np.isfinite(e.entry_vwap) & e.entry_vwap.gt(0) & e.entry_volume.gt(0)
    e['entry_fill_status'] = np.select([~liquid, ~e.entry_volume.mul(.1).ge(e.decision_shares),
        e.entry_vwap.mul(1.0005).ge(e.upper_limit-.005)], ['no_liquidity', 'volume_cap', 'estimated_upper_limit'], default='filled')
    e['entry_recorded'] = e.entry_fill_status.eq('filled')
    e['entry_queue_unknown'] = e.entry_recorded & (~e.entry_bounds_valid | e.entry_high.round(2).ge(e.upper_limit))
    r = obs.merge(e, on=LABEL_KEYS, validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    valid_ref = np.isfinite(r.next_preclose) & r.next_preclose.gt(0) & (r.next_preclose-r.next_preclose.round(2)).abs().le(.0001)
    valid_close = np.isfinite(r.day_close) & r.day_close.gt(0) & (r.day_close-r.day_close.round(2)).abs().le(.0001)
    r['corporate_unknown'] = ~r.catalog_covered | r.action_exposure | ~valid_ref | ~valid_close | (r.next_preclose-r.day_close).abs().gt(.005)
    r['next_daily_valid'] = valid_ref & r.next_trade_status.eq(1) & r.next_isST.isin([0, 1]) & r.next_adjustflag.eq(3)
    r['entry_source_unknown'] = ~r.entry_source_valid | r.period_entry_bad_day | r.period_bad_symbol
    r['known_no_trade'] = ~r.entry_source_unknown & ~r.entry_recorded
    r['observation_status'] = np.select([r.entry_source_unknown, r.known_no_trade, r.entry_queue_unknown,
        r.corporate_unknown, ~r.next_daily_valid, ~r.source_valid],
        ['entry_source_unknown', 'no_trade', 'entry_queue_unknown', 'corporate_unknown', 'next_daily_unknown', 'morning_source_unknown'], default='known')
    for bps in [5, 15]:
        buy = r.entry_vwap + np.maximum(r.entry_vwap*bps/10000, .005); value = r.decision_shares*buy
        r[f'buy_cash{bps}'] = value + np.maximum(5, value*.0003) + value*.00001
        r[f'entry_stress_unknown{bps}'] = r.entry_recorded & buy.ge(r.upper_limit-.005)
        known = r.observation_status.eq('known') & ~r[f'entry_stress_unknown{bps}']; r[f'known{bps}'] = known
        for name, price in [('sustained', r.sustained_close), ('any_close', r.max_close), ('mark_0959', r.price_0959), ('adverse', r.min_low)]:
            r[f'{name}_return{bps}'] = mark(price, r.decision_shares, r[f'buy_cash{bps}'], bps).where(known)
        r[f'opportunity{bps}'] = r[f'sustained_return{bps}'].gt(0).astype(float).where(known)
        r[f'any_opportunity{bps}'] = r[f'any_close_return{bps}'].gt(0).astype(float).where(known)
        r[f'one_percent{bps}'] = r[f'sustained_return{bps}'].ge(.01).astype(float).where(known)
        r[f'unknown{bps}'] = ~known & ~r.known_no_trade; r[f'sensitive_known{bps}'] = known & ~r.period_exit_bad_day
    return r

def mark(price, shares, cash, bps):
    with np.errstate(invalid='ignore', divide='ignore'):
        value = shares * (price - np.maximum(.005, price * bps / 10000))
        return (value - np.maximum(5., value * .0003) - value * .00051) / cash - 1

def first(flags):
    return np.where(flags.any(axis=1), flags.argmax(axis=1), -1)

def verify_states(got,entry,obs):
    source=entry.merge(obs,on=LABEL_KEYS,validate='one_to_one');c=conn();c.register('source',source)
    c.execute('''CREATE VIEW facts AS SELECT *,
        CASE WHEN NOT necessary_tradeable THEN 'not_submitted'
             WHEN NOT coalesce(isfinite(entry_vwap) AND entry_vwap>0 AND entry_volume>0,false) THEN 'no_liquidity'
             WHEN NOT coalesce(entry_volume*.1>=decision_shares,false) THEN 'volume_cap'
             WHEN entry_vwap*1.0005>=upper_limit-.005 THEN 'estimated_upper_limit' ELSE 'filled' END AS fill,
        coalesce(isfinite(next_preclose) AND next_preclose>0 AND abs(next_preclose-round(next_preclose,2))<=.0001,false) AS valid_ref,
        coalesce(isfinite(day_close) AND day_close>0 AND abs(day_close-round(day_close,2))<=.0001,false) AS valid_close,
        NOT entry_source_valid OR period_entry_bad_day OR period_bad_symbol AS entry_source_unknown FROM source''')
    c.execute('''CREATE VIEW checked AS SELECT *,fill='filled' AS recorded,
        NOT entry_source_unknown AND fill<>'filled' AS known_no_trade,
        NOT catalog_covered OR action_exposure OR NOT valid_ref OR NOT valid_close OR abs(next_preclose-day_close)>.005 AS corporate_unknown,
        coalesce(valid_ref AND next_trade_status=1 AND next_isST IN (0,1) AND next_adjustflag=3,false) AS next_daily_valid,
        fill='filled' AND (NOT entry_bounds_valid OR round(entry_high,2)>=upper_limit) AS queue_unknown FROM facts''')
    c.execute('''CREATE VIEW states AS SELECT *,CASE WHEN entry_source_unknown THEN 'entry_source_unknown'
        WHEN known_no_trade THEN 'no_trade' WHEN queue_unknown THEN 'entry_queue_unknown'
        WHEN corporate_unknown THEN 'corporate_unknown' WHEN NOT next_daily_valid THEN 'next_daily_unknown'
        WHEN NOT source_valid THEN 'morning_source_unknown' ELSE 'known' END AS observation_status FROM checked''')
    expected = c.sql('''SELECT date,code,corporate_unknown,next_daily_valid,entry_source_unknown,known_no_trade,observation_status,
        recorded AS entry_recorded,queue_unknown AS entry_queue_unknown,fill AS entry_fill_status FROM states ORDER BY date,code''').df()
    equal(got[expected.columns], expected, exact=True); count = len(got)*(len(expected.columns)-2)
    for bps in [5, 15]:
        c.execute(f'''CREATE OR REPLACE VIEW cash AS WITH b AS(SELECT *,entry_vwap+greatest(entry_vwap*{bps}/10000.,.005) AS buy_price FROM states),
            v AS(SELECT *,decision_shares*buy_price AS buy_value FROM b)
            SELECT *,buy_value+greatest(buy_value*.0003,5)+buy_value*.00001 AS buy_cash,
                recorded AND buy_price>=upper_limit-.005 AS stress,
                observation_status='known' AND NOT(recorded AND buy_price>=upper_limit-.005) AS known FROM v''')
        ex = c.sql('''SELECT date,code,buy_cash,stress,known,known AND NOT period_exit_bad_day AS sensitive_known,
            NOT known AND NOT known_no_trade AS unknown FROM cash ORDER BY date,code''').df()
        ex = ex.rename(columns={n: f'{n}{bps}' for n in ['buy_cash', 'known', 'sensitive_known', 'unknown']}).rename(columns={'stress': f'entry_stress_unknown{bps}'})
        equal(got[ex.columns], ex); count += len(got)*5
        for name, price in [('sustained', 'sustained_close'), ('any_close', 'max_close'), ('mark_0959', 'price_0959'), ('adverse', 'min_low')]:
            ex = c.sql(f'''WITH v AS(SELECT *,decision_shares*({price}-greatest({price}*{bps}/10000.,.005)) AS mark_value FROM cash)
                SELECT date,code,CASE WHEN known THEN (mark_value-greatest(mark_value*.0003,5)-mark_value*.00051)/buy_cash-1 END AS value
                FROM v ORDER BY date,code''').df()
            np.testing.assert_allclose(got[f'{name}_return{bps}'], ex.value, atol=2e-10, rtol=0, equal_nan=True); count += len(got)
            if name in ['sustained', 'any_close']:
                target = f'opportunity{bps}' if name == 'sustained' else f'any_opportunity{bps}'
                np.testing.assert_array_equal(got[target], np.where(got[f'known{bps}'], ex.value.gt(0).astype(float), np.nan)); count += len(got)
                if name == 'sustained':
                    np.testing.assert_array_equal(got[f'one_percent{bps}'], np.where(got[f'known{bps}'], ex.value.ge(.01).astype(float), np.nan)); count += len(got)
    c.close()
    return count


def events(parent):
    q=parent.loc[parent.known5|parent.known15].sort_values(LABEL_KEYS).reset_index(drop=True)
    if not len(q):
        result=parent.copy()
        for bps in [5,15]:
            for name in NAMES:result[name+str(bps)]=np.nan
        return result
    closes = np.stack(q.close_values)[:, :29]; lows = np.stack(q.low_values)[:, :29]
    assert closes.shape == lows.shape == (len(q),29)
    active = (q.active_mask.to_numpy(dtype='int64')[:,None] & (1 << np.arange(29))) != 0
    triple_active = active[:,:-2] & active[:,1:-1] & active[:,2:]
    triple_close = np.minimum(np.minimum(closes[:,:-2],closes[:,1:-1]),closes[:,2:])
    best = np.where(triple_active,triple_close,-np.inf).max(axis=1); best[~np.isfinite(best)] = np.nan
    adverse_price = np.where(active,lows,np.inf).min(axis=1); adverse_price[~np.isfinite(adverse_price)] = np.nan
    out = q[LABEL_KEYS].copy(); sql = conn(); sql.register('quotes',q)
    for bps in [5,15]:
        known = q[f'known{bps}'].to_numpy(); shares=q.decision_shares.to_numpy()[:,None]; cash=q[f'buy_cash{bps}'].to_numpy()[:,None]
        cm=mark(closes,shares,cash,bps); lm=mark(lows,shares,cash,bps)
        positive=active & (cm>0); space=active & (cm>=.01)
        a=first(positive[:,:-2] & positive[:,1:-1] & positive[:,2:]); a=np.where(a>=0,a+2,-1)
        b=first(space[:,:-2] & space[:,1:-1] & space[:,2:]); b=np.where(b>=0,b+2,-1)
        z=first(active & (lm<=-.03))
        before=lambda x:(x>=0) & ((z<0) | (x<z))
        def until(x):
            value=np.where(active & (np.arange(29)[None,:]<=x[:,None]),lm,np.inf).min(axis=1)
            value[~np.isfinite(value)]=np.nan;return value
        values=[a,b,z,before(a),before(b),(a>=0)&(z>a),(z>=0)&(a>=z),until(a),until(b),active.any(axis=1)]
        for name,value in zip(NAMES,values):out[name+str(bps)]=np.where(known,np.asarray(value,dtype=float),np.nan)
        sustained=mark(best,q.decision_shares.to_numpy(),q[f'buy_cash{bps}'].to_numpy(),bps)
        adverse=mark(adverse_price,q.decision_shares.to_numpy(),q[f'buy_cash{bps}'].to_numpy(),bps)
        np.testing.assert_allclose(sustained[known],q.loc[known,f'sustained_return{bps}'],rtol=0,atol=2e-10,equal_nan=True)
        np.testing.assert_allclose(adverse[known],q.loc[known,f'adverse_return{bps}'],rtol=0,atol=2e-10,equal_nan=True)
        np.testing.assert_array_equal(a[known]>=0,q.loc[known,f'opportunity{bps}'].eq(1))
        np.testing.assert_array_equal(b[known]>=0,q.loc[known,f'one_percent{bps}'].eq(1))
        fields=f'''WITH bars AS(SELECT date,code,pos,known{bps} AS known,
            (active_mask & (1::BIGINT<<pos))<>0 AS active,decision_shares,buy_cash{bps} AS cash,
            list_extract(close_values,pos+1) AS close,list_extract(low_values,pos+1) AS low
            FROM quotes CROSS JOIN range(29) t(pos)),
            cash_marks AS(SELECT *,decision_shares*(close-greatest(.005,close*{bps}/10000.)) AS cv,
                decision_shares*(low-greatest(.005,low*{bps}/10000.)) AS lv FROM bars),
            marks AS(SELECT *, (cv-greatest(5.,cv*.0003)-cv*.00051)/cash-1 AS cm,
                (lv-greatest(5.,lv*.0003)-lv*.00051)/cash-1 AS lm FROM cash_marks),
            windows AS(SELECT *,count(*) OVER w AS n,
                count(*) FILTER(WHERE active AND cm>0) OVER w AS positive,
                count(*) FILTER(WHERE active AND cm>=.01) OVER w AS space FROM marks
                WINDOW w AS(PARTITION BY date,code ORDER BY pos ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)),
            events AS(SELECT date,code,bool_or(known) AS known,
                coalesce(min(pos) FILTER(WHERE n=3 AND positive=3),-1) AS a,
                coalesce(min(pos) FILTER(WHERE n=3 AND space=3),-1) AS b,
                coalesce(min(pos) FILTER(WHERE active AND lm<=-.03),-1) AS z,
                bool_or(active) AS risk FROM windows GROUP BY date,code),
            prefixes AS(SELECT e.date,e.code,min(m.lm) FILTER(WHERE m.active AND m.pos<=e.a) AS ap,
                min(m.lm) FILTER(WHERE m.active AND m.pos<=e.b) AS bp
                FROM events e JOIN marks m USING(date,code) GROUP BY e.date,e.code)
            SELECT e.date,e.code,CASE WHEN known THEN a END AS first_positive_end{bps},
                CASE WHEN known THEN b END AS first_space_end{bps},CASE WHEN known THEN z END AS first_bad3{bps},
                CASE WHEN known THEN (a>=0 AND (z<0 OR a<z))::INT END AS positive_before_bad3{bps},
                CASE WHEN known THEN (b>=0 AND (z<0 OR b<z))::INT END AS space_before_bad3{bps},
                CASE WHEN known THEN (a>=0 AND z>a)::INT END AS positive_then_bad3{bps},
                CASE WHEN known THEN (z>=0 AND a>=z)::INT END AS bad3_before_positive{bps},
                CASE WHEN known THEN ap END AS adverse_until_positive{bps},
                CASE WHEN known THEN bp END AS adverse_until_space{bps},
                CASE WHEN known THEN risk::INT END AS risk_observed{bps}
                FROM events e JOIN prefixes USING(date,code) ORDER BY e.date,e.code'''
        ex=sql.sql(fields).df();pd.testing.assert_frame_equal(out[['date','code']],ex[['date','code']],check_exact=True)
        columns=[name+str(bps) for name in NAMES]
        np.testing.assert_allclose(out[columns].to_numpy(dtype=float),
            ex[columns].to_numpy(dtype=float,na_value=np.nan),rtol=0,atol=2e-12,equal_nan=True)
    sql.close()
    return parent.merge(out,on=LABEL_KEYS,how='left',validate='one_to_one')



def labels(p,root,joint):
    folder=root/'labels';folder.mkdir(exist_ok=False)
    extra=pd.read_parquet(p['extra_inputs'])
    c=conn();c.register('extra',extra);c.read_parquet(p['calendar']).create_view('calendar')
    keys=c.sql("""WITH days AS(SELECT calendar_date AS date,lead(calendar_date) OVER(ORDER BY calendar_date) AS next_date
        FROM calendar WHERE is_trading_day='1' AND calendar_date BETWEEN '2024-01-01' AND '2025-12-31')
        SELECT e.*,d.next_date FROM extra e JOIN days d USING(date) ORDER BY date,code""").df()
    assert len(keys)==2491 and keys.next_date.gt(keys.date).all() and keys.next_date.lt('2026-01-01').all()
    c.register('keys',keys[LABEL_KEYS])
    entry=c.execute('SELECT '+','.join('o.'+n for n in BUY_COLUMNS)+
        ' FROM read_parquet(?) o JOIN keys USING(date,code,next_date) ORDER BY date,code',[p['entry_cache']]).df()
    equal(entry[KEYS+['next_date','price_1449','preclose','upper_limit']],keys[KEYS+['next_date','price_1449','preclose','upper_limit']],True)
    assert len(entry)==2491 and entry.necessary_tradeable.all()
    c.read_parquet(list(p['minute_files'].values())).create_view('original')
    raw=c.sql("""WITH s AS(SELECT lower(exchange)||'.'||symbol AS code,strftime(timestamp,'%Y-%m-%d') AS obs_date,
        strftime(timestamp,'%H:%M') AS clock,timestamp,open::DOUBLE AS open,high::DOUBLE AS high,
        low::DOUBLE AS low,close::DOUBLE AS close,volume::DOUBLE AS volume,turnover::DOUBLE AS amount
        FROM original WHERE timestamp>=TIMESTAMP '2024-01-01' AND timestamp<TIMESTAMP '2026-01-01'
        AND (strftime(timestamp,'%H:%M') BETWEEN '09:31' AND '10:00'
            OR strftime(timestamp,'%H:%M') BETWEEN '14:52' AND '14:55')))
        SELECT k.date,k.next_date,s.*, 'entry' AS kind FROM s JOIN keys k ON s.code=k.code AND s.obs_date=k.date
        WHERE clock BETWEEN '14:52' AND '14:55'
        UNION ALL SELECT k.date,k.next_date,s.*, 'morning' AS kind FROM s JOIN keys k ON s.code=k.code AND s.obs_date=k.next_date
        WHERE clock BETWEEN '09:31' AND '10:00' ORDER BY date,code,kind,timestamp""").df()
    assert raw.obs_date.lt('2026-01-01').all() and not raw.duplicated(['date','code','kind','timestamp']).any()
    raw.to_parquet(folder/'raw.parquet',index=False,compression='zstd')
    # Entry cache is rejoined by complete key and checked from original minutes.
    e=raw.loc[raw.kind.eq('entry')].copy();prices=e[['open','high','low','close']]
    e['good']=(np.isfinite(e[['open','high','low','close','volume','amount']]).all(axis=1)
        &prices.gt(0).all(axis=1)&e.timestamp.eq(e.timestamp.dt.floor('min'))
        &e.high.add(.0001).ge(prices.max(axis=1))&e.low.sub(.0001).le(prices[['open','close']].min(axis=1))
        &e.volume.ge(0)&e.amount.ge(0)&e.volume.eq(0).eq(e.amount.eq(0))
        &(e.volume.eq(0)|(e.amount/e.volume).between(e.low-.0101,e.high+.0101)))
    e['bad_cent']=e.volume.gt(0)&~((e.high-e.high.round(2)).abs().le(.0001)&(e.low-e.low.round(2)).abs().le(.0001))
    e['active_low']=e.low.where(e.volume.gt(0));e['active_high']=e.high.where(e.volume.gt(0));e['active']=e.volume.gt(0)
    rebuilt=e.groupby(LABEL_KEYS).agg(entry_bars=('timestamp','size'),entry_labels=('clock','nunique'),good=('good','sum'),
        entry_volume=('volume','sum'),amount=('amount','sum'),entry_low=('active_low','min'),entry_high=('active_high','max'),
        active=('active','sum'),bad_cent=('bad_cent','sum')).reset_index()
    rebuilt['entry_vwap']=rebuilt.amount/rebuilt.entry_volume.replace(0,np.nan)
    rebuilt['entry_source_valid']=rebuilt.entry_bars.eq(4)&rebuilt.entry_labels.eq(4)&rebuilt.good.eq(4)
    rebuilt['entry_bounds_valid']=rebuilt.active.gt(0)&rebuilt.bad_cent.eq(0)
    original=entry.merge(rebuilt[LABEL_KEYS],on=LABEL_KEYS,validate='one_to_one')
    names=['entry_bars','entry_labels','entry_source_valid','entry_bounds_valid','entry_volume','entry_vwap','entry_low','entry_high']
    equal(original[LABEL_KEYS+names],rebuilt[LABEL_KEYS+names])
    absent=entry.merge(rebuilt[LABEL_KEYS].assign(present=True),on=LABEL_KEYS,how='left',validate='one_to_one')
    assert not absent.loc[~absent.present.eq(True),'entry_source_valid'].any()
    morning=raw.loc[raw.kind.eq('morning')].copy();c.register('raw',morning)
    obs=c.sql("""WITH b AS(SELECT *,coalesce(timestamp=date_trunc('minute',timestamp)
        AND isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close) AND isfinite(volume) AND isfinite(amount)
        AND least(open,high,low,close)>0 AND volume>=0 AND amount>=0 AND (volume=0)=(amount=0)
        AND high+.0001>=greatest(open,close,low) AND low-.0001<=least(open,close)
        AND abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
        AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001
        AND (volume=0 OR amount/volume BETWEEN low-.0101 AND high+.0101),false) AS valid FROM raw),
        rolling AS(SELECT *,min(close) OVER w AS low_three,count(*) OVER w AS n_three,
        count(*) FILTER(WHERE valid AND volume>0) OVER w AS active_three,min(timestamp) OVER w AS first_three
        FROM b WINDOW w AS(PARTITION BY date,code ORDER BY timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)),
        a AS(SELECT date,code,next_date,count(*) AS bars30,count(DISTINCT clock) AS labels30,
        count(*) FILTER(WHERE valid) AS good30,count(*) FILTER(WHERE clock<='09:59') AS bars29,
        count(DISTINCT clock) FILTER(WHERE clock<='09:59') AS labels29,count(*) FILTER(WHERE clock<='09:59' AND valid) AS good29,
        count(*) FILTER(WHERE clock<='09:59' AND valid AND volume>0) AS active_minutes,
        max(close) FILTER(WHERE clock<='09:59' AND valid AND volume>0) AS max_close,
        max(low_three) FILTER(WHERE clock<='09:59' AND n_three=3 AND active_three=3 AND timestamp-first_three=INTERVAL 2 MINUTE) AS sustained_close,
        min(low) FILTER(WHERE clock<='09:59' AND valid AND volume>0) AS min_low,
        max(close) FILTER(WHERE clock='09:59' AND valid AND volume>0) AS price_0959,
        list(close ORDER BY timestamp) AS close_values,list(low ORDER BY timestamp) AS low_values,
        bit_or(CASE WHEN valid AND volume>0 THEN 1::BIGINT<<(extract(hour FROM timestamp)*60+extract(minute FROM timestamp)-571)::INT ELSE 0 END) AS active_mask
        FROM rolling GROUP BY date,code,next_date)
        SELECT k.*,a.* EXCLUDE(date,code,next_date),coalesce(bars30=30 AND labels30=30 AND good30=30,false) AS source_valid,
        coalesce(bars29=29 AND labels29=29 AND good29=29,false) AS source_valid_0959
        FROM keys k LEFT JOIN a USING(date,code,next_date) ORDER BY date,code""").df();c.close()
    independent=observations_pandas(morning)
    expected=keys[LABEL_KEYS].merge(independent,on=LABEL_KEYS,how='left',validate='one_to_one')
    expected['source_valid']=expected.bars30.eq(30)&expected.labels30.eq(30)&expected.good30.eq(30)
    expected['source_valid_0959']=expected.bars29.eq(29)&expected.labels29.eq(29)&expected.good29.eq(29)
    equal(obs[expected.columns],expected,True)
    parent=classify(entry,obs)
    verify_states(parent,entry,obs)
    ordered=events(parent)
    old=pd.read_parquet(p['original_ordered_labels'])
    refs=pd.read_parquet(p['original_reference_labels'],columns=['date','code','mark_0959_return5','mark_0959_return15'])
    old=old.merge(refs,on=['date','code'],validate='one_to_one')
    new=ordered[old.columns]
    full=pd.concat([old,new],ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    pool=pd.read_parquet(p['visible_pool'],columns=[*KEYS,'original_pool'])
    equal(full[KEYS],pool[KEYS],True);equal(full.loc[pool.original_pool],old,True)
    assert len(full)==1260576 and not full.duplicated(['date','code']).any()
    sources=dict(p['source_hashes'])
    for name,frame in [('keys',keys),('entry',entry),('observations',obs),('extra_details',ordered),('extra_labels',new),('labels',full)]:
        file=folder/(name+'.parquet');frame.to_parquet(file,index=False,compression='zstd');sources[str(file)]=sha(file)
    sources[str(folder/'raw.parquet')]=sha(folder/'raw.parquet')
    save_json(folder/'complete_receipt.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),source_hashes=sources,
        all_original_labels_flags_and_references_exact=True,extra_rows=len(new),all_extra_entry_original_minutes_checked=True,
        all_extra_morning_aggregates_independently_pandas_rebuilt=True,all_states_cash_events_independently_SQL_rebuilt=True,
        original_30bar_quality_guard_retained=True,ten_oclock_used_only_for_source_guard=True,
        economic_window=['09:31','09:59'],new_fits=0,new_economic_groups_read=False,new_2026_prices_read=False,no_exit_rules=True))
    print(json.dumps(dict(label_complete_sha256=sha(folder/'complete_receipt.json'),extra_rows=len(new))),flush=True)


def analyze(p,root,joint):
    receipt=root/'labels/complete_receipt.json';gate=json.loads(receipt.read_text());assert gate['passed'];check_sources(gate['source_hashes'])
    assert sha(receipt) in subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    assert not (root/'economic_report.json').exists()
    labels=pd.read_parquet(root/'labels/labels.parquet')
    frames={item['group']:checked_selection(Path(item['root'])) for item in joint['selections']}
    frames.update({f'original{year}':checked_selection(Path(folder)) for year,folder in p['original_lists'].items()})
    summaries=[];daily=[];comparisons=[];parts=[]
    for item in joint['selections']:
        group,year=item['group'],item['year'];frame=frames[group]
        rows=labels.merge(frame,on=KEYS,validate='one_to_one');selected=rows.loc[rows.selected]
        for arm,subset in [('formula',selected),('base_same_dates',rows.loc[rows.date.isin(selected.date.unique())])]:
            for bps in [5,15]:
                for sensitive in [False,True]:
                    d=aggregate(subset,bps,sensitive);summaries.extend(summarize(d,year,group,arm,bps,sensitive))
                    d['group']=group;d['arm']=arm;d['bps']=bps;d['sensitive']=sensitive;daily.append(d)
        comp,dd=paired(frame,frames['original'+year],labels,year,group,'original'+year);comparisons.append(comp);parts.extend(dd)
    for year in ['2024','2025']:
        comp,dd=paired(frames['expanded'+year],frames['extra'+year],labels,year,'expanded'+year,'extra'+year)
        comparisons.append(comp);parts.extend(dd)
    d=pd.concat(daily,ignore_index=True);d.to_parquet(root/'economic_daily.parquet',index=False,compression='zstd')
    pd.concat(parts,ignore_index=True).to_parquet(root/'paired_daily.parquet',index=False,compression='zstd')
    checks=0
    for s in summaries:
        q=d.loc[d.group.eq(s['group'])&d.arm.eq(s['arm'])&d.bps.eq(s['bps'])&d.sensitive.eq(s['sensitive'])]
        q=period(q,s['period']);eq(s['days'],len(q),'days')
        for name in COUNTS:eq(s[name],int(q[name].sum()),name);checks+=1
        for name in METRICS:
            v=q[name].mean();eq(s[name],float(v) if pd.notna(v) else None,name);checks+=1
        for name in ['positive_before_rate','space_before_rate','lower','upper']:
            eq(s[name+'_ci'],interval(q,name),name+'_ci');checks+=1
    def get(group,period_name):
        return next(s for s in summaries if s['group']==group and s['period']==period_name and s['arm']=='formula' and s['bps']==15 and not s['sensitive'])
    # Original formula evidence is rechecked on unchanged selected keys and labels.
    prior=json.loads(Path(p['original_economic_report']).read_text())
    controls={y:next(s for s in prior['summaries'] if s['group']=='original50'+y and s['period']==y and s['arm']=='formula' and s['bps']==15 and not s['sensitive']) for y in ['2024','2025']}
    criteria=dict(four_half_coverage=all(get('expanded'+y,y+h)['days']>=20 and get('expanded'+y,y+h)['known']>=100 for y in ['2024','2025'] for h in ['H1','H2']),
        both_year_rates_improved=all(get('expanded'+y,y)[n] is not None and get('expanded'+y,y)[n]>controls[y][n] for y in ['2024','2025'] for n in ['positive_before_rate','space_before_rate']),
        both_year_coupled_intervals_positive=all(s['lower_ci'] is not None and s['lower_ci'][0]>0 for comp in comparisons if comp['left'].startswith('expanded') and comp['right'].startswith('original') for s in comp['summaries'] if s['period']==comp['year'] and s['bps']==15 and not s['sensitive'] and s['target']=='positive_before'),
        both_year_prior_risk_not_worse=all(get('expanded'+y,y)['bad_first_rate'] is not None and get('expanded'+y,y)['bad_first_rate']<=controls[y]['bad_first_rate'] for y in ['2024','2025']),
        both_year_daily_median_at_most_five=all(json.loads((root/('expanded'+y)/'selection_report.json').read_text())['median_daily']<=5 for y in ['2024','2025']))
    sources={**joint['source_hashes'],str(receipt):sha(receipt),str(root/'economic_daily.parquet'):sha(root/'economic_daily.parquet'),str(root/'paired_daily.parquet'):sha(root/'paired_daily.parquet')}
    save_json(root/'economic_report.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),source_hashes=sources,summaries=summaries,
        comparisons=comparisons,criteria=criteria,independent_summary_checks=checks,all_daily_statistics_and_coupled_bounds_SQL_verified=True,
        extra_only_diagnostic_not_release_formula=True,not_realized_profit=True,new_fits=0,new_2026_prices_read=False,no_exit_rules=True))
    print(json.dumps(dict(economic_sha256=sha(root/'economic_report.json'),criteria=criteria,summary_checks=checks)),flush=True)


def main():
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['labels','analyze']);args=a.parse_args()
    p,root,joint=checked();globals()[args.stage](p,root,joint)


if __name__=='__main__':main()
