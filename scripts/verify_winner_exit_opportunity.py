"""Independently reconstruct fixed-list T1 opportunity windows and cash flows."""
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha

root = Path('data/research/winner_exit_opportunity')
fixed = json.loads((root/'input_report.json').read_text())
raw = json.loads((root/'raw_report.json').read_text())
report = json.loads((root/'analysis_report.json').read_text())
for path, digest in fixed['sha256'].items():
    assert sha(Path(path)) == digest
assert sha(root/'positions.parquet') == fixed['positions_sha256']
assert raw['input_report_sha256'] == sha(root/'input_report.json')
assert report['raw_report_sha256'] == sha(root/'raw_report.json')
for name, digest in report['outputs_sha256'].items():
    assert sha(root/name) == digest
positions = pd.read_parquet(root/'positions.parquet')
calendar = pd.read_parquet('data/baostock/market_2020_2026/metadata/calendar.parquet')
days = calendar.loc[calendar.is_trading_day.eq('1'), 'calendar_date'].tolist()
assert positions.date.map(dict(zip(days, days[1:]))).eq(positions.target_exit_date).all()
assert not positions.duplicated(['model', 'date', 'code']).any()
for model, part in positions.groupby('model'):
    original = pd.read_parquet(Path('data/research/winner_direction')/model/'continued/tick_cost_scenario.parquet')
    original = original.loc[original.arm.eq('high') & original.horizon.eq(1)].sort_values(['date', 'code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(part.sort_values(['date','code']).reset_index(drop=True)[original.columns].convert_dtypes(),
                                  original.convert_dtypes(), check_dtype=False, atol=0, rtol=0)
    bought = part.entry_status.eq('filled')
    assert part.loc[bought, 'shares'].eq(part.loc[bought, 'decision_shares']).all()
    assert part.loc[~bought, 'shares'].eq(0).all()
minute_paths, daily_paths = [], []
for source in raw['sources']:
    code = source['code']
    path, daily = root/'raw_parts'/(code+'.parquet'), root/'raw_parts'/(code+'.daily.parquet')
    assert sha(path) == source['minute_output_sha256'] and sha(daily) == source['daily_output_sha256']
    assert json.loads(path.with_suffix('.json').read_text()) == source
    selected = positions.loc[positions.code.eq(code) & positions.entry_status.eq('filled')]
    assert source['target_dates'] == sorted(selected.target_exit_date.unique())
    assert source['daily_dates'] == sorted(set(selected.date) | set(selected.target_exit_date))
    assert source['minute_source_sha256'] == fixed['expected_sources'][code]['minute_sha256']
    assert source['daily_source_sha256'] == fixed['expected_sources'][code]['daily_sha256']
    minute_paths.append(str(path)); daily_paths.append(str(daily))
c = duckdb.connect(); c.execute('SET threads=4')
c.read_parquet(minute_paths).create_view('raw_bars')
c.execute('''CREATE VIEW raw_numbers AS SELECT timestamp,code,date,open::DOUBLE AS open,high::DOUBLE AS high,
    low::DOUBLE AS low,close::DOUBLE AS close,volume::DOUBLE AS volume,turnover::DOUBLE AS turnover FROM raw_bars''')
c.execute('''CREATE TABLE bars AS SELECT *,hour(timestamp)*60+minute(timestamp) AS minute,
    isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close)
    AND isfinite(volume) AND isfinite(turnover) AND least(open,high,low,close)>0
    AND high+.0001>=greatest(open,close,low) AND low-.0001<=least(open,close)
    AND volume>=0 AND turnover>=0 AND (volume=0)=(turnover=0)
    AND (volume=0 OR turnover/volume BETWEEN low-.0101 AND high+.0101)
    AND timestamp=date_trunc('minute',timestamp) AS bar_valid,
    volume<=0 OR (abs(high-round(high,2))<=.0001 AND abs(low-round(low,2))<=.0001
                  AND high>0 AND low>0) AS cent_valid
    FROM raw_numbers''')
assert c.sql('SELECT count(*) FROM bars').fetchone()[0] == raw['minute_rows']
assert c.sql('SELECT count(*)-count(DISTINCT (code,timestamp)) FROM bars').fetchone()[0] == 0
assert c.sql("SELECT count(*) FROM bars WHERE date NOT BETWEEN '2025-01-01' AND '2025-12-31' OR date<>strftime(timestamp,'%Y-%m-%d') OR NOT (minute BETWEEN 575 AND 690 OR minute BETWEEN 781 AND 895)").fetchone()[0] == 0
c.read_parquet(daily_paths).create_view('daily')
daily = c.sql('SELECT * FROM daily').df().set_index(['code','date'])
assert daily.index.is_unique
catalog = pd.read_parquet('data/research/winner_direction/catalog/events_reconciled.parquet')
minute_days = set(c.sql('SELECT DISTINCT code,date FROM bars').fetchall())
entitlements = []
for row in positions.itertuples():
    status, quantity, cash, category = None, int(row.shares), 0., None
    if row.entry_status != 'filled':
        status = 'not_bought'
    elif (row.code,row.target_exit_date) not in daily.index:
        status = 'target_daily_unknown'
    elif daily.loc[(row.code,row.target_exit_date),'tradestatus'] != 1 or (row.code,row.target_exit_date) not in minute_days:
        status = 'no_target_trading_window'
    else:
        events = catalog.loc[catalog.code.eq(row.code) & catalog.dividOperateDate.gt(row.date) & catalog.dividOperateDate.le(row.target_exit_date)]
        assert len(events)<=1
        if len(events):
            e = events.iloc[0]
            assert row.date<=e.dividRegistDate<row.target_exit_date
            reserve, bonus = Decimal(e.dividReserveToStockPs or '0'), Decimal(e.dividStocksPs or '0')
            if e.dividCashPsBeforeTax:
                per_share = Decimal(e.dividCashPsBeforeTax)
            else:
                assert e.primary_terms_verified and Decimal(str(e.primary_reference_cash))==0
                assert Decimal(str(e.primary_reference_reserve))==reserve>0 and bonus==0
                per_share = Decimal(0)
            assert reserve>=0 and bonus==0 and per_share>=0
            entitlement = Decimal(quantity)*(1+reserve)
            assert entitlement==entitlement.to_integral_value()
            if reserve: assert e.dividOperateDate<=e.dividStockMarketDate<=row.target_exit_date
            if per_share: assert e.dividOperateDate<=e.dividPayDate<='2025-12-31'
            cash = float(Decimal(quantity)*per_share*Decimal('.8'))
            quantity = int(entitlement); category='catalogue_scenario'
        else:
            assert (row.code,row.date) in daily.index
            if abs(daily.loc[(row.code,row.target_exit_date),'preclose']-daily.loc[(row.code,row.date),'close'])>.005:
                status='entitlement_unknown'; category='unexplained_reference_gap'
            else: category='none'
    entitlements.append({'model':row.model,'date':row.date,'code':row.code,'pre_status':status,
                         'shares_at_target':quantity,'net_receivable':cash,'entitlement_status':category})
entitlements = pd.DataFrame(entitlements)
base = positions.merge(entitlements,on=['model','date','code'],validate='one_to_one')
c.register('positions',base)
c.execute('''CREATE TABLE quotes AS WITH times AS (
    SELECT range AS start_minute FROM range(575,688) UNION ALL SELECT range FROM range(781,893)),
    days AS (SELECT DISTINCT code,date FROM bars)
    SELECT d.code,d.date,t.start_minute,printf('%02d%02d',t.start_minute//60,t.start_minute%60) AS start,
      count(b.timestamp)=4 AND coalesce(bool_and(b.bar_valid),false) AS source_valid,
      count(b.timestamp)=4 AND bool_and(b.cent_valid) AND count(*) FILTER(WHERE b.volume>0)>0 AS queue_bounds_valid,
      CASE WHEN count(b.timestamp)=4 THEN sum(b.volume) ELSE NULL END AS volume,
      CASE WHEN count(b.timestamp)=4 AND sum(b.volume)>0 THEN sum(b.turnover)/sum(b.volume) ELSE NULL END AS vwap,
      coalesce(min(b.low) FILTER(WHERE b.volume>0),'Infinity'::DOUBLE) AS positive_low
    FROM days d CROSS JOIN times t LEFT JOIN bars b
      ON b.code=d.code AND b.date=d.date AND b.minute BETWEEN t.start_minute AND t.start_minute+3
    GROUP BY d.code,d.date,t.start_minute''')
# All target dates are in 2025: main board 10%, ChiNext/STAR 20%, ST 5%.
c.execute('''CREATE TABLE calculated AS WITH inputs AS (
    SELECT p.model,p.date,p.code,p.target_exit_date AS target_date,q.* EXCLUDE(code,date),
      p.shares,p.shares_at_target,p.net_receivable,p.entry_price/1.0005 AS raw_buy,
      round(d.preclose::DECIMAL(18,4)*CASE WHEN p.code LIKE 'sz.30%' OR p.code LIKE 'sh.68%' THEN .8
             WHEN d.isST=1 THEN .95 ELSE .9 END,2)::DOUBLE AS lower_limit
    FROM positions p JOIN quotes q ON p.code=q.code AND p.target_exit_date=q.date
      JOIN daily d ON d.code=q.code AND d.date=q.date WHERE p.pre_status IS NULL),
    eligibility AS (SELECT *,NOT queue_bounds_valid OR round(positive_low,2)<=lower_limit AS queue_unknown,
      shares_at_target<=volume*.1 AS capacity_ok,vwap*.9995>lower_limit+.005 AND volume>0 AS baseline_sellable,
      shares*(raw_buy+greatest(raw_buy*.0015,.005)) AS buy_value,
      shares_at_target*(vwap-greatest(vwap*.0015,.005)) AS sell_value FROM inputs),
    flags AS (SELECT *,coalesce(source_valid AND capacity_ok AND baseline_sellable AND NOT queue_unknown,false) AS eligible,
      buy_value+greatest(5.,buy_value*.0003)+buy_value*.00001 AS buy_cost FROM eligibility)
    SELECT *,CASE WHEN eligible THEN (sell_value-greatest(5.,sell_value*.0003)-sell_value*.00051+net_receivable)/buy_cost-1
           ELSE NULL END AS net_return FROM flags''')
keys = ['model','date','code','start']
checked = c.sql('SELECT * FROM calculated ORDER BY model,date,code,start').df()
stored = pd.read_parquet(root/'window_scenarios.parquet').sort_values(keys).reset_index(drop=True)
assert len(checked)==len(stored)
for name in keys+['target_date','source_valid','queue_bounds_valid','capacity_ok','baseline_sellable','queue_unknown','eligible']:
    if not checked[name].fillna(False).eq(stored[name].fillna(False)).all():
        mismatch = ~checked[name].fillna(False).eq(stored[name].fillna(False))
        print(name,checked.loc[mismatch,keys+[name]].head().to_dict('records'),flush=True)
    pd.testing.assert_series_equal(checked[name].fillna(False),stored[name].fillna(False),check_dtype=False,check_names=False)
errors={}
for name in ['volume','vwap','positive_low','net_return']:
    a,b=checked[name].to_numpy(dtype=float),stored[name].to_numpy(dtype=float)
    np.testing.assert_allclose(a,b,atol=3e-12,rtol=1e-13,equal_nan=True)
    finite=np.isfinite(a)&np.isfinite(b)
    errors[name]=float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else 0.
# Reconcile the fixed original T1 tail window on all valid on-time exits.
c.read_parquet(str(root/'opportunities.parquet')).create_view('stored_outcomes')
c.register('original',positions)
reconciliation=c.sql('''SELECT w.net_return,o.tick_return15 FROM calculated w JOIN original o USING(model,date,code)
    WHERE w.start='1452' AND w.eligible AND o.exit_date=o.target_exit_date''').df()
np.testing.assert_allclose(reconciliation.net_return,reconciliation.tick_return15,rtol=0,atol=2e-12)
aggregated=c.sql('''SELECT model,date,code,count(*) FILTER(WHERE eligible) AS valid_windows,
    max(net_return) AS best_return,min(net_return) AS worst_return,
    arg_max(start,net_return ORDER BY start) AS best_start,arg_min(start,net_return ORDER BY start) AS worst_start,
    count(*) FILTER(WHERE NOT source_valid) AS source_bad_windows,
    count(*) FILTER(WHERE queue_unknown) AS queue_unknown_windows
    FROM calculated GROUP BY model,date,code''').df()
expected=base.merge(aggregated,on=['model','date','code'],how='left',validate='one_to_one')
expected['valid_windows']=expected.valid_windows.fillna(0).astype(int)
expected['status']=expected.pre_status.where(expected.pre_status.notna(),np.where(expected.valid_windows.gt(0),'conditional_opportunity','no_verified_sell_window'))
expected['bought']=expected.entry_status.eq('filled')
expected['entry_source_valid']=expected.entry_window_status.eq('valid')
expected['entry_queue_unknown']=~expected.entry_limit_touched.eq(False)
expected['original_exit_date']=expected.exit_date
expected['target_date']=expected.target_exit_date
expected['original_return15']=expected.tick_return15
for name,cutoff in [('ever_positive',0),('ever_one_percent',.01),('ever_three_percent',.03)]:
    expected[name]=expected.best_return.gt(0) if cutoff==0 else expected.best_return.ge(cutoff)
outcomes=pd.read_parquet(root/'opportunities.parquet').sort_values(keys[:3]).reset_index(drop=True)
expected=expected.sort_values(keys[:3]).reset_index(drop=True)
for name in keys[:3]+['status','valid_windows','entry_source_valid','entry_queue_unknown','original_exit_date','target_date']:
    pd.testing.assert_series_equal(expected[name],outcomes[name],check_dtype=False,check_names=False)
for name in ['best_return','worst_return','source_bad_windows','queue_unknown_windows']:
    np.testing.assert_allclose(expected[name].to_numpy(dtype=float),outcomes[name].to_numpy(dtype=float),atol=3e-12,rtol=0,equal_nan=True)
priced=expected.status.eq('conditional_opportunity')
for name in ['best_start','worst_start','ever_positive','ever_one_percent','ever_three_percent']:
    pd.testing.assert_series_equal(expected.loc[priced,name],outcomes.loc[priced,name],check_dtype=False,check_names=False)
for name in ['shares_at_target','net_receivable','entitlement_status']:
    eligible=expected.pre_status.isna()
    pd.testing.assert_series_equal(expected.loc[eligible,name],outcomes.loc[eligible,name],check_dtype=False,check_names=False)
summary_cells=0
for summary in report['summaries']:
    start,end=('2025-01-01','2025-12-31') if summary['period']=='2025' else (('2025-01-01','2025-06-30') if summary['period']=='2025H1' else ('2025-07-01','2025-12-31'))
    p=expected.loc[expected.model.eq(summary['model'])&expected.date.between(start,end)]
    s=p.loc[p.status.eq('conditional_opportunity')]
    if summary['scope']=='valid_entry_source_no_touch': s=s.loc[s.entry_source_valid&~s.entry_queue_unknown]
    ontime=s.original_exit_date.eq(s.target_date);loss=s.original_return15.lt(0)&ontime
    values={'orders':len(p),'bought':int(p.bought.sum()),'opportunity_rows':len(s),
      'no_verified_window':int((p.bought&p.status.ne('conditional_opportunity')).sum()),
      'original_losses':int(loss.sum()),'original_delayed_or_missing_exit_rows_in_scope':int((~ontime).sum()),
      'original_source_unknown_rows_in_scope':int((~s.execution_source_valid).sum()),
      'best_return_median':s.best_return.median(),'worst_return_median':s.worst_return.median(),
      'daily_mean_best_hindsight_only':s.groupby('date').best_return.mean().mean()}
    for name in ['ever_positive','ever_one_percent','ever_three_percent']:
        values[name]=int(s[name].sum());values[name+'_but_original_loss']=int((s[name]&loss).sum())
    for name,value in values.items():
        assert abs(value-summary[name])<3e-12,(name,value,summary[name])
        summary_cells+=1
result={'passed':True,'unchanged_orders':len(positions),'raw_minute_rows':raw['minute_rows'],
    'independent_window_scenarios':len(checked),'independent_positions':len(expected),
    'original_T1_tail_cashflows_reconciled':len(reconciliation),'summary_values':summary_cells,
    'maximum_absolute_errors':errors,'analysis_report_sha256':sha(root/'analysis_report.json'),
    'new_2026_prices_read':False}
save_json(root/'verification_report.json',result)
print(json.dumps(result,ensure_ascii=False,indent=2))
