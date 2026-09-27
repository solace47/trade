"""Independently verify source joins, unknown states and all forward cost marks."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research.corporate_cash import DAILY, save_json, sha
from trade_research.tail_formula_forward_inputs import OUT
from trade_research.tail_formula_forward_observations import LABELS, CATALOG, checked_keys


def main():
    p,m,keys=checked_keys()
    report=json.loads((LABELS/'full_label_report.json').read_text())
    assert report['labels_sha256']==sha(LABELS/'full_labels.parquet')
    assert report['classified_inputs_sha256']==sha(LABELS/'classified_inputs.parquet')
    assert report['window_verification_sha256']==sha(LABELS/'window_verification.json')
    assert report['catalog_verification_sha256']==sha(CATALOG/'date_verification.json')
    assert report['quality_report_sha256']==sha(LABELS/'quality/report.json')
    r=pd.read_parquet(LABELS/'classified_inputs.parquet');got=pd.read_parquet(LABELS/'full_labels.parquet')
    pd.testing.assert_frame_equal(got[r.columns],r,check_exact=True)
    pd.testing.assert_frame_equal(r[keys.columns],keys,check_exact=True)
    windows=json.loads((LABELS/'window_verification.json').read_text())
    assert windows['passed']
    for stem in ['entry','morning']:
        path=LABELS/f'independent_{stem}.parquet'; assert windows[f'independent_{stem}_sha256']==sha(path)
        a=pd.read_parquet(path)
        if stem=='morning':
            a=a.rename(columns={'valid_bars':'valid_bars_morning'})
        pd.testing.assert_frame_equal(r[a.columns],a,check_exact=True)
    sources=json.loads((OUT/'source_manifest.json').read_text());day=[];following=[]
    for code in sorted(keys.code.unique()):
        path=DAILY/(code.replace('.','_')+'.parquet');assert sha(path)==sources['daily_sha256'][str(path)]
        day.append(pd.read_parquet(path,columns=['date','code','close'],filters=[('date','>=',p['signal_first']),('date','<=',p['signal_last'])]))
        following.append(pd.read_parquet(path,columns=['date','code','preclose','tradestatus','isST','adjustflag'],
            filters=[('date','>=',p['signal_first']),('date','<=',p['observation_last'])]))
    d=keys[['date','code','next_date']].merge(pd.concat(day).rename(columns={'close':'day_close'}),on=['date','code'],how='left',validate='one_to_one')
    next_data=pd.concat(following).rename(columns={'date':'next_date','preclose':'next_preclose','tradestatus':'next_trade_status','isST':'next_isST','adjustflag':'next_adjustflag'})
    d=d.merge(next_data,on=['next_date','code'],how='left',validate='many_to_one')
    pd.testing.assert_frame_equal(r[d.columns],d,check_dtype=False,check_exact=True)
    catalog=json.loads((CATALOG/'date_report.json').read_text())
    assert catalog['events_sha256']==sha(CATALOG/'date_events.parquet') and catalog['coverage_sha256']==sha(CATALOG/'date_coverage.parquet')
    events=pd.read_parquet(CATALOG/'date_events.parquet');coverage=pd.read_parquet(CATALOG/'date_coverage.parquet')
    covered=set(zip(coverage.code,coverage.year))
    assert all((code,'2026') in covered for code in keys.code.unique()) and r.catalog_covered.all()
    exposure=pd.Series(False,index=r.index)
    for code,group in r.groupby('code'):
        for event in events.loc[events.code.eq(code)].itertuples():
            exposure.loc[group.index] |= group.date.eq(event.dividRegistDate) | (group.date.lt(event.dividOperateDate)&group.next_date.ge(event.dividOperateDate))
    np.testing.assert_array_equal(exposure,r.action_exposure)
    quality=json.loads((LABELS/'quality/report.json').read_text())
    issues=[]
    for path,digest in quality['issue_files_sha256'].items():
        assert sha(Path(path))==digest;issues.append(pd.read_csv(path,dtype=str))
    issues=pd.concat(issues,ignore_index=True)
    kinds=json.loads(Path('config/economic_winner_quality.json').read_text())['bad_day_kinds']
    bad=issues.loc[issues.kind.isin(kinds)&issues.date.between(p['signal_first'],p['observation_last']),['date','code']].drop_duplicates().sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(pd.read_parquet(LABELS/'quality/bad_days.parquet'),bad,check_exact=True)
    bad_keys=set(zip(bad.date,bad.code))
    for target,date in [('period_entry_bad_day','date'),('period_exit_bad_day','next_date')]:
        np.testing.assert_array_equal(r[target],[(day,code) in bad_keys for day,code in zip(r[date],r.code)])
    np.testing.assert_array_equal(r.period_bad_symbol,r.code.isin(quality['period_bad_symbols']))
    c=base.conn();c.register('source',r)
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
        coalesce(valid_ref AND next_trade_status=1 AND next_isST IN(0,1) AND next_adjustflag=3,false) AS next_daily_valid,
        fill='filled' AND (NOT entry_bounds_valid OR round(entry_high,2)>=upper_limit) AS queue_unknown FROM facts''')
    c.execute('''CREATE VIEW states AS SELECT *,CASE WHEN entry_source_unknown THEN 'entry_source_unknown'
        WHEN known_no_trade THEN 'no_trade' WHEN queue_unknown THEN 'entry_queue_unknown'
        WHEN corporate_unknown THEN 'corporate_unknown' WHEN NOT next_daily_valid THEN 'next_daily_unknown'
        WHEN NOT source_valid THEN 'morning_source_unknown' ELSE 'known' END AS observation_status FROM checked''')
    expected=c.sql('''SELECT date,code,corporate_unknown,next_daily_valid,entry_source_unknown,known_no_trade,observation_status,
        recorded AS entry_recorded,queue_unknown AS entry_queue_unknown,fill AS entry_fill_status FROM states ORDER BY date,code''').df()
    pd.testing.assert_frame_equal(got[expected.columns],expected,check_dtype=False,check_exact=True)
    checks=len(expected)*(len(expected.columns)-2)
    for bps in [5,15]:
        c.execute(f'''CREATE OR REPLACE VIEW cash AS WITH b AS(SELECT *,entry_vwap+greatest(entry_vwap*{bps}/10000.,.005) AS buy_price FROM states),
            v AS(SELECT *,decision_shares*buy_price AS buy_value FROM b)
            SELECT *,buy_value+greatest(buy_value*.0003,5)+buy_value*.00001 AS buy_cash,
            recorded AND buy_price>=upper_limit-.005 AS stress,
            observation_status='known' AND NOT(recorded AND buy_price>=upper_limit-.005) AS known FROM v''')
        expected=c.sql('SELECT date,code,buy_cash,stress,known,known AND NOT period_exit_bad_day AS sensitive_known,NOT known AND NOT known_no_trade AS unknown FROM cash ORDER BY date,code').df()
        expected=expected.rename(columns={k:f'{k}{bps}' for k in ['buy_cash','known','sensitive_known','unknown']}).rename(columns={'stress':f'entry_stress_unknown{bps}'})
        pd.testing.assert_frame_equal(got[expected.columns],expected,check_dtype=False,rtol=0,atol=2e-10);checks+=len(got)*5
        for name,price in [('sustained','sustained_close'),('any_close','max_close'),('mark_1000','price_1000'),('adverse','min_low')]:
            values=c.sql(f'''WITH v AS(SELECT *,decision_shares*({price}-greatest({price}*{bps}/10000.,.005)) AS mark_value FROM cash)
                SELECT date,code,CASE WHEN known THEN(mark_value-greatest(mark_value*.0003,5)-mark_value*.00051)/buy_cash-1 END AS value
                FROM v ORDER BY date,code''').df().value
            np.testing.assert_allclose(got[f'{name}_return{bps}'],values,rtol=0,atol=2e-10,equal_nan=True);checks+=len(got)
            if name in ['sustained','any_close']:
                column='opportunity' if name=='sustained' else 'any_opportunity'
                expected=pd.Series(np.where(got[f'known{bps}'],values.gt(0).astype(float),np.nan))
                np.testing.assert_allclose(got[f'{column}{bps}'],expected,rtol=0,atol=0,equal_nan=True);checks+=len(got)
            if name=='sustained':
                expected=pd.Series(np.where(got[f'known{bps}'],values.ge(.01).astype(float),np.nan))
                np.testing.assert_allclose(got[f'one_percent{bps}'],expected,rtol=0,atol=0,equal_nan=True);checks+=len(got)
    c.close()
    proof=dict(passed=True,label_report_sha256=sha(LABELS/'full_label_report.json'),rows=len(got),scalar_checks=checks,
        all_daily_source_joins_and_exposure_dates_rebuilt=True,all_unknown_states_and_cash_marks_rebuilt=True,
        no_fixed_exit_rule_used=True,new_2026_prices_read=True,only_april_morning_prices_read=True,no_exit_rules=True)
    save_json(LABELS/'full_label_verification.json',proof)
    print(json.dumps(proof,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
