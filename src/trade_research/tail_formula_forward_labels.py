"""Apply the original next-morning classification to frozen 2026 observations."""
import json

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from .corporate_cash import save_json, sha
from .tail_formula_1000_analysis import classify
from .tail_formula_forward import PROTOCOL
from .tail_formula_forward_inputs import OUT
from .tail_formula_forward_observations import LABELS, CATALOG, checked_keys


def labels():
    p,m,keys=checked_keys()
    if (LABELS/'full_label_report.json').exists():
        raise ValueError('Do not replace the forward labels')
    wp=json.loads((LABELS/'window_verification.json').read_text())
    assert wp['passed'] and wp['window_report_sha256']==sha(LABELS/'window_report.json')
    wr=json.loads((LABELS/'window_report.json').read_text())
    for stem in ['entry','morning']:
        assert wr[stem+'_windows_sha256']==sha(LABELS/(stem+'_windows.parquet'))
    cp=json.loads((CATALOG/'date_verification.json').read_text())
    assert cp['passed'] and cp['date_report_sha256']==sha(CATALOG/'date_report.json')
    cr=json.loads((CATALOG/'date_report.json').read_text())
    assert cr['complete'] and cr['events_sha256']==sha(CATALOG/'date_events.parquet')
    assert cr['coverage_sha256']==sha(CATALOG/'date_coverage.parquet')
    sources=json.loads((OUT/'source_manifest.json').read_text())
    c=base.conn();c.register('keys',keys)
    c.read_parquet(list(sources['daily_sha256'])).create_view('daily')
    c.read_parquet(str(CATALOG/'date_events.parquet')).create_view('events')
    c.read_parquet(str(CATALOG/'date_coverage.parquet')).create_view('coverage')
    # Project the signal day's close separately; April 1's close is not read.
    current=c.execute('''SELECT k.date,k.code,d.close AS day_close FROM keys k LEFT JOIN
        (SELECT date,code,close FROM daily WHERE date BETWEEN ? AND ?) d USING(date,code)
        ORDER BY k.date,k.code''',[p['signal_first'],p['signal_last']]).df()
    following=c.execute('''SELECT k.date,k.code,d.preclose AS next_preclose,d.tradestatus AS next_trade_status,
        d.isST AS next_isST,d.adjustflag AS next_adjustflag FROM keys k LEFT JOIN
        (SELECT date,code,preclose,tradestatus,isST,adjustflag FROM daily WHERE date BETWEEN ? AND ?) d
        ON d.date=k.next_date AND d.code=k.code ORDER BY k.date,k.code''',[p['signal_first'],p['observation_last']]).df()
    exposure=c.sql('''SELECT k.date,k.code,
        EXISTS(SELECT 1 FROM coverage v WHERE v.code=k.code AND v.year=substr(k.date,1,4))
        AND EXISTS(SELECT 1 FROM coverage v WHERE v.code=k.code AND v.year=substr(k.next_date,1,4)) AS catalog_covered,
        EXISTS(SELECT 1 FROM events e WHERE e.code=k.code AND
            (e.dividRegistDate=k.date OR(e.dividOperateDate>k.date AND e.dividOperateDate<=k.next_date))) AS action_exposure
        FROM keys k ORDER BY k.date,k.code''').df();c.close()
    r=keys.copy()
    for frame in [current,following,exposure,pd.read_parquet(LABELS/'entry_windows.parquet')]:
        r=r.merge(frame,on=['date','code'],validate='one_to_one')
    r=r.merge(pd.read_parquet(LABELS/'morning_windows.parquet'),on=['date','code','next_date'],validate='one_to_one',suffixes=('','_morning'))
    quality=json.loads((LABELS/'quality/report.json').read_text())
    assert quality['input_manifest_sha256']==sha(LABELS/'input_manifest.json')
    assert quality['bad_days_sha256']==sha(LABELS/'quality/bad_days.parquet')
    bad=pd.MultiIndex.from_frame(pd.read_parquet(LABELS/'quality/bad_days.parquet'))
    r['period_entry_bad_day']=pd.MultiIndex.from_frame(r[['date','code']]).isin(bad)
    r['period_exit_bad_day']=pd.MultiIndex.from_frame(r[['next_date','code']]).isin(bad)
    r['period_bad_symbol']=r.code.isin(quality['period_bad_symbols'])
    liquid=np.isfinite(r.entry_vwap)&r.entry_vwap.gt(0)&r.entry_volume.gt(0)
    capacity=r.entry_volume.mul(.1).ge(r.decision_shares)
    at_limit=r.entry_vwap.mul(1.0005).ge(r.upper_limit-.005)
    r['entry_fill_status']=np.select([~r.necessary_tradeable,~liquid,~capacity,at_limit],
        ['not_submitted','no_liquidity','volume_cap','estimated_upper_limit'],default='filled')
    r['entry_recorded']=r.entry_fill_status.eq('filled')
    r['entry_queue_unknown']=r.entry_recorded&(~r.entry_bounds_valid|r.entry_high.round(2).ge(r.upper_limit))
    r.to_parquet(LABELS/'classified_inputs.parquet',index=False,compression='zstd')
    classified=classify(r).sort_values(['date','code']).reset_index(drop=True)
    classified.to_parquet(LABELS/'full_labels.parquet',index=False,compression='zstd')
    report=dict(protocol_sha256=sha(PROTOCOL),input_manifest_sha256=sha(LABELS/'input_manifest.json'),
        window_verification_sha256=sha(LABELS/'window_verification.json'),catalog_verification_sha256=sha(CATALOG/'date_verification.json'),
        quality_report_sha256=sha(LABELS/'quality/report.json'),classified_inputs_sha256=sha(LABELS/'classified_inputs.parquet'),
        labels_sha256=sha(LABELS/'full_labels.parquet'),rows=len(classified),first_signal=classified.date.min(),
        last_signal=classified.date.max(),last_observation=classified.next_date.max(),
        original_classification_and_costs_unchanged=True,no_cash_distribution_adjustments=True,
        new_2026_prices_read=True,only_april_morning_prices_read=True,opportunity_is_not_realized_profit=True,no_exit_rules=True)
    save_json(LABELS/'full_label_report.json',report)
    return report


if __name__=='__main__':
    print(json.dumps(labels(),ensure_ascii=False,indent=2))
