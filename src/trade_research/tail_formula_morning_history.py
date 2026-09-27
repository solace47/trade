"""Two native inputs from the most recent twenty completed morning windows."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_morning_history_raw as raw
from .corporate_cash import save_json, sha
from .turnover_reference import CALENDAR

ROOT = raw.ROOT
PROTOCOL = raw.PROTOCOL
NEW_EXPRESSIONS = {
    'HM01': '(HMR+' + '+'.join(f'REF(HMR,B{i})' for i in range(19)) + ')/20/V01',
    'HM02': '5*(HMP+' + '+'.join(f'REF(HMP,B{i})' for i in range(19)) + ')',
}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER + '''HMR:=VALUEWHEN(TIME=1000,100*(C/Q-1));
HMP:=VALUEWHEN(TIME=1000,HHV(IF(LLV(V,3)>0,LLV(C,3),0),28)>Q);
'''


def checked_raw():
    p, _, schedule = raw.checked()
    r = json.loads((ROOT / 'raw_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['producer_sha256'] == sha(Path(raw.__file__))
    assert r['schedule_sha256'] == sha(raw.SOURCE / 'stock_days.parquet')
    for path, digest in r['parts_sha256'].items():
        assert sha(Path(path)) == digest
    return p, r, schedule


def windows():
    if (ROOT / 'window_report.json').exists():
        raise ValueError('Do not replace frozen historical morning windows')
    _, r, _ = checked_raw(); c = base.conn()
    c.read_parquet(list(r['parts_sha256'])).create_view('raw')
    out = c.sql('''WITH checked AS (
        SELECT *,coalesce(timestamp=date_trunc('minute',timestamp) AND isfinite(close) AND close>0
            AND abs(close-round(close,2))<=.0001 AND isfinite(volume) AND volume>=0
            AND volume=floor(volume),false) AS good FROM raw),
        triples AS (SELECT *,count(*) OVER w AS n3,min(round(close,2)) OVER w AS min3,
            min(volume) OVER w AS volume_min3 FROM checked
            WINDOW w AS (PARTITION BY code,date ORDER BY timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)),
        agg AS (SELECT date,code,count(*) FILTER(WHERE clock<='1000') AS morning_bars,
            count(DISTINCT clock) FILTER(WHERE clock<='1000') AS morning_clocks,
            count(*) FILTER(WHERE clock<='1000' AND good) AS morning_good,
            max(round(close,2)) FILTER(WHERE clock='1000') AS price_1000,
            max(CASE WHEN n3=3 AND volume_min3>0 THEN min3 ELSE 0 END)
                FILTER(WHERE clock<='1000') AS sustained_close,
            count(*) FILTER(WHERE clock='1449') AS tail_bars,
            count(*) FILTER(WHERE clock='1449' AND good) AS tail_good,
            max(round(close,2)) FILTER(WHERE clock='1449') AS price_1449
            FROM triples GROUP BY date,code)
        SELECT *,morning_bars=30 AND morning_clocks=30 AND morning_good=30 AS morning_valid,
            tail_bars=1 AND tail_good=1 AS tail_valid FROM agg ORDER BY code,date''').df()
    assert not out.duplicated(['date', 'code']).any()
    assert int(out.morning_bars.sum() + out.tail_bars.sum()) == r['raw_minutes']
    c.close(); out.to_parquet(ROOT / 'windows.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), raw_report_sha256=sha(ROOT / 'raw_report.json'),
                  windows_sha256=sha(ROOT / 'windows.parquet'), rows=len(out), raw_minutes=r['raw_minutes'],
                  historical_morning_prices_for_features=True, new_selection_outcomes_read=False,
                  new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'window_report.json', report); return report


def checked_old():
    p = json.loads(PROTOCOL.read_text())
    r = json.loads((previous.ROOT / 'feature_report.json').read_text())
    v = json.loads((previous.ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == p['previous_feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert r['features_sha256'] == sha(previous.ROOT / 'features.parquet')
    return pd.read_parquet(previous.ROOT / 'features.parquet')


def calendar_positions():
    assert sha(CALENDAR) == '25d81e17d9e32f25fc7f61cda244f2ea2df818e6a1c1669d0e8bfcf1c1696400'
    f = pd.read_parquet(CALENDAR)
    days = sorted(f.loc[f.is_trading_day.eq('1'), 'calendar_date'])
    return {day: index for index, day in enumerate(days)}


def features():
    if (ROOT / 'feature_report.json').exists():
        raise ValueError('Do not replace frozen morning-history inputs')
    p, _, schedule = checked_raw(); old = checked_old()
    wr = json.loads((ROOT / 'window_report.json').read_text())
    proof = json.loads((ROOT / 'window_verification.json').read_text())
    assert proof['passed'] and proof['window_report_sha256'] == sha(ROOT / 'window_report.json')
    assert wr['windows_sha256'] == sha(ROOT / 'windows.parquet')
    out = schedule.merge(pd.read_parquet(ROOT / 'windows.parquet'), on=['date', 'code'], how='left', validate='one_to_one')
    out = out.sort_values(['code', 'date']).reset_index(drop=True)
    for name in ['morning_valid', 'tail_valid']:
        out[name] = out[name].eq(True)
    positions = calendar_positions(); out['market_position'] = out.date.map(positions)
    assert out.market_position.notna().all()
    frames = []
    for _, d in out.groupby('code', sort=True):
        d = d.copy()
        d['reference_date'] = d.date.shift(1)
        d['reference_price'] = d.price_1449.shift(1)
        d['reference_valid'] = d.tail_valid.shift(1).eq(True)
        d['atom_valid'] = (d.morning_valid & d.reference_valid & d.reference_date.lt(d.date)
                           & d.reference_price.gt(0) & np.isfinite(d.reference_price))
        d['atom_mark'] = (100 * (d.price_1000 / d.reference_price - 1)).where(d.atom_valid)
        d['atom_positive'] = d.sustained_close.gt(d.reference_price).astype(float).where(d.atom_valid)
        d['stock_day_gap'] = d.market_position.diff().gt(1).astype(int)
        d['mh_first_reference'] = d.reference_date.shift(19)
        d['mh_first_morning'] = d.date.shift(19)
        d['mh_last_morning'] = d.date
        d['mh_valid_count'] = d.atom_valid.astype(int).rolling(20, min_periods=20).sum()
        d['mh_gap_count'] = d.stock_day_gap.rolling(20, min_periods=20).sum()
        d['mh_mean_mark'] = d.atom_mark.rolling(20, min_periods=20).mean()
        d['mh_positive_frequency'] = 100 * d.atom_positive.rolling(20, min_periods=20).mean()
        frames.append(d)
    history = pd.concat(frames, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    history.to_parquet(ROOT / 'history.parquet', index=False, compression='zstd')
    columns = ['date', 'code', 'mh_first_reference', 'mh_first_morning', 'mh_last_morning', 'mh_valid_count',
               'mh_gap_count', 'mh_mean_mark', 'mh_positive_frequency']
    f = old.merge(history[columns], on=['date', 'code'], how='left', validate='one_to_one')
    f = f.sort_values(['date', 'code']).reset_index(drop=True)
    good = (f.mh_valid_count.eq(20) & f.mh_first_reference.lt(f.mh_first_morning)
            & f.mh_last_morning.eq(f.date) & f.V01.gt(0)
            & np.isfinite(f[['mh_mean_mark', 'mh_positive_frequency', 'V01']]).all(axis=1))
    f['HM01'] = (f.mh_mean_mark / f.V01).where(good)
    f['HM02'] = f.mh_positive_frequency.where(good)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['morning_history_valid'] = good
    f['formula_input_valid'] &= good & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    pd.testing.assert_frame_equal(old.drop(columns='formula_input_valid'), f[old.columns].drop(columns='formula_input_valid'), check_exact=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), previous_feature_report_sha256=p['previous_feature_report_sha256'],
                  schedule_manifest_sha256=sha(raw.SOURCE / 'input_manifest.json'), calendar_sha256=sha(CALENDAR),
                  window_report_sha256=sha(ROOT / 'window_report.json'), window_verification_sha256=sha(ROOT / 'window_verification.json'),
                  features_sha256=sha(ROOT / 'features.parquet'), history_sha256=sha(ROOT / 'history.parquet'),
                  rows=len(f), previous_valid=int(old.formula_input_valid.sum()), valid=int(f.formula_input_valid.sum()),
                  newly_invalid=int((f.prior_formula_input_valid & ~f.formula_input_valid).sum()),
                  valid_with_stock_day_gap=int((f.formula_input_valid & f.mh_gap_count.gt(0)).sum()),
                  first_reference=f.mh_first_reference.dropna().min() if f.mh_first_reference.notna().any() else None,
                  last_morning=f.mh_last_morning.dropna().max() if f.mh_last_morning.notna().any() else None,
                  current_morning_included_before_1449=True, expressions=EXPRESSIONS, native_header=HEADER,
                  native_source_parity_verified=False, historical_morning_prices_for_features=True,
                  new_selection_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', report)
    return {key: value for key, value in report.items() if key not in ['expressions', 'native_header']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['windows', 'features'])
    a = p.parse_args(); print(json.dumps(globals()[a.stage](), ensure_ascii=False, indent=2))
