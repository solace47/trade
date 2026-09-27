"""Post-freeze morning observations with reused, reauthorized buy windows."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_liquidity as selection
from . import tail_formula_liquidity_inputs as inputs
from .corporate_cash import MINUTES, save_json, sha
from .tail_formula_1000_analysis import BUY_COLUMNS, OBS_COLUMNS, classify, analyze as common_analysis
from .turnover_reference import CALENDAR

ROOT, PROTOCOL = inputs.ROOT, inputs.PROTOCOL
LABELS = ROOT/'labels'
OLD_BUY = Path('data/research/economic_winner/period_quality')
OLD_FULL = Path('data/research/tail_formula_1000')


def checked_old():
    assert sha(OLD_BUY/'label_report.json') == 'df796ea55cbcda5a40cfbd05cfff450bba263347ed499f661d936a8c5386cf68'
    report = json.loads((OLD_BUY/'label_report.json').read_text())
    proof = json.loads((OLD_BUY/'label_verification.json').read_text())
    assert proof['passed'] and proof['label_report_sha256'] == sha(OLD_BUY/'label_report.json')
    assert report['labels_sha256'] == sha(OLD_BUY/'labels.parquet')
    assert sha(OLD_FULL/'full_label_report.json') == '1ae3602185c4162cd7fbfad0cec1061bac246e61c1d4f80c9dd2d88632b3ca3f'
    report = json.loads((OLD_FULL/'full_label_report.json').read_text())
    proof = json.loads((OLD_FULL/'full_label_verification.json').read_text())
    assert proof['passed'] and proof['label_report_sha256'] == sha(OLD_FULL/'full_label_report.json')
    assert report['labels_sha256'] == sha(OLD_FULL/'full_labels.parquet')


def prepare():
    if (LABELS/'input_manifest.json').exists():
        raise ValueError('Do not replace post-selection observation identities')
    inputs.checked_sources(); selection.checked_scores(); checked_old()
    sources = json.loads((inputs.OUT/'source_manifest.json').read_text())
    frozen = {}
    for kind in ['added','combined']:
        r = json.loads((ROOT/kind/'selection_report.json').read_text())
        v = json.loads((ROOT/kind/'selection_verification.json').read_text())
        assert v['passed'] and v['selection_report_sha256'] == sha(ROOT/kind/'selection_report.json')
        assert r['selection_sha256'] == sha(ROOT/kind/'selection.parquet')
        frozen[kind] = sha(ROOT/kind/'selection_report.json')
    chosen = pd.read_parquet(ROOT/'combined/selection.parquet')
    dates = sorted(chosen.loc[chosen.selected,'date'].unique())
    universe = pd.read_parquet(inputs.OUT/'universe.parquet')
    keys = universe.loc[universe.date.isin(dates)].reset_index(drop=True).copy()
    cal = pd.read_parquet(CALENDAR)
    days = sorted(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between('2025-01-01','2025-12-31'),'calendar_date'])
    keys['next_date'] = keys.date.map(dict(zip(days[:-1],days[1:])))
    assert keys.next_date.gt(keys.date).all() and keys.next_date.le('2025-12-31').all()
    LABELS.mkdir(exist_ok=True)
    keys.to_parquet(LABELS/'keys.parquet', index=False, compression='zstd')
    c = base.conn(); c.register('keys', keys)
    # Never project old tail-exit prices, status, or returns.
    entry = c.execute('SELECT '+','.join('o.'+x for x in BUY_COLUMNS)+
        ' FROM read_parquet(?) o JOIN keys USING(date,code) ORDER BY date,code',[str(OLD_BUY/'labels.parquet')]).df()
    c.close()
    assert len(entry) == len(keys)
    same = ['date','code','next_date','half','board','decision_shares','price_1449','preclose','upper_limit']
    pd.testing.assert_frame_equal(entry[same],keys[same],check_exact=True)
    assert not entry.necessary_tradeable.any() and keys.necessary_tradeable.all()
    entry.to_parquet(LABELS/'entry_original.parquet', index=False, compression='zstd')
    result = dict(protocol_sha256=sha(PROTOCOL), selection_report_sha256=frozen,
        source_manifest_sha256=sha(inputs.OUT/'source_manifest.json'), calendar_sha256=sha(CALENDAR),
        keys_sha256=sha(LABELS/'keys.parquet'), entry_original_sha256=sha(LABELS/'entry_original.parquet'),
        old_buy_label_report_sha256=sha(OLD_BUY/'label_report.json'),
        old_buy_label_verification_sha256=sha(OLD_BUY/'label_verification.json'),
        original_full_label_report_sha256=sha(OLD_FULL/'full_label_report.json'),
        original_full_label_verification_sha256=sha(OLD_FULL/'full_label_verification.json'),
        rows=len(keys), days=len(dates), codes=keys.code.nunique(), last_observation=keys.next_date.max(),
        all_added_pool_keys_on_combined_signal_dates=True, reused_buy_windows=True,
        old_tail_exit_outcomes_read=False, new_morning_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(LABELS/'input_manifest.json',result)
    return result


def checked_keys():
    p = json.loads(PROTOCOL.read_text())
    m = json.loads((LABELS/'input_manifest.json').read_text())
    for key,path in [('protocol_sha256',PROTOCOL),('source_manifest_sha256',inputs.OUT/'source_manifest.json'),
                     ('calendar_sha256',CALENDAR),('keys_sha256',LABELS/'keys.parquet'),
                     ('entry_original_sha256',LABELS/'entry_original.parquet')]:
        assert m[key] == sha(path)
    for kind,digest in m['selection_report_sha256'].items():
        assert sha(ROOT/kind/'selection_report.json') == digest
        proof = json.loads((ROOT/kind/'selection_verification.json').read_text())
        assert proof['passed'] and proof['selection_report_sha256'] == digest
    return p,m,pd.read_parquet(LABELS/'keys.parquet')


def raw():
    _,m,keys = checked_keys()
    if (LABELS/'raw_report.json').exists():
        raise ValueError('Do not replace frozen raw morning windows')
    sources = json.loads((inputs.OUT/'source_manifest.json').read_text())
    codes = sorted(keys.code.unique()); folder = LABELS/'raw_parts'; folder.mkdir(exist_ok=True)
    parts = {}; rows = 0
    for start in range(0,len(codes),64):
        subset = codes[start:start+64]; path = folder/f'part_{start//64:03d}.parquet'; receipt = path.with_suffix('.json')
        identity = dict(codes=subset, input_manifest_sha256=sha(LABELS/'input_manifest.json'), extractor_sha256=sha(Path(__file__)))
        if receipt.exists():
            meta = json.loads(receipt.read_text())
            assert all(meta[k] == v for k,v in identity.items()) and meta['sha256'] == sha(path)
        else:
            files = [MINUTES/c[:2].upper()/(c[3:]+'.parquet') for c in subset]
            for file in files:
                assert sha(file) == sources['minute_sha256'][str(file)]
            c = base.conn(); c.register('keys',keys.loc[keys.code.isin(subset),['date','code','next_date']])
            c.read_parquet([str(x) for x in files]).create_view('raw')
            f = c.sql('''WITH s AS(SELECT lower(exchange)||'.'||symbol AS code,
                strftime(timestamp,'%Y-%m-%d') AS next_date,timestamp,strftime(timestamp,'%H:%M') AS clock,
                open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,close::DOUBLE AS close,
                volume::DOUBLE AS volume,turnover::DOUBLE AS amount FROM raw
                WHERE timestamp>=TIMESTAMP '2025-01-01' AND timestamp<TIMESTAMP '2026-01-01'
                AND strftime(timestamp,'%H:%M') BETWEEN '09:31' AND '10:00')
                SELECT k.date,s.* FROM s JOIN keys k USING(code,next_date) ORDER BY date,code,timestamp''').df()
            c.close()
            assert not f.duplicated(['date','code','timestamp']).any()
            f.to_parquet(path,index=False,compression='zstd')
            meta = dict(**identity,sha256=sha(path),rows=len(f)); save_json(receipt,meta)
        parts[str(path)] = meta['sha256']; rows += meta['rows']
        print(json.dumps(dict(codes=start+len(subset),total_codes=len(codes),raw_rows=rows)),flush=True)
    r = dict(input_manifest_sha256=sha(LABELS/'input_manifest.json'), extractor_sha256=sha(Path(__file__)),
        parts_sha256=parts, raw_rows=rows, new_2026_prices_read=False, no_exit_rules=True)
    save_json(LABELS/'raw_report.json',r)
    return {k:v for k,v in r.items() if k != 'parts_sha256'}


def windows():
    _,_,keys = checked_keys()
    if (LABELS/'window_report.json').exists():
        raise ValueError('Do not replace morning aggregates')
    r = json.loads((LABELS/'raw_report.json').read_text())
    assert r['input_manifest_sha256'] == sha(LABELS/'input_manifest.json')
    assert r['extractor_sha256'] == sha(Path(__file__))
    for path,digest in r['parts_sha256'].items():
        assert sha(Path(path)) == digest
    c = base.conn(); c.read_parquet(list(r['parts_sha256'])).create_view('raw'); c.register('keys',keys[['date','code','next_date']])
    f = c.sql('''WITH b AS(SELECT *,coalesce(timestamp=date_trunc('minute',timestamp)
        AND isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close)
        AND isfinite(volume) AND isfinite(amount) AND least(open,high,low,close)>0
        AND high+.0001>=greatest(open,close,low) AND low-.0001<=least(open,close)
        AND abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
        AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001
        AND volume>=0 AND amount>=0 AND (volume=0)=(amount=0)
        AND (volume=0 OR amount/volume BETWEEN low-.0101 AND high+.0101),false) AS valid FROM raw),
        rolling AS(SELECT *,min(close) OVER three AS low_three,count(*) OVER three AS n_three,
        count(*) FILTER(WHERE valid AND volume>0) OVER three AS active_three,min(timestamp) OVER three AS first_three
        FROM b WINDOW three AS(PARTITION BY date,code ORDER BY timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)),
        a AS(SELECT date,code,next_date,count(*) AS bars,count(DISTINCT clock) AS labels,
        count(*) FILTER(WHERE valid) AS valid_bars,count(*) FILTER(WHERE valid AND volume>0) AS active_minutes,
        max(close) FILTER(WHERE valid AND volume>0) AS max_close,
        max(low_three) FILTER(WHERE n_three=3 AND active_three=3 AND timestamp-first_three=INTERVAL 2 MINUTE) AS sustained_close,
        min(low) FILTER(WHERE valid AND volume>0) AS min_low,
        max(close) FILTER(WHERE clock='10:00' AND valid AND volume>0) AS price_1000
        FROM rolling GROUP BY date,code,next_date)
        SELECT k.*,a.* EXCLUDE(date,code,next_date),coalesce(bars=30 AND labels=30 AND valid_bars=30,false) AS source_valid
        FROM keys k LEFT JOIN a USING(date,code,next_date) ORDER BY date,code''').df()
    c.close()
    f.to_parquet(LABELS/'observations.parquet',index=False,compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL),input_manifest_sha256=sha(LABELS/'input_manifest.json'),
        raw_report_sha256=sha(LABELS/'raw_report.json'),observations_sha256=sha(LABELS/'observations.parquet'),
        rows=len(f),complete=int(f.source_valid.sum()),new_2026_prices_read=False,no_exit_rules=True)
    save_json(LABELS/'window_report.json',report)
    return report


def labels():
    _,_,keys = checked_keys(); checked_old()
    if (LABELS/'full_label_report.json').exists():
        raise ValueError('Do not replace classified observations')
    proof = json.loads((LABELS/'window_verification.json').read_text())
    assert proof['passed'] and proof['window_report_sha256'] == sha(LABELS/'window_report.json')
    r = json.loads((LABELS/'window_report.json').read_text())
    assert r['observations_sha256'] == sha(LABELS/'observations.parquet')
    entry = pd.read_parquet(LABELS/'entry_original.parquet')
    pd.testing.assert_frame_equal(entry[['date','code','next_date']],keys[['date','code','next_date']],check_exact=True)
    entry['necessary_tradeable'] = keys.necessary_tradeable
    liquid = np.isfinite(entry.entry_vwap) & entry.entry_vwap.gt(0) & entry.entry_volume.gt(0)
    capacity = entry.entry_volume.mul(.1).ge(entry.decision_shares)
    at_limit = entry.entry_vwap.mul(1.0005).ge(entry.upper_limit-.005)
    entry['entry_fill_status'] = np.select([~entry.necessary_tradeable,~liquid,~capacity,at_limit],
        ['not_submitted','no_liquidity','volume_cap','estimated_upper_limit'],default='filled')
    entry['entry_recorded'] = entry.entry_fill_status.eq('filled')
    entry['entry_queue_unknown'] = entry.entry_recorded & (~entry.entry_bounds_valid | entry.entry_high.round(2).ge(entry.upper_limit))
    observations = pd.read_parquet(LABELS/'observations.parquet',columns=OBS_COLUMNS)
    raw = observations.merge(entry,on=['date','code','next_date'],validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
    raw.to_parquet(LABELS/'classified_inputs.parquet',index=False,compression='zstd')
    added = classify(raw)
    old = pd.read_parquet(OLD_FULL/'full_labels.parquet')
    assert set(added.columns) == set(old.columns)
    added = added[old.columns]
    added.to_parquet(LABELS/'added_labels.parquet',index=False,compression='zstd')
    combined = pd.concat([old,added],ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    assert not combined.duplicated(['date','code']).any()
    chosen = pd.read_parquet(ROOT/'combined/selection.parquet')
    assert chosen.loc[chosen.selected,['date','code']].merge(combined[['date','code']],how='left',indicator=True)._merge.eq('both').all()
    dates = chosen.loc[chosen.selected,'date'].unique()
    pd.testing.assert_frame_equal(combined.loc[combined.date.isin(dates),['date','code']].reset_index(drop=True),
        chosen.loc[chosen.date.isin(dates),['date','code']].reset_index(drop=True),check_exact=True)
    combined.to_parquet(LABELS/'full_labels.parquet',index=False,compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), input_manifest_sha256=sha(LABELS/'input_manifest.json'),
        window_verification_sha256=sha(LABELS/'window_verification.json'),
        classified_inputs_sha256=sha(LABELS/'classified_inputs.parquet'),added_labels_sha256=sha(LABELS/'added_labels.parquet'),
        original_full_labels_sha256=sha(OLD_FULL/'full_labels.parquet'), labels_sha256=sha(LABELS/'full_labels.parquet'),
        rows=len(combined),added_rows=len(added),original_rows=len(old),original_labels_unchanged=True,
        all_selected_keys_and_signal_date_baselines_covered=True,
        buy_windows_reused_and_fill_status_recomputed=True, old_tail_exit_outcomes_read=False,
        year_2025_is_exploratory=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(LABELS/'full_label_report.json',report)
    return report


def analyze(kind):
    proof = json.loads((LABELS/'full_label_verification.json').read_text())
    assert proof['passed'] and proof['label_report_sha256'] == sha(LABELS/'full_label_report.json')
    path = ROOT/kind
    proof = json.loads((path/'selection_verification.json').read_text())
    assert proof['passed'] and proof['selection_report_sha256'] == sha(path/'selection_report.json')
    for name in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        if not (path/name).exists():
            (path/name).symlink_to((LABELS/name).resolve())
        assert sha(path/name) == sha(LABELS/name)
    return common_analysis(path,PROTOCOL)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','raw','windows','labels','analyze'])
    p.add_argument('--kind',choices=['added','combined'],default='combined')
    a = p.parse_args()
    result = analyze(a.kind) if a.stage == 'analyze' else globals()[a.stage]()
    print(json.dumps(result,ensure_ascii=False,indent=2))
