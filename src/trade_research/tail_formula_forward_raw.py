"""Read only frozen buy/morning windows and bounded source-quality evidence."""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import tail_formula_additive as base
from .corporate_cash import DAILY, MINUTES, save_json, sha
from .hf_audit import audit_symbol
from .market_study import BAD_STOCK_FIELDS, _quality_symbols
from .tail_formula_forward import ROOT, PROTOCOL
from .tail_formula_forward_inputs import OUT
from .tail_formula_forward_observations import LABELS, checked_keys


def raw():
    p,m,keys = checked_keys()
    if (LABELS/'raw_report.json').exists():
        raise ValueError('Do not replace frozen forward raw windows')
    sources = json.loads((OUT/'source_manifest.json').read_text())
    codes = sorted(keys.code.unique()); folder = LABELS/'raw_parts'; folder.mkdir(exist_ok=True)
    parts = {}; rows = 0; extractor = sha(Path(__file__))
    for start in range(0,len(codes),64):
        subset = codes[start:start+64]; path = folder/f'part_{start//64:03d}.parquet'; meta_path = path.with_suffix('.json')
        identity = dict(codes=subset,input_manifest_sha256=sha(LABELS/'input_manifest.json'),extractor_sha256=extractor)
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            assert all(meta[k]==v for k,v in identity.items()) and meta['sha256']==sha(path)
        else:
            q = keys.loc[keys.code.isin(subset),['date','code','next_date']]
            windows = pd.concat([q[['date','code']].assign(source_date=q.date,kind='entry'),
                q[['date','code']].assign(source_date=q.next_date,kind='morning')],ignore_index=True)
            files = [MINUTES/code[:2].upper()/(code[3:]+'.parquet') for code in subset]
            for file in files:
                assert sha(file)==sources['minute_sha256'][str(file)]
            c = base.conn(); c.register('windows',windows); c.read_parquet([str(x) for x in files]).create_view('source')
            f = c.execute('''WITH s AS(SELECT lower(exchange)||'.'||symbol AS code,
                strftime(timestamp,'%Y-%m-%d') AS source_date,strftime(timestamp,'%H:%M') AS clock,
                timestamp,open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,close::DOUBLE AS close,
                volume::DOUBLE AS volume,turnover::DOUBLE AS amount FROM source
                WHERE timestamp>=CAST(? AS TIMESTAMP) AND timestamp<CAST(? AS TIMESTAMP)
                AND (strftime(timestamp,'%H:%M') BETWEEN '14:52' AND '14:55'
                    OR strftime(timestamp,'%H:%M') BETWEEN '09:31' AND '10:00'))
                SELECT w.date,w.code,w.kind,s.* EXCLUDE(code) FROM s JOIN windows w USING(source_date,code)
                WHERE (w.kind='entry' AND clock BETWEEN '14:52' AND '14:55')
                    OR (w.kind='morning' AND clock BETWEEN '09:31' AND '10:00')
                ORDER BY w.date,w.code,w.kind,timestamp''',[p['signal_first'],p['observation_last']+' 10:01']).df()
            c.close()
            assert not f.duplicated(['date','code','kind','timestamp']).any()
            f.to_parquet(path,index=False,compression='zstd')
            meta = dict(**identity,sha256=sha(path),rows=len(f)); save_json(meta_path,meta)
        parts[str(path)] = meta['sha256']; rows += meta['rows']
        print(json.dumps(dict(codes=min(start+64,len(codes)),total=len(codes),raw_rows=rows)),flush=True)
    report = dict(protocol_sha256=sha(PROTOCOL),input_manifest_sha256=sha(LABELS/'input_manifest.json'),
        parts_sha256=parts,raw_rows=rows,first_signal=p['signal_first'],last_signal=p['signal_last'],
        last_observation=p['observation_last'],new_2026_prices_read=True,only_april_morning_prices_read=True,no_exit_rules=True)
    save_json(LABELS/'raw_report.json',report)
    return {k:v for k,v in report.items() if k!='parts_sha256'}


def windows():
    p,m,keys = checked_keys()
    if (LABELS/'window_report.json').exists():
        raise ValueError('Do not replace frozen forward aggregates')
    raw_report = json.loads((LABELS/'raw_report.json').read_text())
    assert raw_report['input_manifest_sha256']==sha(LABELS/'input_manifest.json')
    for name,digest in raw_report['parts_sha256'].items():
        assert sha(Path(name))==digest
    c = base.conn(); c.read_parquet(list(raw_report['parts_sha256'])).create_view('raw'); c.register('keys',keys[['date','code','next_date']])
    c.execute('''CREATE VIEW bars AS SELECT *,
        coalesce(timestamp=date_trunc('minute',timestamp) AND isfinite(open) AND isfinite(high) AND isfinite(low)
        AND isfinite(close) AND isfinite(volume) AND isfinite(amount) AND least(open,high,low,close)>0
        AND high+.0001>=greatest(open,close,low) AND low-.0001<=least(open,close)
        AND volume>=0 AND amount>=0 AND (volume=0)=(amount=0)
        AND (volume=0 OR amount/volume BETWEEN low-.0101 AND high+.0101),false) AS valid,
        coalesce(abs(high-round(high,2))<=.0001 AND abs(low-round(low,2))<=.0001,false) AS cent_bounds,
        coalesce(abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
            AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001,false) AS cent_prices
        FROM raw''')
    entry = c.sql('''WITH a AS(SELECT date,code,count(*) AS entry_bars,count(DISTINCT clock) AS entry_labels,
        count(*) FILTER(WHERE valid) AS valid_bars,count(*) FILTER(WHERE volume>0) AS positive_bars,
        count(*) FILTER(WHERE volume>0 AND NOT cent_bounds) AS invalid_bounds,
        sum(volume) AS entry_volume,sum(amount)/nullif(sum(volume),0) AS entry_vwap,
        min(low) FILTER(WHERE volume>0) AS entry_low,max(high) FILTER(WHERE volume>0) AS entry_high
        FROM bars WHERE kind='entry' GROUP BY date,code)
        SELECT k.date,k.code,a.* EXCLUDE(date,code),
        coalesce(entry_bars=4 AND entry_labels=4 AND valid_bars=4,false) AS entry_source_valid,
        coalesce(positive_bars>0 AND invalid_bounds=0,false) AS entry_bounds_valid
        FROM keys k LEFT JOIN a USING(date,code) ORDER BY date,code''').df()
    morning = c.sql('''WITH b AS(SELECT *,valid AND cent_prices AS morning_valid FROM bars WHERE kind='morning'),
        rolling AS(SELECT *,min(close) OVER three AS low_three,count(*) OVER three AS n_three,
        count(*) FILTER(WHERE morning_valid AND volume>0) OVER three AS active_three,
        min(timestamp) OVER three AS first_three FROM b
        WINDOW three AS(PARTITION BY date,code ORDER BY timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)),
        a AS(SELECT date,code,count(*) AS bars,count(DISTINCT clock) AS labels,
        count(*) FILTER(WHERE morning_valid) AS valid_bars,count(*) FILTER(WHERE morning_valid AND volume>0) AS active_minutes,
        max(close) FILTER(WHERE morning_valid AND volume>0) AS max_close,
        max(low_three) FILTER(WHERE n_three=3 AND active_three=3 AND timestamp-first_three=INTERVAL 2 MINUTE) AS sustained_close,
        min(low) FILTER(WHERE morning_valid AND volume>0) AS min_low,
        max(close) FILTER(WHERE clock='10:00' AND morning_valid AND volume>0) AS price_1000 FROM rolling GROUP BY date,code)
        SELECT k.date,k.code,k.next_date,a.* EXCLUDE(date,code),coalesce(bars=30 AND labels=30 AND valid_bars=30,false) AS source_valid
        FROM keys k LEFT JOIN a USING(date,code) ORDER BY date,code''').df()
    c.close()
    entry.to_parquet(LABELS/'entry_windows.parquet',index=False,compression='zstd')
    morning.to_parquet(LABELS/'morning_windows.parquet',index=False,compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL),input_manifest_sha256=sha(LABELS/'input_manifest.json'),
        raw_report_sha256=sha(LABELS/'raw_report.json'),rows=len(keys),entry_windows_sha256=sha(LABELS/'entry_windows.parquet'),
        morning_windows_sha256=sha(LABELS/'morning_windows.parquet'),new_2026_prices_read=True,
        no_new_exit_rules=True,only_april_morning_prices_read=True)
    save_json(LABELS/'window_report.json',report)
    return report


def quality():
    p,m,keys = checked_keys()
    folder = LABELS/'quality'; folder.mkdir(exist_ok=True)
    if (folder/'report.json').exists():
        raise ValueError('Do not replace the bounded quarter quality audit')
    sources = json.loads((OUT/'source_manifest.json').read_text())
    issues_folder = Path('data/research/market_issues_ci')
    files = sorted(issues_folder.glob('shard_*.csv')); assert len(files)==20
    issues = pd.concat([pd.read_csv(path,dtype=str) for path in files],ignore_index=True)
    bad_kinds = json.loads(Path('config/economic_winner_quality.json').read_text())['bad_day_kinds']
    bad = issues.loc[issues.kind.isin(bad_kinds) & issues.date.between(p['signal_first'],p['observation_last']),['date','code']]
    bad = bad.drop_duplicates().sort_values(['date','code']).reset_index(drop=True)
    bad.to_parquet(folder/'bad_days.parquet',index=False,compression='zstd')
    codes = sorted(set(_quality_symbols(issues_folder).code)&set(keys.code))
    bounded = folder/'daily'; bounded.mkdir(exist_ok=True)
    checks = []; severe = []
    for code in codes:
        original_daily = DAILY/(code.replace('.','_')+'.parquet')
        minute = MINUTES/code[:2].upper()/(code[3:]+'.parquet')
        assert sha(original_daily)==sources['daily_sha256'][str(original_daily)]
        assert sha(minute)==sources['minute_sha256'][str(minute)]
        daily = bounded/original_daily.name
        pd.read_parquet(original_daily,filters=[('date','>=',p['signal_first']),('date','<=',p['signal_last'])]).to_parquet(daily,index=False)
        # The auditor's daily reader receives an already-bounded file, so it
        # cannot pull later-2026 prices into this quarter's source check.
        audit,_ = audit_symbol(code,minute,daily,p['signal_first'],p['signal_last'])
        is_bad = audit.get('status')!='ok' or any(int(audit.get(k,0))>0 for k in BAD_STOCK_FIELDS)
        if is_bad:
            severe.append(code)
        checks.append(dict(code=code,severe_in_period=bool(is_bad),status=audit.get('status'),
            fields={k:int(audit.get(k,0)) for k in BAD_STOCK_FIELDS},bounded_daily_sha256=sha(daily)))
    r = dict(protocol_sha256=sha(PROTOCOL),input_manifest_sha256=sha(LABELS/'input_manifest.json'),
        issue_files_sha256={str(x):sha(x) for x in files},bad_days_sha256=sha(folder/'bad_days.parquet'),
        period_bad_symbols=severe,checked_symbols=len(checks),checks=checks,
        symbol_audit_first=p['signal_first'],symbol_audit_last=p['signal_last'],
        day_issue_keys_last=p['observation_last'],existing_april_full_day_issue_flags_reused=True,
        new_april_full_day_prices_read=False,new_2026_prices_read=True,not_used_for_selection=True)
    save_json(folder/'report.json',r)
    return {k:v for k,v in r.items() if k not in ['issue_files_sha256','checks']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['raw','windows','quality'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
