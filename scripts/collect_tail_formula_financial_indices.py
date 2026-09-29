"""Collect fixed brokerage/bank history, with independent dated daily controls."""
import json
from pathlib import Path
import socket
from urllib.parse import urlencode

import requests

import pandas as pd

from probe_historical_ticks import connect, save_json
from probe_index_minutes import request
from trade_research.corporate_cash import sha
from trade_research.turnover_reference import CALENDAR

ROOT = Path('data/research/tail_formula_financial_context')
PROTOCOL = Path('config/tail_formula_financial_context_protocol.json')
PILOT = Path('data/research/tail_formula_financial_index_probe')
OLD = Path('data/research/tail_formula_float')


def collect_daily(p):
    if p.get('daily_provider') == 'tencent_raw_day':
        return collect_tencent_daily(p)
    frames = []; raw_paths = {}; blocks = []
    for code in p['symbols']:
        records = []
        for start, end in p['daily_blocks']:
            assert '2023-06-01' <= start <= end <= '2025-12-31'
            checkpoint = ROOT / 'daily' / f'{code}_{start}_{end}.json'
            params = dict(secid='0.' + code[3:], fields1='f1,f2,f3', fields2='f51,f52,f53,f54,f55',
                          klt='101', fqt='0', beg=start.replace('-', ''), end=end.replace('-', ''))
            url = p['daily_endpoint'] + '?' + urlencode(params)
            if checkpoint.exists():
                entry = json.loads(checkpoint.read_text())
                assert entry['url'] == url and entry['code'] == code
            else:
                entry = dict(code=code, start=start, end=end, url=url, attempts=[])
                for attempt in range(2):
                    folder = ROOT / 'daily/raw' / f'{code}_{start}_{end}' / str(attempt)
                    folder.mkdir(parents=True, exist_ok=True)
                    item = dict(attempt=attempt)
                    try:
                        r = requests.get(url, timeout=5)
                        file = folder / 'response.bin'; file.write_bytes(r.content)
                        item.update(path=str(file), sha256=sha(file), http_status=r.status_code)
                        r.raise_for_status(); data = r.json().get('data')
                        count = len((data or {}).get('klines', []))
                        item.update(status='nonempty' if count else 'empty', rows=count)
                    except Exception as exc:
                        item.update(status='error', error=str(exc), error_type=type(exc).__name__)
                    entry['attempts'].append(item)
                    if item['status'] == 'nonempty': break
                save_json(checkpoint, entry)
            blocks.append(entry)
            successful = [a for a in entry['attempts'] if a['status'] == 'nonempty']
            assert successful, f'No daily control for {code} {start}; preserve failure and stop'
            a = successful[0]; file = Path(a['path']); assert sha(file) == a['sha256']
            data = json.loads(file.read_bytes())['data']
            assert data['code'] == code[3:] and data['market'] == 0
            for bar in data['klines']:
                fields = bar.split(','); assert len(fields) == 5 and start <= fields[0] <= end
                records.append(dict(date=fields[0], code=code, open=fields[1], high=fields[3], low=fields[4], close=fields[2]))
        path = ROOT / 'daily' / (code + '.json')
        save_json(path, dict(symbol=code, start=p['history_first'], end=p['history_last'], records=records))
        f = pd.DataFrame(records)
        assert len(f) and f.code.eq(code).all() and not f.date.duplicated().any()
        for name in ['open', 'high', 'low', 'close']: f[name] = pd.to_numeric(f[name], errors='raise')
        frames.append(f); raw_paths[str(path)] = sha(path)
    return raw_paths, frames, blocks


def collect_tencent_daily(p):
    from probe_tail_formula_financial_tencent import day_rows
    probe = Path('data/research/tail_formula_financial_tencent_probe')
    for key, path in [('tencent_probe_protocol_sha256', Path('config/tail_formula_financial_tencent_probe_protocol.json')),
                      ('tencent_probe_source_sha256', probe / 'source_report.json'),
                      ('tencent_probe_verification_sha256', probe / 'verification.json')]:
        assert p[key] == sha(path)
    assert json.loads((probe / 'verification.json').read_text())['source_usable']
    frames, raw_paths, blocks = [], {}, []
    for code in p['symbols']:
        params = dict(_var='kline_dayqfq', param=f"{code.replace('.', '')},day,{p['history_first']},{p['history_last']},640,qfq", r='0.8205512681390605')
        url = p['daily_endpoint'] + '?' + urlencode(params)
        checkpoint = ROOT / 'daily/tencent' / (code + '.json')
        if checkpoint.exists():
            entry = json.loads(checkpoint.read_text()); assert entry['url'] == url
        else:
            entry = dict(code=code, url=url, attempts=[])
            for attempt in range(2):
                folder = ROOT / 'daily/tencent/raw' / code / str(attempt); folder.mkdir(parents=True, exist_ok=True)
                item = dict(attempt=attempt)
                try:
                    r = requests.get(url, timeout=5)
                    path = folder / 'response.bin'; path.write_bytes(r.content)
                    item.update(path=str(path), sha256=sha(path), http_status=r.status_code)
                    r.raise_for_status(); count = len(day_rows(r.content, code))
                    item.update(status='nonempty' if count else 'empty', rows=count)
                except Exception as exc:
                    item.update(status='error', error_type=type(exc).__name__, error=str(exc))
                entry['attempts'].append(item)
                if item['status'] == 'nonempty': break
            save_json(checkpoint, entry)
        blocks.append(entry)
        a = next((v for v in entry['attempts'] if v['status'] == 'nonempty'), None)
        assert a is not None, 'Preserve bounded Tencent source failure; no alternate date or symbol'
        path = Path(a['path']); assert sha(path) == a['sha256']
        rows = day_rows(path.read_bytes(), code); dates = [r[0] for r in rows]
        assert dates == sorted(set(dates)) and dates[-1] == p['history_last']
        records = [dict(date=r[0], code=code, open=r[1], close=r[2], high=r[3], low=r[4])
                   for r in rows if p['history_first'] <= r[0] <= p['history_last']]
        normalized = ROOT / 'daily/tencent' / (code + '_normalized.json')
        save_json(normalized, dict(symbol=code, start=p['history_first'], end=p['history_last'], records=records))
        raw_paths[str(normalized)] = sha(normalized)
        f = pd.DataFrame(records)
        for name in ['open', 'high', 'low', 'close']: f[name] = pd.to_numeric(f[name], errors='raise')
        frames.append(f)
    return raw_paths, frames, blocks


def main():
    assert not (ROOT / 'source_report.json').exists(), 'Do not replace frozen source grid'
    p = json.loads(PROTOCOL.read_text())
    cfg = json.loads(Path('config/tail_formula_financial_index_probe_protocol.json').read_text())
    assert p['probe_protocol_sha256'] == sha(Path('config/tail_formula_financial_index_probe_protocol.json'))
    assert p['probe_source_report_sha256'] == sha(PILOT / 'source_report.json')
    assert p['probe_verification_sha256'] == sha(PILOT / 'verification.json')
    assert p['previous_feature_report_sha256'] == sha(OLD / 'feature_report.json')
    assert p['calendar_sha256'] == sha(CALENDAR)
    dp = Path('data/research/tail_formula_financial_daily_probe')
    for key, file in [('daily_probe_protocol_sha256', Path('config/tail_formula_financial_daily_probe_protocol.json')), ('daily_probe_source_sha256', dp / 'source_report.json'), ('daily_probe_verification_sha256', dp / 'verification.json')]:
        assert p[key] == sha(file)
    assert json.loads((dp / 'verification.json').read_text())['source_usable']
    assert json.loads((OLD / 'feature_report.json').read_text())['features_sha256'] == sha(OLD / 'features.parquet')
    days = sorted(pd.read_parquet(OLD / 'features.parquet', columns=['date']).date.unique())
    assert days[0] >= '2024-01-01' and days[-1] <= '2025-12-30'
    pilot = {(s['symbol'], s['date']): s for s in json.loads((PILOT / 'source_report.json').read_text())['sessions']}
    raw_paths, frames, blocks = collect_daily(p)
    daily = pd.concat(frames).sort_values(['code', 'date']).reset_index(drop=True)
    calendar = pd.read_parquet(CALENDAR)
    expected = sorted(calendar.loc[calendar.is_trading_day.eq('1') &
        calendar.calendar_date.between(p['history_first'], p['history_last']), 'calendar_date'])
    for code in p['symbols']:
        assert daily.loc[daily.code.eq(code), 'date'].tolist() == expected
    daily.to_parquet(ROOT / 'indices.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), source_feature_sha256=sha(OLD / 'features.parquet'),
        probe_verification_sha256=sha(PILOT / 'verification.json'), daily_raw_sha256=raw_paths,
        indices_sha256=sha(ROOT / 'indices.parquet'), daily_blocks=blocks, sessions=[], expected_sessions=2 * len(days),
        stock_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    sock = None
    try:
        for code in p['symbols']:
            for day in days:
                checkpoint = ROOT / 'sessions' / f'{code}_{day}.json'
                if checkpoint.exists():
                    item = json.loads(checkpoint.read_text())
                    assert item['symbol'] == code and item['date'] == day
                elif (code, day) in pilot:
                    old = pilot[code, day]
                    attempt = next(a for a in old['attempts'] if a['status'] == 'nonempty')
                    folder = Path(attempt['folder'])
                    item = dict(symbol=code, date=day, path=attempt['path'], sha256=attempt['sha256'],
                        request_path=str(folder / 'minute.request.bin'), response_path=str(folder / 'minute.response.bin'),
                        reused_probe=True, attempts=old['attempts'], rows=attempt['rows'])
                    save_json(checkpoint, item)
                else:
                    item = dict(symbol=code, date=day, reused_probe=False, attempts=[])
                    for attempt in range(cfg['attempts']):
                        folder = ROOT / 'wire' / f'{code}_{day}' / str(attempt)
                        entry = dict(attempt=attempt, folder=str(folder))
                        try:
                            if sock is None:
                                sock = connect(cfg['host'], cfg, folder)
                            values = request(sock, code, day, folder / 'history')
                            path = folder / 'decoded.json'; save_json(path, values)
                            entry.update(status='nonempty' if values else 'empty', rows=len(values))
                            item.update(path=str(path), sha256=sha(path), rows=len(values),
                                request_path=str(folder / 'history.request.bin'), response_path=str(folder / 'history.response.bin'))
                        except Exception as exc:
                            entry.update(status='error', error_type=type(exc).__name__, error=str(exc))
                            if sock is not None:
                                sock.close(); sock = None
                        entry['wire_sha256'] = {str(f): sha(f) for f in sorted(folder.glob('*.bin'))}
                        item['attempts'].append(entry)
                        if entry['status'] == 'nonempty':
                            break
                    save_json(checkpoint, item)
                if 'path' in item:
                    assert sha(Path(item['path'])) == item['sha256']
                report['sessions'].append(item)
                if len(report['sessions']) % 100 == 0:
                    print(json.dumps(dict(completed=len(report['sessions']), expected=report['expected_sessions'])), flush=True)
        assert len(report['sessions']) == report['expected_sessions']
        save_json(ROOT / 'source_report.json', report)
    finally:
        if sock is not None:
            sock.close()
    return dict(sessions=len(report['sessions']), source_report_sha256=sha(ROOT / 'source_report.json'),
        reused_probes=sum(s['reused_probe'] for s in report['sessions']), new_2026_prices_read=False)


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
