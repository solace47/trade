"""Collect the frozen two-index historical grid, reusing the six source probes."""
import json
from pathlib import Path
import socket

import baostock as bs
import pandas as pd

from probe_historical_ticks import connect, save_json
from probe_index_minutes import request
from trade_research.corporate_cash import sha
from trade_research.ingest import _login, _rows
from trade_research.turnover_reference import CALENDAR

ROOT = Path('data/research/tail_formula_size_context')
PROTOCOL = Path('config/tail_formula_size_context_protocol.json')
PILOT = Path('data/research/tail_formula_size_index_probe')
OLD = Path('data/research/tail_formula_float')


def main():
    assert not (ROOT / 'source_report.json').exists(), 'Do not replace frozen source grid'
    p = json.loads(PROTOCOL.read_text())
    cfg = json.loads(Path('config/tail_formula_size_index_probe_protocol.json').read_text())
    assert p['probe_protocol_sha256'] == sha(Path('config/tail_formula_size_index_probe_protocol.json'))
    assert p['probe_source_report_sha256'] == sha(PILOT / 'source_report.json')
    assert p['probe_verification_sha256'] == sha(PILOT / 'verification.json')
    assert p['previous_feature_report_sha256'] == sha(OLD / 'feature_report.json')
    assert p['calendar_sha256'] == sha(CALENDAR)
    assert json.loads((PILOT / 'verification.json').read_text())['source_usable']
    assert json.loads((OLD / 'feature_report.json').read_text())['features_sha256'] == sha(OLD / 'features.parquet')
    days = sorted(pd.read_parquet(OLD / 'features.parquet', columns=['date']).date.unique())
    assert days[0] >= '2024-01-01' and days[-1] <= '2025-12-30'
    pilot = {(s['symbol'], s['date']): s for s in json.loads((PILOT / 'source_report.json').read_text())['sessions']}
    raw_paths = {}; frames = []; socket.setdefaulttimeout(20)
    fields = 'date,code,open,high,low,close'; _login()
    try:
        for code in p['symbols']:
            path = ROOT / 'daily' / (code + '.json')
            if not path.exists():
                data = _rows(bs.query_history_k_data_plus(code, fields,
                    start_date=p['history_first'], end_date=p['history_last'], frequency='d', adjustflag='3'))
                save_json(path, dict(symbol=code, start=p['history_first'], end=p['history_last'], records=data.to_dict('records')))
            raw = json.loads(path.read_text())
            assert raw['symbol'] == code and raw['start'] == p['history_first'] and raw['end'] == p['history_last']
            f = pd.DataFrame(raw['records']); assert len(f) and f.code.eq(code).all()
            assert f.date.between(p['history_first'], p['history_last']).all() and not f.date.duplicated().any()
            for name in ['open', 'high', 'low', 'close']:
                f[name] = pd.to_numeric(f[name], errors='raise')
            frames.append(f); raw_paths[str(path)] = sha(path)
    finally:
        bs.logout()
    daily = pd.concat(frames).sort_values(['code', 'date']).reset_index(drop=True)
    calendar = pd.read_parquet(CALENDAR)
    expected = sorted(calendar.loc[calendar.is_trading_day.eq('1') &
        calendar.calendar_date.between(p['history_first'], p['history_last']), 'calendar_date'])
    for code in p['symbols']:
        assert daily.loc[daily.code.eq(code), 'date'].tolist() == expected
    daily.to_parquet(ROOT / 'indices.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), source_feature_sha256=sha(OLD / 'features.parquet'),
        probe_verification_sha256=sha(PILOT / 'verification.json'), daily_raw_sha256=raw_paths,
        indices_sha256=sha(ROOT / 'indices.parquet'), sessions=[], expected_sessions=2 * len(days),
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
