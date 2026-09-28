"""Six frozen brokerage/bank index sessions; independent SZ packet verification.

Adapted from the prior size-index probe without changing its frozen source."""
import argparse
import json
from pathlib import Path
import socket
import struct

import baostock as bs
import numpy as np

from probe_historical_ticks import connect, save_json
from probe_index_minutes import request
from verify_index_minute_probe import replay
from trade_research.corporate_cash import sha
from trade_research.ingest import _login, _rows

ROOT = Path('data/research/tail_formula_financial_index_probe')
PROTOCOL = Path('config/tail_formula_financial_index_probe_protocol.json')


def collect():
    assert not (ROOT / 'source_report.json').exists(), 'Do not replace the frozen source probe'
    cfg = json.loads(PROTOCOL.read_text())
    report = dict(protocol_sha256=sha(PROTOCOL), sessions=[], completed=False,
        stock_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    socket.setdefaulttimeout(20)
    fields = 'date,code,open,high,low,close'
    _login()
    try:
        for code in cfg['symbols']:
            for day in cfg['dates']:
                assert day[:4] in ['2024', '2025']
                item = dict(symbol=code, date=day, attempts=[])
                try:
                    raw = _rows(bs.query_history_k_data_plus(code, fields,
                        start_date=day, end_date=day, frequency='d', adjustflag='3'))
                    path = ROOT / 'daily' / f'{code}_{day}.json'
                    save_json(path, dict(symbol=code, date=day, fields=fields.split(','), records=raw.to_dict('records')))
                    item.update(daily_path=str(path), daily_sha256=sha(path), daily_rows=len(raw))
                except Exception as exc:
                    item.update(daily_error=str(exc), daily_error_type=type(exc).__name__)
                for attempt in range(cfg['attempts']):
                    folder = ROOT / 'wire' / f'{code}_{day}' / str(attempt)
                    entry = dict(attempt=attempt, folder=str(folder))
                    try:
                        with connect(cfg['host'], cfg, folder) as sock:
                            values = request(sock, code, day, folder / 'minute')
                        path = folder / 'decoded.json'; save_json(path, values)
                        entry.update(path=str(path), sha256=sha(path), rows=len(values), status='nonempty' if values else 'empty')
                    except Exception as exc:
                        entry.update(status='error', error=str(exc), error_type=type(exc).__name__)
                    entry['wire_sha256'] = {str(p): sha(p) for p in sorted(folder.glob('*.bin'))}
                    item['attempts'].append(entry)
                    if entry['status'] == 'nonempty':
                        break
                report['sessions'].append(item); save_json(ROOT / 'source_report.json', report)
                print(json.dumps(dict(code=code, date=day, daily_rows=item.get('daily_rows'),
                    minute_status=item['attempts'][-1]['status'], minute_rows=item['attempts'][-1].get('rows')), ensure_ascii=False), flush=True)
    finally:
        bs.logout()
    report['completed'] = True; save_json(ROOT / 'source_report.json', report)
    return dict(completed=True, sessions=len(report['sessions']), stock_outcomes_read=False, new_2026_prices_read=False)


def verify():
    cfg = json.loads(PROTOCOL.read_text()); r = json.loads((ROOT / 'source_report.json').read_text())
    assert r['completed'] and r['protocol_sha256'] == sha(PROTOCOL)
    old = Path('data/research/index_minute_probe')
    assert cfg['original_probe_sha256'] == sha(old / 'source_report.json')
    assert cfg['original_probe_verification_sha256'] == sha(old / 'verification.json')
    proof = json.loads((old / 'verification.json').read_text())
    assert proof['passed'] and proof['source_report_sha256'] == sha(old / 'source_report.json')
    for name, digest in proof['wire_sha256'].items():
        assert sha(Path(name)) == digest
    controls = [d for d in proof['details'] if d['code'] == 'sh.600000']
    assert sorted(d['date'] for d in controls) == sorted(cfg['dates'])
    assert all(d['all_within_same_minute_envelope'] for d in controls)
    assert {(s['symbol'], s['date']) for s in r['sessions']} == {(c, d) for c in cfg['symbols'] for d in cfg['dates']}
    results = []
    for s in r['sessions']:
        usable = False; checks = dict(symbol=s['symbol'], date=s['date'])
        if 'daily_path' in s:
            path = Path(s['daily_path']); assert sha(path) == s['daily_sha256']
            daily = json.loads(path.read_text())
            assert daily['symbol'] == s['symbol'] and daily['date'] == s['date']
            assert len(daily['records']) == s['daily_rows']
            for d in daily['records']:
                assert d['code'] == s['symbol'] and d['date'] == s['date']
        else:
            daily = dict(records=[])
        assert 1 <= len(s['attempts']) <= cfg['attempts']
        for index, attempt in enumerate(s['attempts']):
            assert attempt['attempt'] == index
            for name, digest in attempt['wire_sha256'].items():
                assert sha(Path(name)) == digest
            folder = Path(attempt['folder']); req = folder / 'minute.request.bin'
            if req.exists():
                packet = req.read_bytes()
                assert packet[:12] == bytes.fromhex('0c01300001010d000d00b40f')
                day, market, code = struct.unpack('<IB6s', packet[12:])
                assert (day, market, code.decode()) == (int(s['date'].replace('-', '')), int(s['symbol'].startswith('sh.')), s['symbol'][3:])
            if attempt['status'] != 'error':
                path = Path(attempt['path']); assert sha(path) == attempt['sha256']
                rows = replay((folder / 'minute.response.bin').read_bytes())
                assert rows == json.loads(path.read_text()) and len(rows) == attempt['rows']
                assert len(rows) == 0 or len(rows) == 240
                if rows and len(daily['records']) == 1:
                    d = daily['records'][0]; values = np.array([x['price_raw'] for x in rows]) / 100
                    assert (values > 0).all()
                    inside = bool((values >= float(d['low']) - .0051).all() and (values <= float(d['high']) + .0051).all())
                    difference = float(values[-1] - float(d['close']))
                    usable = inside and abs(difference) <= .0051
                    checks.update(points=len(rows), all_points_inside_daily_envelope=inside, close_difference=difference)
        checks['source_usable'] = usable; results.append(checks)
    result = dict(passed=True, source_report_sha256=sha(ROOT / 'source_report.json'),
        all_requests_responses_and_source_dates_rebuilt=True, reused_stock_clock_control_sha256=cfg['original_probe_verification_sha256'],
        sessions=results, source_usable=all(s['source_usable'] for s in results),
        native_cross_symbol_price_parity_verified=False, conservative_research_cutoff='14:48',
        stock_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'verification.json', result); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['collect', 'verify'])
    a = p.parse_args(); print(json.dumps(globals()[a.stage](), ensure_ascii=False, indent=2))
