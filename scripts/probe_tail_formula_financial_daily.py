"""Six exact-date daily controls for the frozen brokerage/bank minute probe."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlencode, parse_qs, urlsplit

import numpy as np
import requests

from probe_historical_ticks import save_json
from verify_index_minute_probe import replay
from trade_research.corporate_cash import sha

ROOT = Path('data/research/tail_formula_financial_daily_probe')
PILOT = Path('data/research/tail_formula_financial_index_probe')
PROTOCOL = Path('config/tail_formula_financial_daily_probe_protocol.json')


def config():
    p = json.loads(PROTOCOL.read_text())
    for key, file in [('probe_protocol_sha256', Path('config/tail_formula_financial_index_probe_protocol.json')),
                      ('probe_source_sha256', PILOT / 'source_report.json'),
                      ('probe_verification_sha256', PILOT / 'verification.json'),
                      ('reference_code_sha256', PILOT / 'source/index_stock_zh.py')]:
        assert p[key] == sha(file)
    proof = json.loads((PILOT / 'verification.json').read_text())
    assert proof['passed'] and not proof['source_usable']
    return p


def collect():
    assert not (ROOT / 'source_report.json').exists(), 'Keep the original source probe'
    p = config()
    report = dict(protocol_sha256=sha(PROTOCOL), completed=False, sessions=[],
                  stock_outcomes_read=False, new_2026_prices_read=False)
    for code in p['symbols']:
        for day in p['dates']:
            assert code.startswith('sz.') and day[:4] in ['2024', '2025']
            params = {k: p[k] for k in ['fields1', 'fields2', 'klt', 'fqt']}
            params.update(secid='0.' + code[3:], beg=day.replace('-', ''), end=day.replace('-', ''))
            url = p['endpoint'] + '?' + urlencode(params)
            item = dict(code=code, date=day, url=url, attempts=[])
            for attempt in range(p['attempts']):
                folder = ROOT / 'raw' / f'{code}_{day}' / str(attempt)
                folder.mkdir(parents=True, exist_ok=True)
                entry = dict(attempt=attempt)
                try:
                    response = requests.get(url, timeout=p['timeout_seconds'])
                    path = folder / 'response.bin'; path.write_bytes(response.content)
                    entry.update(path=str(path), sha256=sha(path), http_status=response.status_code)
                    response.raise_for_status()
                    data = response.json().get('data')
                    count = len((data or {}).get('klines', []))
                    entry.update(status='nonempty' if count else 'empty', rows=count)
                except Exception as exc:
                    entry.update(status='error', error_type=type(exc).__name__, error=str(exc))
                item['attempts'].append(entry)
                if entry['status'] == 'nonempty':
                    break
            report['sessions'].append(item)
            save_json(ROOT / 'source_report.json', report)
            print(json.dumps(dict(code=code, date=day, status=entry['status'], rows=entry.get('rows')), ensure_ascii=False), flush=True)
    report['completed'] = True; save_json(ROOT / 'source_report.json', report)
    return dict(completed=True, sessions=len(report['sessions']))


def verify():
    p = config(); r = json.loads((ROOT / 'source_report.json').read_text())
    assert r['completed'] and r['protocol_sha256'] == sha(PROTOCOL)
    grid = {(c, d) for c in p['symbols'] for d in p['dates']}
    assert len(r['sessions']) == len(grid) and {(s['code'], s['date']) for s in r['sessions']} == grid
    original = json.loads((PILOT / 'source_report.json').read_text())
    minute = {(s['symbol'], s['date']): s for s in original['sessions']}
    details = []
    for s in r['sessions']:
        url = urlsplit(s['url']); query = parse_qs(url.query)
        assert url.scheme + '://' + url.netloc + url.path == p['endpoint']
        expected = {k: [p[k]] for k in ['fields1', 'fields2', 'klt', 'fqt']}
        expected.update(secid=['0.' + s['code'][3:]], beg=[s['date'].replace('-', '')], end=[s['date'].replace('-', '')])
        assert query == expected
        assert 1 <= len(s['attempts']) <= p['attempts']
        result = dict(code=s['code'], date=s['date'], source_usable=False)
        for number, entry in enumerate(s['attempts']):
            assert entry['attempt'] == number
            if 'path' in entry:
                assert sha(Path(entry['path'])) == entry['sha256']
            if entry['status'] == 'error':
                continue
            raw = json.loads(Path(entry['path']).read_bytes())
            data = raw.get('data'); bars = (data or {}).get('klines', [])
            assert len(bars) == entry['rows']
            if not bars:
                continue
            assert data['code'] == s['code'][3:] and data['market'] == 0
            assert len(bars) == 1
            fields = bars[0].split(','); assert len(fields) == 5 and fields[0] == s['date']
            o, c, h, l = map(float, fields[1:])
            assert np.isfinite([o, c, h, l]).all() and 0 < l <= min(o, c) <= max(o, c) <= h
            source = minute[s['code'], s['date']]
            a = next(a for a in source['attempts'] if a['status'] == 'nonempty')
            for file, digest in a['wire_sha256'].items():
                assert sha(Path(file)) == digest
            rows = replay((Path(a['folder']) / 'minute.response.bin').read_bytes())
            assert rows == json.loads(Path(a['path']).read_text()) and len(rows) == 240
            prices = np.array([v['price_raw'] for v in rows]) / 100
            inside = bool((prices > 0).all() and (prices >= l - .0051).all() and (prices <= h + .0051).all())
            difference = float(prices[-1] - c)
            result.update(points=len(rows), inside_daily_envelope=inside, last_minus_daily_close=difference,
                          source_usable=inside and abs(difference) <= .0051)
        details.append(result)
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), source_report_sha256=sha(ROOT / 'source_report.json'),
                 original_minute_source_sha256=p['probe_source_sha256'], sessions=details,
                 source_usable=all(d['source_usable'] for d in details), native_source_price_parity_verified=False,
                 stock_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'verification.json', proof); return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['collect', 'verify'])
    a = p.parse_args(); print(json.dumps(globals()[a.stage](), ensure_ascii=False, indent=2))
