"""Bounded independent Tencent daily controls; never inspect quote side branches."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlencode, parse_qs, urlsplit

import numpy as np
import requests

from probe_historical_ticks import save_json
from trade_research.corporate_cash import sha

ROOT = Path('data/research/tail_formula_financial_tencent_probe')
PROTOCOL = Path('config/tail_formula_financial_tencent_probe_protocol.json')
REFERENCE = Path('data/research/tail_formula_financial_daily_probe')


def config():
    p = json.loads(PROTOCOL.read_text())
    for key, path in [
        ('primary_source_sha256', Path('data/research/tail_formula_financial_index_probe/source/index_stock_zh.py')),
        ('prior_source_incomplete_sha256', Path('data/research/tail_formula_financial_context/source_incomplete.json')),
        ('reference_source_sha256', REFERENCE / 'source_report.json'),
        ('reference_verification_sha256', REFERENCE / 'verification.json'),
    ]:
        assert p[key] == sha(path)
    assert json.loads((REFERENCE / 'verification.json').read_text())['source_usable']
    return p


def params(code, day):
    assert code in ['sz.399975', 'sz.399986'] and day[:4] in ['2024', '2025']
    return dict(_var='kline_dayqfq', param=f"{code.replace('.', '')},day,{day},{day},640,qfq", r='0.8205512681390605')


def day_rows(raw, code):
    text = raw.decode('utf-8')
    payload = json.loads(text[text.find('=')+1:])
    assert payload['code'] == 0
    # Do not expose or use any current quote/metadata branch supplied by the endpoint.
    return payload['data'][code.replace('.', '')].get('day', [])


def collect():
    p = config(); assert not (ROOT / 'source_report.json').exists()
    r = dict(protocol_sha256=sha(PROTOCOL), completed=False, sessions=[], new_stock_outcomes_read=False,
             current_quote_side_branches_not_used=True)
    for code in p['symbols']:
        for day in p['dates']:
            url = p['endpoint'] + '?' + urlencode(params(code, day))
            s = dict(code=code, date=day, url=url, attempts=[])
            for number in range(p['attempts']):
                folder = ROOT / 'raw' / f'{code}_{day}' / str(number); folder.mkdir(parents=True, exist_ok=True)
                a = dict(attempt=number)
                try:
                    response = requests.get(url, timeout=p['timeout_seconds'])
                    path = folder / 'response.bin'; path.write_bytes(response.content)
                    a.update(path=str(path), sha256=sha(path), http_status=response.status_code)
                    response.raise_for_status()
                    rows = day_rows(response.content, code)
                    a.update(status='nonempty' if rows else 'empty', rows=len(rows))
                except Exception as exc:
                    a.update(status='error', error_type=type(exc).__name__, error=str(exc))
                s['attempts'].append(a)
                if a['status'] == 'nonempty': break
            r['sessions'].append(s); save_json(ROOT / 'source_report.json', r)
            print(json.dumps(dict(code=code, date=day, status=a['status'], rows=a.get('rows'))), flush=True)
    r['completed'] = True; save_json(ROOT / 'source_report.json', r)
    return dict(completed=True, sessions=len(r['sessions']))


def verify():
    p = config(); r = json.loads((ROOT / 'source_report.json').read_text())
    assert r['completed'] and r['protocol_sha256'] == p.get('raw_source_report_protocol_sha256', sha(PROTOCOL))
    if 'fixed_raw_source_report_sha256' in p:
        assert p['fixed_raw_source_report_sha256'] == sha(ROOT / 'source_report.json')
        assert p['date_boundary_diagnosis_sha256'] == sha(ROOT / 'date_boundary_failure.json')
    assert len(r['sessions']) == 6
    assert {(s['code'], s['date']) for s in r['sessions']} == {(c, d) for c in p['symbols'] for d in p['dates']}
    reference = {(s['code'], s['date']): s for s in json.loads((REFERENCE / 'source_report.json').read_text())['sessions']}
    details = []
    for s in r['sessions']:
        url = urlsplit(s['url'])
        assert url.scheme + '://' + url.netloc + url.path == p['endpoint']
        assert parse_qs(url.query) == {k: [v] for k, v in params(s['code'], s['date']).items()}
        assert 1 <= len(s['attempts']) <= p['attempts']
        result = dict(code=s['code'], date=s['date'], source_usable=False)
        for i, a in enumerate(s['attempts']):
            assert a['attempt'] == i
            if 'path' in a: assert sha(Path(a['path'])) == a['sha256']
            if a['status'] == 'error': continue
            bars = day_rows(Path(a['path']).read_bytes(), s['code']); assert len(bars) == a['rows']
            if not bars: continue
            dates = [b[0] for b in bars]
            assert dates == sorted(set(dates)) and dates[-1] == s['date']
            assert dates.count(s['date']) == 1
            bars = [b for b in bars if b[0] == s['date']]
            assert len(bars) == 1 and bars[0][0] == s['date'] and len(bars[0]) >= 6
            o, c, h, l = map(float, bars[0][1:5])
            assert np.isfinite([o, c, h, l]).all() and 0 < l <= min(o, c) <= max(o, c) <= h
            old = next(v for v in reference[s['code'], s['date']]['attempts'] if v['status'] == 'nonempty')
            assert sha(Path(old['path'])) == old['sha256']
            rows = json.loads(Path(old['path']).read_bytes())['data']['klines']
            assert len(rows) == 1 and rows[0].split(',')[0] == s['date']
            expected = list(map(float, rows[0].split(',')[1:]))
            np.testing.assert_array_equal(np.rint(100*np.array([o, c, h, l])), np.rint(100*np.array(expected)))
            result.update(source_usable=True, four_prices_equal_at_cent_precision=True)
        details.append(result)
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), source_report_sha256=sha(ROOT / 'source_report.json'),
                 source_usable=all(d['source_usable'] for d in details), sessions=details,
                 reference_verification_sha256=p['reference_verification_sha256'],
                 new_stock_outcomes_read=False, current_quote_side_branches_not_used=True, no_exit_rules=True)
    save_json(ROOT / 'verification.json', proof); return proof


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['collect', 'verify'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
