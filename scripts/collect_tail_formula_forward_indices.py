"""Collect only the frozen forward study's index history, preserving wire evidence."""
import argparse
import json
import socket
import struct
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd

from probe_historical_ticks import connect, exchange
from probe_index_minutes import decode
from verify_index_minute_probe import replay
from trade_research.corporate_cash import save_json, sha
from trade_research.ingest import _login, _rows
from trade_research.tail_formula_forward import ROOT, PROTOCOL, checked_model, policy
from trade_research.turnover_reference import CALENDAR

OUT = ROOT / 'indices'
CODES = ['sh.000001', 'sz.399001']


def calendar(first, last):
    f = pd.read_parquet(CALENDAR)
    return sorted(f.loc[f.is_trading_day.eq('1') & f.calendar_date.between(first, last), 'calendar_date'])


def daily():
    checked_model(); p = policy()
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'daily_report.json').exists():
        raise ValueError('Do not replace frozen forward index days')
    socket.setdefaulttimeout(20)
    frames = []; raw_files = {}
    _login()
    try:
        for code in CODES:
            path = OUT / f'{code}_daily.json'
            if not path.exists():
                fields = 'date,code,open,high,low,close'
                data = _rows(bs.query_history_k_data_plus(code, fields, start_date=p['warmup_first'],
                    end_date=p['signal_last'], frequency='d', adjustflag='3'))
                save_json(path, dict(code=code, first=p['warmup_first'], last=p['signal_last'], fields=fields,
                    records=data.to_dict('records')))
            r = json.loads(path.read_text())
            assert r['first'] == p['warmup_first'] and r['last'] == p['signal_last'] and r['code'] == code
            f = pd.DataFrame(r['records'])
            for name in ['open', 'high', 'low', 'close']:
                f[name] = pd.to_numeric(f[name], errors='raise')
            assert f.code.eq(code).all() and f.date.tolist() == calendar(p['warmup_first'], p['signal_last'])
            prices = f[['open', 'high', 'low', 'close']]
            assert np.isfinite(prices).all().all() and prices.gt(0).all().all()
            assert f.high.add(.0001).ge(prices.max(axis=1)).all() and f.low.sub(.0001).le(prices.min(axis=1)).all()
            frames.append(f); raw_files[str(path)] = sha(path)
    finally:
        bs.logout()
    f = pd.concat(frames).sort_values(['code', 'date']).reset_index(drop=True)
    f.to_parquet(OUT / 'daily.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), model_freeze_report_sha256=sha(ROOT / 'model_freeze_report.json'),
        raw_files_sha256=raw_files, daily_sha256=sha(OUT / 'daily.parquet'), rows=len(f),
        first=p['warmup_first'], last=p['signal_last'], new_2026_prices_read=True,
        no_2026_after_q1_signal_prices_read=True, next_morning_stock_outcomes_read=False)
    save_json(OUT / 'daily_report.json', r)
    return r


def minutes():
    checked_model(); p = policy()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'raw').mkdir(exist_ok=True)
    (OUT / 'attempts').mkdir(exist_ok=True)
    if (OUT / 'minute_report.json').exists():
        raise ValueError('Do not replace frozen forward index minutes')
    probe = Path('data/research/index_minute_probe/verification.json')
    proof = json.loads(probe.read_text())
    assert proof['passed'] and proof['historical_data_nonempty']
    cfg = json.loads(Path('config/index_minute_probe_protocol.json').read_text())
    days = calendar(p['signal_first'], p['signal_last'])
    sessions = []; sock = None
    try:
        for symbol in CODES:
            for date in days:
                assert p['signal_first'] <= date <= p['signal_last']
                path = OUT / 'raw' / f'{symbol}_{date}.json'
                attempts = []
                # A local output error must not discard an already received wire
                # response or cause the source data to be requested again.
                if not path.exists():
                    for response in sorted((OUT / 'wire' / f'{symbol}_{date}').glob('*/history.response.bin')):
                        try:
                            rows = replay(response.read_bytes())
                            save_json(path, rows)
                            attempts.append(dict(status='recovered_archived_response', response=str(response), rows=len(rows)))
                            break
                        except (ValueError, AssertionError, IndexError, struct.error):
                            continue
                    if attempts:
                        save_json(OUT / 'attempts' / f'{symbol}_{date}.json', attempts)
                if not path.exists():
                    for attempt in range(2):
                        folder = OUT / 'wire' / f'{symbol}_{date}' / str(attempt)
                        try:
                            if sock is None:
                                sock = connect(cfg['host'], cfg, folder)
                            packet = bytes.fromhex('0c01300001010d000d00b40f') + struct.pack('<IB6s',
                                int(date.replace('-', '')), int(symbol.startswith('sh.')), symbol[3:].encode())
                            rows = decode(exchange(sock, packet, folder / 'history'))
                            assert rows == replay((folder / 'history.response.bin').read_bytes())
                            save_json(path, rows)
                            attempts.append(dict(attempt=attempt, status='nonempty' if rows else 'empty', rows=len(rows)))
                            break
                        except Exception as exc:
                            attempts.append(dict(attempt=attempt, status='error', error_type=type(exc).__name__, error=str(exc)))
                            if sock is not None:
                                sock.close(); sock = None
                    save_json(OUT / 'attempts' / f'{symbol}_{date}.json', attempts)
                item = dict(symbol=symbol, date=date, status='cached' if path.exists() else 'unavailable')
                if path.exists():
                    rows = json.loads(path.read_text())
                    matched = []
                    for response in sorted((OUT / 'wire' / f'{symbol}_{date}').glob('*/history.response.bin')):
                        try:
                            if replay(response.read_bytes()) == rows:
                                request = response.with_name('history.request.bin')
                                raw = request.read_bytes()
                                assert raw[:12] == bytes.fromhex('0c01300001010d000d00b40f')
                                assert struct.unpack('<IB6s', raw[12:]) == (int(date.replace('-', '')), int(symbol.startswith('sh.')), symbol[3:].encode())
                                matched.append(dict(response=str(response), response_sha256=sha(response), request=str(request), request_sha256=sha(request)))
                        except (ValueError, AssertionError, IndexError, struct.error):
                            continue
                    assert matched
                    item.update(path=str(path), sha256=sha(path), rows=len(rows), wire=matched)
                sessions.append(item)
                if len(sessions) % 20 == 0:
                    print(json.dumps(dict(completed=len(sessions), expected=2*len(days))), flush=True)
    finally:
        if sock is not None:
            sock.close()
    r = dict(protocol_sha256=sha(PROTOCOL), model_freeze_report_sha256=sha(ROOT / 'model_freeze_report.json'),
        probe_verification_sha256=sha(probe), sessions=sessions, all_nonmissing_responses_independently_replayed=True,
        new_2026_prices_read=True, no_2026_after_q1_signal_prices_read=True, next_morning_stock_outcomes_read=False)
    save_json(OUT / 'minute_report.json', r)
    return {k:v for k,v in r.items() if k != 'sessions'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['daily', 'minutes'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
