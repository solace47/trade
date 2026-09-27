"""Collect frozen prior-day index breadth without decoding quote prices."""
from datetime import date
import json
from pathlib import Path
import struct
import zlib

import pandas as pd

from probe_historical_ticks import connect, exchange, save_json
from trade_research.corporate_cash import sha

ROOT = Path('data/research/tail_formula_prior_breadth')
PROTOCOL = Path('config/tail_formula_prior_breadth_protocol.json')
CALENDAR = Path('data/research/tail_formula_prior_breadth_calendar/calendar.json')
DAILY = Path('data/research/tail_formula_prior_breadth_probe/report.json')
INTRADAY = Path('data/research/tail_formula_intraday_breadth_probe/report.json')


def decode(body, cfg):
    count = struct.unpack_from('<H', body)[0]; assert count <= cfg['count']
    if not count:
        assert len(body) in [2, 6]
        return []
    pos = 2; rows = []
    for _ in range(count):
        stamp = struct.unpack_from('<I', body, pos)[0]; pos += 4
        day = date(stamp // 10000, stamp // 100 % 100, stamp % 100).isoformat()
        assert cfg['history_first'] <= day <= cfg['history_last'], 'Date outside frozen scope'
        for _ in range(4):
            for length in range(9):
                value = body[pos]; pos += 1
                if value < 128:
                    break
            else:
                raise ValueError('Unbounded price encoding')
        pos += 8
        up, down = struct.unpack_from('<HH', body, pos); pos += 4
        rows.append(dict(date=day, up_count=up, down_count=down))
    assert pos == len(body)
    return rows


def replay(path, cfg):
    raw = path.read_bytes(); compressed, expanded = struct.unpack_from('<HH', raw, 12)
    assert len(raw) == 16 + compressed
    data = raw[16:] if compressed == expanded else zlib.decompress(raw[16:])
    assert len(data) == expanded
    count = int.from_bytes(data[:2], 'little'); assert count <= cfg['count']
    if not count:
        assert len(data) in [2, 6]
        return []
    pointer = 2; records = []
    for _ in range(count):
        stamp = str(int.from_bytes(data[pointer:pointer + 4], 'little')); pointer += 4
        assert len(stamp) == 8
        day = date.fromisoformat(stamp[:4] + '-' + stamp[4:6] + '-' + stamp[6:]).isoformat()
        assert cfg['history_first'] <= day <= cfg['history_last']
        for _ in range(4):
            terminal = next(j for j in range(pointer, min(pointer + 9, len(data))) if data[j] < 128)
            pointer = terminal + 1
        pointer += 8
        up = int.from_bytes(data[pointer:pointer + 2], 'little')
        down = int.from_bytes(data[pointer + 2:pointer + 4], 'little'); pointer += 4
        records.append(dict(date=day, up_count=up, down_count=down))
    assert pointer == len(data)
    return records


def main():
    assert not (ROOT / 'source_report.json').exists(), 'Do not replace frozen source'
    cfg = json.loads(PROTOCOL.read_text())
    for key, path in [('calendar_source_sha256', CALENDAR), ('daily_probe_report_sha256', DAILY),
            ('intraday_probe_report_sha256', INTRADAY)]:
        assert cfg[key] == sha(path)
    days = [r['calendar_date'] for r in json.loads(CALENDAR.read_text())['records'] if r['is_trading_day'] == '1']
    expected = [d for d in days if cfg['history_first'] <= d <= cfg['history_last']]
    assert len(expected) == cfg['count']
    assert days.index(cfg['inferred_server_anchor']) - days.index(cfg['history_last']) == cfg['start']
    result = dict(protocol_sha256=sha(PROTOCOL), sessions=[], prices_decoded=False,
        stock_outcomes_read=False, new_2026_prices_read=False, client_parity_verified=False)
    values = []
    for symbol in cfg['symbols']:
        item = dict(symbol=symbol, attempts=[])
        for attempt_id in range(cfg['attempts']):
            folder = ROOT / 'wire' / symbol / str(attempt_id); attempt = dict(attempt=attempt_id)
            packet = struct.pack('<HIHHHH6sHHHHIIH', 0x10c, 0x01016408, 0x1c, 0x1c, 0x052d,
                int(symbol.startswith('sh.')), symbol[3:].encode(), cfg['category'], 1,
                cfg['start'], cfg['count'], 0, 0, 0)
            try:
                with connect(cfg['host'], cfg, folder) as sock:
                    body = exchange(sock, packet, folder / 'breadth')
                records = decode(body, cfg)
                assert records == replay(folder / 'breadth.response.bin', cfg)
                assert (folder / 'breadth.request.bin').read_bytes() == packet
                assert [r['date'] for r in records] == expected
                assert all(r['up_count'] + r['down_count'] > 0 for r in records)
                attempt.update(status='passed', records=records, independent_replay_passed=True,
                    request_path=str(folder / 'breadth.request.bin'), response_path=str(folder / 'breadth.response.bin'))
            except Exception as exc:
                attempt.update(status='error', error=str(exc), error_type=type(exc).__name__)
            attempt['wire_sha256'] = {str(p): sha(p) for p in sorted(folder.glob('*.bin'))}
            item['attempts'].append(attempt)
            if attempt['status'] == 'passed':
                values.extend(dict(code=symbol, **row) for row in records)
                break
        result['sessions'].append(item); save_json(ROOT / 'source_report.json', result)
        print(json.dumps(dict(symbol=symbol, status=attempt['status'], error=attempt.get('error')), ensure_ascii=False), flush=True)
    actual = {(r['code'], r['date']): (r['up_count'], r['down_count']) for r in values}
    result['overlap_checks'] = []
    for path, tag in [(DAILY, 'daily_probe'), (INTRADAY, 'five_minute_close')]:
        for session in json.loads(path.read_text())['sessions']:
            records = next((a['records'] for a in session['attempts'] if a['status'] == 'nonempty'), [])
            if tag == 'five_minute_close':
                records = [r for r in records if r['minute'] == 900]
            for r in records:
                match = actual.get((session['symbol'], r['date'])) == (r['up_count'], r['down_count'])
                result['overlap_checks'].append(dict(symbol=session['symbol'], date=r['date'], source=tag, matched=match))
    result['passed'] = (len(values) == 2 * len(expected) and len(result['overlap_checks']) == 14
        and all(r['matched'] for r in result['overlap_checks']))
    if result['passed']:
        frame = pd.DataFrame(values).sort_values(['code', 'date']).reset_index(drop=True)
        frame.to_parquet(ROOT / 'counts.parquet', index=False, compression='zstd')
        result.update(counts_sha256=sha(ROOT / 'counts.parquet'), rows=len(frame))
    result['completed'] = True; save_json(ROOT / 'source_report.json', result)
    return {k: v for k, v in result.items() if k != 'sessions'}


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
