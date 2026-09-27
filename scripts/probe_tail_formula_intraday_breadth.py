"""Bounded historical intraday breadth probe; price fields are skipped."""
from datetime import date, timedelta
import json
from pathlib import Path
import struct
import zlib

from probe_historical_ticks import connect, exchange, save_json
from trade_research.corporate_cash import sha

ROOT = Path('data/research/tail_formula_intraday_breadth_probe')
PROTOCOL = Path('config/tail_formula_intraday_breadth_probe_protocol.json')


def decode_counts(body, maximum):
    count = struct.unpack_from('<H', body)[0]; assert count <= maximum
    if count == 0:
        assert len(body) in [2, 6]
        return []
    cursor = 2; records = []
    for _ in range(count):
        packed, minute = struct.unpack_from('<HH', body, cursor); cursor += 4
        day = date((packed >> 11) + 2004, (packed % 2048) // 100, (packed % 2048) % 100)
        assert date(2024, 1, 1) <= day <= date(2025, 12, 31), 'Outside frozen historical date scope'
        assert 0 <= minute < 1440
        for _ in range(4):
            length = 0
            while True:
                byte = body[cursor]; cursor += 1; length += 1
                assert length <= 9
                if not byte & 128:
                    break
        cursor += 8  # Volume and amount are not interpreted.
        up, down = struct.unpack_from('<HH', body, cursor); cursor += 4
        records.append(dict(date=day.isoformat(), minute=minute, up_count=up, down_count=down))
    assert cursor == len(body)
    return records


def independent_replay(path, maximum):
    raw = path.read_bytes(); compressed, expanded = struct.unpack_from('<HH', raw, 12)
    assert len(raw) == 16 + compressed
    body = raw[16:] if compressed == expanded else zlib.decompress(raw[16:])
    assert len(body) == expanded
    count = int.from_bytes(body[:2], 'little'); assert count <= maximum
    if not count:
        assert len(body) in [2, 6]
        return []
    pos = 2; records = []
    for _ in range(count):
        packed = int.from_bytes(body[pos:pos + 2], 'little')
        year, rem = divmod(packed, 2048); month, day = divmod(rem, 100)
        stamp = date(2004 + year, month, day).isoformat()
        assert '2024-01-01' <= stamp <= '2025-12-31'
        minute = int.from_bytes(body[pos + 2:pos + 4], 'little'); pos += 4
        assert minute < 1440
        for _ in range(4):
            terminal = next(j for j in range(pos, min(pos + 9, len(body))) if body[j] < 128)
            pos = terminal + 1
        pos += 8
        up = int.from_bytes(body[pos:pos + 2], 'little')
        down = int.from_bytes(body[pos + 2:pos + 4], 'little'); pos += 4
        records.append(dict(date=stamp, minute=minute, up_count=up, down_count=down))
    assert pos == len(body)
    return records


def main():
    assert not (ROOT / 'report.json').exists(), 'Do not overwrite a frozen probe'
    cfg = json.loads(PROTOCOL.read_text()); asof = date.fromisoformat(cfg['asof_date'])
    assert asof.year == 2026
    weekdays = sum((date(2026, 1, 1) + timedelta(days=i)).weekday() < 5
        for i in range((asof - date(2026, 1, 1)).days + 1))
    for query in cfg['queries']:
        assert query['start'] > weekdays * query['maximum_bars_per_weekday']
        assert query['category'] in [0, 8] and query['start'] <= 65535 and query['count'] <= 800
    result = dict(protocol_sha256=sha(PROTOCOL), sessions=[], comparisons=[], weekday_upper_bound=weekdays,
        prices_decoded=False, stock_outcomes_read=False, new_2026_prices_read=False,
        native_counts_scope_verified=False, client_parity_verified=False, no_exit_rules=True)
    for symbol in cfg['symbols']:
        for query in cfg['queries']:
            item = dict(symbol=symbol, query=query['id'], attempts=[])
            for attempt_id in range(cfg['attempts']):
                folder = ROOT / 'wire' / symbol / query['id'] / str(attempt_id)
                attempt = dict(attempt=attempt_id)
                packet = struct.pack('<HIHHHH6sHHHHIIH', 0x10c, 0x01016408, 0x1c, 0x1c, 0x052d,
                    int(symbol.startswith('sh.')), symbol[3:].encode(), query['category'], 1,
                    query['start'], query['count'], 0, 0, 0)
                try:
                    with connect(cfg['host'], cfg, folder) as sock:
                        body = exchange(sock, packet, folder / 'breadth')
                    records = decode_counts(body, query['count'])
                    assert records == independent_replay(folder / 'breadth.response.bin', query['count'])
                    assert (folder / 'breadth.request.bin').read_bytes() == packet
                    assert len({(r['date'], r['minute']) for r in records}) == len(records)
                    attempt.update(status='nonempty' if records else 'empty', records=records,
                        independent_replay_passed=True)
                except Exception as exc:
                    attempt.update(status='error', error=str(exc), error_type=type(exc).__name__)
                attempt['wire_sha256'] = {str(p): sha(p) for p in sorted(folder.glob('*.bin'))}
                item['attempts'].append(attempt)
                if attempt['status'] == 'nonempty':
                    break
            result['sessions'].append(item); save_json(ROOT / 'report.json', result)
            print(json.dumps(dict(symbol=symbol, query=query['id'], attempts=[
                {k: v for k, v in a.items() if k not in ['records', 'wire_sha256']} |
                dict(rows=len(a.get('records', [])), dates=sorted({r['date'] for r in a.get('records', [])}))
                for a in item['attempts']]), ensure_ascii=False), flush=True)
    valid_sessions = 0
    for symbol in cfg['symbols']:
        groups = {}
        for session in result['sessions']:
            if session['symbol'] != symbol:
                continue
            records = next((a['records'] for a in session['attempts'] if a['status'] == 'nonempty'), [])
            groups[session['query']] = {(r['date'], r['minute']): r for r in records}
            required_year = '2024' if session['query'] == 'older5' else '2025'
            valid_sessions += bool(records and all(r['up_count'] + r['down_count'] > 0 for r in records)
                and any(r['date'].startswith(required_year) for r in records))
        common = sorted(groups['paired5'].keys() & groups['paired1'].keys())
        different = [dict(date=key[0], minute=key[1], five=groups['paired5'][key], one=groups['paired1'][key])
            for key in common if groups['paired5'][key] != groups['paired1'][key]]
        targets = [key for key in common if key[1] in [14 * 60 + 20, 14 * 60 + 45]]
        result['comparisons'].append(dict(symbol=symbol, common_timestamps=len(common), target_timestamps=len(targets),
            different_counts=different, passed=len(common) >= 100 and len(targets) >= 4 and not different))
    result.update(completed=True, valid_sessions=valid_sessions,
        passed=valid_sessions == 6 and all(c['passed'] for c in result['comparisons']))
    save_json(ROOT / 'report.json', result)
    return {k: v for k, v in result.items() if k != 'sessions'}


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
