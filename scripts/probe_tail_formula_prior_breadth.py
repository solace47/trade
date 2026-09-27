"""Read only historical dates and breadth counts, never decode quote prices."""
from datetime import date
import json
from pathlib import Path
import struct
import zlib

from probe_historical_ticks import connect, exchange, save_json
from trade_research.corporate_cash import sha

ROOT = Path('data/research/tail_formula_prior_breadth_probe')
PROTOCOL = Path('config/tail_formula_prior_breadth_probe_protocol.json')


def decode_counts(body, maximum):
    count = struct.unpack_from('<H', body)[0]; assert count <= maximum
    if count == 0:
        assert len(body) in [2, 6]
        return []
    cursor = 2; records = []
    for _ in range(count):
        stamp = struct.unpack_from('<I', body, cursor)[0]; cursor += 4
        # Reject dates before skipping even the first price field.
        assert 20240101 <= stamp <= 20251231, 'Outside frozen historical date scope'
        day = date(stamp // 10000, stamp // 100 % 100, stamp % 100).isoformat()
        for _ in range(4):
            length = 0
            while True:
                byte = body[cursor]; cursor += 1; length += 1
                assert length <= 9
                if not byte & 128:
                    break
        cursor += 8  # Skip volume and amount, without interpreting either.
        up, down = struct.unpack_from('<HH', body, cursor); cursor += 4
        records.append(dict(date=day, up_count=up, down_count=down))
    assert cursor == len(body)
    return records


def independent_replay(response):
    raw = response.read_bytes(); compressed, expanded = struct.unpack_from('<HH', raw, 12)
    assert len(raw) == 16 + compressed
    data = raw[16:] if compressed == expanded else zlib.decompress(raw[16:])
    assert len(data) == expanded
    count = int.from_bytes(data[:2], 'little'); assert count <= 3
    if not count:
        assert len(data) in [2, 6]
        return []
    pointer = 2; records = []
    for _ in range(count):
        day = str(int.from_bytes(data[pointer:pointer + 4], 'little')); pointer += 4
        assert len(day) == 8 and '20240101' <= day <= '20251231'
        day = date.fromisoformat(day[:4] + '-' + day[4:6] + '-' + day[6:]).isoformat()
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
    assert not (ROOT / 'report.json').exists(), 'Do not replace frozen breadth probe'
    cfg = json.loads(PROTOCOL.read_text())
    assert cfg['category'] == 9 and cfg['start'] == 400 and cfg['count'] == 3
    assert cfg['start'] > (date.fromisoformat(cfg['asof_date']) - date(2026, 1, 1)).days + 1
    result = dict(protocol_sha256=sha(PROTOCOL), sessions=[], prices_decoded=False,
        native_counts_scope_verified=False, stock_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    for symbol in cfg['symbols']:
        item = dict(symbol=symbol, attempts=[])
        for index in range(cfg['attempts']):
            folder = ROOT / 'wire' / symbol / str(index); attempt = dict(attempt=index)
            packet = struct.pack('<HIHHHH6sHHHHIIH', 0x10c, 0x01016408, 0x1c, 0x1c, 0x052d,
                int(symbol.startswith('sh.')), symbol[3:].encode(), cfg['category'], 1, cfg['start'], cfg['count'], 0, 0, 0)
            try:
                with connect(cfg['host'], cfg, folder) as sock:
                    body = exchange(sock, packet, folder / 'breadth')
                records = decode_counts(body, cfg['count'])
                assert records == independent_replay(folder / 'breadth.response.bin')
                request = (folder / 'breadth.request.bin').read_bytes()
                assert request == packet and len({r['date'] for r in records}) == len(records)
                attempt.update(status='nonempty' if records else 'empty', records=records, independent_replay_passed=True)
            except Exception as exc:
                attempt.update(status='error', error=str(exc), error_type=type(exc).__name__)
            attempt['wire_sha256'] = {str(p): sha(p) for p in sorted(folder.glob('*.bin'))}
            item['attempts'].append(attempt)
            if attempt['status'] == 'nonempty':
                break
        result['sessions'].append(item); save_json(ROOT / 'report.json', result)
        print(json.dumps(item, ensure_ascii=False), flush=True)
    result['nonempty_sessions'] = sum(any(a['status'] == 'nonempty' for a in s['attempts']) for s in result['sessions'])
    result['completed'] = True; save_json(ROOT / 'report.json', result)
    return result


if __name__ == '__main__':
    main()
