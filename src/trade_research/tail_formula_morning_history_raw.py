"""Immutable raw-minute extraction for historical tail-to-morning input features."""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import tail_formula_additive as base
from .corporate_cash import MINUTES, save_json, sha

ROOT = Path('data/research/tail_formula_morning_history')
PROTOCOL = Path('config/tail_formula_morning_history_protocol.json')
SOURCE = Path('data/research/tail_formula_volume_history')
MINUTE_MANIFEST = Path('data/research/economic_winner/input_manifest.json')


def checked():
    p = json.loads(PROTOCOL.read_text())
    for key, path in [('schedule_manifest_sha256', SOURCE / 'input_manifest.json'),
                      ('schedule_sha256', SOURCE / 'stock_days.parquet'),
                      ('schedule_feature_verification_sha256', SOURCE / 'feature_verification.json'),
                      ('minute_source_manifest_sha256', MINUTE_MANIFEST)]:
        assert p[key] == sha(path)
    v = json.loads((SOURCE / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(SOURCE / 'feature_report.json')
    m = json.loads((SOURCE / 'input_manifest.json').read_text())
    assert m['stock_days_sha256'] == p['schedule_sha256']
    assert m['history_first'] == p['history_first'] == '2023-06-01'
    assert m['signal_last'] == p['signal_last'] == '2025-12-30'
    assert m['minute_source_manifest_sha256'] == p['minute_source_manifest_sha256']
    original = json.loads(MINUTE_MANIFEST.read_text())['source_sha256']
    assert all(original[path] == digest for path, digest in m['minute_source_sha256'].items())
    schedule = pd.read_parquet(SOURCE / 'stock_days.parquet')
    assert len(schedule) == m['stock_days'] and schedule.code.nunique() == m['codes']
    assert not schedule.duplicated(['date', 'code']).any()
    assert schedule.date.between(p['history_first'], p['signal_last']).all()
    return p, m, schedule


def extract():
    if (ROOT / 'raw_report.json').exists():
        raise ValueError('Do not replace frozen morning-history raw minutes')
    p, m, schedule = checked()
    ROOT.mkdir(parents=True, exist_ok=True); folder = ROOT / 'raw_parts'; folder.mkdir(exist_ok=True)
    codes = sorted(schedule.code.unique()); parts = {}; raw_rows = 0
    producer = sha(Path(__file__))
    for offset in range(0, len(codes), 64):
        subset = codes[offset:offset + 64]
        path = folder / f'part_{offset // 64:03d}.parquet'; meta = path.with_suffix('.json')
        sources = [MINUTES / code[:2].upper() / (code[3:] + '.parquet') for code in subset]
        digests = {str(source): m['minute_source_sha256'][str(source)] for source in sources}
        if meta.exists():
            r = json.loads(meta.read_text())
            assert r['codes'] == subset and r['protocol_sha256'] == sha(PROTOCOL)
            assert r['producer_sha256'] == producer and r['source_sha256'] == digests and r['raw_sha256'] == sha(path)
        else:
            for source in sources:
                assert sha(source) == digests[str(source)]
            c = base.conn(); c.read_parquet([str(source) for source in sources]).create_view('raw')
            c.register('schedule', schedule.loc[schedule.code.isin(subset)])
            f = c.execute('''WITH bounded AS (
                SELECT lower(exchange)||'.'||symbol AS code,strftime(timestamp,'%Y-%m-%d') AS date,
                    strftime(timestamp,'%H%M') AS clock,timestamp,close::DOUBLE AS close,volume::DOUBLE AS volume
                FROM raw WHERE timestamp>=?::DATE AND timestamp<?::DATE+INTERVAL 1 DAY
                AND (strftime(timestamp,'%H%M') BETWEEN '0931' AND '1000' OR strftime(timestamp,'%H%M')='1449'))
                SELECT b.* FROM bounded b JOIN schedule s USING(date,code) ORDER BY b.code,b.date,b.timestamp''',
                          [p['history_first'], p['signal_last']]).df(); c.close()
            assert set(f.code).issubset(subset) and f.date.between(p['history_first'], p['signal_last']).all()
            assert (f.clock.between('0931', '1000') | f.clock.eq('1449')).all()
            # Keep original floating fields and any duplicates; the window stage decides validity.
            f.to_parquet(path, index=False, compression='zstd')
            r = dict(codes=subset, protocol_sha256=sha(PROTOCOL), producer_sha256=producer,
                     source_sha256=digests, raw_sha256=sha(path), rows=len(f))
            save_json(meta, r)
        parts[str(path)] = r['raw_sha256']; raw_rows += r['rows']
        print(json.dumps(dict(codes=offset + len(subset), total_codes=len(codes), raw_minutes=raw_rows)), flush=True)
    r = dict(protocol_sha256=sha(PROTOCOL), schedule_manifest_sha256=sha(SOURCE / 'input_manifest.json'),
             schedule_sha256=sha(SOURCE / 'stock_days.parquet'), minute_source_manifest_sha256=sha(MINUTE_MANIFEST),
             producer_sha256=producer, parts_sha256=parts, raw_minutes=raw_rows, stock_days=len(schedule),
             history_first=p['history_first'], signal_last=p['signal_last'],
             projection=['date', 'code', 'clock', 'timestamp', 'close', 'volume'],
             historical_morning_prices_for_features=True, new_selection_outcomes_read=False,
             new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'raw_report.json', r)
    return {key: value for key, value in r.items() if key != 'parts_sha256'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['extract'])
    p.parse_args(); print(json.dumps(extract(), ensure_ascii=False, indent=2))
