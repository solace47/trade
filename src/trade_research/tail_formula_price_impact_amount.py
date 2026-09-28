"""Extract only the newly required 29 minute amounts from pinned 2024–2025 files."""
import json
from pathlib import Path

import pandas as pd

from . import tail_formula_additive as base
from .corporate_cash import MINUTES, save_json, sha

ROOT = Path('data/research/tail_formula_price_impact')
PROTOCOL = Path('config/tail_formula_price_impact_protocol.json')
MANIFEST = Path('data/research/economic_winner/input_manifest.json')
OLD = Path('data/research/tail_formula_float/features.parquet')
COLUMNS = [f'il_a{n:02d}' for n in range(21, 50)]


def windows():
    assert not (ROOT / 'window_report.json').exists(), 'Do not replace frozen amount windows'
    p = json.loads(PROTOCOL.read_text())
    for path in [OLD, MANIFEST]:
        assert sha(path) == p['source_hashes'][str(path)]
    assert p['window_first'] == '1421' and p['window_last'] == '1449'
    assert p['signal_first'] == '2024-01-01' and p['signal_last'] == '2025-12-30'
    assert not p['new_2026_prices_allowed']
    keys = pd.read_parquet(OLD, columns=['date', 'code'])
    source_hashes = json.loads(MANIFEST.read_text())['source_sha256']
    folder = ROOT / 'amount_parts'; folder.mkdir(parents=True, exist_ok=True)
    codes = sorted(keys.code.unique()); parts = {}; sources = {}
    for offset in range(0, len(codes), 64):
        subset = codes[offset:offset + 64]
        path = folder / f'part_{offset // 64:03d}.parquet'; meta_path = path.with_suffix('.json')
        paths = [MINUTES / code[:2].upper() / (code[3:] + '.parquet') for code in subset]
        for source in paths:
            assert sha(source) == source_hashes[str(source)]
            sources[str(source)] = source_hashes[str(source)]
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            assert meta['codes'] == subset and meta['protocol_sha256'] == sha(PROTOCOL)
            assert meta['extractor_sha256'] == sha(Path(__file__)) and meta['sha256'] == sha(path)
        else:
            c = base.conn(); c.read_parquet([str(q) for q in paths]).create_view('raw')
            c.register('keys', keys.loc[keys.code.isin(subset)])
            pivots = ','.join(f"max(amount) FILTER(WHERE clock='14{n:02d}') AS il_a{n:02d}" for n in range(21, 50))
            d = c.sql(f"""WITH s AS (
                SELECT lower(exchange)||'.'||symbol AS code, strftime(timestamp,'%Y-%m-%d') AS date,
                strftime(timestamp,'%H%M') AS clock, timestamp, turnover::DOUBLE AS amount
                FROM raw WHERE timestamp>=TIMESTAMP '2024-01-01' AND timestamp<TIMESTAMP '2026-01-01'
                AND strftime(timestamp,'%H%M') BETWEEN '1421' AND '1449'),
                selected AS (SELECT s.*, coalesce(timestamp=date_trunc('minute',timestamp)
                AND isfinite(amount) AND amount>=0,false) AS good FROM s JOIN keys USING(date,code))
                SELECT date,code,count(*) AS il_bars,count(DISTINCT clock) AS il_clocks,
                count(*) FILTER(WHERE good) AS il_good_bars,{pivots}
                FROM selected GROUP BY date,code ORDER BY date,code""").df(); c.close()
            d.to_parquet(path, index=False, compression='zstd')
            meta = dict(codes=subset, protocol_sha256=sha(PROTOCOL), extractor_sha256=sha(Path(__file__)),
                        sha256=sha(path), rows=len(d))
            save_json(meta_path, meta)
        parts[str(path)] = meta['sha256']
        print(json.dumps(dict(codes=offset + len(subset), total_codes=len(codes))), flush=True)
    r = dict(protocol_sha256=sha(PROTOCOL), extractor_sha256=sha(Path(__file__)),
             source_sha256=sources, parts_sha256=parts, minute_manifest_sha256=sha(MANIFEST),
             window_first='1421', window_last='1449', fields_read=['timestamp', 'exchange', 'symbol', 'turnover'],
             outcomes_read=False, new_2026_prices_read=False)
    save_json(ROOT / 'window_report.json', r)
    return {k: v for k, v in r.items() if k not in ['source_sha256', 'parts_sha256']}


if __name__ == '__main__':
    print(json.dumps(windows(), ensure_ascii=False, indent=2))
