"""Freeze strictly prior stock dates, then extract only their closing input bars."""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as original
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_prior_close'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
SCHEDULE = Path('data/research/tail_formula_volume_history')


def checked():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    m = json.loads((SCHEDULE / 'input_manifest.json').read_text())
    assert m['stock_days_sha256'] == sha(SCHEDULE / 'stock_days.parquet')
    r = json.loads((original.ROOT / 'feature_report.json').read_text())
    assert r['features_sha256'] == sha(original.ROOT / 'features.parquet')
    assert p['source_window'] == ['1432', '1500'] and p['price_clock'] == '1456'
    assert not p['new_2026_prices_allowed']
    return p, m


def prepare():
    p, m = checked()
    assert not (ROOT / 'schedule_report.json').exists()
    s = pd.read_parquet(SCHEDULE / 'stock_days.parquet').sort_values(['code', 'date'])
    s['source_date'] = s.groupby('code').date.shift(1)
    keys = pd.read_parquet(original.ROOT / 'features.parquet', columns=['date', 'code'])
    out = keys.merge(s, on=['date', 'code'], how='left', validate='one_to_one')
    assert out.source_date.notna().all() and out.source_date.lt(out.date).all()
    assert out.source_date.between(p['history_first'], p['history_last']).all()
    c = base.conn()
    c.register('keys', keys)
    expected = c.sql(f'''WITH history AS(SELECT date,code,lag(date) OVER(PARTITION BY code ORDER BY date) AS source_date
        FROM read_parquet('{SCHEDULE}/stock_days.parquet'))
        SELECT k.*,h.source_date FROM keys k LEFT JOIN history h USING(date,code) ORDER BY date,code''').df()
    c.close()
    pd.testing.assert_frame_equal(out, expected, check_exact=True)
    needed = out[['source_date', 'code']].drop_duplicates().rename(columns={'source_date': 'date'}).sort_values(['code', 'date']).reset_index(drop=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    out.to_parquet(ROOT / 'prior_dates.parquet', index=False, compression='zstd')
    needed.to_parquet(ROOT / 'source_dates.parquet', index=False, compression='zstd')
    report = dict(passed=True, protocol_sha256=sha(PROTOCOL), prior_dates_sha256=sha(ROOT / 'prior_dates.parquet'),
        source_dates_sha256=sha(ROOT / 'source_dates.parquet'), schedule_manifest_sha256=sha(SCHEDULE / 'input_manifest.json'),
        signal_rows=len(out), source_days=len(needed), codes=needed.code.nunique(), first=needed.date.min(), last=needed.date.max(),
        all_prior_dates_independently_rebuilt=True, no_skipping_missing_minute_days=True,
        new_selection_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'schedule_report.json', report)
    return report


def extract():
    p, m = checked()
    assert not (ROOT / 'raw_report.json').exists()
    sr = json.loads((ROOT / 'schedule_report.json').read_text())
    assert sr['passed'] and sr['protocol_sha256'] == sha(PROTOCOL)
    assert sr['source_dates_sha256'] == sha(ROOT / 'source_dates.parquet')
    needed = pd.read_parquet(ROOT / 'source_dates.parquet')
    folder = ROOT / 'raw_parts'
    folder.mkdir(exist_ok=True)
    codes, parts, total = sorted(needed.code.unique()), {}, 0
    producer = sha(Path(__file__))
    for offset in range(0, len(codes), 64):
        subset = codes[offset:offset + 64]
        path = folder / f'part_{offset // 64:03d}.parquet'
        meta = path.with_suffix('.json')
        sources = [MINUTES / code[:2].upper() / (code[3:] + '.parquet') for code in subset]
        digests = {str(s): m['minute_source_sha256'][str(s)] for s in sources}
        if meta.exists():
            r = json.loads(meta.read_text())
            assert r['codes'] == subset and r['protocol_sha256'] == sha(PROTOCOL)
            assert r['producer_sha256'] == producer and r['source_sha256'] == digests and r['raw_sha256'] == sha(path)
        else:
            for source in sources:
                assert sha(source) == digests[str(source)]
            c = base.conn()
            c.read_parquet([str(s) for s in sources]).create_view('raw')
            c.register('needed', needed.loc[needed.code.isin(subset)])
            f = c.execute('''WITH bounded AS(SELECT lower(exchange)||'.'||symbol AS code,
                strftime(timestamp,'%Y-%m-%d') AS date,strftime(timestamp,'%H%M') AS clock,timestamp,
                open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,close::DOUBLE AS close,volume::DOUBLE AS volume
                FROM raw WHERE timestamp>=?::DATE AND timestamp<?::DATE+INTERVAL 1 DAY
                AND strftime(timestamp,'%H%M') BETWEEN '1432' AND '1500')
                SELECT b.* FROM bounded b JOIN needed n USING(date,code) ORDER BY code,date,timestamp''',
                [p['history_first'], p['history_last']]).df()
            c.close()
            assert f.date.between(p['history_first'], p['history_last']).all() and f.clock.between('1432', '1500').all()
            # Do not hide duplicates or repair bad prices: validity is evaluated later.
            f.to_parquet(path, index=False, compression='zstd')
            r = dict(codes=subset, protocol_sha256=sha(PROTOCOL), producer_sha256=producer,
                source_sha256=digests, raw_sha256=sha(path), rows=len(f))
            save_json(meta, r)
        parts[str(path)] = r['raw_sha256']
        total += r['rows']
        print(json.dumps(dict(codes=offset + len(subset), total_codes=len(codes), raw_minutes=total)), flush=True)
    r = dict(protocol_sha256=sha(PROTOCOL), schedule_report_sha256=sha(ROOT / 'schedule_report.json'),
        producer_sha256=producer, parts_sha256=parts, raw_minutes=total,
        projection=['code', 'date', 'clock', 'timestamp', 'open', 'high', 'low', 'close', 'volume'],
        only_strictly_prior_input_windows=True, new_selection_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'raw_report.json', r)
    return {k: v for k, v in r.items() if k != 'parts_sha256'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'extract'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
