"""A zero-fit identity gate before any new cross-board price inputs."""
import json
import subprocess
from pathlib import Path

import pandas as pd

from trade_research.tail_formula_additive import conn
from trade_research.research_io import check_runtime, check_sources, minute_sources, save_json, sha

ROOT = Path('data/research/tail_formula_board_context')
PROTOCOL = Path('config/tail_formula_board_context_sources.json')


def audit():
    check_runtime()
    assert subprocess.check_output(['git', 'show', 'HEAD:'+str(PROTOCOL)]) == PROTOCOL.read_bytes()
    p = json.loads(PROTOCOL.read_text())
    check_sources(p['source_hashes'])
    destination = ROOT/'source_gate.json'
    assert not destination.exists()
    m = json.loads(Path('data/research/economic_winner/input_manifest.json').read_text())
    d = json.loads(Path('data/research/next_day_winner/source_manifest.json').read_text())
    declared = minute_sources(m['source_sha256'])
    minutes = {code: path for code, path in declared.items() if code.startswith('sz.30')}
    daily = sorted(path for path in d['sha256']
                   if Path(path).parent.name == 'daily' and Path(path).stem.startswith('sz_30'))
    assert minutes and daily
    for path in daily:
        assert sha(Path(path)) == d['sha256'][path], path
    for code, path in minutes.items():
        assert Path(path).exists() and sha(Path(path)) == m['source_sha256'][path], code
    frames = [pd.read_parquet(path, columns=p['allowed_daily_columns'],
        filters=[('date', '>=', p['first_date']), ('date', '<=', p['last_date'])]) for path in daily]
    expected = pd.concat(frames, ignore_index=True)
    assert not expected.duplicated(['date', 'code']).any()
    assert expected.code.str.startswith('sz.30').all()
    expected = expected.loc[expected.tradestatus.eq(1), ['date', 'code']].sort_values(['date', 'code']).reset_index(drop=True)
    c = conn()
    rebuilt = c.execute('''SELECT date,code FROM read_parquet(?)
        WHERE tradestatus=1 AND date>=? AND date<=? ORDER BY date,code''',
        [daily, p['first_date'], p['last_date']]).df()
    pd.testing.assert_frame_equal(expected, rebuilt, check_exact=True)
    expected['declared_minute_file'] = expected.code.isin(minutes)
    c.register('active', expected)
    counts = c.sql('''SELECT date,count(*) AS active_local_members,
        sum(declared_minute_file::INT) AS declared_minute_members,
        sum((NOT declared_minute_file)::INT) AS missing_minute_members
        FROM active GROUP BY date ORDER BY date''').df()
    by_date = expected.groupby('date').declared_minute_file.agg(['size', 'sum'])
    assert counts.date.tolist() == by_date.index.tolist()
    assert counts.active_local_members.tolist() == by_date['size'].tolist()
    assert counts.declared_minute_members.tolist() == by_date['sum'].tolist()
    c.close()
    missing = expected.loc[~expected.declared_minute_file].groupby('code').date.agg(['min', 'max', 'size']).reset_index()
    ROOT.mkdir(exist_ok=True)
    expected.to_parquet(ROOT/'active_identity_keys.parquet', index=False, compression='zstd')
    counts.to_parquet(ROOT/'identity_counts.parquet', index=False, compression='zstd')
    missing.to_parquet(ROOT/'missing_minute_codes.parquet', index=False, compression='zstd')
    receipts = dict(p['source_hashes'])
    for path in daily: receipts[path] = d['sha256'][path]
    for path in minutes.values(): receipts[path] = m['source_sha256'][path]
    for name in ['active_identity_keys.parquet', 'identity_counts.parquet', 'missing_minute_codes.parquet']:
        receipts[str(ROOT/name)] = sha(ROOT/name)
    result = dict(passed=True, protocol_sha256=sha(PROTOCOL), source_hashes=receipts,
        all_active_identity_keys_and_daily_counts_SQL_verified=True,
        daily_files=len(daily), minute_files=len(minutes), active_keys=len(expected),
        days=len(counts), missing_codes=missing.to_dict('records'),
        source_population_gate_passed=missing.empty,
        file_existence_is_necessary_not_complete_minute_or_native_parity=True,
        no_new_fits=True, no_new_price_features=True, no_new_economic_groups=True,
        new_2026_price_rows_read=False)
    save_json(destination, result)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_hashes'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    audit()
