"""Input-only audit of the research's additional sixty-session listing gate."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research.tail_formula_additive import conn
from trade_research.research_io import check_runtime, check_sources, sha, save_json

PROTOCOL = Path('config/tail_formula_listing_scope.json')
META = ['date', 'code', 'half', 'board', 'decision_shares']
FIELDS = [*META, 'listing_age_sessions', 'price_1449', 'preclose', 'upper_limit',
          'volume_1449', 'amount_1449', 'reference_gap', 'known_delisting',
          'isST', 'tradestatus', 'necessary_tradeable']


def minute_sources(manifest):
    """Use the frozen source identity, including exchange, instead of guessing a root."""
    mapping = {}
    for file in manifest:
        path = Path(file)
        assert path.suffix == '.parquet' and path.parent.name in ['SH', 'SZ']
        assert len(path.stem) == 6 and path.stem.isdigit()
        code = path.parent.name.lower() + '.' + path.stem
        assert code not in mapping, 'Ambiguous minute source for ' + code
        mapping[code] = file
    return mapping


def main():
    check_runtime(); p = json.loads(PROTOCOL.read_text()); check_sources(p['source_hashes'])
    assert subprocess.check_output(['git', 'show', f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    assert sha(PROTOCOL) in subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    gate = json.loads(Path(p['completed_receipt']).read_text())
    assert gate['passed']; check_sources(gate['source_hashes'])
    root = Path(p['output_root']); root.mkdir(exist_ok=False)
    b = pd.read_parquet(p['visible_base'], columns=FIELDS).sort_values(['date', 'code']).reset_index(drop=True)
    assert len(b) == 2404280 and b.date.ge('2024-01-01').all() and b.date.lt('2026-01-01').all()
    assert not b.duplicated(['date', 'code']).any()
    b = b.loc[b.board.eq('main')].reset_index(drop=True)
    assert b.isST.eq(0).all() and b.tradestatus.eq(1).all() and b.listing_age_sessions.ge(20).all()
    assert b.code.str.startswith(('sh.60', 'sz.00')).all()
    cents = np.rint(b.price_1449 * 100).astype('int64')
    shares = (2000000 // cents) // 100 * 100
    np.testing.assert_array_equal(shares, b.decision_shares)
    upper = ((np.rint(b.preclose * 100).astype('int64') * 110 + 50) // 100) / 100
    np.testing.assert_array_equal(upper, b.upper_limit)
    necessary = ((b.price_1449 + np.maximum(b.price_1449 * .0015, .005) < upper - .005)
                 & shares.gt(0) & b.amount_1449.ge(3e7) & ~b.reference_gap & ~b.known_delisting)
    np.testing.assert_array_equal(necessary, b.necessary_tradeable)
    expanded = b.loc[necessary & b.price_1449.le(200)].copy()
    expanded['original_pool'] = expanded.listing_age_sessions.ge(60)
    expanded['extra_pool'] = ~expanded.original_pool
    parent = pd.read_parquet(p['original_features'], columns=META,
                             filters=[('date', '>=', '2024-01-01')])
    old = expanded.loc[expanded.original_pool, META].reset_index(drop=True)
    pd.testing.assert_frame_equal(old, parent, check_exact=True)
    c = conn(); c.register('base', b)
    expected = c.sql('''SELECT date,code,half,board,decision_shares,listing_age_sessions,
        listing_age_sessions>=60 AS original_pool,listing_age_sessions<60 AS extra_pool
        FROM base WHERE price_1449+greatest(price_1449*.0015,.005)<upper_limit-.005
        AND decision_shares>0 AND amount_1449>=30000000 AND NOT reference_gap
        AND NOT known_delisting AND price_1449<=200 ORDER BY date,code''').df()
    pd.testing.assert_frame_equal(expanded[expected.columns].reset_index(drop=True), expected, check_exact=True)
    raw = json.loads(Path(p['raw_report']).read_text()); cached = []
    sources = dict(p['source_hashes'])
    for file, digest in raw['batch_manifests_sha256'].items():
        q = json.loads(Path(file).read_text()); path = Path(file).with_suffix('.parquet')
        assert sha(Path(file)) == digest and sha(path) == q['sha256']
        cached.append(pd.read_parquet(path, columns=['date', 'code']))
        sources[file] = digest; sources[str(path)] = q['sha256']
    keys = pd.concat(cached, ignore_index=True)
    assert keys.date.ge('2024-01-01').all() and keys.date.lt('2026-01-01').all()
    assert not keys.duplicated(['date', 'code']).any()
    extra = expanded.loc[expanded.extra_pool].copy()
    coverage = extra[META].merge(keys.assign(prefix_cache_present=True), on=['date', 'code'],
                                how='left', validate='one_to_one')
    coverage['prefix_cache_present'] = coverage.prefix_cache_present.eq(True)
    manifest = json.loads(Path(p['minute_manifest']).read_text())['source_sha256']
    extra['minute_path'] = extra.code.map(minute_sources(manifest))
    extra['minute_manifest_present'] = extra.minute_path.isin(manifest)
    extra['minute_file_present'] = extra.minute_path.map(lambda file: isinstance(file, str) and Path(file).exists())
    expanded.to_parquet(root / 'visible_pool.parquet', index=False, compression='zstd')
    coverage.to_parquet(root / 'prefix_cache_coverage.parquet', index=False, compression='zstd')
    extra.to_parquet(root / 'extra_inputs.parquet', index=False, compression='zstd')
    c.register('extra', extra)
    groups = c.sql('SELECT half,count(*)::BIGINT AS rows,count(DISTINCT date)::BIGINT AS days,count(DISTINCT code)::BIGINT AS codes,min(listing_age_sessions) AS youngest,max(listing_age_sessions) AS oldest FROM extra GROUP BY half ORDER BY half').df().to_dict('records')
    months = c.sql('SELECT substr(date,1,7) AS month,count(*)::BIGINT AS rows,count(DISTINCT code)::BIGINT AS codes FROM extra GROUP BY month ORDER BY month').df().to_dict('records')
    c.close()
    for name in ['visible_pool', 'prefix_cache_coverage', 'extra_inputs']:
        path = root / (name + '.parquet'); sources[str(path)] = sha(path)
    save_json(root / 'report.json', dict(passed=True, protocol_sha256=sha(PROTOCOL), source_hashes=sources,
        original_rows=len(old), expanded_rows=len(expanded), extra_rows=len(extra), extra_by_half=groups,
        extra_by_month=months, prefix_cached_rows=int(coverage.prefix_cache_present.sum()),
        minute_manifest_rows=int(extra.minute_manifest_present.sum()), minute_file_rows=int(extra.minute_file_present.sum()),
        all_original_keys_and_metadata_exactly_preserved=True, all_extra_eligibility_SQL_verified=True,
        source_coverage_not_quality_or_native_parity=True, research_listing_gate_only_changed=True,
        original_volume_and_price_restrictions_unchanged=True, previous_history_of_21_stock_days_still_required=True,
        new_fits=0, new_selection_lists=0, new_outcomes_or_scores_read=False,
        new_2026_prices_read=False, no_exit_rules=True))
    print(json.dumps(dict(report_sha256=sha(root / 'report.json'), extra_rows=len(extra), groups=groups)), flush=True)


if __name__ == '__main__':
    main()
