"""Extend training inputs while reusing the earlier low-amount 2025 replay.

Only the 2024 low-amount keys require new afternoon extraction. No execution,
morning observations, labels, scores or model fitting belong to this stage.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_feature_subsample as original
from . import tail_formula_liquidity_inputs as cached
from . import tail_formula_replay48 as replay
from .corporate_cash import DAILY, MINUTES, save_json, sha

STEM = 'tail_formula_pool_scope'
ROOT = Path('data/research') / STEM
OUT = ROOT / 'inputs'
PROTOCOL = Path('config/tail_formula_pool_scope_inputs_protocol.json')
META = original.META
EXPRESSIONS = original.ARMS['norm']
HEADER = original.HEADER
VISIBLE = cached.VISIBLE
COLUMNS = cached.UNIVERSE_COLUMNS


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['input_only'] and not p['new_2026_prices_allowed']
    assert p['expressions'] == EXPRESSIONS and p['native_header'] == HEADER
    assert p['extract_first'] == '2024-01-01' and p['extract_last'] == '2024-12-31'
    assert p['warmup_first'] == '2023-06-01'
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    gate = json.loads((Path('data/research/tail_formula_dense_bars') / 'expansion_gate.json').read_text())
    assert gate['passed'] and not gate['expand_2024']
    coverage = json.loads((ROOT / 'coverage_verification.json').read_text())
    assert coverage['passed'] and coverage['visible_pool_sha256'] == sha(ROOT / 'visible_pool.parquet')
    for parent in [original.INPUTS, cached.OUT]:
        r = json.loads((parent / 'feature_report.json').read_text())
        v = json.loads((parent / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(parent / 'feature_report.json')
        assert r['features_sha256'] == sha(parent / 'features.parquet')
    return p


def universe():
    visible = pd.read_parquet(VISIBLE, columns=COLUMNS,
        filters=[('date', '>=', '2024-01-01'), ('date', '<', '2026-01-01')])
    member = pd.read_parquet(ROOT / 'visible_pool.parquet')
    f = member[['date', 'code', 'original_pool']].merge(visible, on=['date', 'code'],
        how='left', validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    assert len(f) == 1443629 and f.isST.eq(0).all() and f.tradestatus.eq(1).all()
    assert f.board.eq('main').all() and f.listing_age_sessions.ge(60).all()
    # Keep the historical flag under its own name. The changed submission
    # domain is a visible-input choice, not a claim about a future fill.
    f['historical_necessary_tradeable'] = f.necessary_tradeable
    assert f.historical_necessary_tradeable.equals(f.original_pool)
    f['necessary_tradeable'] = True
    pd.testing.assert_frame_equal(f[member.columns], member, check_exact=True)
    return f


def inventory():
    checked(); assert not (ROOT / 'input_cache_reuse_verification.json').exists()
    u = universe(); extra = u.loc[~u.original_pool]
    low = pd.read_parquet(cached.OUT / 'features.parquet', columns=META)
    expected = extra.loc[extra.date.ge('2025-01-01'), META[:-1]].reset_index(drop=True)
    pd.testing.assert_frame_equal(low[META[:-1]], expected, check_exact=True)
    assert len(low) == 48473
    assert not extra.loc[extra.date.lt('2025-01-01'), ['date', 'code']].merge(
        low[['date', 'code']], on=['date', 'code']).shape[0]
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL),
        previous_limited_inventory_sha256=sha(ROOT / 'coverage_verification.json'),
        prior_low_amount_feature_report_sha256=sha(cached.OUT / 'feature_report.json'),
        prior_low_amount_feature_verification_sha256=sha(cached.OUT / 'feature_verification.json'),
        all_2025_extra_keys_metadata_and_existing_validity_retained=True,
        extra_rows=185544, cached_extra_rows=48473, extra_rows_requiring_replay=137071,
        cached_valid=int(low.formula_input_valid.sum()),
        earlier_old_formula_extrapolation_is_not_a_new_experiment=True,
        new_training_scope_still_requires_separate_label_and_model_protocol=True,
        no_new_prices_or_future_fields_or_scores_read=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'input_cache_reuse_verification.json', proof)
    return proof


def sources():
    p = checked(); proof = json.loads((ROOT / 'input_cache_reuse_verification.json').read_text())
    assert proof['passed'] and proof['protocol_sha256'] == sha(PROTOCOL)
    daily = json.loads(VISIBLE.with_name('source_manifest.json').read_text())['sha256']
    minute = json.loads(cached.MINUTE_MANIFEST.read_text())['source_sha256']
    return p, daily, minute


def prepare():
    p, daily, minute = sources(); assert not (OUT / 'base_report.json').exists()
    OUT.mkdir(parents=True, exist_ok=True)
    u = universe(); u.to_parquet(OUT / 'universe.parquet', index=False, compression='zstd')
    u[['date', 'code', 'original_pool']].to_parquet(OUT / 'membership.parquet', index=False, compression='zstd')
    new = u.loc[~u.original_pool & u.date.lt('2025-01-01'), COLUMNS].reset_index(drop=True)
    assert len(new) == 137071
    new.to_parquet(OUT / 'new_2024_universe.parquet', index=False, compression='zstd')
    codes = sorted(new.code.unique())
    manifest = dict(protocol_sha256=sha(PROTOCOL),
        daily_sha256={str(DAILY / (c.replace('.', '_') + '.parquet')):
            daily[str(DAILY / (c.replace('.', '_') + '.parquet'))] for c in codes},
        minute_sha256={str(MINUTES / c[:2].upper() / (c[3:] + '.parquet')):
            minute[str(MINUTES / c[:2].upper() / (c[3:] + '.parquet'))] for c in codes},
        source_manifest_sha256=sha(VISIBLE.with_name('source_manifest.json')),
        raw_manifest_sha256=sha(cached.MINUTE_MANIFEST), only_new_2024_keys_require_prices=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(OUT / 'source_manifest.json', manifest)
    r = dict(protocol_sha256=sha(PROTOCOL), universe_sha256=sha(OUT / 'universe.parquet'),
        membership_sha256=sha(OUT / 'membership.parquet'),
        new_2024_universe_sha256=sha(OUT / 'new_2024_universe.parquet'),
        source_manifest_sha256=sha(OUT / 'source_manifest.json'),
        cache_reuse_verification_sha256=sha(ROOT / 'input_cache_reuse_verification.json'),
        rows=len(u), old_rows=int(u.original_pool.sum()), cached_2025_extra_rows=48473,
        new_2024_extra_rows=len(new), new_2024_codes=len(codes),
        no_execution_or_morning_or_labels_or_scores_read=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(OUT / 'base_report.json', r); return r


def checked_base():
    p, _, _ = sources(); r = json.loads((OUT / 'base_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL)
    for key, file in [('universe', 'universe.parquet'), ('membership', 'membership.parquet'),
        ('new_2024_universe', 'new_2024_universe.parquet'), ('source_manifest', 'source_manifest.json')]:
        assert r[key + '_sha256'] == sha(OUT / file)
    return p, json.loads((OUT / 'source_manifest.json').read_text())


def features():
    p, manifest = checked_base(); assert not (OUT / 'feature_report.json').exists()
    new = pd.read_parquet(OUT / 'new_2024_universe.parquet')
    indices = pd.read_parquet(cached.OUT / 'indices.parquet')
    indices = indices.loc[indices.date.le('2024-12-31')].reset_index(drop=True)
    points = pd.read_parquet(cached.OUT / 'index_points.parquet')
    points = points.loc[points.date.lt('2025-01-01')].reset_index(drop=True)
    indices.to_parquet(OUT / 'indices.parquet', index=False, compression='zstd')
    points.to_parquet(OUT / 'index_points.parquet', index=False, compression='zstd')
    codes = sorted(new.code.unique()); folder = OUT / 'parts'; folder.mkdir(exist_ok=True); parts = {}
    for offset in range(0, len(codes), 64):
        subset = codes[offset:offset + 64]; path = folder / f'part_{offset // 64:03d}.parquet'
        receipt = path.with_suffix('.json')
        identity = dict(codes=subset, protocol_sha256=sha(PROTOCOL),
            base_report_sha256=sha(OUT / 'base_report.json'),
            source_manifest_sha256=sha(OUT / 'source_manifest.json'),
            extractor_sha256=sha(Path(__file__)), adapter_sha256=sha(Path(replay.__file__)),
            indices_sha256=sha(OUT / 'indices.parquet'), index_points_sha256=sha(OUT / 'index_points.parquet'))
        if receipt.exists():
            meta = json.loads(receipt.read_text())
            assert all(meta[k] == v for k, v in identity.items()) and meta['sha256'] == sha(path)
        else:
            files = [MINUTES / c[:2].upper() / (c[3:] + '.parquet') for c in subset]
            daily = [DAILY / (c.replace('.', '_') + '.parquet') for c in subset]
            for name, paths in [('minute', files), ('daily', daily)]:
                for file in paths:
                    assert sha(file) == manifest[name + '_sha256'][str(file)], file
            keys = new.loc[new.code.isin(subset)]
            agg = replay.afternoon_aggregates(keys, files, p['extract_first'], p['extract_last'])
            history = replay.read_daily(daily, p['warmup_first'], p['extract_last'])
            f = replay.combine(keys, agg, history, indices, points)
            f.to_parquet(path, index=False, compression='zstd')
            meta = dict(**identity, sha256=sha(path), rows=len(f)); save_json(receipt, meta)
        parts[str(path)] = meta['sha256']
        print(json.dumps(dict(codes=offset + len(subset), total_codes=len(codes))), flush=True)
    new = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    new.to_parquet(OUT / 'new_2024_features.parquet', index=False, compression='zstd')
    fields = [*META, *EXPRESSIONS]
    old = pd.read_parquet(original.INPUTS / 'features.parquet', columns=fields,
        filters=[('date', '>=', '2024-01-01'), ('date', '<', '2026-01-01')])
    low = pd.read_parquet(cached.OUT / 'features.parquet', columns=fields)
    all_rows = pd.concat([old, low, new[fields]], ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    u = pd.read_parquet(OUT / 'universe.parquet')
    pd.testing.assert_frame_equal(all_rows[META[:-1]], u[META[:-1]], check_exact=True)
    assert len(all_rows) == 1443629 and not all_rows[['date', 'code']].duplicated().any()
    member = pd.read_parquet(OUT / 'membership.parquet')
    pd.testing.assert_frame_equal(all_rows.loc[member.original_pool].reset_index(drop=True), old, check_exact=True)
    all_rows.to_parquet(OUT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), base_report_sha256=sha(OUT / 'base_report.json'),
        source_manifest_sha256=sha(OUT / 'source_manifest.json'), implementation_sha256=sha(Path(__file__)),
        adapter_sha256=sha(Path(replay.__file__)), parts_sha256=parts,
        features_sha256=sha(OUT / 'features.parquet'),
        new_2024_features_sha256=sha(OUT / 'new_2024_features.parquet'),
        indices_sha256=sha(OUT / 'indices.parquet'), index_points_sha256=sha(OUT / 'index_points.parquet'),
        rows=len(all_rows), valid=int(all_rows.formula_input_valid.sum()), old_valid=int(old.formula_input_valid.sum()),
        cached_2025_extra_valid=int(low.formula_input_valid.sum()), new_2024_extra_valid=int(new.formula_input_valid.sum()),
        expressions=EXPRESSIONS, native_header=HEADER,
        by_half=all_rows.groupby('half').agg(rows=('code', 'size'), valid=('formula_input_valid', 'sum')).reset_index().to_dict('records'),
        old_values_and_validity_exactly_preserved=True, no_duplicate_2025_raw_extraction=True,
        no_new_execution_or_morning_or_labels_or_scores_read=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(OUT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['expressions', 'native_header', 'parts_sha256']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['inventory', 'prepare', 'features'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
