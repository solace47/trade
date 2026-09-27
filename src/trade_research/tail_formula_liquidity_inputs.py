"""Replay the original inputs only for 2025 keys below the old amount floor."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as original
from . import tail_formula_replay48 as replay
from .corporate_cash import DAILY, MINUTES, save_json, sha

ROOT = Path('data/research/tail_formula_liquidity')
OUT = ROOT/'inputs'
PROTOCOL = Path('config/tail_formula_liquidity_protocol.json')
VISIBLE = Path('data/research/next_day_winner/visible_base.parquet')
MINUTE_MANIFEST = Path('data/research/economic_winner/input_manifest.json')
LEGACY_PROOF = Path('data/research/tail_formula_forward_2026q1/legacy_replay_verification.json')
UNIVERSE_COLUMNS = ['date','code','half','board','necessary_tradeable','price_1449','high_1449','low_1449',
    'daily_open','volume_1449','amount_1449','preclose','listing_age_sessions','reference_gap','known_delisting',
    'decision_shares','upper_limit','isST','tradestatus']


def source(fold):
    p = json.loads(PROTOCOL.read_text())
    cfg = p['folds'][fold]
    prior = Path(cfg['source_root'])
    for name, digest in cfg['source_sha256'].items():
        assert sha(prior/name) == digest
    for kind in ['model', 'score', 'selection']:
        proof = json.loads((prior/f'{kind}_verification.json').read_text())
        assert proof['passed'] and proof[f'{kind}_report_sha256'] == sha(prior/f'{kind}_report.json')
    model = json.loads((prior/'model_report.json').read_text())
    selected = json.loads((prior/'selection_report.json').read_text())
    assert model['feature_names'] == list(original.EXPRESSIONS) and model['variant'] == 'relative'
    assert model['last_observation'] < cfg['evaluation_start']
    assert selected['chosen_threshold']['training_quantile'] == .995
    assert selected['selection_sha256'] == sha(prior/'selection.parquet')
    assert selected['core_sha256'] == sha(prior/'frozen_numeric_core.tdx')
    return cfg, prior


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['first'] == '2025-01-01' and p['last'] == '2025-12-30'
    assert p['warmup_first'] == '2023-06-01' and not p['model_refitted']
    assert p['base_report_sha256'] == sha(VISIBLE.with_name('base_report.json'))
    r = json.loads(VISIBLE.with_name('base_report.json').read_text())
    assert r['base_sha256'] == sha(VISIBLE)
    daily_manifest = VISIBLE.with_name('source_manifest.json')
    assert r['source_manifest_sha256'] == sha(daily_manifest)
    assert sha(MINUTE_MANIFEST) == '26ec21da15c414c1d725c8ff26208ef129eaa5f5873c61de56c915ae25ff6a70'
    assert p['original_feature_report_sha256'] == sha(original.ROOT/'feature_report.json')
    r = json.loads((original.ROOT/'feature_report.json').read_text())
    proof = json.loads((original.ROOT/'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(original.ROOT/'feature_report.json')
    assert r['features_sha256'] == sha(original.ROOT/'features.parquet')
    assert p['replay_adapter_sha256'] == sha(Path(replay.__file__))
    assert sha(LEGACY_PROOF) == 'fdf218aadb205b5abbe8d19741642852cc0d5bbe6816523e90e94a05e896e77c'
    proof = json.loads(LEGACY_PROOF.read_text())
    assert proof['passed'] and proof['adapter_sha256'] == sha(Path(replay.__file__))
    assert not proof['new_2026_prices_read']
    return p, json.loads(daily_manifest.read_text())['sha256'], json.loads(MINUTE_MANIFEST.read_text())['source_sha256']


def prepare():
    if (OUT/'base_report.json').exists():
        raise ValueError('Do not replace the frozen expanded universe')
    p, daily_hashes, minute_hashes = checked_sources()
    for fold in ['2024','recent']:
        source(fold)
    f = pd.read_parquet(VISIBLE, columns=UNIVERSE_COLUMNS)
    mask = (f.date.between(p['first'], p['last']) & f.board.eq('main') & f.listing_age_sessions.ge(60)
        & f.price_1449.le(200) & f.decision_shares.gt(0) & ~f.reference_gap & ~f.known_delisting
        & (f.price_1449+np.maximum(f.price_1449*.0015, .005) < f.upper_limit-.005))
    f = f.loc[mask & ~f.necessary_tradeable].sort_values(['date','code']).reset_index(drop=True)
    assert f.amount_1449.gt(0).all() and f.amount_1449.lt(3e7).all()
    assert f.isST.eq(0).all() and f.tradestatus.eq(1).all()
    assert not f.code.str[3:].str.startswith(('92','688','300','301')).any()
    f['necessary_tradeable'] = True
    OUT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(OUT/'universe.parquet', index=False, compression='zstd')
    manifest = dict(protocol_sha256=sha(PROTOCOL), daily_manifest_sha256=sha(VISIBLE.with_name('source_manifest.json')),
        minute_manifest_sha256=sha(MINUTE_MANIFEST),
        daily_sha256={str(DAILY/(c.replace('.','_')+'.parquet')):daily_hashes[str(DAILY/(c.replace('.','_')+'.parquet'))]
                      for c in sorted(f.code.unique())},
        minute_sha256={str(MINUTES/c[:2].upper()/(c[3:]+'.parquet')):minute_hashes[str(MINUTES/c[:2].upper()/(c[3:]+'.parquet'))]
                       for c in sorted(f.code.unique())})
    save_json(OUT/'source_manifest.json', manifest)
    report = dict(protocol_sha256=sha(PROTOCOL), original_base_report_sha256=sha(VISIBLE.with_name('base_report.json')),
        source_manifest_sha256=sha(OUT/'source_manifest.json'), universe_sha256=sha(OUT/'universe.parquet'),
        rows=len(f), codes=f.code.nunique(), by_half=f.groupby('half').size().to_dict(),
        only_amount_floor_removed=True, original_formula_and_thresholds_frozen=True,
        outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(OUT/'base_report.json', report)
    return report


def index_inputs():
    market, context = original.context.market.ROOT, original.context.ROOT
    assert sha(market/'index_source_report.json') == '26d62966871660c596bce49a24bdcf007e30628d91948a9f87dca49363ccf166'
    r = json.loads((market/'index_source_report.json').read_text())
    assert r['indices_sha256'] == sha(market/'indices.parquet')
    assert sha(context/'feature_report.json') == '9bfc158fcea31442b21cd2afbc97d384877a6bece25635d4c4768f7b34ab463a'
    r = json.loads((context/'feature_report.json').read_text())
    proof = json.loads((context/'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(context/'feature_report.json')
    assert r['index_points_sha256'] == sha(context/'index_points.parquet')
    return pd.read_parquet(market/'indices.parquet'), pd.read_parquet(context/'index_points.parquet')


def features():
    if (OUT/'feature_report.json').exists():
        raise ValueError('Do not replace frozen added-key inputs')
    p, _, _ = checked_sources()
    r = json.loads((OUT/'base_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL)
    assert r['source_manifest_sha256'] == sha(OUT/'source_manifest.json')
    assert r['universe_sha256'] == sha(OUT/'universe.parquet')
    sources = json.loads((OUT/'source_manifest.json').read_text())
    universe = pd.read_parquet(OUT/'universe.parquet')
    indices, points = index_inputs()
    indices.to_parquet(OUT/'indices.parquet', index=False, compression='zstd')
    points.to_parquet(OUT/'index_points.parquet', index=False, compression='zstd')
    codes = sorted(universe.code.unique())
    folder = OUT/'parts'; folder.mkdir(exist_ok=True); parts = {}
    for start in range(0, len(codes), 64):
        subset = codes[start:start+64]; path = folder/f'part_{start//64:03d}.parquet'; receipt = path.with_suffix('.json')
        identity = dict(codes=subset, protocol_sha256=sha(PROTOCOL), base_report_sha256=sha(OUT/'base_report.json'),
            source_manifest_sha256=sha(OUT/'source_manifest.json'), extractor_sha256=sha(Path(__file__)),
            adapter_sha256=sha(Path(replay.__file__)), indices_sha256=sha(OUT/'indices.parquet'),
            index_points_sha256=sha(OUT/'index_points.parquet'))
        if receipt.exists():
            meta = json.loads(receipt.read_text())
            assert all(meta[k] == v for k,v in identity.items()) and meta['sha256'] == sha(path)
        else:
            files = [MINUTES/c[:2].upper()/(c[3:]+'.parquet') for c in subset]
            daily_files = [DAILY/(c.replace('.','_')+'.parquet') for c in subset]
            for file in files:
                assert sha(file) == sources['minute_sha256'][str(file)]
            for file in daily_files:
                assert sha(file) == sources['daily_sha256'][str(file)]
            keys = universe.loc[universe.code.isin(subset)]
            agg = replay.afternoon_aggregates(keys, files, p['first'], p['last'])
            daily = replay.read_daily(daily_files, p['warmup_first'], p['last'])
            f = replay.combine(keys, agg, daily, indices, points)
            f.to_parquet(path, index=False, compression='zstd')
            meta = dict(**identity, sha256=sha(path), rows=len(f)); save_json(receipt, meta)
        parts[str(path)] = meta['sha256']
        print(json.dumps(dict(codes=start+len(subset), total_codes=len(codes))), flush=True)
    f = pd.concat([pd.read_parquet(path) for path in parts], ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(f[UNIVERSE_COLUMNS], universe, check_exact=True)
    f.to_parquet(OUT/'features.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), base_report_sha256=sha(OUT/'base_report.json'),
        source_manifest_sha256=sha(OUT/'source_manifest.json'), adapter_sha256=sha(Path(replay.__file__)),
        extractor_sha256=sha(Path(__file__)), legacy_replay_verification_sha256=sha(LEGACY_PROOF),
        parts_sha256=parts, features_sha256=sha(OUT/'features.parquet'), indices_sha256=sha(OUT/'indices.parquet'),
        index_points_sha256=sha(OUT/'index_points.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        by_half=f.groupby('half').agg(rows=('code','size'),valid=('formula_input_valid','sum')).reset_index().to_dict('records'),
        expressions=original.EXPRESSIONS, native_header=original.HEADER,
        original_rows_not_recomputed=True, model_refitted=False, outcomes_read=False,
        new_2026_prices_read=False, no_exit_rules=True, native_source_parity_verified=False)
    save_json(OUT/'feature_report.json', report)
    return {k:v for k,v in report.items() if k not in ['parts_sha256','expressions','native_header']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare','features'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
