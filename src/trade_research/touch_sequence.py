"""Frozen ordering of current-day limit touches, without future selection."""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import MINUTES, save_json, sha
from .tick_morning_exit import classify

ROOT = Path('data/research/touch_sequence')
BASE = Path('data/research/next_day_winner/visible_base.parquet')
PROTOCOL = Path('config/touch_sequence_protocol.json')
LABELS = Path('data/research/economic_winner/period_quality')
GROUPS = ['early_open_rising', 'early_open_not_rising', 'late_first_touch_open',
          'tail_retouch_open', 'unknown_source']
BALANCE = ['return_1449', 'return20_prior_adjusted', 'price_1449', 'amount_1449']


def connection():
    c = duckdb.connect()
    c.execute('SET threads=4')
    c.execute("SET memory_limit='5GB'")
    return c


def freeze():
    if (ROOT / 'manifest.json').exists():
        raise ValueError('Do not replace frozen touch inputs')
    assert sha(BASE) == json.loads(BASE.with_name('base_report.json').read_text())['base_sha256']
    base = pd.read_parquet(BASE)
    base = base.loc[base.board.eq('main') & base.necessary_tradeable].copy()
    assert base.date.between('2024-01-01', '2025-12-30').all()
    base['touched'] = np.rint(base.high_1449 * 100).eq(np.rint(base.upper_limit * 100))
    touched = base.loc[base.touched].sort_values(['date', 'code'])
    references = {d: p.sort_values('code') for d, p in base.loc[base.high_1449.lt(base.upper_limit-.005)].groupby('date')}
    pairs = []
    for d, candidates in touched.groupby('date', sort=True):
        available = references[d]
        for r in candidates.itertuples():
            if not np.isfinite(r.return20_prior_adjusted):
                continue
            choices = available.loc[available.code.str[:2].eq(r.code[:2])
                & (available.return_1449-r.return_1449).abs().le(.01)
                & (available.return20_prior_adjusted-r.return20_prior_adjusted).abs().le(.05)
                & (available.price_1449/r.price_1449).between(.5, 2)
                & (available.amount_1449/r.amount_1449).between(.5, 2)]
            if choices.empty:
                continue
            distances = ((choices.return_1449-r.return_1449).abs()/.01
                + (choices.return20_prior_adjusted-r.return20_prior_adjusted).abs()/.05
                + np.abs(np.log2(choices.price_1449/r.price_1449))
                + np.abs(np.log2(choices.amount_1449/r.amount_1449)))
            peer = choices.loc[distances.idxmin()]
            pairs.append(dict(date=d, code=r.code, half=r.half, control_code=peer.code,
                distance=float(distances.min()), **{k+'_difference': float(getattr(r, k)-peer[k]) for k in BALANCE}))
    pairs = pd.DataFrame(pairs).sort_values(['date', 'code']).reset_index(drop=True)
    keys = pd.concat([touched[['date', 'code']], pairs[['date', 'control_code']].rename(columns={'control_code': 'code'})]).drop_duplicates()
    cohort = keys.merge(base, on=['date', 'code'], validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    cohort.to_parquet(ROOT / 'frozen_cohort.parquet', index=False, compression='zstd')
    pairs.to_parquet(ROOT / 'frozen_pairs.parquet', index=False, compression='zstd')
    source_path = Path('data/research/economic_winner/input_manifest.json')
    sources = json.loads(source_path.read_text())['source_sha256']
    for code in cohort.code.unique():
        assert str(MINUTES / code[:2].upper() / (code[3:]+'.parquet')) in sources
    report = dict(protocol_sha256=sha(PROTOCOL), base_sha256=sha(BASE), economic_manifest_sha256=sha(source_path),
        necessary_main=len(base), touched=len(touched), above_limit=int(base.high_1449.gt(base.upper_limit+.005).sum()),
        paired=len(pairs), unique_controls=len(cohort)-len(touched), rows=len(cohort),
        counts=cohort.groupby(['half', 'touched']).size().rename('rows').reset_index().to_dict('records'),
        output_sha256={n: sha(ROOT / n) for n in ['frozen_cohort.parquet', 'frozen_pairs.parquet']},
        outcomes_read=False, new_2026_prices_read=False)
    save_json(ROOT / 'manifest.json', report)
    return report


def extract():
    if (ROOT / 'raw_report.json').exists():
        raise ValueError('Do not replace extracted prefixes')
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    assert sha(PROTOCOL) == manifest['protocol_sha256']
    for n, h in manifest['output_sha256'].items():
        assert sha(ROOT / n) == h
    cohort = pd.read_parquet(ROOT / 'frozen_cohort.parquet')
    source_path = Path('data/research/economic_winner/input_manifest.json')
    assert sha(source_path) == manifest['economic_manifest_sha256']
    sources = json.loads(source_path.read_text())['source_sha256']
    codes = sorted(cohort.code.unique())
    folder = ROOT / 'raw_parts'
    folder.mkdir(exist_ok=True)
    parts = {}
    for offset in range(0, len(codes), 64):
        subset = codes[offset:offset+64]
        dest = folder / f'part_{offset//64:03d}.parquet'
        meta = dest.with_suffix('.json')
        if dest.exists() and meta.exists():
            info = json.loads(meta.read_text())
            assert info['codes'] == subset and info['manifest_sha256'] == sha(ROOT / 'manifest.json')
            assert info['code_sha256'] == sha(Path(__file__)) and info['sha256'] == sha(dest)
        else:
            paths = [MINUTES / code[:2].upper() / (code[3:]+'.parquet') for code in subset]
            for path in paths:
                assert sha(path) == sources[str(path)]
            c = connection()
            c.read_parquet([str(p) for p in paths]).create_view('original')
            c.register('keys', cohort.loc[cohort.code.isin(subset), ['date', 'code']])
            bars = c.sql("""WITH s AS (SELECT lower(exchange)||'.'||symbol AS code,
                strftime(timestamp,'%Y-%m-%d') AS date,timestamp,
                hour(timestamp)*60+minute(timestamp) AS minute,
                open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,close::DOUBLE AS close,
                volume::DOUBLE AS volume,turnover::DOUBLE AS amount FROM original
                WHERE timestamp>=TIMESTAMP '2024-01-01' AND timestamp<TIMESTAMP '2026-01-01')
                SELECT s.* FROM s JOIN keys USING(date,code)
                WHERE minute BETWEEN 570 AND 690 OR minute BETWEEN 781 AND 889
                ORDER BY date,code,timestamp""").df()
            bars.to_parquet(dest, index=False, compression='zstd')
            c.close()
            info = dict(codes=subset, sha256=sha(dest), rows=len(bars), manifest_sha256=sha(ROOT / 'manifest.json'),
                        code_sha256=sha(Path(__file__)))
            save_json(meta, info)
        parts[str(dest)] = info['sha256']
        print(json.dumps(dict(codes=offset+len(subset), total=len(codes), raw_rows=info['rows'])), flush=True)
    c = connection()
    c.read_parquet(list(parts)).create_view('raw')
    c.register('cohort', cohort)
    frame = c.sql("""WITH b AS (SELECT r.*,f.upper_limit,
        floor((round(f.preclose*100)*90+50)/100)/100 AS lower_limit,
        coalesce(timestamp=date_trunc('minute',timestamp) AND isfinite(open) AND isfinite(high) AND isfinite(low)
          AND isfinite(close) AND isfinite(volume) AND isfinite(amount) AND least(open,high,low,close)>0
          AND high+.0001>=greatest(open,close,low) AND low-.0001<=least(open,close)
          AND abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
          AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001
          AND volume>=0 AND amount>=0 AND (volume=0)=(amount=0)
          AND (volume=0 OR amount/volume BETWEEN low-.0101 AND high+.0101),false) AS valid
        FROM raw r JOIN cohort f USING(date,code)),
        a AS (SELECT date,code,count(*) AS bars,count(DISTINCT timestamp) AS labels,
          bool_and(valid) AS valid_bars,
          bool_and(volume=0 OR (high<=upper_limit+.0001 AND low>=lower_limit-.0001)) AS within_limits,
          min(minute) FILTER(WHERE volume>0 AND round(high*100)=round(upper_limit*100)) AS first_touch,
          max(minute) FILTER(WHERE volume>0 AND round(high*100)=round(upper_limit*100)) AS last_touch,
          max(close) FILTER(WHERE minute=860) AS raw_1420,max(close) FILTER(WHERE minute=889) AS raw_1449,
          max(volume) FILTER(WHERE minute=860) AS raw_volume_1420,max(volume) FILTER(WHERE minute=889) AS raw_volume_1449,
          max(high) FILTER(WHERE volume>0) AS raw_high FROM b GROUP BY date,code)
        SELECT f.*,a.* EXCLUDE(date,code),
          coalesce(a.bars=230 AND a.labels=230 AND valid_bars AND within_limits
            AND raw_volume_1420>0 AND raw_volume_1449>0 AND abs(raw_1420-f.price_1420)<=.0001
            AND abs(raw_1449-f.price_1449)<=.0001 AND abs(raw_high-f.high_1449)<=.0001
            AND (f.touched=(first_touch IS NOT NULL)),false) AS source_valid
        FROM cohort f LEFT JOIN a USING(date,code) ORDER BY date,code""").df()
    conditions = [
        ~frame.source_valid, ~frame.touched, frame.first_touch.ge(860).fillna(False),
        frame.last_touch.ge(860).fillna(False), frame.price_1449.gt(frame.price_1420)]
    frame['group'] = np.select([x.to_numpy(dtype=bool, na_value=False) for x in conditions],
        ['unknown_source', 'untouched_control', 'late_first_touch_open', 'tail_retouch_open', 'early_open_rising'],
        default='early_open_not_rising')
    frame['primary'] = frame.touched & frame.group.eq('early_open_rising')
    assert len(frame) == manifest['rows'] and not frame.duplicated(['date','code']).any()
    frame.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    report = dict(manifest_sha256=sha(ROOT / 'manifest.json'), parts_sha256=parts,
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(frame), new_2026_prices_read=False)
    save_json(ROOT / 'raw_report.json', report)
    return {k: v for k, v in report.items() if k != 'parts_sha256'}


def inputs():
    if (ROOT / 'input_report.json').exists():
        raise ValueError('Do not replace final frozen inputs')
    check = json.loads((ROOT / 'prefix_verification.json').read_text())
    assert check['passed'] and check['raw_report_sha256'] == sha(ROOT / 'raw_report.json')
    raw = json.loads((ROOT / 'raw_report.json').read_text())
    assert sha(ROOT / 'features.parquet') == raw['features_sha256']
    f = pd.read_parquet(ROOT / 'features.parquet')
    pairs = pd.read_parquet(ROOT / 'frozen_pairs.parquet')
    primary = f.loc[f.primary, ['date', 'code']]
    pairs = primary.merge(pairs, on=['date', 'code'], how='left', validate='one_to_one')
    pairs.to_parquet(ROOT / 'primary_pairs.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), manifest_sha256=sha(ROOT / 'manifest.json'),
        prefix_verification_sha256=sha(ROOT / 'prefix_verification.json'), rows=len(f), primary=len(primary),
        paired=int(pairs.control_code.notna().sum()),
        counts=f.groupby(['half', 'touched', 'group']).size().rename('rows').reset_index().to_dict('records'),
        balance={k: pairs[k+'_difference'].describe().to_dict() for k in BALANCE},
        output_sha256={n: sha(ROOT / n) for n in ['features.parquet', 'primary_pairs.parquet']},
        outcomes_read=False, new_2026_prices_read=False)
    save_json(ROOT / 'input_report.json', report)
    save_json(ROOT / 'input_verification.json', dict(passed=True, input_report_sha256=sha(ROOT / 'input_report.json'),
        prefix_verification_sha256=sha(ROOT / 'prefix_verification.json'), primary=len(primary)))
    return report


def prepare_exits():
    target = ROOT / 'morning'
    if (target / 'input_report.json').exists():
        raise ValueError('Do not replace fixed exit inputs')
    check = json.loads((ROOT / 'input_verification.json').read_text())
    assert check['passed'] and check['input_report_sha256'] == sha(ROOT / 'input_report.json')
    inputs = json.loads((ROOT / 'input_report.json').read_text())
    for n, h in inputs['output_sha256'].items():
        assert sha(ROOT / n) == h
    proof = json.loads((LABELS / 'analysis_verification.json').read_text())
    assert proof['passed'] and proof['analysis_report_sha256'] == sha(LABELS / 'analysis_report.json')
    assert sha(LABELS / 'labels.parquet') == json.loads((LABELS / 'label_report.json').read_text())['labels_sha256']
    f = pd.read_parquet(ROOT / 'features.parquet')
    c = connection()
    c.register('keys', f[['date', 'code']])
    old = c.execute('SELECT l.* FROM read_parquet(?) l JOIN keys USING(date,code) ORDER BY date,code',
                    [str(LABELS / 'labels.parquet')]).df()
    assert len(old) == len(f)
    pd.testing.assert_frame_equal(old, classify(old)[old.columns], check_dtype=False, atol=1e-10, rtol=0)
    pd.testing.assert_series_equal(old.decision_shares, f.decision_shares, check_dtype=False)
    assert old.next_date.gt(old.date).all() and old.next_date.le('2025-12-31').all()
    target.mkdir(exist_ok=True)
    old.to_parquet(target / 'old_tail_labels.parquet', index=False, compression='zstd')
    keys = old[['next_date', 'code']].rename(columns={'next_date': 'date'}).sort_values(['date', 'code'])
    assert not keys.duplicated().any()
    keys.to_parquet(target / 'window_keys.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL), source_input_report_sha256=sha(ROOT / 'input_report.json'),
        economic_manifest_sha256=sha(Path('data/research/economic_winner/input_manifest.json')),
        old_label_report_sha256=sha(LABELS / 'label_report.json'), rows=len(old),
        output_sha256={n: sha(target / n) for n in ['old_tail_labels.parquet', 'window_keys.parquet']},
        tail_reproduction_columns=len(old.columns), new_2026_prices_read=False)
    save_json(target / 'input_report.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'extract', 'inputs', 'prepare_exits'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
