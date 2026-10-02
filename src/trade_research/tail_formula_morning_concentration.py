"""Frozen completed-morning volume concentration; no investor identity claim."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from . import tail_formula_additive as numeric
from . import tail_formula_baseline as baseline
from .research_io import check_runtime, check_sources, minute_sources, save_json, sha

ROOT = Path('data/research/tail_formula_morning_concentration')
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config/tail_formula_morning_concentration_inputs.json')
EXTRA_HEADER = '''AVDT:=VALUEWHEN(TIME=1130,DATE);
AVBC:=VALUEWHEN(TIME=1130,B0);
AVNC:=VALUEWHEN(TIME=1130,COUNT(TIME>=931 AND TIME<=1130,120));
AVNG:=VALUEWHEN(TIME=1130,COUNT(V>=0,120));
AVTO:=VALUEWHEN(TIME=1130,SUM(V,120));
AVSQ:=VALUEWHEN(TIME=1130,SUM(V*V,120));
AVOK:=AMREADY AND AVDT=DATE AND AVBC=121 AND AVNC=120 AND AVNG=120;
'''
NEW_EXPRESSION = 'IF(AVOK,IF(AVTO=0,-1,100*(120*AVSQ/IF(AVTO>0,AVTO*AVTO,1)-1)/119),DRAWNULL)'
EXPRESSIONS = {**baseline.EXPRESSIONS, 'AMVC': NEW_EXPRESSION}
HEADER = baseline.HEADER + EXTRA_HEADER


def measure(total, squares, valid):
    total, squares = np.asarray(total, float), np.asarray(squares, float)
    with np.errstate(all='ignore'):
        value = np.where(total == 0, -1., 100*(120*squares / np.where(total > 0, total*total, 1) - 1)/119)
    return np.where(valid, value, np.nan)


def checked():
    check_runtime()
    assert subprocess.check_output(['git', 'show', f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    p = json.loads(PROTOCOL.read_text())
    assert sha(PROTOCOL) in subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    assert p['expressions'] == EXPRESSIONS and p['native_header'] == HEADER
    assert p['window_labels'] == 120 and p['known_empty_code'] == -1
    assert p['maximum_new_fits'] == 0 and not p['new_2026_prices_allowed']
    check_sources(p['source_hashes'])
    return p


def cached_parts(p):
    parts = {}
    for path in p['raw_reports']:
        r = json.loads(Path(path).read_text())
        assert not r['new_2026_prices_read']
        for file, digest in r['parts_sha256'].items():
            assert file not in parts and sha(Path(file)) == digest, file
            parts[file] = digest
    return parts


def volume_rows(file):
    d = pd.read_parquet(file, columns=['date', 'code', 'timestamp', 'volume'])
    assert d.date.ge('2023-01-01').all() and d.date.lt('2026-01-01').all()
    clock = d.timestamp.dt.hour*60+d.timestamp.dt.minute
    return d.loc[clock.between(571, 690)].sort_values(['date', 'code', 'timestamp']).reset_index(drop=True)


def aggregate_rows(d):
    d = d.copy()
    d['good'] = (d.timestamp.dt.floor('min').eq(d.timestamp)
                 & d.timestamp.dt.strftime('%Y-%m-%d').eq(d.date)
                 & np.isfinite(d.volume) & d.volume.ge(0))
    d['square'] = d.volume*d.volume
    return d.groupby(['date', 'code'], sort=True).agg(
        bars=('volume', 'size'), labels=('timestamp', 'nunique'),
        good=('good', 'all'), total=('volume', 'sum'), squares=('square', 'sum')).reset_index()


def prepare(p):
    assert not (INPUTS / 'feature_report.json').exists()
    INPUTS.mkdir(parents=True, exist_ok=True)
    parts = cached_parts(p)
    values = []
    for i, file in enumerate(parts):
        values.append(aggregate_rows(volume_rows(file)))
        if i % 8 == 0 or i+1 == len(parts):
            print(json.dumps(dict(aggregated_parts=i+1, total_parts=len(parts))), flush=True)
    a = pd.concat(values, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    assert not a.duplicated(['date', 'code']).any()
    old = pd.read_parquet(p['original_features'])
    assert len(old) == 1815129 and int(old.formula_input_valid.sum()) == 1602413
    a = old[['date', 'code']].merge(a, on=['date', 'code'], how='left', validate='one_to_one')
    valid = (a.bars.eq(120) & a.labels.eq(120) & a.good.eq(True)
             & np.isfinite(a.total) & np.isfinite(a.squares) & a.total.ge(0))
    f = old.copy()
    f['morning_concentration_valid'] = valid
    f['prior_formula_input_valid'] = old.formula_input_valid
    f['AMVC'] = measure(a.total, a.squares, valid)
    f['formula_input_valid'] &= valid & np.isfinite(f.AMVC)
    a.to_parquet(INPUTS / 'aggregates.parquet', index=False, compression='zstd')
    f.to_parquet(INPUTS / 'features.parquet', index=False, compression='zstd')
    save_json(INPUTS / 'feature_report.json', dict(
        protocol_sha256=sha(PROTOCOL), features_sha256=sha(INPUTS / 'features.parquet'),
        aggregates_sha256=sha(INPUTS / 'aggregates.parquet'), source_hashes={**p['source_hashes'], **parts},
        rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        expressions=EXPRESSIONS, native_header=HEADER,
        known_empty_among_old_valid=int((old.formula_input_valid & valid & a.total.eq(0)).sum()),
        all_original_fifty_values_and_metadata_preserved=True, no_new_raw_extraction=True,
        new_economic_groups_read=False, new_2026_prices_read=False, no_exit_rules=True,
        software_compilation_verified=False, native_source_parity_verified=False))
    return dict(rows=len(f), valid=int(f.formula_input_valid.sum()))


def verify(p):
    r = json.loads((INPUTS / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(INPUTS / 'features.parquet')
    parts = cached_parts(p)
    c = numeric.conn()
    c.read_parquet(list(parts)).create_view('cached')
    # Independently retain the whole ordered vector rather than pandas grouped squares.
    expected = c.sql('''WITH w AS (
        SELECT *,extract(hour FROM timestamp)*60+extract(minute FROM timestamp) AS minute
        FROM cached WHERE date>='2023-01-01' AND date<'2026-01-01'),
        a AS (SELECT date,code,count(*) AS bars,count(DISTINCT timestamp) AS labels,
        bool_and(coalesce(timestamp=date_trunc('minute',timestamp)
        AND strftime(timestamp,'%Y-%m-%d')=date AND isfinite(volume) AND volume>=0,false)) AS good,
        list(volume ORDER BY timestamp) AS volumes
        FROM w WHERE minute BETWEEN 571 AND 690 GROUP BY date,code)
        SELECT date,code,bars,labels,good,list_sum(volumes)::DOUBLE AS total,
        list_sum(list_transform(volumes,v->v*v))::DOUBLE AS squares
        FROM a ORDER BY date,code''').df()
    c.close()
    old = pd.read_parquet(p['original_features'])
    expected = old[['date', 'code']].merge(expected, how='left', on=['date', 'code'], validate='one_to_one')
    a = pd.read_parquet(INPUTS / 'aggregates.parquet')
    f = pd.read_parquet(INPUTS / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_frame_equal(a[['date', 'code']], expected[['date', 'code']], check_exact=True)
    for n in ['bars', 'labels', 'good']:
        np.testing.assert_array_equal(a[n], expected[n])
    for n in ['total', 'squares']:
        np.testing.assert_allclose(a[n], expected[n], rtol=3e-13, atol=1e-6, equal_nan=True)
    valid = (expected.bars.eq(120) & expected.labels.eq(120) & expected.good.eq(True)
             & np.isfinite(expected.total) & np.isfinite(expected.squares) & expected.total.ge(0))
    env = dict(AVOK=valid, AVTO=expected.total, AVSQ=expected.squares, IF=np.where, DRAWNULL=np.nan)
    literal = eval(re.sub(r'(?<![<>=!])=(?!=)', '==', NEW_EXPRESSION), {'__builtins__': {}}, env)
    np.testing.assert_allclose(f.AMVC, literal, rtol=0, atol=2e-10, equal_nan=True)
    ok = f.formula_input_valid
    encode = lambda x: np.floor(np.clip(100*np.asarray(x)+10000+.000001, 0, 999999))
    np.testing.assert_array_equal(encode(f.loc[ok, 'AMVC']), encode(literal[ok]))
    np.testing.assert_array_equal(f.morning_concentration_valid, valid)
    np.testing.assert_array_equal(f.prior_formula_input_valid, old.formula_input_valid)
    np.testing.assert_array_equal(f.formula_input_valid, old.formula_input_valid & valid)
    assert f.loc[valid, 'AMVC'].between(-1-2e-10, 100+2e-10).all()
    unchanged = f.formula_input_valid.equals(old.formula_input_valid)
    save_json(INPUTS / 'feature_verification.json', dict(
        passed=True, feature_report_sha256=sha(INPUTS / 'feature_report.json'),
        rows=len(f), valid=int(ok.sum()), all_original_values_and_keys_unchanged=True,
        all_volume_vectors_clock_quality_totals_squares_and_encodings_independently_rebuilt=True,
        effective_input_intersection_unchanged=unchanged, new_fitting_allowed=False,
        mathematical_native_replay_only=True, software_compilation_verified=False,
        native_source_parity_verified=False, new_economic_groups_read=False,
        new_2026_prices_read=False, no_exit_rules=True))
    return dict(passed=True, effective_input_intersection_unchanged=unchanged)


def native(p):
    v = json.loads((INPUTS / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(INPUTS / 'feature_report.json')
    f = pd.read_parquet(INPUTS / 'features.parquet')
    candidates = f.loc[f.formula_input_valid, ['date', 'code', 'half', 'AMVC']].copy()
    candidates['hash'] = [hashlib.sha256((d+'|'+s+'|amvc-v1').encode()).hexdigest()
                          for d, s in zip(candidates.date, candidates.code)]
    sample = candidates.sort_values('hash').groupby('half', sort=True).head(8).sort_values(['date', 'code'])
    manifest = json.loads(Path(p['minute_manifest']).read_text())['source_sha256']
    mapping = minute_sources(manifest)
    receipts = []
    for row in sample.itertuples():
        file = mapping[row.code]
        assert sha(Path(file)) == manifest[file]
        d = pd.read_parquet(file, columns=['timestamp', 'volume'], filters=[
            ('timestamp', '>=', pd.Timestamp(row.date+' 09:31')),
            ('timestamp', '<=', pd.Timestamp(row.date+' 11:30'))]).sort_values('timestamp')
        assert d.timestamp.tolist() == list(pd.date_range(row.date+' 09:31', row.date+' 11:30', freq='min'))
        values = d.volume.to_numpy(float)
        assert np.isfinite(values).all() and (values >= 0).all()
        total = sum(float(x) for x in values)
        squares = sum(float(x)*float(x) for x in values)
        numerator = sum((float(x)-float(y))**2 for i,x in enumerate(values) for y in values[i+1:])
        difference_identity = -1. if total == 0 else 100*numerator/(119*total*total)
        with np.errstate(all='ignore'):
            got = measure([total], [squares], [True])[0]
            hands = values/100
            native_hands = measure([sum(hands)], [sum(x*x for x in hands)], [True])[0]
        np.testing.assert_allclose([got, native_hands, difference_identity], row.AMVC, rtol=0, atol=2e-10)
        enc = lambda x: np.floor(100*np.asarray(x)+10000+.000001)
        np.testing.assert_array_equal(enc([got, native_hands, difference_identity]), np.repeat(enc(row.AMVC), 3))
        receipts.append(dict(date=row.date, code=row.code, source=file, sha256=manifest[file], bars=len(d)))
    assert len(receipts) == 48
    helper = INPUTS / 'morning_concentration_inputs.tdx'
    helper.write_text(HEADER+'AMVC:='+NEW_EXPRESSION+';\n')
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER)+list(EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names})
    sources = dict(json.loads((INPUTS / 'feature_report.json').read_text())['source_hashes'])
    for path in [INPUTS/'features.parquet', INPUTS/'aggregates.parquet', INPUTS/'feature_report.json', INPUTS/'feature_verification.json', helper]:
        sources[str(path)] = sha(path)
    sources.update({r['source']:r['sha256'] for r in receipts})
    save_json(INPUTS / 'complete_receipt.json', dict(
        passed=True, protocol_sha256=sha(PROTOCOL), source_hashes=sources, source_samples=receipts,
        sample_bars=5760, all_fifty_one_integer_encodings_SQL_verified=True,
        volume_hands_and_pairwise_difference_identity_verified=True,
        effective_input_intersection_unchanged=v['effective_input_intersection_unchanged'],
        no_new_raw_extraction=True, no_new_fits=True, no_new_economic_groups=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_2026_prices_read=False, no_exit_rules=True))
    return dict(passed=True, receipt_sha256=sha(INPUTS/'complete_receipt.json'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'verify', 'native'])
    args = parser.parse_args()
    print(json.dumps(globals()[args.stage](checked()), ensure_ascii=False), flush=True)
