"""Two early index observations added to the unchanged original fifty inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_morning_range as prior
from . import tail_formula_context as source
from .corporate_cash import save_json, sha

STEM = 'tail_formula_index_morning'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
INTENT = Path('config') / (STEM + '_intent.json')
META = prior.META
EXTRA_HEADER = '''IQ31:=VALUEWHEN(TIME=931,INDEXC);
IQ59:=VALUEWHEN(TIME=959,INDEXC);
IQD31:=VALUEWHEN(TIME=931,DATE);
IQD59:=VALUEWHEN(TIME=959,DATE);
IQREADY:=AMREADY AND RTREADY AND IQD31=DATE AND IQD59=DATE AND IQ31>0 AND IQ59>0 AND ICP1>0;
'''
HEADER = prior.HEADER + EXTRA_HEADER
CORE_GATE = 'IQREADY'
NEW_EXPRESSIONS = {'IM01': 'IF(IQREADY,100*(IQ59/IQ31-1),DRAWNULL)',
                   'IM02': 'IF(IQREADY,100*(IQ59/ICP1-1),DRAWNULL)'}
ARMS = {'control': prior.EXPRESSIONS, 'early': {**prior.EXPRESSIONS, **NEW_EXPRESSIONS}}


def checked():
    prior.checked(); p = json.loads(PROTOCOL.read_text())
    assert p['intent_sha256'] == sha(INTENT) and p['arms'] == ARMS and p['native_header'] == HEADER
    assert p['expected_keys'] == 1258085 and p['expected_valid'] == 1117397
    assert p['no_new_raw_extraction'] and not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items(): assert sha(Path(file)) == digest, file
    complete = json.loads(Path(p['conditional_completion']).read_text())
    gate = json.loads(Path(p['conditional_gate']).read_text())
    assert complete['passed'] and gate['passed'] and not gate['supports_2024_extension']
    sr = json.loads((source.ROOT / 'source_report.json').read_text())
    assert len(sr['sessions']) == 968
    for item in sr['sessions']:
        assert '2024-01-01' <= item['date'] < '2026-01-01'
        if 'path' in item: assert sha(Path(item['path'])) == item['sha256']
    fr = json.loads((source.ROOT / 'feature_report.json').read_text())
    fv = json.loads((source.ROOT / 'feature_verification.json').read_text())
    assert fv['passed'] and fv['feature_report_sha256'] == sha(source.ROOT / 'feature_report.json')
    assert fr['features_sha256'] == sha(source.ROOT / 'features.parquet')
    return p


def measure(first, end, reference, parent_valid):
    assert first.shape == end.shape == reference.shape == parent_valid.shape
    good = parent_valid.copy() & np.isfinite(first) & np.isfinite(end) & np.isfinite(reference)
    good &= (first > 0) & (end > 0) & (reference > 0)
    with np.errstate(all='ignore'):
        return dict(index_morning_input_valid=good,
            IM01=np.where(good, 100*(end/first-1), np.nan),
            IM02=np.where(good, 100*(end/reference-1), np.nan))


def raw_sessions():
    sr = json.loads((source.ROOT / 'source_report.json').read_text())
    for item in sr['sessions']:
        # Only the already visible prefix is interpreted, never the closing suffix.
        rows = json.loads(Path(item['path']).read_text())[:228] if 'path' in item else []
        yield item, rows


def features():
    checked(); INPUTS.mkdir(parents=True, exist_ok=True)
    assert not (INPUTS / 'feature_report.json').exists()
    old = pd.read_parquet(prior.INPUTS / 'features.parquet')
    context = pd.read_parquet(source.ROOT / 'features.parquet', columns=[
        'date', 'code', 'index_code', 'index_prior_close', 'prefix_valid', 'J01'])
    points = []
    for item, rows in raw_sessions():
        good = source.points(rows)['prefix_valid']
        good &= all(np.isfinite(r['price_raw']) for r in rows)
        points.append(dict(date=item['date'], index_code=item['symbol'], iq_prefix_valid=good,
            iq31=rows[0]['price_raw']/100 if good else np.nan,
            iq59=rows[28]['price_raw']/100 if good else np.nan))
    f = old.merge(context.rename(columns={'J01': 'iq_original_J01'}), on=['date', 'code'], how='left', validate='one_to_one')
    f = f.merge(pd.DataFrame(points), on=['date', 'index_code'], how='left', validate='many_to_one', sort=False)
    pd.testing.assert_frame_equal(f[old.columns], old, check_exact=True)
    np.testing.assert_array_equal(f.J01, f.iq_original_J01)
    expected_codes = np.where(f.code.str.startswith('sh.'), 'sh.000001', 'sz.399001')
    np.testing.assert_array_equal(f.index_code, expected_codes)
    np.testing.assert_array_equal(f.prefix_valid, f.iq_prefix_valid)
    result = measure(f.iq31.to_numpy(float), f.iq59.to_numpy(float), f.index_prior_close.to_numpy(float), f.iq_prefix_valid.eq(True).to_numpy())
    for name, value in result.items(): f[name] = value
    f['formula_input_valid'] &= f.index_morning_input_valid
    np.testing.assert_array_equal(f.formula_input_valid, old.formula_input_valid)
    assert len(f) == 1258085 and f.formula_input_valid.sum() == 1117397
    f.to_parquet(INPUTS / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), features_sha256=sha(INPUTS / 'features.parquet'), rows=len(f),
        valid=int(f.formula_input_valid.sum()), parent_feature_report_sha256=sha(prior.INPUTS / 'feature_report.json'),
        all_original_50_values_metadata_keys_and_effective_domain_unchanged=True,
        original_index_mapping_reference_J01_and_prefix_validity_exactly_matched=True,
        expressions=ARMS['early'], native_header=HEADER, no_new_raw_extraction=True,
        no_future_index_suffix_interpreted=True, new_group_outcomes_read=False, new_2026_prices_read=False,
        no_exit_rules=True, software_compilation_verified=False, native_source_parity_verified=False)
    save_json(INPUTS / 'feature_report.json', r)
    for name in ['full_labels.parquet', 'full_label_report.json', 'full_label_verification.json']:
        (INPUTS / name).symlink_to((prior.INPUTS / name).resolve())
    return {k:r[k] for k in ['rows', 'valid', 'all_original_50_values_metadata_keys_and_effective_domain_unchanged']}


def verify():
    checked(); r = json.loads((INPUTS / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(INPUTS / 'features.parquet')
    f = pd.read_parquet(INPUTS / 'features.parquet'); old = pd.read_parquet(prior.INPUTS / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns], old, check_exact=True)
    contexts = pd.read_parquet(source.ROOT / 'features.parquet', columns=[
        'date', 'code', 'index_code', 'index_prior_close', 'prefix_valid', 'J01'])
    atoms, sessions = [], []
    for item, rows in raw_sessions():
        sessions.append(dict(date=item['date'], index_code=item['symbol']))
        atoms.extend(dict(date=item['date'], index_code=item['symbol'], ordinal=i,
            sequence=row['sequence'], price_raw=row['price_raw']) for i, row in enumerate(rows))
    c = base.conn(); c.register('atoms', pd.DataFrame(atoms)); c.register('sessions', pd.DataFrame(sessions))
    c.register('parent', old); c.register('contexts', contexts)
    c.sql('''WITH p AS(SELECT date,index_code,count(*)=228 AND
        bool_and(ordinal=sequence AND isfinite(price_raw) AND price_raw>0) AS good,
        max(price_raw) FILTER(WHERE ordinal=0)/100. AS p31,
        max(price_raw) FILTER(WHERE ordinal=28)/100. AS p59
        FROM atoms GROUP BY date,index_code) SELECT s.*,coalesce(good,false) AS good,p31,p59
        FROM sessions s LEFT JOIN p USING(date,index_code)''').create_view('points')
    d = c.sql('''WITH z AS(SELECT f.date,f.code,c.index_prior_close,c.J01 AS original_J01,
        c.prefix_valid,p.good,c.index_code,
        coalesce(p.good AND isfinite(p31) AND p31>0 AND isfinite(p59) AND p59>0
            AND isfinite(index_prior_close) AND index_prior_close>0,false) AS ok,p31,p59
        FROM parent f LEFT JOIN contexts c USING(date,code)
        LEFT JOIN points p ON f.date=p.date AND c.index_code=p.index_code)
        SELECT *,CASE WHEN ok THEN 100.*(p59/p31-1) END AS IM01,
        CASE WHEN ok THEN 100.*(p59/index_prior_close-1) END AS IM02 FROM z ORDER BY date,code''').df()
    c.close()
    pd.testing.assert_frame_equal(f[['date', 'code']], d[['date', 'code']], check_exact=True)
    np.testing.assert_array_equal(f.index_prior_close, d.index_prior_close)
    np.testing.assert_array_equal(f.J01, d.original_J01)
    np.testing.assert_array_equal(d.prefix_valid, d.good)
    np.testing.assert_array_equal(f.index_morning_input_valid, d.ok)
    maximum = 0.; encode = lambda x: np.floor(np.clip(100*x+10000+.000001,0,999999))
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name], d[name], atol=2e-12, rtol=0, equal_nan=True)
        np.testing.assert_array_equal(encode(f.loc[d.ok,name]), encode(d.loc[d.ok,name]))
        maximum = max(maximum, float((f[name]-d[name]).abs().max()))
    np.testing.assert_array_equal(f.formula_input_valid, old.formula_input_valid & d.ok)
    out = dict(passed=True, feature_report_sha256=sha(INPUTS / 'feature_report.json'), rows=len(f),
        valid=int(f.formula_input_valid.sum()), max_difference=maximum,
        all_original_50_values_metadata_and_effective_domain_unchanged=True,
        all_index_prefix_quality_points_references_mappings_values_and_codes_sql_rebuilt=True,
        effective_input_intersection_unchanged=True, domain_reference=str(prior.INPUTS),
        no_future_index_suffix_interpreted=True, new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'feature_verification.json', out); return out


def native():
    checked(); proof = json.loads((INPUTS / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(INPUTS / 'feature_report.json')
    f = pd.read_parquet(INPUTS / 'features.parquet')
    good = f.index_morning_input_valid
    # Replay the two literal ratios for every row, retaining missing references.
    for name, divisor in [('IM01', f.iq31), ('IM02', f.index_prior_close)]:
        expected = (100*(f.iq59/divisor-1)).where(good)
        np.testing.assert_allclose(f[name], expected, atol=2e-12, rtol=0, equal_nan=True)
        enc = lambda x: np.floor(np.clip(100*x+10000+.000001,0,999999))
        np.testing.assert_array_equal(enc(f.loc[good,name]), enc(expected.loc[good]))
    assert 'VALUEWHEN(TIME=931,INDEXC)' in EXTRA_HEADER and 'VALUEWHEN(TIME=959,INDEXC)' in EXTRA_HEADER
    assert 'IQD31=DATE AND IQD59=DATE' in EXTRA_HEADER
    identity = json.loads((prior.INPUTS / 'native_input_verification.json').read_text())['source_samples']
    keys = list(dict.fromkeys((x['date'],x['code']) for x in identity)); indexed=f.set_index(['date','code'])
    sessions = {(item['date'],item['symbol']):(item,rows) for item,rows in raw_sessions()}; cases=[]
    for day,code in keys:
        row=indexed.loc[(day,code)]; item,rows=sessions[(day,row.index_code)]
        ready = source.points(rows)['prefix_valid'] and np.isfinite(row.index_prior_close) and row.index_prior_close>0
        assert ready == row.index_morning_input_valid
        first,end=(rows[0]['price_raw']/100,rows[28]['price_raw']/100) if ready else (np.nan,np.nan)
        got=[100*(end/first-1),100*(end/row.index_prior_close-1)] if ready else [np.nan,np.nan]
        np.testing.assert_allclose(got,row[list(NEW_EXPRESSIONS)].to_numpy(float),atol=2e-12,rtol=0)
        cases.append(dict(date=day,code=code,index_code=row.index_code,index_source_sha256=item['sha256'],
            existing_stock_source_identity_reused=True,literal_index_points_and_reference_replayed=True))
    helper=INPUTS/'native_inputs.tdx';helper.write_text(HEADER+'\n'.join(f'{k}:={v};' for k,v in NEW_EXPRESSIONS.items())+'\n')
    out=dict(passed=True,feature_report_sha256=sha(INPUTS / 'feature_report.json'),helper_sha256=sha(helper),
        feature_verification_sha256=sha(INPUTS / 'feature_verification.json'),samples=cases,
        all_new_full_literal_values_and_codes_checked=True,fixed_old_case_identities_only=True,
        no_future_index_suffix_interpreted=True,no_new_raw_extraction=True,
        software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS / 'native_input_verification.json',out);return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['features','verify','native']);args=p.parse_args()
    print(json.dumps(globals()[args.stage](),ensure_ascii=False,indent=2))
