"""Current morning-to-tail movement relative to strictly prior same-time moves."""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from .corporate_cash import save_json, sha

STEM = 'tail_formula_intraday_rank'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
WINDOWS = Path('data/research/tail_formula_morning_history')
SCHEDULE = Path('data/research/tail_formula_volume_history')
HEADER = previous.HEADER + 'ITMORN:=VALUEWHEN(TIME=1000,C);\nITQ:=INTPART(Q*100+0.5);\nITP:=INTPART(ITMORN*100+0.5);\n'
for i in range(1, 21):
    HEADER += f'ITQ{i}:=INTPART(REF(Q,B{i-1})*100+0.5);\nITP{i}:=INTPART(REF(ITMORN,B{i-1})*100+0.5);\n'
NEW_EXPRESSIONS = {
    'IT01': '100*(ITQ/MAX(ITP,1)-1)/V01',
    'IT02': '100*(' + '+'.join(f'IF(ITQ*ITP{i}>ITQ{i}*ITP,1,IF(ITQ*ITP{i}=ITQ{i}*ITP,0.5,0))' for i in range(1, 21)) + ')/20'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for folder, name, parquet in [(previous.ROOT, 'feature', 'features'), (WINDOWS, 'window', 'windows')]:
        r = json.loads((folder / (name + '_report.json')).read_text())
        v = json.loads((folder / (name + '_verification.json')).read_text())
        assert v['passed'] and v[name + '_report_sha256'] == sha(folder / (name + '_report.json'))
        assert r[parquet + '_sha256'] == sha(folder / (parquet + '.parquet'))
    m = json.loads((SCHEDULE / 'input_manifest.json').read_text())
    assert m['stock_days_sha256'] == sha(SCHEDULE / 'stock_days.parquet')
    assert (m['history_first'], m['signal_last']) == ('2023-06-01', '2025-12-30')
    assert p['history_days'] == 20 and p['new_fields'] == list(NEW_EXPRESSIONS)
    assert not p['new_2026_prices_allowed']
    return p


def rank_sequence(tail_cents, morning_cents, good, n=20):
    """Exact rational ranks, using only the preceding n original stock days."""
    q = np.asarray(tail_cents, dtype=float)
    p = np.asarray(morning_cents, dtype=float)
    good = np.asarray(good, dtype=bool) & np.isfinite(q) & np.isfinite(p) & (q > 0) & (p > 0) & (q <= 2**26) & (p <= 2**26)
    good &= (q == np.floor(q)) & (p == np.floor(p))
    count = np.zeros(len(q))
    for lag in range(1, n + 1):
        left, right = q[lag:] * p[:-lag], q[:-lag] * p[lag:]
        count[lag:] += (left > right) + .5 * (left == right)
    prior_good = pd.Series(good.astype(int)).rolling(n, min_periods=n).sum().shift().to_numpy()
    valid = good & (prior_good == n)
    return np.where(valid, 100 * count / n, np.nan)


def source_frame():
    schedule = pd.read_parquet(SCHEDULE / 'stock_days.parquet', columns=['date', 'code'])
    w = pd.read_parquet(WINDOWS / 'windows.parquet', columns=['date', 'code', 'price_1000', 'price_1449', 'morning_valid', 'tail_valid'])
    f = schedule.merge(w, on=['date', 'code'], how='left', validate='one_to_one').sort_values(['code', 'date']).reset_index(drop=True)
    assert f.date.between('2023-06-01', '2025-12-30').all()
    f['it_q'] = np.floor(100 * f.price_1449 + .5)
    f['it_p'] = np.floor(100 * f.price_1000 + .5)
    f['it_good'] = (f.morning_valid.eq(True) & f.tail_valid.eq(True) & np.isfinite(f[['it_q', 'it_p']]).all(axis=1)
                    & f.it_q.between(1, 2**26) & f.it_p.between(1, 2**26))
    return f


def features():
    p = checked_sources()
    assert not (ROOT / 'feature_report.json').exists(), 'Do not replace frozen inputs'
    pieces = []
    for _, d in source_frame().groupby('code', sort=True):
        d = d.reset_index(drop=True)
        q = d[['date', 'code', 'it_q', 'it_p', 'it_good']].copy()
        q['it_rows'] = np.minimum(np.arange(len(d)), 20)
        q['it_prior_good'] = d.it_good.astype(int).rolling(20, min_periods=1).sum().shift()
        q['it_start'] = d.date.shift(20).fillna(d.date.iloc[0]); q.loc[0, 'it_start'] = None
        q['it_end'] = d.date.shift()
        q['it_rank'] = rank_sequence(d.it_q, d.it_p, d.it_good)
        pieces.append(q.loc[q.date.ge('2024-01-01')])
    h = pd.concat(pieces, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = old.merge(h, on=['date', 'code'], how='left', validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    overlap = f.it_q.notna()
    np.testing.assert_array_equal(f.loc[overlap, 'it_q'], np.floor(100 * f.loc[overlap, 'price_1449'] + .5))
    valid = (f.it_rows.eq(20) & f.it_prior_good.eq(20) & f.it_good.eq(True) & f.it_end.lt(f.date)
             & np.isfinite(f[['it_rank', 'V01']]).all(axis=1) & f.V01.gt(0))
    f['IT01'] = (100 * (f.it_q / f.it_p - 1) / f.V01).where(valid)
    f['IT02'] = f.it_rank.where(valid)
    f['intraday_history_valid'] = valid
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True, exist_ok=True)
    h.to_parquet(ROOT / 'history.parquet', index=False, compression='zstd')
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
             features_sha256=sha(ROOT / 'features.parquet'), history_sha256=sha(ROOT / 'history.parquet'),
             rows=len(f), valid=int(f.formula_input_valid.sum()), newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
             first_history_date=f.it_start.dropna().min(), last_history_date=f.it_end.dropna().max(),
             expressions=EXPRESSIONS, native_header=HEADER, arms=p['arms'],
             native_source_parity_verified=False, software_compilation_verified=False,
             new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['source_hashes', 'expressions', 'native_header']}


def verify_features():
    checked_sources()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT / 'features.parquet')
    assert r['history_sha256'] == sha(ROOT / 'history.parquet')
    c = base.conn()
    c.read_parquet(str(SCHEDULE / 'stock_days.parquet')).create_view('schedule')
    c.read_parquet(str(WINDOWS / 'windows.parquet')).create_view('quotes')
    lagged = ','.join(f'lag(it_q,{i}) OVER w AS q{i},lag(it_p,{i}) OVER w AS p{i}' for i in range(1, 21))
    counts = '+'.join(f'(CASE WHEN it_q*p{i}>q{i}*it_p THEN 1 WHEN it_q*p{i}=q{i}*it_p THEN .5 ELSE 0 END)' for i in range(1, 21))
    h = c.sql(f'''WITH cents AS(SELECT s.date,s.code,round(q.price_1449*100) AS it_q,
        round(q.price_1000*100) AS it_p,q.morning_valid,q.tail_valid FROM schedule s LEFT JOIN quotes q USING(date,code)),
        good AS(SELECT *,coalesce(morning_valid AND tail_valid AND isfinite(it_q) AND isfinite(it_p)
            AND it_q BETWEEN 1 AND 67108864 AND it_p BETWEEN 1 AND 67108864,false) AS it_good FROM cents),
        histories AS(SELECT date,code,it_q,it_p,it_good,{lagged},count(*) OVER h AS it_rows,
            sum(it_good::INT) OVER h AS it_prior_good,min(date) OVER h AS it_start,max(date) OVER h AS it_end
            FROM good WINDOW w AS(PARTITION BY code ORDER BY date),
            h AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING))
        SELECT date,code,it_q,it_p,it_good,it_rows,it_prior_good,it_start,it_end,
            CASE WHEN it_good AND it_rows=20 AND it_prior_good=20 THEN 100*({counts})/20 ELSE NULL END AS it_rank
        FROM histories WHERE date>='2024-01-01' ORDER BY date,code''').df()
    c.close()
    actual = pd.read_parquet(ROOT / 'history.parquet')
    pd.testing.assert_frame_equal(actual, h[actual.columns], check_dtype=False, rtol=0, atol=0)
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_names=False, check_exact=True)
    e = old[['date', 'code']].merge(h, on=['date', 'code'], how='left', validate='one_to_one')
    valid = (e.it_rows.eq(20) & e.it_prior_good.eq(20) & e.it_good.eq(True) & e.it_end.lt(e.date)
             & np.isfinite(e.it_rank) & np.isfinite(old.V01) & old.V01.gt(0))
    values = np.column_stack([100 * (e.it_q - e.it_p) / e.it_p / old.V01, e.it_rank])
    values[~valid] = np.nan
    np.testing.assert_allclose(f[list(NEW_EXPRESSIONS)], values, rtol=0, atol=2e-10, equal_nan=True)
    final = old.formula_input_valid & valid & np.isfinite(values).all(axis=1)
    np.testing.assert_array_equal(f.intraday_history_valid, valid)
    np.testing.assert_array_equal(f.formula_input_valid, final)
    encode = lambda a: np.floor(np.clip(100 * a + 10000 + .000001, 0, 999999))
    np.testing.assert_array_equal(encode(values[final]), encode(f.loc[final, list(NEW_EXPRESSIONS)].to_numpy()))
    assert int(final.sum()) == r['valid'] and int((old.formula_input_valid & ~final).sum()) == r['newly_invalid']
    assert f.loc[final, 'IT02'].between(0, 100).all()
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len({name.casefold() for name in names})
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'),
                 all_20_prior_dates_exact_ranks_current_levels_validity_and_encodings_rebuilt=True,
                 all_original_48_values_and_keys_unchanged=True,
                 effective_input_intersection_unchanged=bool(np.array_equal(final, old.formula_input_valid)),
                 new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof)
    return proof


def native():
    checked_sources()
    v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    prior = json.loads((WINDOWS / 'native_input_verification.json').read_text())
    assert prior['passed'] and len(prior['cases']) == 32
    h = source_frame()
    f = pd.read_parquet(ROOT / 'features.parquet', columns=['date', 'code', 'formula_input_valid', 'V01', 'IT01', 'IT02']).set_index(['date', 'code'])
    cases = []
    for case in prior['cases']:
        row = f.loc[(case['date'], case['code'])]
        d = h.loc[h.code.eq(case['code']) & h.date.le(case['date'])].tail(21)
        record = dict(date=case['date'], code=case['code'], formula_input_valid=bool(row.formula_input_valid), source_days=len(d))
        if row.formula_input_valid:
            assert len(d) == 21 and d.it_good.all() and d.date.iloc[-1] == case['date']
            prices = [(int(q), int(p)) for q, p in zip(d.it_q, d.it_p)]
            q0, p0 = prices[-1]
            count = sum(1 if q0 * p > q * p0 else .5 if q0 * p == q * p0 else 0 for q, p in prices[:-1])
            vals = [100 * (q0 / p0 - 1) / row.V01, 100 * count / 20]
            np.testing.assert_allclose(vals, row[['IT01', 'IT02']].to_numpy(float), rtol=0, atol=2e-10)
            np.testing.assert_array_equal(np.floor(100 * np.array(vals) + 10000 + .000001),
                                          np.floor(100 * row[['IT01', 'IT02']].to_numpy(float) + 10000 + .000001))
            record.update(IT01=vals[0], IT02=vals[1], first_history_date=d.date.iloc[0])
        cases.append(record)
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
                 feature_verification_sha256=sha(ROOT / 'feature_verification.json'),
                 reused_chronological_raw_proof_sha256=sha(WINDOWS / 'native_input_verification.json'), cases=cases,
                 all_current_and_strictly_prior_native_comparisons_and_encodings_rebuilt=True,
                 original_raw_minutes_not_reextracted=True, native_source_parity_verified=False,
                 software_compilation_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', proof)
    return {k: v for k, v in proof.items() if k != 'cases'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
