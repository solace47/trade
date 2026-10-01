"""Finite volume-adaptive prior-price reference; no inventory interpretation."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from . import tail_formula_additive as base
from . import tail_formula_morning_range as prior
from .corporate_cash import save_json, sha

STEM = 'tail_formula_volume_memory'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_protocol.json')
META = prior.META
CONTROL = prior.EXPRESSIONS
EXPRESSIONS = {**CONTROL, 'VM01': '100*(Q/VMREF-1)/VP20'}
HEADER = prior.HEADER + 'VMREF:=YJVMA01.AN60#DAY;\n'
HELPER = '''PG:=C>0 AND ABS(C*100-ROUND(C*100,0))<=0.01;
VG:=V>0;
TA:=IF(COUNT(VG,20)=20,V/SUM(V,20),0.5);
EG:=DMA(IF(PG,ROUND(C*100,0),100),TA);
PM:=MULAR(1-TA,60);
READY:=BARSCOUNT(C)>=80 AND REF(COUNT(PG AND VG,79),1)=79;
AN60:IF(READY,REF((EG-REF(EG,60)*PM)/(1-PM),1)/100,DRAWNULL);
'''


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['helper_source'] == HELPER and p['native_header'] == HEADER
    assert p['arms'] == {'control': CONTROL, 'memory': EXPRESSIONS}
    assert p['volume_window'] == 20 and p['price_window'] == 60
    assert p['history_rows_required'] == 79 and p['expected_keys'] == 1815129
    assert p['history_first'] == '2022-06-01' and p['history_last'] == '2025-12-30'
    assert not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    assert sha(INPUTS / 'daily_source_manifest.json') == p['daily_source_manifest_sha256']
    return p


def original():
    # 2024 is present in both sources and must agree, rather than being
    # silently deduplicated. Historical stocks absent in 2024 are retained.
    a = pd.read_parquet('data/research/tail_formula_stock_2024/inputs/features.parquet',
                        columns=[*META, *CONTROL])
    b = pd.read_parquet(prior.INPUTS / 'features.parquet', columns=[*META, *CONTROL])
    pd.testing.assert_frame_equal(a.loc[a.date.ge('2024-01-01')].reset_index(drop=True),
                                  b.loc[b.date.lt('2025-01-01')].reset_index(drop=True), check_exact=True)
    f = pd.concat([a.loc[a.date.lt('2024-01-01')], b], ignore_index=True)
    assert len(f) == 1815129 and not f.duplicated(['date', 'code']).any()
    assert f.date.ge('2023-01-01').all() and f.date.lt('2026-01-01').all()
    return f.sort_values(['date', 'code']).reset_index(drop=True)


def daily(p):
    manifest = json.loads((INPUTS / 'daily_source_manifest.json').read_text())
    for file, digest in manifest['source_sha256'].items():
        assert sha(Path(file)) == digest, file
    c = base.conn()
    c.read_parquet(list(manifest['source_sha256'])).create_view('raw_daily')
    d = c.sql(f'''SELECT date,code,try_cast(close AS DOUBLE) AS close,try_cast(volume AS DOUBLE) AS volume,
          try_cast(preclose AS DOUBLE) AS preclose,try_cast(adjustflag AS DOUBLE) AS adjustflag
        FROM raw_daily WHERE tradestatus=1 AND date BETWEEN '{p['history_first']}' AND '{p['history_last']}'
        ORDER BY code,date''').df()
    c.close()
    assert not d.duplicated(['date', 'code']).any()
    d['good'] = (np.isfinite(d.close) & d.close.gt(0)
                 & (d.close - np.round(100 * d.close) / 100).abs().le(.0001)
                 & np.isfinite(d.volume) & d.volume.gt(0) & d.volume.eq(np.floor(d.volume))
                 & d.adjustflag.eq(3))
    return d


def finite_states(closes, volumes, good):
    """Direct finite products, newest observations are more influential.

    Every active row, including a bad one, occupies its historical position.
    Invalid rows are never skipped to find an older usable observation.
    """
    closes, volumes = np.asarray(closes, float), np.asarray(volumes, float)
    good = np.asarray(good, bool)
    n = len(closes)
    anchor, residual = np.full(n, np.nan), np.full(n, np.nan)
    valid = np.zeros(n, bool)
    if n < 79:
        return anchor, residual, valid
    # Local window sums avoid cancellation from large lifetime volumes.
    v20 = sliding_window_view(volumes, 20).sum(axis=1)
    alpha = np.full(n, np.nan)
    with np.errstate(all='ignore'):
        alpha[19:] = volumes[19:] / v20
        a = sliding_window_view(alpha[19:], 60)
        prices = sliding_window_view(np.round(100 * closes)[19:], 60) / 100
        later = np.concatenate([np.cumprod((1-a[:, 1:])[:, ::-1], axis=1)[:, ::-1],
                                np.ones((len(a), 1))], axis=1)
        weights = a * later
        value = np.sum(weights * prices, axis=1) / np.sum(weights, axis=1)
        product = np.prod(1-a, axis=1)
    ready = (sliding_window_view(good, 79).all(axis=1)
             & np.isfinite(a).all(axis=1) & (a > 0).all(axis=1) & (a < 1).all(axis=1)
             & np.isfinite(value) & (value > 0) & (product >= 0) & (product < 1))
    anchor[78:] = np.where(ready, value, np.nan)
    residual[78:] = np.where(ready, product, np.nan)
    valid[78:] = ready
    return anchor, residual, valid


def state_frame(d, independent=False):
    records = []
    for code, g in d.groupby('code', sort=False):
        g = g.reset_index(drop=True)
        if independent:
            anchor, residual, valid = native_states(g.close.to_numpy(), g.volume.to_numpy(), g.good.to_numpy())
        else:
            anchor, residual, valid = finite_states(g.close.to_numpy(), g.volume.to_numpy(), g.good.to_numpy())
        s = pd.DataFrame(dict(code=code, history_date=g.date, history_rows=np.minimum(np.arange(len(g))+1, 79),
                              anchor=anchor, residual=residual, history_input_valid=valid))
        s['first_history_date'] = g.date.shift(78)
        s['first_reference_date'] = g.date.shift(59)
        breaks = (g.preclose - g.close.shift(1)).abs().gt(.005).astype(int)
        s['reference_breaks_60'] = breaks.rolling(60, min_periods=60).sum()
        s['history_good'] = g.good.astype(int).rolling(79, min_periods=1).sum().astype(int)
        records.append(s)
    return pd.concat(records, ignore_index=True)


def map_states(keys, states):
    c = base.conn()
    c.register('keys', keys[['date', 'code']]); c.register('states', states)
    out = c.sql('''SELECT k.date,k.code,s.* EXCLUDE(code)
        FROM keys k ASOF LEFT JOIN states s ON k.code=s.code AND k.date>s.history_date
        ORDER BY k.date,k.code''').df()
    c.close()
    out['history_input_valid'] = out.history_input_valid.fillna(False).astype(bool)
    assert len(out) == len(keys) and not out.duplicated(['date', 'code']).any()
    assert out.loc[out.history_date.notna(), 'history_date'].lt(out.loc[out.history_date.notna(), 'date']).all()
    return out


def prepare():
    p = checked(); assert not (INPUTS / 'feature_report.json').exists()
    old = original(); d = daily(p)
    states = state_frame(d)
    h = map_states(old, states)
    f = old.copy(); f['prior_formula_input_valid'] = old.formula_input_valid
    f['history_input_valid'] = h.history_input_valid
    f['VM01'] = (100 * (old.A04 / h.anchor - 1) / old.V01).where(h.history_input_valid)
    f['formula_input_valid'] &= h.history_input_valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    states.to_parquet(INPUTS / 'states.parquet', index=False, compression='zstd')
    h.to_parquet(INPUTS / 'history.parquet', index=False, compression='zstd')
    f.to_parquet(INPUTS / 'features.parquet', index=False, compression='zstd')
    helper = ROOT / 'completed_daily_helper.tdx'; helper.write_text(HELPER)
    r = dict(protocol_sha256=sha(PROTOCOL), daily_source_manifest_sha256=sha(INPUTS / 'daily_source_manifest.json'),
        states_sha256=sha(INPUTS / 'states.parquet'), history_sha256=sha(INPUTS / 'history.parquet'),
        features_sha256=sha(INPUTS / 'features.parquet'), completed_helper_sha256=sha(helper),
        rows=len(f), historical_active_rows=len(d), valid=int(f.formula_input_valid.sum()),
        prior_valid=int(old.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        by_year=f.assign(year=f.date.str[:4]).groupby('year').agg(rows=('code','size'),
            prior_valid=('prior_formula_input_valid','sum'),valid=('formula_input_valid','sum')).to_dict('index'),
        valid_with_raw_reference_break=int((f.formula_input_valid & h.reference_breaks_60.gt(0)).sum()),
        first_history_date=h.first_history_date.min(), last_history_date=h.history_date.max(),
        expressions=EXPRESSIONS, native_header=HEADER, helper_source=HELPER,
        seed_independent_finite_window=True, actual_holder_cost_or_main_force_identity=False,
        bad_active_rows_retained=True, strictly_prior_asof_not_current_daily_join=True,
        matched_control_same_quality=True, no_new_labels=True, new_2026_prices_read=False,
        software_compilation_verified=False, native_source_parity_verified=False, no_exit_rules=True)
    save_json(INPUTS / 'feature_report.json', r)
    for name in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        (INPUTS / name).symlink_to((prior.INPUTS / name).resolve())
    return {k:r[k] for k in ['rows','historical_active_rows','valid','prior_valid','newly_invalid','by_year']}


def native_states(closes, volumes, good, seed=None):
    """Independent DMA recurrence and MULAR subtraction, including startup.

    Startup uses alpha=.5 and a legal dummy cent price outside valid raw
    inputs. A published finite value requires 79 good prior active rows;
    the startup state then cancels from the 60-row expression.
    """
    closes, volumes = np.asarray(closes, float), np.asarray(volumes, float)
    good = np.asarray(good, bool)
    n = len(closes)
    alpha = np.full(n, .5); ema = np.zeros(n)
    legal_v = np.isfinite(volumes) & (volumes > 0)
    cents = np.where(np.isfinite(closes) & (closes > 0), np.round(100*closes), 100)
    previous = cents[0] if seed is None and n else seed or 0.
    for t in range(n):
        if t >= 19 and legal_v[t-19:t+1].all():
            alpha[t] = volumes[t] / np.sum(volumes[t-19:t+1])
        previous = alpha[t]*cents[t] + (1-alpha[t])*previous
        ema[t] = previous
    anchor, residual = np.full(n, np.nan), np.full(n, np.nan)
    valid = np.zeros(n, bool)
    for t in range(78, n):
        if good[t-78:t+1].all() and ((alpha[t-59:t+1] > 0) & (alpha[t-59:t+1] < 1)).all():
            product = float(np.prod(1-alpha[t-59:t+1]))
            value = (ema[t]-ema[t-60]*product)/(1-product)/100
            if np.isfinite(value) and value > 0 and 0 <= product < 1:
                anchor[t], residual[t], valid[t] = value, product, True
    return anchor, residual, valid


def verify():
    p = checked(); r = json.loads((INPUTS / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL)
    for name in ['features','history','states']:
        assert r[name+'_sha256'] == sha(INPUTS / (name+'.parquet'))
    old = original(); f = pd.read_parquet(INPUTS / 'features.parquet')
    pd.testing.assert_frame_equal(f[[n for n in old if n != 'formula_input_valid']],
                                  old.drop(columns='formula_input_valid'), check_exact=True)
    np.testing.assert_array_equal(f.prior_formula_input_valid, old.formula_input_valid)
    d = daily(p)
    # Independent SQL rolling sums/quality and recurrence provide a second
    # production implementation; they do not consume saved finite weights.
    c = base.conn(); c.register('daily', d)
    q = c.sql('''WITH quality AS(SELECT *,coalesce(isfinite(close) AND close>0 AND
        abs(close-round(100*close)/100)<=.0001 AND isfinite(volume) AND volume>0
        AND volume=floor(volume) AND adjustflag=3,false) AS independent_good FROM daily)
        SELECT date,code,independent_good,count(*) OVER w AS rows79,
        sum(independent_good::INT) OVER w AS good79,sum(volume) OVER v AS sum20 FROM quality
        WINDOW w AS(PARTITION BY code ORDER BY date ROWS BETWEEN 78 PRECEDING AND CURRENT ROW),
        v AS(PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
        ORDER BY code,date''').df(); c.close()
    production = pd.read_parquet(INPUTS / 'states.parquet')
    pd.testing.assert_frame_equal(production[['history_date','code']].rename(columns={'history_date':'date'}),
                                  q[['date','code']], check_exact=True)
    np.testing.assert_array_equal(production.history_rows, q.rows79)
    np.testing.assert_array_equal(production.history_good, q.good79)
    np.testing.assert_array_equal(d.good, q.independent_good)
    independent_daily = d.copy(); independent_daily['good'] = q.independent_good
    rebuilt = state_frame(independent_daily, independent=True)
    np.testing.assert_array_equal(production.history_input_valid, rebuilt.history_input_valid)
    np.testing.assert_allclose(production.anchor, rebuilt.anchor, rtol=0, atol=2e-9, equal_nan=True)
    np.testing.assert_allclose(production.residual, rebuilt.residual, rtol=0, atol=2e-15, equal_nan=True)
    h = map_states(old, rebuilt)
    expected = (100*(old.A04/h.anchor-1)/old.V01).where(h.history_input_valid)
    np.testing.assert_allclose(f.VM01, expected, rtol=0, atol=2e-8, equal_nan=True)
    enc = lambda x: np.floor(np.clip(100*x+10000+.000001,0,999999))
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(enc(f.loc[finite,'VM01']), enc(expected[finite]))
    expected_valid = old.formula_input_valid & h.history_input_valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    np.testing.assert_array_equal(f.formula_input_valid, expected_valid)
    saved_h = pd.read_parquet(INPUTS / 'history.parquet')
    for column in ['date','code','history_date','first_history_date','first_reference_date',
                   'history_rows','history_good','history_input_valid','reference_breaks_60']:
        pd.testing.assert_series_equal(saved_h[column], h[column], check_exact=True)
    # Changing any current/future state cannot affect a signal's strict
    # historical lookup. Actual-source comparisons retain every original key.
    sample_keys = old[['date','code']].iloc[np.linspace(0,len(old)-1,48).astype(int)]
    checks = []
    for row in sample_keys.itertuples(index=False):
        g = d.loc[d.code.eq(row.code) & d.date.lt(row.date)]
        if len(g) < 79 or not g.good.iloc[-79:].all():
            continue
        a,_,ready = finite_states(g.close, g.volume, g.good)
        for seed in [0., 100., 1000000.]:
            b,_,valid = native_states(g.close, g.volume, g.good, seed=seed)
            assert ready[-1] and valid[-1]
            np.testing.assert_allclose(a[-1], b[-1], rtol=0, atol=2e-9)
        checks.append(dict(date=row.date,code=row.code,prior_last_date=g.date.iloc[-1]))
    assert len(checks) >= 24
    proof = dict(passed=True, feature_report_sha256=sha(INPUTS / 'feature_report.json'),
        rows=len(f), valid=int(expected_valid.sum()), all_original_keys_and_50_values_exact=True,
        all_states_independent_dma_mular_reconstruction=True, sql_history_counts_and_quality_equal=True,
        all_new_integer_encodings_equal=True, strict_prior_asof_dates_equal=True,
        arbitrary_seed_checks=checks, matched_control_validity_exact=True,
        current_and_future_daily_prices_not_used=True, no_new_training_or_evaluation_labels=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'feature_verification.json', proof)
    save_json(INPUTS / 'native_input_verification.json', dict(**proof,
        mathematical_replay_only=True, software_compilation_verified=False,
        native_source_parity_verified=False, official_source_sha256=p['official_source_sha256']))
    return {k:proof[k] for k in ['passed','rows','valid','all_new_integer_encodings_equal']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare','verify'])
    a = parser.parse_args()
    print(json.dumps(globals()[a.stage](), ensure_ascii=False, indent=2), flush=True)
