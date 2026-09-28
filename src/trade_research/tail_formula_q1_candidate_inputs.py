"""Bounded Q1 extensions; unchanged 48 inputs are inherited from their prior audit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_daily_efficiency as paths
from . import tail_formula_q1_candidates as study
from .corporate_cash import DAILY, MINUTES, save_json, sha
from .turnover_reference import CALENDAR

ROOT = study.ROOT/'inputs'
OLD = study.OLD/'inputs'
PATH_COLS = paths.PRICE_COLUMNS
STATE_COLS = ['date', 'code', 'isST', 'tradestatus', 'listing_age_sessions', 'preclose', 'open_1450']


def checked_sources():
    p, gate = study.checked_models()
    r = json.loads((OLD/'feature_report.json').read_text())
    v = json.loads((OLD/'feature_verification.json').read_text())
    b = json.loads((OLD/'base_report.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(OLD/'feature_report.json')
    assert r['features_sha256'] == sha(OLD/'features.parquet')
    assert b['prefix_sha256'] == sha(OLD/'prefix.parquet') and b['universe_sha256'] == sha(OLD/'universe.parquet')
    assert b['source_manifest_sha256'] == sha(OLD/'source_manifest.json')
    manifest = json.loads((OLD/'source_manifest.json').read_text())
    assert manifest['calendar_sha256'] == sha(CALENDAR)
    return p, gate, manifest


def active_daily(files):
    c = base.conn(); c.read_parquet(files).create_view('source_daily')
    # The lag construction below prevents any signal-day close from entering
    # that day's inputs, including when all quarter dates are processed together.
    c.sql('''SELECT date,code,close::DOUBLE AS raw_close,round(close::DOUBLE,2) AS cl,
        preclose::DOUBLE AS pc,adjustflag::DOUBLE AS adj FROM source_daily
        WHERE tradestatus=1 AND date BETWEEN '2025-01-01' AND '2026-03-31' ''').create_view('active')
    return c


def build_path_history(files):
    c = active_daily(files)
    lags = ','.join(f'lag(cl,{n}) OVER w AS de_c{n:02d}' for n in range(1, 22))
    h = c.sql(f'''WITH atoms AS(SELECT *,coalesce(isfinite(raw_close) AND raw_close>0
        AND abs(raw_close-cl)<=.0001 AND adj=3,false) AS good,
        coalesce(abs(pc-lag(cl) OVER(PARTITION BY code ORDER BY date))>.005,false) AS reference_break FROM active),
        h AS(SELECT date,code,{lags},count(*) OVER hw AS de_rows,sum(good::INT) OVER hw AS de_good,
            min(date) OVER hw AS de_first_date,max(date) OVER hw AS de_last_date,
            sum(reference_break::INT) OVER h20 AS de_reference_breaks
        FROM atoms WINDOW w AS(PARTITION BY code ORDER BY date),
            hw AS(PARTITION BY code ORDER BY date ROWS BETWEEN 21 PRECEDING AND 1 PRECEDING),
            h20 AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING))
        SELECT * FROM h WHERE date>='2026-01-01' ORDER BY date,code''').df(); c.close()
    return h


def member_tables(manifest):
    for file, digest in manifest['snapshot_sha256'].items():
        assert sha(Path(file)) == digest
    c = base.conn(); c.read_parquet(str(OLD/'prefix.parquet')).create_view('prefix')
    c.read_parquet(list(manifest['snapshot_sha256'])).create_view('snapshots')
    c.sql('''SELECT date,code,isST,tradestatus,listing_age_sessions,preclose,open_1450
        FROM snapshots WHERE date BETWEEN '2026-01-01' AND '2026-03-31'
        AND (code LIKE 'sh.60%' OR code LIKE 'sz.00%') AND isST=0 AND tradestatus=1
        AND listing_age_sessions>=20 AND preclose>0 AND open_1450>0''').create_view('states')
    m = c.sql('''SELECT p.date,p.code,round(p.price_1420*100)::BIGINT AS p20,
        round(p.price_1449*100)::BIGINT AS p49,round(s.preclose*100)::BIGINT AS pc
        FROM prefix p JOIN states s USING(date,code)
        WHERE NOT p.quote_outside_traded_range AND p.amount_1449>0 AND p.volume_1449>0
        AND p.price_1449>0 AND p.price_1420>0 AND abs(p.price_1449-round(p.price_1449,2))<=.0001
        ORDER BY p.date,p.code''').df()
    coverage = c.sql('SELECT date,count(*) AS expected_state_members FROM states GROUP BY date ORDER BY date').df(); c.close()
    assert not m.duplicated(['date', 'code']).any()
    m['ew_day_atom'] = 100*(m.p49-m.pc)/m.pc
    m['ew_tail_atom'] = 100*(m.p49-m.p20)/m.p20
    daily = m.groupby('date', sort=True).agg(ew_members=('code', 'size'),
        ew_day_mean=('ew_day_atom', 'mean'), ew_tail_mean=('ew_tail_atom', 'mean')).reset_index()
    daily = daily.merge(coverage, on='date', how='outer', validate='one_to_one').sort_values('date').reset_index(drop=True)
    daily['missing_state_members'] = daily.expected_state_members-daily.ew_members
    assert daily.missing_state_members.ge(0).all()
    daily['equal_weight_valid'] = daily.ew_members.ge(2000)
    return m, daily


def features():
    p, gate, manifest = checked_sources(); assert not (ROOT/'feature_report.json').exists()
    old = pd.read_parquet(OLD/'features.parquet')
    assert old.date.between(p['signal_first'], p['signal_last']).all()
    assert (len(old), int(old.formula_input_valid.sum()), old.date.nunique()) == (161781, 139353, 56)
    f = old.copy(); ROOT.mkdir(parents=True, exist_ok=True)
    artifacts = {}; consumed_daily = {}; counts = {}
    if 'path' in gate['active_variants']:
        for code in sorted(old.code.unique()):
            file = str(DAILY/(code.replace('.', '_')+'.parquet'))
            assert file in manifest['daily_sha256'] and sha(Path(file)) == manifest['daily_sha256'][file]
            consumed_daily[file] = manifest['daily_sha256'][file]
        h = build_path_history(list(consumed_daily))
        f = f.merge(h, on=['date', 'code'], how='left', validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)
        prices = f[PATH_COLS].to_numpy(float)
        valid = (f.de_rows.eq(21) & f.de_good.eq(21) & f.de_last_date.lt(f.date)
            & np.isfinite(prices).all(axis=1) & (prices>0).all(axis=1))
        for n in [5, 20]:
            f[f'DE{n:02d}'] = pd.Series(paths.efficiency(prices, n)).where(valid)
        f['path_input_valid'] = old.formula_input_valid & valid
        cal = pd.read_parquet(CALENDAR)
        dates = sorted(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between('2025-01-01', '2026-03-31'), 'calendar_date'])
        rank = {d:i for i,d in enumerate(dates)}
        f['de_market_span'] = f.de_last_date.map(rank)-f.de_first_date.map(rank)+1
        f['de_last_gap'] = f.date.map(rank)-f.de_last_date.map(rank)
        h.to_parquet(ROOT/'path_history.parquet', index=False, compression='zstd')
        artifacts['path_history.parquet'] = sha(ROOT/'path_history.parquet')
        counts['path'] = dict(valid=int(f.path_input_valid.sum()), newly_invalid=int((old.formula_input_valid & ~f.path_input_valid).sum()),
            reference_breaks=int((f.path_input_valid & f.de_reference_breaks.gt(0)).sum()),
            stock_day_gaps=int((f.path_input_valid & (f.de_market_span.gt(21) | f.de_last_gap.gt(1))).sum()))
    if 'equal_weight' in gate['active_variants']:
        m, daily = member_tables(manifest)
        f = f.merge(daily, on='date', how='left', validate='many_to_one').sort_values(['date', 'code']).reset_index(drop=True)
        valid = f.equal_weight_valid.fillna(False) & f.V01.gt(0)
        f['EW01'] = ((f.A01-f.ew_day_mean)/f.V01).where(valid)
        f['EW02'] = ((f.A05-f.ew_tail_mean)/f.V01).where(valid)
        f['equal_weight_input_valid'] = old.formula_input_valid & valid & np.isfinite(f[['EW01', 'EW02']]).all(axis=1)
        for name, table in [('members', m), ('market_reference', daily)]:
            table.to_parquet(ROOT/(name+'.parquet'), index=False, compression='zstd')
            artifacts[name+'.parquet'] = sha(ROOT/(name+'.parquet'))
        counts['equal_weight'] = dict(valid=int(f.equal_weight_input_valid.sum()),
            newly_invalid=int((old.formula_input_valid & ~f.equal_weight_input_valid).sum()),
            members=len(m), days=len(daily), minimum_members=int(daily.ew_members.min()),
            maximum_members=int(daily.ew_members.max()), missing_state_members=int(daily.missing_state_members.sum()))
    pd.testing.assert_frame_equal(f[old.columns], old, check_exact=True)
    f.to_parquet(ROOT/'features.parquet', index=False, compression='zstd'); artifacts['features.parquet'] = sha(ROOT/'features.parquet')
    r = dict(protocol_sha256=sha(study.PROTOCOL), models_gate_sha256=sha(study.ROOT/'models_gate.json'),
        old_feature_report_sha256=sha(OLD/'feature_report.json'), old_feature_verification_sha256=sha(OLD/'feature_verification.json'),
        source_manifest_sha256=sha(OLD/'source_manifest.json'), consumed_daily_sha256=consumed_daily,
        artifacts_sha256=artifacts, variants=counts, rows=len(f), original_valid=int(old.formula_input_valid.sum()),
        original_48_and_keys_unchanged=True, q1_input_values_read=True, q1_new_group_outcomes_read=False,
        no_q2_signal_prices_read=True, source_hashing_is_not_out_of_scope_price_evaluation=True,
        native_source_parity_verified=False, no_exit_rules=True)
    save_json(ROOT/'feature_report.json', r); return {k:v for k,v in r.items() if k != 'consumed_daily_sha256'}


def independent_path_history(files):
    histories = []
    for file in files:
        d = pd.read_parquet(file, columns=['date', 'code', 'close', 'preclose', 'adjustflag', 'tradestatus'],
            filters=[('date', '>=', '2025-01-01'), ('date', '<=', '2026-03-31')])
        d = d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True)
        cl = d.close.astype(float).round(2)
        raw_close = d.close.astype(float)
        good = np.isfinite(raw_close) & raw_close.gt(0) & (raw_close-cl).abs().le(.0001) & d.adjustflag.eq(3)
        h = d[['date', 'code']].copy()
        for i, name in enumerate(PATH_COLS, 1):
            h[name] = cl.shift(i)
        h['de_rows'] = pd.Series(np.minimum(np.arange(len(d)), 21), index=d.index)
        h['de_good'] = good.astype(int).shift(1).rolling(21, min_periods=1).sum()
        h['de_first_date'] = [d.date.iloc[max(0, i-21)] if i else None for i in range(len(d))]
        h['de_last_date'] = d.date.shift(1)
        breaks = (d.preclose.astype(float)-cl.shift(1)).abs().gt(.005)
        h['de_reference_breaks'] = breaks.astype(int).shift(1).rolling(20, min_periods=1).sum()
        histories.append(h.loc[h.date.ge('2026-01-01')])
    return pd.concat(histories, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)


def verify_features():
    p, gate, manifest = checked_sources(); r = json.loads((ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(study.PROTOCOL) and r['models_gate_sha256'] == sha(study.ROOT/'models_gate.json')
    for file, digest in r['artifacts_sha256'].items():
        assert sha(ROOT/file) == digest
    old = pd.read_parquet(OLD/'features.parquet'); f = pd.read_parquet(ROOT/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns], old, check_exact=True)
    checked_names = []
    if 'path' in gate['active_variants']:
        for file, digest in r['consumed_daily_sha256'].items():
            assert sha(Path(file)) == digest == manifest['daily_sha256'][file]
        h = independent_path_history(r['consumed_daily_sha256'])
        got = pd.read_parquet(ROOT/'path_history.parquet')
        pd.testing.assert_frame_equal(got[h.columns], h, check_dtype=False, check_exact=True)
        aligned = old[['date', 'code']].merge(h, on=['date', 'code'], how='left', validate='one_to_one')
        prices = aligned[PATH_COLS].to_numpy(float)
        valid = aligned.de_rows.eq(21) & aligned.de_good.eq(21) & aligned.de_last_date.lt(aligned.date)
        for n in [5, 20]:
            # Alternate up/down-sum identity, using integer cents throughout.
            a = np.rint(prices[:, :n+1]*100); changes = a[:, :-1]-a[:, 1:]
            up = np.maximum(changes, 0).sum(axis=1); down = np.maximum(-changes, 0).sum(axis=1)
            wanted = np.divide(100*(up-down), up+down, out=np.zeros(len(a)), where=(up+down)>0)
            wanted = pd.Series(wanted).where(valid & np.isfinite(a).all(axis=1) & (a>0).all(axis=1))
            np.testing.assert_allclose(f[f'DE{n:02d}'], wanted, rtol=0, atol=2e-10, equal_nan=True)
            enc = lambda x: np.floor(np.clip(100*x+10000+.000001, 0, 999999))
            np.testing.assert_array_equal(enc(f.loc[f.path_input_valid, f'DE{n:02d}']), enc(wanted[f.path_input_valid]))
            checked_names.append(f'DE{n:02d}')
        np.testing.assert_array_equal(f.path_input_valid, old.formula_input_valid & valid)
    if 'equal_weight' in gate['active_variants']:
        for file, digest in manifest['snapshot_sha256'].items():
            assert sha(Path(file)) == digest
        states = pd.concat([pd.read_parquet(file, columns=STATE_COLS,
            filters=[('date', '>=', '2026-01-01'), ('date', '<=', '2026-03-31')])
            for file in manifest['snapshot_sha256']], ignore_index=True)
        states = states.loc[states.code.str.startswith(('sh.60', 'sz.00')) & states.isST.eq(0)
            & states.tradestatus.eq(1) & states.listing_age_sessions.ge(20) & states.preclose.gt(0) & states.open_1450.gt(0)]
        assert not states.duplicated(['date', 'code']).any()
        prefix = pd.read_parquet(OLD/'prefix.parquet')
        m = prefix.merge(states, on=['date', 'code'], validate='one_to_one')
        m = m.loc[~m.quote_outside_traded_range & m.amount_1449.gt(0) & m.volume_1449.gt(0)
            & m.price_1449.gt(0) & m.price_1420.gt(0) & (m.price_1449-m.price_1449.round(2)).abs().le(.0001)]
        m = m.sort_values(['date', 'code']).reset_index(drop=True)
        for column, source in [('p20', 'price_1420'), ('p49', 'price_1449'), ('pc', 'preclose')]:
            m[column] = np.rint(m[source]*100).astype('int64')
        actual = pd.read_parquet(ROOT/'members.parquet')
        pd.testing.assert_frame_equal(actual[['date', 'code', 'p20', 'p49', 'pc']], m[['date', 'code', 'p20', 'p49', 'pc']], check_exact=True)
        m['day'] = 100*(m.p49/m.pc-1); m['tail'] = 100*(m.p49/m.p20-1)
        d = m.groupby('date', sort=True).agg(ew_members=('code', 'size'), ew_day_mean=('day', 'mean'), ew_tail_mean=('tail', 'mean')).reset_index()
        d['expected_state_members'] = d.date.map(states.groupby('date').size())
        d['missing_state_members'] = d.expected_state_members-d.ew_members
        d['equal_weight_valid'] = d.ew_members.ge(2000)
        actual = pd.read_parquet(ROOT/'market_reference.parquet')
        pd.testing.assert_frame_equal(actual[d.columns], d, check_dtype=False, rtol=0, atol=2e-12)
        reference = old[['date', 'code', 'V01']].merge(d, on='date', validate='many_to_one').merge(m[['date', 'code', 'day', 'tail']], on=['date', 'code'], how='left', validate='one_to_one')
        pd.testing.assert_frame_equal(reference[['date', 'code']], old[['date', 'code']], check_exact=True)
        valid = reference.equal_weight_valid & reference.V01.gt(0)
        for name, own, peer in [('EW01', 'day', 'ew_day_mean'), ('EW02', 'tail', 'ew_tail_mean')]:
            wanted = ((reference[own]-reference[peer])/reference.V01).where(valid)
            np.testing.assert_allclose(f[name], wanted, rtol=0, atol=2e-10, equal_nan=True)
            enc = lambda x: np.floor(np.clip(100*x+10000+.000001, 0, 999999))
            np.testing.assert_array_equal(enc(f.loc[f.equal_weight_input_valid, name]), enc(wanted[f.equal_weight_input_valid]))
            checked_names.append(name)
        np.testing.assert_array_equal(f.equal_weight_input_valid, old.formula_input_valid & valid)
    assert f.date.between('2026-01-01', '2026-03-31').all()
    proof = dict(passed=True, feature_report_sha256=sha(ROOT/'feature_report.json'), rows=len(f),
        original_48_fields_and_keys_exactly_unchanged=True, independently_rebuilt_inputs=checked_names,
        all_new_validity_and_integer_encodings_rebuilt=True, q1_input_values_read=True,
        q1_new_group_outcomes_read=False, no_q2_signal_prices_read=True, native_source_parity_verified=False, no_exit_rules=True)
    save_json(ROOT/'feature_verification.json', proof); return proof


def native():
    p, gate, manifest = checked_sources()
    proof = json.loads((ROOT/'feature_verification.json').read_text())
    report = json.loads((ROOT/'feature_report.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(ROOT/'feature_report.json')
    assert report['artifacts_sha256']['features.parquet'] == sha(ROOT/'features.parquet')
    f = pd.read_parquet(ROOT/'features.parquet')
    valid = f.formula_input_valid.copy()
    for variant in gate['active_variants']:
        if variant != 'control':
            valid &= f[variant+'_input_valid']
    f = f.loc[valid].copy()
    f['month'] = f.date.str[:7]
    f['sample_hash'] = [hashlib.sha256((d+'|'+s+'|q1-candidate-native-v1').encode()).hexdigest()
        for d,s in zip(f.date, f.code)]
    sample = f.sort_values('sample_hash').groupby('month', sort=True).head(8).sort_values(['date', 'code'])
    assert len(sample) == 24 and sample.groupby('month').size().eq(8).all()
    receipts = []; total = 0
    for row in sample.itertuples():
        daily = DAILY/(row.code.replace('.', '_')+'.parquet')
        minute = MINUTES/row.code[:2].upper()/(row.code[3:]+'.parquet')
        assert sha(daily) == manifest['daily_sha256'][str(daily)]
        assert sha(minute) == manifest['minute_sha256'][str(minute)]
        first = row.date
        if 'path' in gate['active_variants']:
            d = pd.read_parquet(daily, columns=['date', 'close', 'tradestatus'],
                filters=[('date', '>=', p['warmup_first']), ('date', '<', row.date)])
            days = d.loc[d.tradestatus.eq(1)].sort_values('date').tail(21)
            assert len(days) == 21
            first = days.date.iloc[0]
        q = pd.read_parquet(minute, columns=['timestamp', 'close', 'volume', 'turnover'], filters=[
            ('timestamp', '>=', pd.Timestamp(first)), ('timestamp', '<=', pd.Timestamp(row.date+' 14:49'))])
        q['date'] = q.timestamp.dt.strftime('%Y-%m-%d')
        if 'path' in gate['active_variants']:
            q = q.loc[q.date.isin(days.date.tolist()+[row.date])]
        q = q.sort_values('timestamp')
        assert not q.timestamp.duplicated().any() and q.timestamp.max() == pd.Timestamp(row.date+' 14:49')
        current = q.loc[q.date.eq(row.date)]
        assert len(current) == 230
        values = {}; enc = lambda x: np.floor(np.clip(100*np.asarray(x)+10000+.000001, 0, 999999))
        if 'path' in gate['active_variants']:
            prior = q.loc[q.date.lt(row.date)]
            assert prior.groupby('date').size().eq(241).all()
            ends = prior.groupby('date', sort=True).tail(1).set_index('date')
            prices = [round(float(ends.loc[day, 'close']), 2) for day in days.date.iloc[::-1]]
            np.testing.assert_allclose(prices, [getattr(row, name) for name in PATH_COLS], rtol=0, atol=2e-12)
            for n in [5, 20]:
                distance = sum(abs(prices[i]-prices[i+1]) for i in range(n))
                value = 100*(prices[0]-prices[n])/distance if distance else 0.
                np.testing.assert_allclose(value, getattr(row, f'DE{n:02d}'), rtol=0, atol=2e-10)
                np.testing.assert_array_equal(enc(value), enc(getattr(row, f'DE{n:02d}')))
                values[f'DE{n:02d}'] = value
        if 'equal_weight' in gate['active_variants']:
            state = pd.read_parquet(daily, columns=['date', 'preclose', 'isST', 'tradestatus'],
                filters=[('date', '==', row.date)]).iloc[0]
            clocks = current.timestamp.dt.strftime('%H%M')
            p20, p49 = [int(round(float(current.loc[clocks.eq(t), 'close'].iloc[0])*100)) for t in ['1420', '1449']]
            pc = int(round(float(state.preclose)*100))
            assert state.isST == 0 and state.tradestatus == 1 and current.volume.sum()>0 and current.turnover.sum()>0
            for name, own, peer in [('EW01', 100*(p49/pc-1), row.ew_day_mean),
                                     ('EW02', 100*(p49/p20-1), row.ew_tail_mean)]:
                value = (own-peer)/row.V01
                np.testing.assert_allclose(value, getattr(row, name), rtol=0, atol=2e-10)
                np.testing.assert_array_equal(enc(value), enc(getattr(row, name)))
                values[name] = value
        total += len(q)
        receipts.append(dict(date=row.date, code=row.code, first_history_date=first, minutes=len(q),
            expressions=values, minute_sha256=manifest['minute_sha256'][str(minute)],
            daily_sha256=manifest['daily_sha256'][str(daily)]))
    result = dict(passed=True, feature_report_sha256=sha(ROOT/'feature_report.json'),
        feature_verification_sha256=sha(ROOT/'feature_verification.json'), samples=receipts, raw_minutes=total,
        prior_day_ends_and_current_visible_quotes_rebuilt=True, full_reference_pool_independently_verified=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        q1_input_values_read=True, q1_new_group_outcomes_read=False, no_q2_signal_prices_read=True, no_exit_rules=True)
    save_json(ROOT/'native_input_verification.json', result); return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify_features', 'native']); args = parser.parse_args()
    print(json.dumps(globals()[args.stage](), ensure_ascii=False, indent=2))
