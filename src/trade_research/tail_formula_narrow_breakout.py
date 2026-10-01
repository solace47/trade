"""A fixed strict-NR7 breakout, using completed unadjusted stock days only."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from .corporate_cash import save_json, sha
from .tail_formula_reversal_change_daily_native import expression_to_python

STEM = 'tail_formula_narrow_breakout'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_protocol.json')
META = ['date', 'code', 'half', 'board', 'decision_shares', 'formula_input_valid']
DAY_HELPER = '''HC:=ROUND(H*100);
LC:=ROUND(L*100);
OC:=ROUND(O*100);
CC:=ROUND(C*100);
DG:=O>0 AND L>0 AND C>0 AND H>=MAX(O,C) AND L<=MIN(O,C) AND H>=L AND V>0 AND ABS(H*100-HC)<=0.01 AND ABS(L*100-LC)<=0.01 AND ABS(O*100-OC)<=0.01 AND ABS(C*100-CC)<=0.01;
''' + '\n'.join(f'R{i}:=REF(HC,{i})-REF(LC,{i});' for i in range(1, 8)) + '''
READY:=BARSCOUNT(C)>=8 AND REF(COUNT(DG,7),1)=7 AND MIN(R1,MIN(R2,MIN(R3,MIN(R4,MIN(R5,MIN(R6,R7))))))>0;
PH:IF(READY,REF(HC,1),DRAWNULL);
N7:IF(READY,IF(R1<MIN(MIN(R2,R3),MIN(MIN(R4,R5),MIN(R6,R7))),1,0),DRAWNULL);
'''
MINUTE_CORE = '''QC:=ROUND(C*100);
PH:=YJNR7A.PH#DAY;
N7:=YJNR7A.N7#DAY;
CORE:PH>0 AND N7=1 AND QC>PH;
'''


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['day_helper'] == DAY_HELPER and p['minute_core'] == MINUTE_CORE
    assert p['history_days'] == 7 and p['maximum_new_fits'] == 0
    assert not p['new_2026_prices_allowed'] and p['expected_keys'] == 1258085
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    audit = json.loads(Path(p['source_metadata_audit']).read_text())
    for file, digest in audit['daily_source_hashes'].items():
        assert sha(Path(file)) == digest, file
    gate = json.loads(Path(p['conditional_gate']).read_text())
    assert gate['passed'] and not gate['supports_further_validation']
    committed = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'],
                               text=True, capture_output=True, check=True).stdout
    assert sha(PROTOCOL) in committed
    return p


def replay_helper(d):
    """Replay the actual exported helper; the current bar's OHLC is not used."""
    env = {name: d[col].to_numpy(float) for name, col in
           [('H', 'high'), ('L', 'low'), ('O', 'open'), ('C', 'close'), ('V', 'volume')]}
    env.update(ROUND=lambda x: np.floor(x + .5), ABS=np.abs, MIN=np.minimum, MAX=np.maximum,
        REF=lambda a, n: pd.Series(a).shift(int(n)).to_numpy(),
        COUNT=lambda a, n: pd.Series(np.asarray(a, float)).rolling(int(n), min_periods=int(n)).sum().to_numpy(),
        BARSCOUNT=lambda a: np.arange(1, len(a) + 1), IF=np.where, DRAWNULL=np.nan)
    for line in DAY_HELPER.splitlines():
        if not line: continue
        name, expr = line.rstrip(';').split(':=') if ':=' in line else line.rstrip(';').split(':')
        env[name] = eval(expression_to_python(expr), {'__builtins__': {}}, env)
    return pd.DataFrame(dict(ready=env['READY'], previous_high_cents=env['PH'], nr7=env['N7']))


def pandas_history(d):
    """Bad active rows retain their places; no search for an older good row."""
    results = []
    for code, g in d.groupby('code', sort=False):
        g = g.reset_index(drop=True)
        prices = g[['open', 'high', 'low', 'close']].to_numpy(float)
        cents = np.floor(100 * prices + .5)
        good = (np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
                & (np.abs(100 * prices - cents) <= .01).all(axis=1)
                & g.high.ge(g[['open', 'close']].max(axis=1))
                & g.low.le(g[['open', 'close']].min(axis=1)) & g.high.ge(g.low)
                & np.isfinite(g.volume) & g.volume.gt(0) & g.adjustflag.eq(3))
        width = pd.Series(cents[:, 1] - cents[:, 2])
        out = pd.DataFrame(dict(code=code, history_date=g.date, first_history_date=g.date.shift(6),
                                good_days=pd.Series(good).rolling(7, min_periods=1).sum(),
                                history_rows=np.minimum(np.arange(len(g)) + 1, 7),
                                prior_high_cents=cents[:, 1]))
        for i in range(1, 8): out[f'r{i}'] = width.shift(i - 1)
        ready = (out.history_rows.eq(7) & out.good_days.eq(7)
                 & out[[f'r{i}' for i in range(1, 8)]].gt(0).all(axis=1))
        out['history_input_valid'] = ready
        out['nr7'] = ready & out.r1.lt(out[[f'r{i}' for i in range(2, 8)]].min(axis=1))
        out['reference_breaks'] = g.preclose.sub(g.close.shift()).abs().gt(.005).astype(int).rolling(7, min_periods=1).sum()
        # Append an unused current row so every completed seven-day state,
        # including the last source day, has an actual REF-only replay.
        dummy = {c: 10. for c in ['open', 'high', 'low', 'close', 'volume']}
        replay = replay_helper(pd.concat([g, pd.DataFrame([dummy])], ignore_index=True)).iloc[1:].reset_index(drop=True)
        # adjustflag is a source/setting requirement, not a client quote field.
        assert g.adjustflag.eq(3).all()
        np.testing.assert_array_equal(replay.ready, ready)
        np.testing.assert_allclose(replay.previous_high_cents, out.prior_high_cents.where(ready), rtol=0, atol=0, equal_nan=True)
        np.testing.assert_array_equal(replay.nr7.fillna(0).eq(1), out.nr7)
        results.append(out)
    return pd.concat(results, ignore_index=True).sort_values(['code', 'history_date']).reset_index(drop=True)


def map_history(keys, states):
    pieces = []
    by_code = {c: g.reset_index(drop=True) for c, g in states.groupby('code', sort=False)}
    fields = list(states.columns.drop('code'))
    for code, group in keys.groupby('code', sort=False):
        s = by_code[code]
        positions = np.searchsorted(s.history_date.to_numpy(), group.date.to_numpy(), side='left') - 1
        mapped = s[fields].reindex(np.where(positions >= 0, positions, -1)).reset_index(drop=True)
        out = group[['date', 'code']].reset_index(drop=True).join(mapped)
        pieces.append(out)
    return pd.concat(pieces, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)


def prepare():
    p = checked(); audit = json.loads(Path(p['source_metadata_audit']).read_text())
    assert not (INPUTS / 'feature_report.json').exists()
    INPUTS.mkdir(parents=True, exist_ok=True)
    original = pd.read_parquet(p['feature_file'], columns=META + ['A04'])
    assert len(original) == p['expected_keys'] and not original.duplicated(['date', 'code']).any()
    c = base.conn(); c.read_parquet(list(audit['daily_source_hashes'])).create_view('raw')
    d = c.sql(f'''SELECT date,code,open::DOUBLE AS open,high::DOUBLE AS high,low::DOUBLE AS low,
        close::DOUBLE AS close,preclose::DOUBLE AS preclose,volume::DOUBLE AS volume,
        adjustflag::DOUBLE AS adjustflag FROM raw WHERE tradestatus=1
        AND date BETWEEN '{p['history_first']}' AND '{p['history_last']}' ORDER BY code,date''').df()
    assert not d.duplicated(['date', 'code']).any() and d.date.lt('2026-01-01').all()
    c.register('daily', d); c.register('keys', original)
    lag_ranges = ','.join(f'lag(w,{i-1}) OVER z AS r{i}' for i in range(1, 8))
    states = c.sql(f'''WITH a AS(SELECT *,floor(high*100+.5)-floor(low*100+.5) AS w,
        coalesce(isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close)
        AND open>0 AND low>0 AND close>0 AND high>=greatest(open,close) AND low<=least(open,close)
        AND high>=low AND isfinite(volume) AND volume>0 AND adjustflag=3
        AND abs(high*100-round(high*100))<=.01 AND abs(low*100-round(low*100))<=.01
        AND abs(open*100-round(open*100))<=.01 AND abs(close*100-round(close*100))<=.01,false) AS good,
        coalesce(abs(preclose-lag(close) OVER z)>.005,false)::INT AS ref_break FROM daily
        WINDOW z AS(PARTITION BY code ORDER BY date)), b AS(SELECT code,date AS history_date,
        lag(date,6) OVER z AS first_history_date,count(*) OVER w AS history_rows,
        sum(good::INT) OVER w AS good_days,floor(high*100+.5) AS prior_high_cents,
        sum(ref_break) OVER w AS reference_breaks,{lag_ranges} FROM a
        WINDOW z AS(PARTITION BY code ORDER BY date),
        w AS(PARTITION BY code ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)),
        q AS(SELECT *,coalesce(history_rows=7 AND good_days=7 AND least(r1,r2,r3,r4,r5,r6,r7)>0,false) AS history_input_valid FROM b)
        SELECT *,history_input_valid AND r1<least(r2,r3,r4,r5,r6,r7) AS nr7 FROM q ORDER BY code,history_date''').df()
    c.register('states', states)
    actual = c.sql('''SELECT k.*,s.* EXCLUDE(code),floor(A04*100+.5) AS quote_cents
        FROM keys k ASOF LEFT JOIN states s ON k.code=s.code AND k.date>s.history_date ORDER BY date,code''').df()
    expected_states = pandas_history(d)
    pd.testing.assert_frame_equal(states[expected_states.columns], expected_states, check_dtype=False, check_exact=True)
    expected = map_history(original, expected_states)
    pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False, check_exact=True)
    pd.testing.assert_frame_equal(actual[META + ['A04']], original, check_exact=True)
    actual['prior_formula_input_valid'] = actual.formula_input_valid
    actual['formula_input_valid'] &= actual.history_input_valid.fillna(False) & actual.history_date.lt(actual.date)
    actual['breakout'] = actual.formula_input_valid & actual.quote_cents.gt(actual.prior_high_cents)
    actual['narrow_breakout'] = actual.breakout & actual.nr7.fillna(False)
    # Direct unrounded decimal comparisons are independently equivalent on
    # valid cents, and all native PH/N7 outputs were replayed above.
    np.testing.assert_array_equal(actual.breakout, actual.formula_input_valid & actual.A04.gt(actual.prior_high_cents / 100))
    assert actual.loc[actual.formula_input_valid, 'history_date'].lt(actual.loc[actual.formula_input_valid, 'date']).all()
    states.to_parquet(INPUTS / 'history.parquet', index=False, compression='zstd')
    actual.to_parquet(INPUTS / 'features.parquet', index=False, compression='zstd'); c.close()
    (INPUTS / 'YJNR7A.tdx').write_text(DAY_HELPER)
    (INPUTS / 'frozen_numeric_core.tdx').write_text(MINUTE_CORE)
    report = dict(protocol_sha256=sha(PROTOCOL), source_metadata_audit_sha256=sha(Path(p['source_metadata_audit'])),
        features_sha256=sha(INPUTS / 'features.parquet'), history_sha256=sha(INPUTS / 'history.parquet'),
        rows=len(actual), valid=int(actual.formula_input_valid.sum()),
        newly_invalid=int((actual.prior_formula_input_valid & ~actual.formula_input_valid).sum()),
        native_replayed_states=len(states), first_source_date=d.date.min(), last_source_date=d.date.max(),
        valid_with_reference_breaks=int((actual.formula_input_valid & actual.reference_breaks.gt(0)).sum()),
        source_hashes={str(INPUTS / file):sha(INPUTS / file) for file in ['YJNR7A.tdx', 'frozen_numeric_core.tdx']},
        software_compilation_verified=False,native_source_parity_verified=False,
        no_new_group_outcomes_read=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS / 'feature_report.json', report)
    proof = dict(passed=True, feature_report_sha256=sha(INPUTS / 'feature_report.json'),
        all_states_raw_rolling_and_actual_native_helper_exact=True,
        strict_prior_mapping_independent_searchsorted_exact=True,
        all_original_metadata_quote_and_keys_exact=True,
        all_bad_active_rows_and_new_invalid_keys_retained=True,
        strict_ties_positive_ranges_and_one_cent_breakout_verified=True,
        no_current_completed_day_values_used=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS / 'feature_verification.json', proof)
    return report


def freeze():
    p = checked(); path = ROOT / 'joint_selection_freeze.json'; assert not path.exists()
    report = json.loads((INPUTS / 'feature_report.json').read_text())
    proof = json.loads((INPUTS / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(INPUTS / 'feature_report.json')
    assert report['features_sha256'] == sha(INPUTS / 'features.parquet')
    f = pd.read_parquet(INPUTS / 'features.parquet')
    c = base.conn(); c.register('f', f)
    receipts = {str(INPUTS / file):sha(INPUTS / file) for file in
        ['features.parquet','history.parquet','feature_report.json','feature_verification.json','YJNR7A.tdx','frozen_numeric_core.tdx']}
    selections = []
    for arm, column in [('control','breakout'),('memory','narrow_breakout')]:
        for year in ['2024','2025']:
            root = ROOT / (arm + year); root.mkdir(parents=True,exist_ok=True)
            out = f[META[:-1]].copy(); out['selected'] = f[column] & f.date.str.startswith(year)
            expected = c.sql(f"SELECT date,code,half,board,decision_shares,coalesce(formula_input_valid AND quote_cents>prior_high_cents {'AND nr7' if arm=='memory' else ''} AND starts_with(date,'{year}'),false) AS selected FROM f ORDER BY date,code").df()
            pd.testing.assert_frame_equal(out,expected,check_exact=True)
            out.to_parquet(root / 'selection.parquet',index=False,compression='zstd')
            r = dict(protocol_sha256=sha(PROTOCOL),selection_sha256=sha(root / 'selection.parquet'),group=arm+year,
                rows=len(out),selected=int(out.selected.sum()),days=out.loc[out.selected,'date'].nunique(),
                no_new_group_outcomes_read=True,no_future_fill_or_label_filtering=True,new_2026_prices_read=False,no_exit_rules=True)
            save_json(root / 'selection_report.json',r)
            save_json(root / 'selection_verification.json',dict(passed=True,selection_report_sha256=sha(root / 'selection_report.json'),
                all_complete_keys_metadata_flags_independent_sql_equal=True,invalid_and_other_year_keys_retained=True))
            selections.append(dict(group=arm+year,root=str(root),rows=len(out),selected=r['selected'],days=r['days']))
            for file in ['selection.parquet','selection_report.json','selection_verification.json']:
                receipts[str(root / file)] = sha(root / file)
    c.close()
    save_json(path,dict(passed=True,protocol_sha256=sha(PROTOCOL),source_hashes=receipts,selections=selections,
        arm_display_names={'control':'同质量单纯突破','memory':'严格NR7突破'},no_new_fits=True,
        all_four_full_year_frames_fixed_together=True,no_new_group_outcomes_read=True,
        software_compilation_verified=False,native_source_parity_verified=False,new_2026_prices_read=False,no_exit_rules=True))
    return dict(joint_sha256=sha(path),selections=selections)
