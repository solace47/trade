"""Visible quote cost distance and recent volume capacity, never future fills."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_before1000 as labels
from . import tail_formula_float as previous
from . import tail_formula_volume_response as probes
from .corporate_cash import MINUTES, save_json, sha

STEM = 'tail_formula_entry_friction'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
EXTRA_HEADER = '''FRPC:=INTPART(100*Q+0.5);
FRQ:=FRPC/100;
FRN:=100*INTPART(2000000/(100*FRPC));
FRBV:=FRN*(FRQ+MAX(0.0015*FRQ,0.005));
FRCA:=FRBV+MAX(0.0003*FRBV,5)+0.00001*FRBV;
FRSV:=MAX(FRCA/(1-0.00081),(FRCA+5)/(1-0.00051));
FRPX:=FRSV/FRN;
FRBE:=MAX(FRPX/(1-0.0015),FRPX+0.005);
FRV4:=100*VALUEWHEN(TIME=1449,SUM(V,4));
'''
NEW_EXPRESSIONS = {'FR01': '100*(FRBE/FRQ-1)/VP20', 'FR02': 'LN(1+FRN/(0.1*FRV4))'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER + EXTRA_HEADER
native_core = base.native_core


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for path, digest in p['source_hashes'].items():
        assert sha(Path(path)) == digest
    for folder in [previous.ROOT, Path('data/research/tail_formula_intraday')]:
        r = json.loads((folder / 'feature_report.json').read_text())
        v = json.loads((folder / 'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(folder / 'feature_report.json')
        assert r['features_sha256'] == sha(folder / 'features.parquet')
    assert p['new_fields'] == list(NEW_EXPRESSIONS) and p['expected_features'] == len(EXPRESSIONS)
    assert not p['new_2026_prices_allowed']
    return p


def measure(quotes, atr_pct, recent_shares):
    quotes, atr_pct, recent_shares = np.broadcast_arrays(
        np.asarray(quotes, float), np.asarray(atr_pct, float), np.asarray(recent_shares, float))
    cents = np.floor(quotes * 100 + .5)
    quote = cents / 100
    with np.errstate(divide='ignore', invalid='ignore'):
        shares = 100 * np.floor(2000000 / (100 * cents))
        buy_value = shares * (quote + np.maximum(.0015 * quote, .005))
        cash = buy_value + np.maximum(.0003 * buy_value, 5) + .00001 * buy_value
        required_value = np.maximum(cash / (1 - .00081), (cash + 5) / (1 - .00051))
        net_price = required_value / shares
        breakeven = np.maximum(net_price / (1 - .0015), net_price + .005)
        distance = 100 * (breakeven / quote - 1) / atr_pct
        utilization = np.log1p(shares / (.1 * recent_shares))
    valid = (np.isfinite(quotes) & (np.abs(quote - quotes) <= .0001) & (quote > 0)
             & np.isfinite(atr_pct) & (atr_pct > 0) & np.isfinite(recent_shares) & (recent_shares > 0)
             & (shares >= 100) & np.isfinite(distance) & np.isfinite(utilization))
    return dict(valid=valid, fr_quote=quote, fr_shares=shares, fr_buy_value=buy_value,
        fr_cash=cash, fr_required_value=required_value, fr_breakeven=breakeven,
        FR01=np.where(valid, distance, np.nan), FR02=np.where(valid, utilization, np.nan))


def binary_zero(quote, shares, cash):
    """Independently invert the original direct fee calculation, not our inverse."""
    low, high = quote.copy(), 2 * quote + 1
    assert (labels.net_mark(low, shares, cash, 15) < 0).all()
    assert (labels.net_mark(high, shares, cash, 15) > 0).all()
    for _ in range(64):
        mid = (low + high) / 2
        positive = labels.net_mark(mid, shares, cash, 15) >= 0
        high = np.where(positive, mid, high)
        low = np.where(positive, low, mid)
    return (low + high) / 2


def encoding_counts(f):
    out = {}
    for name in NEW_EXPRESSIONS:
        x = 100 * f.loc[f.formula_input_valid, name] + 10000 + .000001
        out[name] = dict(below_clip=int((x < 0).sum()), above_clip=int((x > 999999).sum()),
            near_integer_boundary=int((np.abs(x - np.rint(x)) < 1e-8).sum()),
            minimum=float(f.loc[f.formula_input_valid, name].min()),
            maximum=float(f.loc[f.formula_input_valid, name].max()))
    return out


def features():
    p = checked_sources()
    assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    result = measure(old.price_1449, old.V01, old.v4)
    valid = result.pop('valid')
    f = old.copy()
    for name, value in result.items():
        f[name] = value
    f['entry_friction_valid'] = valid
    f['prior_formula_input_valid'] = old.formula_input_valid
    f['formula_input_valid'] &= valid
    np.testing.assert_array_equal(f.fr_shares, old.decision_shares)
    assert f.date.between('2024-01-01', '2025-12-30').all()
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
        features_sha256=sha(ROOT / 'features.parquet'), rows=len(f), valid=int(f.formula_input_valid.sum()),
        newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        encoding_counts=encoding_counts(f), expressions=EXPRESSIONS, native_header=HEADER,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['source_hashes', 'expressions', 'native_header']}


def verify_features():
    checked_sources()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT / 'features.parquet')
    old, f = [pd.read_parquet(folder / 'features.parquet') for folder in [previous.ROOT, ROOT]]
    c = base.conn()
    c.register('old', old)
    e = c.sql('''WITH a AS(SELECT *, round(price_1449*100) AS pc, pc/100 AS fr_quote,
        100*floor(2000000/(100*pc)) AS fr_shares,
        fr_shares*(fr_quote+greatest(.0015*fr_quote,.005)) AS fr_buy_value,
        fr_buy_value+greatest(.0003*fr_buy_value,5)+.00001*fr_buy_value AS fr_cash,
        greatest(fr_cash/(1-.00081),(fr_cash+5)/(1-.00051)) AS fr_required_value,
        fr_required_value/fr_shares AS e, greatest(e/(1-.0015),e+.005) AS fr_breakeven,
        coalesce(isfinite(price_1449) AND abs(fr_quote-price_1449)<=.0001 AND fr_quote>0
            AND isfinite(V01) AND V01>0 AND isfinite(v4) AND v4>0 AND fr_shares>=100,false) AS good
        FROM old)
        SELECT date,code,fr_quote,fr_shares,fr_buy_value,fr_cash,fr_required_value,fr_breakeven,
        CASE WHEN good THEN 100*(fr_breakeven/fr_quote-1)/V01 END AS FR01,
        CASE WHEN good THEN ln(1+fr_shares/(.1*v4)) END AS FR02,
        good AND isfinite(FR01) AND isfinite(FR02) AS valid FROM a ORDER BY date,code''').df()
    c.close()
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_names=False, check_exact=True)
    pd.testing.assert_frame_equal(f[['date', 'code']], e[['date', 'code']], check_exact=True)
    np.testing.assert_array_equal(f.entry_friction_valid, e.valid)
    np.testing.assert_array_equal(f.fr_shares, old.decision_shares)
    final = old.formula_input_valid & e.valid
    np.testing.assert_array_equal(f.formula_input_valid, final)
    for name in ['fr_quote', 'fr_shares', 'fr_buy_value', 'fr_cash', 'fr_required_value', 'fr_breakeven', *NEW_EXPRESSIONS]:
        np.testing.assert_allclose(f[name], e[name], rtol=3e-14, atol=2e-12, equal_nan=True)
    for name in NEW_EXPRESSIONS:
        encode = lambda x: np.floor(np.clip(100*x+10000+.000001, 0, 999999)).astype('int32')
        np.testing.assert_array_equal(encode(f.loc[final, name]), encode(e.loc[final, name]))
    active = f.loc[final]
    zero = binary_zero(active.fr_quote.to_numpy(), active.fr_shares.to_numpy(), active.fr_cash.to_numpy())
    target = active.fr_breakeven.to_numpy()
    np.testing.assert_allclose(zero, target, rtol=2e-14, atol=2e-12)
    delta = np.maximum(target, 1) * 1e-8
    assert (labels.net_mark(target-delta, active.fr_shares, active.fr_cash, 15) < 0).all()
    assert (labels.net_mark(target+delta, active.fr_shares, active.fr_cash, 15) > 0).all()
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names})
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    assert r['rows'] == len(f) and r['valid'] == int(final.sum())
    assert r['newly_invalid'] == int((old.formula_input_valid & ~final).sum())
    assert r['encoding_counts'] == encoding_counts(f)
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f),
        valid=int(final.sum()), all_original_48_values_and_keys_unchanged=True,
        all_quote_shares_costs_inverse_capacity_validity_and_encodings_sql_rebuilt=True,
        direct_fee_binary_zero_rows=len(active), max_binary_zero_error=float(np.max(np.abs(zero-target))),
        every_direct_fee_zero_bracket_checked=True, effective_input_intersection_unchanged=bool(final.equals(old.formula_input_valid)),
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof)
    return proof


def native_value(quote, atr_pct, lots, outside=1000000.):
    assert len(lots) == 4
    env = dict(Q=float(quote), VP20=float(atr_pct), V=np.r_[9999., lots, 9999., 9999.],
        TIME=np.r_[1445, np.arange(1446, 1450), 1450, 1451], INTPART=np.floor,
        MAX=np.maximum, LN=np.log, SUM=lambda x, n: pd.Series(x).rolling(int(n)).sum().to_numpy(),
        VALUEWHEN=lambda mask, values: np.asarray(values)[np.flatnonzero(mask)[-1]])
    env['C'] = np.r_[outside, [quote]*4, outside, outside]
    env['Q'] = env['VALUEWHEN'](env['TIME'] == 1449, env['C'])
    for line in EXTRA_HEADER.splitlines():
        name, expr = line.rstrip(';').split(':=')
        expr = re.sub(r'(?<![<>=!])=(?!=)', '==', expr)
        env[name] = eval(expr, {'__builtins__': {}}, env)
    return {name: float(eval(expr, {'__builtins__': {}}, env)) for name, expr in NEW_EXPRESSIONS.items()}


def native():
    checked_sources()
    v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    prior = json.loads((probes.ROOT / 'native_input_verification.json').read_text())
    assert prior['passed'] and prior['feature_report_sha256'] == sha(probes.ROOT / 'feature_report.json')
    f = pd.read_parquet(ROOT / 'features.parquet').set_index(['date', 'code'])
    samples = []
    for item in prior['samples']:
        date, code = item['date'], item['code']
        assert '2024-01-01' <= date <= '2025-12-30'
        path = MINUTES / code[:2].upper() / (code[3:] + '.parquet')
        assert sha(path) == item['source_sha256']
        raw = pd.read_parquet(path, columns=['timestamp', 'close', 'volume'],
            filters=[('timestamp', '>=', pd.Timestamp(date+' 14:46')), ('timestamp', '<=', pd.Timestamp(date+' 14:49'))]).sort_values('timestamp')
        assert raw.timestamp.dt.strftime('%H:%M').tolist() == [f'14:{i}' for i in range(46, 50)]
        row = f.loc[(date, code)]
        assert row.formula_input_valid
        q = float(raw.close.iloc[-1])
        assert abs(q-row.fr_quote) <= .0001 and raw.volume.sum() == row.v4
        value = native_value(q, row.V01, raw.volume.to_numpy(float)/100)
        changed = native_value(q, row.V01, raw.volume.to_numpy(float)/100, .01)
        for name in NEW_EXPRESSIONS:
            np.testing.assert_allclose(value[name], row[name], rtol=0, atol=2e-12)
            assert value[name] == changed[name]
            assert np.floor(100*value[name]+10000+.000001) == np.floor(100*row[name]+10000+.000001)
        samples.append(dict(date=date, code=code, source_sha256=item['source_sha256'], minutes=4, **value))
    assert len(samples) == 32
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'),
        existing_raw_probe_receipt_sha256=sha(probes.ROOT / 'native_input_verification.json'), samples=samples, raw_minutes=128,
        actual_anchored_quote_cost_inverse_and_share_lot_units_rebuilt=True,
        original_atr_numerical_proof_reused=True, outside_and_future_quotes_do_not_change_features=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', proof)
    return {k: v for k, v in proof.items() if k != 'samples'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
