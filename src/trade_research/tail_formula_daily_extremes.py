"""Extreme raw close-to-close changes over the 20 completed stock days."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_daily_efficiency as history
from .corporate_cash import DAILY, save_json, sha

STEM = 'tail_formula_daily_extremes'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')


def nested_expression(function):
    terms = [f'100*(DCP{i}/DCP{i+1}-1)' for i in range(1,21)]
    expr = terms[-1]
    for term in reversed(terms[:-1]):
        expr = f'{function}({term},{expr})'
    return expr + '/V01'


NEW_EXPRESSIONS = {'DX01':nested_expression('MAX'), 'DX02':nested_expression('MIN')}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER
HISTORY_COLUMNS = ['date','code','daily_efficiency_valid',*history.PRICE_COLUMNS,
    'de_rows','de_good','de_first_date','de_last_date','de_reference_breaks','de_market_span','de_last_gap']


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['history_returns'] == 20 and p['history_prices'] == 21
    assert p['expressions'] == NEW_EXPRESSIONS and p['native_header'] == ''
    assert not p['new_2026_prices_allowed']
    for file,digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for root in [previous.ROOT,history.ROOT]:
        r = json.loads((root/'feature_report.json').read_text())
        v = json.loads((root/'feature_verification.json').read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(root/'feature_report.json')
        assert r['features_sha256'] == sha(root/'features.parquet')
    return p


def extremes(prices,atr):
    prices = np.asarray(prices,dtype=float)
    atr = np.asarray(atr,dtype=float)
    assert prices.ndim == 2 and prices.shape[1] == 21 and atr.shape == (len(prices),)
    valid = np.isfinite(prices).all(axis=1) & (prices>0).all(axis=1) & np.isfinite(atr) & (atr>0)
    with np.errstate(invalid='ignore',divide='ignore'):
        changes = 100*(prices[:,:-1]/prices[:,1:]-1)
        values = np.column_stack([changes.max(axis=1),changes.min(axis=1)])/atr[:,None]
    return np.where(valid[:,None],values,np.nan)


def features():
    checked_sources()
    assert not (ROOT/'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT/'features.parquet')
    raw = pd.read_parquet(history.ROOT/'features.parquet',columns=HISTORY_COLUMNS)
    f = old.merge(raw,on=['date','code'],how='left',validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
    values = extremes(f[history.PRICE_COLUMNS].to_numpy(float),f.V01.to_numpy(float))
    valid = f.daily_efficiency_valid.eq(True) & np.isfinite(values).all(axis=1)
    for i,name in enumerate(NEW_EXPRESSIONS):
        f[name] = pd.Series(values[:,i]).where(valid)
    f['daily_extremes_valid'] = valid
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid
    ROOT.mkdir(parents=True,exist_ok=True)
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(ROOT/'features.parquet'),
        rows=len(f),valid=int(f.formula_input_valid.sum()),newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
        valid_with_reference_breaks=int((f.formula_input_valid & f.de_reference_breaks.gt(0)).sum()),
        valid_with_stock_day_gaps=int((f.formula_input_valid & (f.de_market_span.gt(21) | f.de_last_gap.gt(1))).sum()),
        all_positive_valid=int((f.formula_input_valid & f.DX02.gt(0)).sum()),
        all_negative_valid=int((f.formula_input_valid & f.DX01.lt(0)).sum()),
        flat_valid=int((f.formula_input_valid & f.DX01.eq(0) & f.DX02.eq(0)).sum()),
        expressions=EXPRESSIONS,native_header=HEADER,raw_unadjusted_price_path=True,
        software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',r)
    return {k:v for k,v in r.items() if k not in ['expressions','native_header']}


def verify_features():
    p = checked_sources()
    r = json.loads((ROOT/'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT/'features.parquet')
    old = pd.read_parquet(previous.ROOT/'features.parquet')
    raw = pd.read_parquet(history.ROOT/'features.parquet',columns=HISTORY_COLUMNS)
    got = pd.read_parquet(ROOT/'features.parquet')
    c = base.conn();c.register('raw',raw);c.register('old',old[['date','code','V01']])
    atoms = ','.join(f'(100*(de_c{i:02d}/de_c{i+1:02d}-1))' for i in range(1,21))
    good = ' AND '.join(f'isfinite({name}) AND {name}>0' for name in [*history.PRICE_COLUMNS,'V01'])
    ex = c.sql(f'''SELECT date,code,coalesce(daily_efficiency_valid AND {good},false) AS valid,
        CASE WHEN valid THEN greatest({atoms})/V01 END AS DX01,
        CASE WHEN valid THEN least({atoms})/V01 END AS DX02
        FROM raw JOIN old USING(date,code) ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(got[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    pd.testing.assert_frame_equal(got[raw.columns],raw,check_exact=True)
    pd.testing.assert_series_equal(got.prior_formula_input_valid,old.formula_input_valid,check_exact=True,check_names=False)
    np.testing.assert_array_equal(got.daily_extremes_valid,ex.valid)
    np.testing.assert_allclose(got[list(NEW_EXPRESSIONS)],ex[list(NEW_EXPRESSIONS)],rtol=0,atol=2e-10,equal_nan=True)
    valid = old.formula_input_valid & ex.valid
    np.testing.assert_array_equal(got.formula_input_valid,valid)
    assert got.loc[valid,'DX01'].ge(got.loc[valid,'DX02']).all()
    encode = lambda a:np.floor(np.clip(100*a+10000+.000001,0,999999)).astype('int32')
    np.testing.assert_array_equal(encode(got.loc[valid,list(NEW_EXPRESSIONS)]),encode(ex.loc[valid,list(NEW_EXPRESSIONS)]))
    assert r['rows'] == len(got) == p['expected_keys']
    assert r['valid'] == int(valid.sum()) == p['expected_valid']
    assert r['newly_invalid'] == int((old.formula_input_valid & ~valid).sum())
    for key,mask in [('valid_with_reference_breaks',raw.de_reference_breaks.gt(0)),
        ('valid_with_stock_day_gaps',raw.de_market_span.gt(21)|raw.de_last_gap.gt(1)),
        ('all_positive_valid',ex.DX02.gt(0)),('all_negative_valid',ex.DX01.lt(0)),
        ('flat_valid',ex.DX01.eq(0)&ex.DX02.eq(0))]:
        assert r[key] == int((valid & mask).sum())
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names})
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    proof = dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=len(got),
        effective_input_intersection_unchanged=bool(valid.equals(old.formula_input_valid)),
        all_20_close_ratios_extremes_and_integer_encodings_independently_rebuilt=True,
        all_old_values_keys_and_validity_unchanged=True,history_gaps_and_reference_breaks_preserved=True,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_verification.json',proof)
    return proof


def native_value(prices,atr):
    assert len(prices) == 21
    env = {f'DCP{i}':float(price) for i,price in enumerate(prices,1)}
    env.update(V01=float(atr),MAX=max,MIN=min)
    return [eval(expr,{'__builtins__':{}},env) for expr in NEW_EXPRESSIONS.values()]


def native():
    checked_sources()
    v = json.loads((ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT/'feature_report.json')
    source = history.ROOT/'native_input_verification.json'
    samples = json.loads(source.read_text())
    assert samples['passed'] and samples['feature_report_sha256'] == sha(history.ROOT/'feature_report.json')
    assert samples['feature_verification_sha256'] == sha(history.ROOT/'feature_verification.json')
    hashes = json.loads(history.DAILY_REPORT.read_text())['source_sha256']
    f = pd.read_parquet(ROOT/'features.parquet').set_index(['date','code'])
    receipts = []
    for sample in samples['samples']:
        date,code = sample['date'],sample['code']
        assert '2024-01-01' <= date <= '2025-12-30'
        path = DAILY/(code.replace('.','_')+'.parquet')
        assert sha(path) == hashes[str(path)]
        d = pd.read_parquet(path,columns=['date','close','tradestatus','adjustflag'],
            filters=[('date','>=','2023-06-01'),('date','<',date)])
        days = d.loc[d.tradestatus.eq(1)].sort_values('date').tail(21)
        assert len(days) == 21 and days.adjustflag.eq(3).all()
        assert days.date.iloc[0] == sample['first_history_date'] and days.date.max() < date
        values = days.close.iloc[::-1].to_numpy(float).round(2)
        row = f.loc[(date,code)]
        assert row.formula_input_valid
        np.testing.assert_allclose(values,row[history.PRICE_COLUMNS].to_numpy(float),rtol=0,atol=2e-12)
        got = native_value(values,row.V01)
        expected = row[list(NEW_EXPRESSIONS)].to_numpy(float)
        np.testing.assert_allclose(got,expected,rtol=0,atol=2e-10)
        encode = lambda a:np.floor(np.clip(100*np.asarray(a)+10000+.000001,0,999999))
        np.testing.assert_array_equal(encode(got),encode(expected))
        receipts.append(dict(date=date,code=code,source_sha256=hashes[str(path)],stock_days=21,
            first_history_date=days.date.iloc[0],last_history_date=days.date.iloc[-1]))
    assert len(receipts) == 32
    proof = dict(passed=True,protocol_sha256=sha(PROTOCOL),feature_report_sha256=sha(ROOT/'feature_report.json'),
        feature_verification_sha256=sha(ROOT/'feature_verification.json'),fixed_sample_source_sha256=sha(source),
        samples=receipts,raw_daily_rows=672,actual_nested_max_min_and_dcps_rebuilt=True,
        prior_daily_close_to_final_minute_parity_reused=True,
        software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'native_input_verification.json',proof)
    return {k:v for k,v in proof.items() if k!='samples'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['features','verify_features','native'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
