"""Two independent ETF activity inputs with an explicit same-date prefix."""
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_morning_range as prior
from . import tail_formula_volume_memory as original_source
from .corporate_cash import save_json, sha

STEM = 'tail_formula_etf_activity'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
INTENT = Path('config') / (STEM + '_intent.json')
META, CONTROL = prior.META, prior.EXPRESSIONS
WINDOW_CLOCKS = (list(range(931, 960)) + list(range(1000, 1060)) + list(range(1100, 1131))
                 + list(range(1301, 1360)) + list(range(1400, 1449)))
assert len(WINDOW_CLOCKS) == 228
SEQUENCE_GUARD = '+'.join(f'IF(REF(TIME,{227-i})={clock},IF(REF(DATE,{227-i})=DATE,1,0),0)'
                          for i, clock in enumerate(WINDOW_CLOCKS))
ETF_HELPER = '''EFDT:=VALUEWHEN(TIME=1448,DATE);
''' + f'EFCLK:=VALUEWHEN(TIME=1448,{SEQUENCE_GUARD});\n' + '''EFGOOD:=VALUEWHEN(TIME=1448,COUNT(V>=0 AND V-V=0,228));
EFTOTAL:=VALUEWHEN(TIME=1448,SUM(V,228));
EFAM:=VALUEWHEN(TIME=1448,SUM(REF(V,108),120));
EFLATE:=VALUEWHEN(TIME=1448,SUM(V,29));
EFREADY:=EFDT=DATE AND EFCLK=228 AND EFGOOD=228 AND EFTOTAL>=0;
EA:IF(EFREADY,IF(EFTOTAL>0,100*EFAM/EFTOTAL,0),DRAWNULL);
EL:IF(EFREADY,IF(EFTOTAL>0,100*EFLATE/EFTOTAL,0),DRAWNULL);
ED:IF(EFREADY,EFDT,DRAWNULL);
'''
EXTRA_HEADER = '''E3DT:=CALCSTOCKINDEX('SH510300','YJETFQ',3);
E5DT:=CALCSTOCKINDEX('SH510500','YJETFQ',3);
E3AM:=CALCSTOCKINDEX('SH510300','YJETFQ',1);
E3LT:=CALCSTOCKINDEX('SH510300','YJETFQ',2);
E5AM:=CALCSTOCKINDEX('SH510500','YJETFQ',1);
E5LT:=CALCSTOCKINDEX('SH510500','YJETFQ',2);
EFVALID:=E3DT=DATE AND E5DT=DATE;
'''
NEW_EXPRESSIONS = {name: f'IF(EFVALID,{value},DRAWNULL)' for name, value in
                   [('E300A', 'E3AM'), ('E300L', 'E3LT'), ('E500A', 'E5AM'), ('E500L', 'E5LT')]}
EXPRESSIONS = {**CONTROL, **NEW_EXPRESSIONS}
HEADER = prior.HEADER + EXTRA_HEADER


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['intent_sha256'] == sha(INTENT) and p['arms'] == {'control': CONTROL, 'memory': EXPRESSIONS}
    assert p['native_header'] == HEADER and p['etf_helper'] == ETF_HELPER
    assert p['expected_keys'] == 1815129 and p['window_labels'] == 228
    assert p['maximum_new_fits'] == 4 and not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    gate = json.loads(Path(p['conditional_gate']).read_text())
    assert gate['passed'] and not gate['supports_further_validation']
    text = subprocess.run(['git', 'show', 'HEAD:docs/selection-formula.md'], text=True,
                          capture_output=True, check=True).stdout
    assert sha(PROTOCOL) in text and sha(INTENT) in text
    return p


def measure(volumes):
    v = np.asarray(volumes, float)
    assert v.ndim == 2 and v.shape[1] == 228
    good = (np.isfinite(v) & (v >= 0)).all(axis=1)
    total = v.sum(axis=1)
    with np.errstate(all='ignore'):
        result = np.column_stack([np.where(total > 0, 100*v[:, :120].sum(axis=1)/total, 0),
                                  np.where(total > 0, 100*v[:, -29:].sum(axis=1)/total, 0)])
    result[~good] = np.nan
    return result, good


def native_value(volumes, outside=1000000., window_clocks=None, window_dates=None,
                 current_date=1240101, quote_date=None):
    clocks = WINDOW_CLOCKS if window_clocks is None else list(window_clocks)
    dates = np.full(228, current_date) if window_dates is None else np.asarray(window_dates)
    env = dict(V=np.r_[outside, volumes, outside, outside], TIME=np.array([930]+clocks+[1449,1450]),
               DATE=np.r_[current_date, dates, current_date, current_date], IF=np.where, DRAWNULL=np.nan,
               REF=lambda x, n: pd.Series(x).shift(int(n)).to_numpy(),
               SUM=lambda x, n: pd.Series(np.asarray(x, float)).rolling(int(n), min_periods=int(n)).sum().to_numpy(),
               COUNT=lambda x, n: pd.Series(np.asarray(x, float)).rolling(int(n), min_periods=int(n)).sum().to_numpy(),
               VALUEWHEN=lambda mask, value: np.asarray(value)[np.flatnonzero(mask)[-1]] if np.ndim(value) else value)
    output = []
    with np.errstate(all='ignore'):
        for line in ETF_HELPER.splitlines():
            definition = ':=' if ':=' in line else ':'
            name, expression = line.rstrip(';').split(definition, 1)
            expression = re.sub(r'(?<![<>=!])=(?!=)', '==', expression)
            if name == 'EFGOOD':
                expression = 'VALUEWHEN(TIME==1448,COUNT((V>=0) & ((V-V)==0),228))'
            elif name == 'EFREADY':
                expression = expression.replace('EFDT==DATE', 'EFDT==DATE[-2]').replace(' AND ', ' and ')
            env[name] = eval(expression, {'__builtins__': {}}, env)
            if definition == ':':
                output.append(float(env[name]))
    # This is a replay of the literal stock-side CALCSTOCKINDEX adapter;
    # it does not assert the client's data alignment or compilation.
    date = current_date if quote_date is None else quote_date
    return stock_value(output, output, date), np.asarray(output)


def stock_value(first, second, date):
    reference = {'SH510300': first, 'SH510500': second}
    adapter = dict(DATE=date, IF=np.where, DRAWNULL=np.nan,
                   CALCSTOCKINDEX=lambda code, name, index: reference[code][index-1])
    for line in EXTRA_HEADER.splitlines():
        name, expression = line.rstrip(';').split(':=')
        expression = re.sub(r'(?<![<>=!])=(?!=)', '==', expression).replace(' AND ', ' and ')
        adapter[name] = eval(expression, {'__builtins__': {}}, adapter)
    values = [float(eval(e, {'__builtins__': {}}, adapter)) for e in NEW_EXPRESSIONS.values()]
    return np.asarray(values)


def prepare():
    p = checked(); path = INPUTS / 'feature_report.json'
    assert not path.exists(); INPUTS.mkdir(parents=True, exist_ok=True)
    daily = []; source_receipts = []; native_probes = []; zero = 0; literal_outputs = {}
    for symbol in ['510300', '510500']:
        source = Path(p['etf_sources'][symbol]); c = base.conn()
        c.read_parquet(str(source)).create_view('source')
        raw = c.sql('''SELECT timestamp, volume FROM source
            WHERE timestamp>=TIMESTAMP '2023-01-01' AND timestamp<TIMESTAMP '2026-01-01'
            AND (strftime(timestamp,'%H%M') BETWEEN '0931' AND '1130'
                 OR strftime(timestamp,'%H%M') BETWEEN '1301' AND '1448') ORDER BY timestamp''').df()
        raw['date'] = raw.timestamp.dt.strftime('%Y-%m-%d')
        assert not raw.timestamp.duplicated().any() and raw.timestamp.eq(raw.timestamp.dt.floor('min')).all()
        grouped = raw.groupby('date', sort=True)
        assert grouped.size().eq(228).all()
        for _, group in grouped:
            np.testing.assert_array_equal(group.timestamp.dt.hour*100+group.timestamp.dt.minute, WINDOW_CLOCKS)
        dates = grouped.size().index.to_numpy(); volumes = raw.volume.to_numpy(float).reshape(len(dates),228)
        values, good = measure(volumes)
        c.register('raw', raw)
        sql = c.sql('''WITH sums AS(SELECT date,count(*) AS bars,count(DISTINCT timestamp) AS clocks,
            sum(coalesce(isfinite(volume) AND volume>=0,false)::INT) AS good,
            sum(volume::HUGEINT)::DOUBLE AS total,
            sum(volume::HUGEINT) FILTER(WHERE strftime(timestamp,'%H%M')<='1130')::DOUBLE AS am,
            sum(volume::HUGEINT) FILTER(WHERE strftime(timestamp,'%H%M')>='1420')::DOUBLE AS late
            FROM raw GROUP BY date) SELECT date,bars=228 AND clocks=228 AND good=228 AS valid,
            CASE WHEN good=228 AND total>0 THEN 100*am/total WHEN good=228 THEN 0. END AS am_share,
            CASE WHEN good=228 AND total>0 THEN 100*late/total WHEN good=228 THEN 0. END AS late_share
            FROM sums ORDER BY date''').df(); c.close()
        np.testing.assert_array_equal(dates, sql.date)
        np.testing.assert_array_equal(good, sql.valid)
        np.testing.assert_allclose(values, sql[['am_share','late_share']], rtol=0, atol=2e-11, equal_nan=True)
        encode = lambda a: np.floor(np.clip(100*a+10000+.000001,0,999999))
        np.testing.assert_array_equal(encode(values), encode(sql[['am_share','late_share']].to_numpy()))
        for i, date in enumerate(dates):
            actual, outputs = native_value(volumes[i], current_date=int(date.replace('-',''))-19000000)
            literal_outputs[(symbol, date)] = outputs
            np.testing.assert_allclose(actual[:2], values[i], rtol=0, atol=2e-11, equal_nan=True)
            np.testing.assert_allclose(actual[2:], values[i], rtol=0, atol=2e-11, equal_nan=True)
            np.testing.assert_array_equal(encode(actual[:2]), encode(values[i]))
            if i % 31 == 0 or i == len(dates)-1:
                future, _ = native_value(volumes[i], outside=.01, current_date=int(date.replace('-',''))-19000000)
                scaled, _ = native_value(volumes[i]*5, current_date=int(date.replace('-',''))-19000000)
                np.testing.assert_array_equal(actual, future)
                np.testing.assert_allclose(actual, scaled, rtol=0, atol=2e-11, equal_nan=True)
                np.testing.assert_array_equal(encode(actual), encode(scaled))
                native_probes.append(dict(symbol=symbol,date=date,bars=228,same_date_output=int(outputs[2])))
        zero += int((good & (volumes.sum(axis=1)==0)).sum())
        prefix = 'E300' if symbol == '510300' else 'E500'
        table = pd.DataFrame({'date':dates,prefix+'A':values[:,0],prefix+'L':values[:,1],prefix+'_valid':good})
        file = INPUTS / (symbol+'_daily.parquet'); table.to_parquet(file,index=False,compression='zstd')
        daily.append(table); source_receipts.append(dict(file=str(file),sha256=sha(file),raw_source=str(source),
                                                       raw_sha256=sha(source),days=len(table),raw_bars=len(raw)))
        print(json.dumps(dict(source=symbol,days=len(table),all_native_daily_math_verified=True)),flush=True)
    etf = daily[0].merge(daily[1],on='date',how='outer',validate='one_to_one').sort_values('date')
    for row in etf.itertuples(index=False):
        actual = stock_value(literal_outputs[('510300',row.date)],literal_outputs[('510500',row.date)],
                             int(row.date.replace('-',''))-19000000)
        expected = np.array([row.E300A,row.E300L,row.E500A,row.E500L])
        np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-11,equal_nan=True)
        np.testing.assert_array_equal(encode(actual),encode(expected))
    old = original_source.original(); out = old.merge(etf,on='date',how='left',validate='many_to_one')
    pd.testing.assert_frame_equal(out[[*META,*CONTROL]],old[[*META,*CONTROL]],check_exact=True)
    new_good = out.E300_valid.fillna(False) & out.E500_valid.fillna(False)
    invalid = int((old.formula_input_valid & ~new_good).sum())
    save_json(INPUTS/'input_domain_audit.json',dict(passed=invalid==0,original_valid=1602413,
        new_valid=int((old.formula_input_valid & new_good).sum()),newly_invalid=invalid,
        original_keys_values_metadata_exact=True,missing_current_ETF_dates_not_filled=True,stop_fit_if_domain_changes=True))
    assert invalid==0
    out['prior_formula_input_valid'] = old.formula_input_valid
    out['etf_activity_valid'] = new_good
    out['formula_input_valid'] = old.formula_input_valid & new_good
    pd.testing.assert_series_equal(out.formula_input_valid,old.formula_input_valid,check_names=True)
    c = base.conn(); c.register('old',old); c.register('etf',etf)
    independent = c.sql('SELECT old.*,'+','.join(NEW_EXPRESSIONS)+' FROM old LEFT JOIN etf USING(date) ORDER BY date,code').df();c.close()
    pd.testing.assert_frame_equal(out[[*META,*EXPRESSIONS]],independent[[*META,*EXPRESSIONS]],check_exact=True)
    out.to_parquet(INPUTS/'features.parquet',index=False,compression='zstd')
    pd.testing.assert_frame_equal(pd.read_parquet(INPUTS/'features.parquet'),out,check_exact=True)
    for name in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        (INPUTS/name).symlink_to((prior.INPUTS/name).resolve())
    helper = INPUTS/'YJETFQ.tdx'; helper.write_text(ETF_HELPER)
    adapter = INPUTS/'etf_inputs.tdx'; adapter.write_text(EXTRA_HEADER+'\n'.join(f'{n}:={e};' for n,e in NEW_EXPRESSIONS.items())+'\n')
    save_json(path,dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(INPUTS/'features.parquet'),
        source_receipts=source_receipts,rows=len(out),valid=int(out.formula_input_valid.sum()),newly_invalid=invalid,
        expressions=EXPRESSIONS,native_header=HEADER,ETF_helper_sha256=sha(helper),ETF_adapter_sha256=sha(adapter),
        scalar_feature_and_encoding_checks=len(out)*4,known_zero_volume_windows=zero,
        new_group_outcomes_read=False,software_compilation_verified=False,native_source_parity_verified=False,
        new_2026_values_read=False,no_exit_rules=True))
    save_json(INPUTS/'feature_verification.json',dict(passed=True,feature_report_sha256=sha(path),
        all_original_50_metadata_keys_and_domain_exact=True,all_new_values_SQL_native_and_encodings_equal=True,
        all_date_join_values_independently_equal=True,new_2026_prices_read=False,no_exit_rules=True))
    save_json(INPUTS/'native_input_verification.json',dict(passed=True,feature_report_sha256=sha(path),
        feature_verification_sha256=sha(INPUTS/'feature_verification.json'),helper_sha256=sha(helper),adapter_sha256=sha(adapter),
        all_1454_ETF_source_windows_replayed=True,samples=native_probes,
        all_current_native_helper_math_and_literal_cross_security_adapter_verified=True,
        all_distinct_ETF_outputs_identity_order_and_date_joins_replayed=True,
        zero_bad_values_duplicates_previous_date_and_future_tests_passed=True,
        software_compilation_verified=False,native_source_parity_verified=False,new_2026_prices_read=False,no_exit_rules=True))
    return dict(feature_report_sha256=sha(path),rows=len(out),valid=int(out.formula_input_valid.sum()),newly_invalid=invalid)
