"""Replace four absolute style inputs with contemporaneous population z-scores."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_market_turnover as sample_source
from .corporate_cash import save_json, sha

STEM = 'tail_formula_style_zscore'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
FIELDS = ['V01', 'S01', 'S02', 'S03']
EPSILON = 1e-12
EXPRESSIONS = {('Z'+n if n in FIELDS else n):
    (f'({n}-CSM{FIELDS.index(n)+1})/SQRT(MAX(CSV{FIELDS.index(n)+1},0.000000000001))' if n in FIELDS else expr)
    for n, expr in previous.EXPRESSIONS.items()}
HEADER = previous.HEADER + '\n'.join(f'{n}:={previous.EXPRESSIONS[n]};' for n in FIELDS) + '\n'
HEADER += '{YJSTY1/2是数学接口，完整客户端有效成员及历史股本来源仍未核准。}\n'
HEADER += "CSN1:=INSUM('沪深Ａ股','YJSTY1',1,0);\nCSN2:=INSUM('沪深Ａ股','YJSTY2',1,0);\n"
for i in range(4):
    helper, output = i//2+1, 2*(i % 2)+2
    HEADER += f"CSM{i+1}:=INSUM('沪深Ａ股','YJSTY{helper}',{output},0)/MAX(CSN{helper},1);\n"
    HEADER += f"CSV{i+1}:=INSUM('沪深Ａ股','YJSTY{helper}',{output+1},0)/MAX(CSN{helper},1)-CSM{i+1}*CSM{i+1};\n"
GATE = 'CSN1>=2 AND CSN1=CSN2 AND ' + ' AND '.join(f'CSV{i}>0.000000000001' for i in range(1, 5))
ORIGINAL_NATIVE_CORE = base.native_core


def native_core(*args, **kwargs):
    text = ORIGINAL_NATIVE_CORE(*args, **kwargs)
    assert text.count('CORE:SC>') == 1
    return text.replace('CORE:SC>', 'CORE:'+GATE+' AND SC>')


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    r = json.loads((previous.ROOT / 'feature_report.json').read_text())
    v = json.loads((previous.ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert r['features_sha256'] == sha(previous.ROOT / 'features.parquet')
    assert p['fields'] == FIELDS and p['minimum_variance'] == EPSILON and p['minimum_members'] == 2
    return p


def transform(old):
    """The pool depends on input validity only; use centered population variance."""
    assert not old.duplicated(['date', 'code']).any()
    m = old.loc[old.formula_input_valid, ['date', 'code', *FIELDS]].copy()
    assert np.isfinite(m[FIELDS]).all().all()
    groups = m.groupby('date', sort=True)
    stats = groups.size().rename('style_members').to_frame()
    for name in FIELDS:
        stats[name+'_mean'] = groups[name].mean()
        centered = m[name]-groups[name].transform('mean')
        stats[name+'_var'] = centered.pow(2).groupby(m.date, sort=True).mean()
    stats = stats.reset_index()
    f = old.merge(stats, on='date', how='left', validate='many_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    valid = f.style_members.ge(2)
    for name in FIELDS:
        valid &= np.isfinite(f[name+'_mean']) & np.isfinite(f[name+'_var']) & f[name+'_var'].gt(EPSILON)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid
    for name in FIELDS:
        f['Z'+name] = ((f[name]-f[name+'_mean'])/np.sqrt(f[name+'_var'].clip(lower=EPSILON))).where(f.formula_input_valid)
    return f, stats


def features():
    p = checked_sources(); assert not (ROOT / 'feature_report.json').exists()
    old = pd.read_parquet(previous.ROOT / 'features.parquet'); f, stats = transform(old)
    assert len(f) == p['all_keys_expected'] and int(old.formula_input_valid.sum()) == p['all_members_count_expected']
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    stats.to_parquet(ROOT / 'market_reference.parquet', index=False, compression='zstd')
    interface = {f'YJSTY{i+1}': dict(membership='完整原硬过滤且全部原48项有效，与原MT成员一致',
        outputs=['成员为1，否则0', *[x for n in FIELDS[2*i:2*i+2] for x in [f'成员为原{n}，否则0', f'成员为原{n}平方，否则0']]],
        mathematical_interface_only=True, full_client_membership_implementation_verified=False) for i in range(2)}
    save_json(ROOT / 'helper_interface.json', interface)
    r = dict(protocol_sha256=sha(PROTOCOL), features_sha256=sha(ROOT / 'features.parquet'),
        market_reference_sha256=sha(ROOT / 'market_reference.parquet'), helper_interface_sha256=sha(ROOT / 'helper_interface.json'),
        rows=len(f), valid=int(f.formula_input_valid.sum()), newly_invalid=int((f.prior_formula_input_valid & ~f.formula_input_valid).sum()),
        member_rows=int(stats.style_members.sum()), days=len(stats), minimum_members=int(stats.style_members.min()),
        maximum_members=int(stats.style_members.max()), minimum_variances={n:float(stats[n+'_var'].min()) for n in FIELDS},
        expressions=EXPRESSIONS, native_header=HEADER, native_core_gate=GATE, all_original_fields_preserved=True,
        replaced_model_inputs=FIELDS, expected_features=len(EXPRESSIONS), mathematical_interface_only=True,
        native_source_parity_verified=False, software_compilation_verified=False,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k:v for k,v in r.items() if k not in ['expressions', 'native_header']}


def verify_features():
    p = checked_sources(); r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['expected_features'] == 48
    for key, file in [('features','features.parquet'), ('market_reference','market_reference.parquet'), ('helper_interface','helper_interface.json')]:
        assert r[key+'_sha256'] == sha(ROOT / file)
    old = pd.read_parquet(previous.ROOT / 'features.parquet'); f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_names=False, check_exact=True)
    source = old[['date','code','formula_input_valid','atr20','preclose','price_1449','float_prior_volume','float_prior_turn','volume_1449','v29']]
    c = base.conn(); c.register('source', source)
    c.execute('''CREATE VIEW rebuilt AS SELECT date,code,formula_input_valid,
        100*atr20/preclose AS V01,
        ln(1+(100*float_prior_volume/float_prior_turn)*price_1449/1e8) AS S01,
        100*volume_1449/(100*float_prior_volume/float_prior_turn) AS S02,
        100*v29/(100*float_prior_volume/float_prior_turn) AS S03 FROM source''')
    members = c.sql('SELECT date,code,V01,S01,S02,S03 FROM rebuilt WHERE formula_input_valid ORDER BY date,code').df()
    original_members = old.loc[old.formula_input_valid]
    pd.testing.assert_frame_equal(members[['date','code']], original_members[['date','code']].reset_index(drop=True), check_exact=True)
    np.testing.assert_allclose(members[FIELDS], original_members[FIELDS], rtol=0, atol=2e-11)
    fields = ','.join(f'avg({n}) AS {n}_mean,var_pop({n}) AS {n}_var' for n in FIELDS)
    c.execute(f'CREATE VIEW stats AS SELECT date,count(*) AS style_members,{fields} FROM rebuilt WHERE formula_input_valid GROUP BY date')
    stats = c.sql('SELECT * FROM stats ORDER BY date').df()
    pd.testing.assert_frame_equal(pd.read_parquet(ROOT / 'market_reference.parquet'), stats, check_dtype=False, rtol=2e-12, atol=2e-10)
    mt = pd.read_parquet(sample_source.ROOT / 'market_reference.parquet', columns=['date','mt_members'])
    pd.testing.assert_frame_equal(stats[['date','style_members']].rename(columns={'style_members':'mt_members'}), mt, check_dtype=False, check_exact=True)
    valid = ' AND '.join(f'isfinite({n}_mean) AND isfinite({n}_var) AND {n}_var>1e-12' for n in FIELDS)
    values = ','.join(f'CASE WHEN valid THEN ({n}-{n}_mean)/sqrt({n}_var) END AS Z{n}' for n in FIELDS)
    expected = c.sql(f'''SELECT date,code,coalesce(formula_input_valid AND style_members>=2 AND {valid},false) AS valid,
        {values} FROM rebuilt LEFT JOIN stats USING(date) ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(f[['date','code']], expected[['date','code']], check_exact=True)
    np.testing.assert_array_equal(f.formula_input_valid, expected.valid)
    names = ['Z'+n for n in FIELDS]
    np.testing.assert_allclose(f[names], expected[names], rtol=0, atol=2e-10, equal_nan=True)
    encode = lambda x: np.floor(np.clip(100*np.asarray(x)+10000+.000001,0,999999))
    np.testing.assert_array_equal(encode(f.loc[expected.valid,names]), encode(expected.loc[expected.valid,names]))
    variables = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER)+list(EXPRESSIONS)
    assert len(variables) == len(set(variables)) and r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    assert all(EXPRESSIONS[n] == expr for n,expr in previous.EXPRESSIONS.items() if n not in FIELDS)
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(f),
        all_raw_variables_members_population_means_variances_and_encodings_sql_rebuilt=True,
        original_48_fields_and_keys_preserved=True, other_44_model_expressions_unchanged=True,
        old_volatility_denominators_preserved=True, same_members_as_relative_turnover=True,
        effective_input_intersection_unchanged=bool(np.array_equal(f.formula_input_valid, old.formula_input_valid)),
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof); return proof


def native():
    checked_sources(); v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    original = json.loads((sample_source.ROOT / 'native_input_verification.json').read_text()); assert original['passed']
    f = pd.read_parquet(ROOT / 'features.parquet'); visible = f.loc[f.prior_formula_input_valid]
    receipts = []
    for sample in original['samples']:
        peers = visible.loc[visible.date.eq(sample['date'])]; row = peers.loc[peers.code.eq(sample['code'])].iloc[0]
        shares = 100*peers.float_prior_volume.to_numpy()/peers.float_prior_turn.to_numpy()
        raw = np.column_stack([100*peers.atr20/peers.preclose, np.log1p(shares*peers.price_1449/1e8),
            10000*(peers.volume_1449/100)/shares, 10000*(peers.v29/100)/shares])
        outputs = []; values = []
        for j in range(2):
            helper = [len(peers)]
            for i in [2*j,2*j+1]:
                xs = raw[:,i].tolist(); total = sum(xs); square = sum(q*q for q in xs)
                helper.extend([total,square]); mean = total/len(xs); var = square/len(xs)-mean*mean
                assert var > EPSILON
                np.testing.assert_allclose([mean,var],[row[FIELDS[i]+'_mean'],row[FIELDS[i]+'_var']],rtol=2e-12,atol=2e-10)
                values.append((row[FIELDS[i]]-mean)/np.sqrt(max(var,EPSILON)))
            outputs.append(helper)
        expected = row[['Z'+n for n in FIELDS]].to_numpy(dtype=float)
        np.testing.assert_allclose(values, expected, rtol=0, atol=2e-10)
        encode = lambda x: np.floor(np.clip(100*np.asarray(x)+10000+.000001,0,999999))
        np.testing.assert_array_equal(encode(values),encode(expected))
        receipts.append(dict(date=sample['date'],code=sample['code'],helper_aggregate_outputs=outputs,values=values))
    assert len(receipts) == 32
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
        feature_verification_sha256=sha(ROOT / 'feature_verification.json'), samples=receipts,
        helper_interface_sha256=sha(ROOT / 'helper_interface.json'), raw_variables_units_and_two_helper_outputs_verified=True,
        original_raw_input_and_fixed_sample_proofs_reused=True, new_raw_prices_extracted=0,
        mathematical_interface_only=True, software_compilation_verified=False, native_source_parity_verified=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', proof)
    return {k:v for k,v in proof.items() if k!='samples'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['features','verify_features','native'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
