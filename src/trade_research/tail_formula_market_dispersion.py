"""Verified main-board dispersion levels as direct market-state inputs."""
import argparse
import json
import math
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_cross_dispersion as parent
from . import tail_formula_float as previous
from .corporate_cash import save_json, sha

STEM = 'tail_formula_market_dispersion'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
NEW_EXPRESSIONS = {'CD01': 'CZDS', 'CD02': 'CZTS'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = parent.HEADER
HELPER = parent.HELPER
native_core = parent.native_core


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    parent.checked_sources()
    r = json.loads((parent.ROOT / 'feature_report.json').read_text())
    v = json.loads((parent.ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(parent.ROOT / 'feature_report.json')
    for key, file in [('features', parent.ROOT / 'features.parquet'),
                      ('market_reference', parent.ROOT / 'market_reference.parquet'),
                      ('members', parent.equal.members_source.ROOT / 'members.parquet'),
                      ('helper', parent.ROOT / 'YJCS20.tdx')]:
        assert r[key + '_sha256'] == sha(file)
    assert p['minimum_members'] == 2000 and p['new_fields'] == list(NEW_EXPRESSIONS)
    assert not p['new_2026_prices_allowed']
    return p


def features():
    p = checked_sources()
    assert not (ROOT / 'feature_report.json').exists(), 'Do not replace frozen inputs'
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    daily = pd.read_parquet(parent.ROOT / 'market_reference.parquet')
    f = old.merge(daily, on='date', how='left', validate='many_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    valid = f.cz_members.ge(2000) & np.isfinite(f[['cz_day_std', 'cz_tail_std']]).all(axis=1)
    f['CD01'] = f.cz_day_std.where(valid)
    f['CD02'] = f.cz_tail_std.where(valid)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True, exist_ok=True)
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    (ROOT / 'YJCS20.tdx').write_text(HELPER)
    r = dict(protocol_sha256=sha(PROTOCOL), source_hashes=p['source_hashes'],
             features_sha256=sha(ROOT / 'features.parquet'), helper_sha256=sha(ROOT / 'YJCS20.tdx'),
             reused_market_reference_sha256=sha(parent.ROOT / 'market_reference.parquet'),
             rows=len(f), valid=int(f.formula_input_valid.sum()), newly_invalid=int((old.formula_input_valid & ~f.formula_input_valid).sum()),
             days=len(daily), minimum_members=int(daily.cz_members.min()),
             expressions=EXPRESSIONS, native_header=HEADER, native_core_gate=parent.GATE,
             native_auxiliary_indicator_required=True, native_source_parity_verified=False, software_compilation_verified=False,
             new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['source_hashes', 'expressions', 'native_header']}


def verify_features():
    checked_sources()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['features_sha256'] == sha(ROOT / 'features.parquet')
    assert r['helper_sha256'] == sha(ROOT / 'YJCS20.tdx') and (ROOT / 'YJCS20.tdx').read_text() == HELPER
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')], old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_series_equal(f.prior_formula_input_valid, old.formula_input_valid, check_names=False, check_exact=True)
    c = base.conn()
    c.read_parquet(str(parent.equal.members_source.ROOT / 'members.parquet')).create_view('members')
    stats = c.sql('''WITH atoms AS(SELECT date,100*(p49::DOUBLE/pc-1) AS d,
        100*(p49::DOUBLE/p20-1) AS t FROM members)
        SELECT date,count(*) AS n,sqrt(greatest(avg(d*d)-avg(d)*avg(d),0)) AS CD01,
            sqrt(greatest(avg(t*t)-avg(t)*avg(t),0)) AS CD02 FROM atoms GROUP BY date ORDER BY date''').df()
    c.close()
    e = old[['date', 'code']].merge(stats, on='date', how='left', validate='many_to_one')
    valid = e.n.ge(2000) & np.isfinite(e[list(NEW_EXPRESSIONS)]).all(axis=1)
    for name in NEW_EXPRESSIONS:
        e[name] = e[name].where(valid)
    np.testing.assert_allclose(f[list(NEW_EXPRESSIONS)], e[list(NEW_EXPRESSIONS)], rtol=0, atol=2e-11, equal_nan=True)
    final = old.formula_input_valid & valid
    np.testing.assert_array_equal(f.formula_input_valid, final)
    encode = lambda v: np.floor(np.clip(100 * v + 10000 + .000001, 0, 999999))
    np.testing.assert_array_equal(encode(f.loc[final, list(NEW_EXPRESSIONS)]), encode(e.loc[final, list(NEW_EXPRESSIONS)]))
    assert len(stats) == r['days'] == 484 and int(final.sum()) == r['valid']
    assert int((old.formula_input_valid & ~final).sum()) == r['newly_invalid']
    names = re.findall(r'(?m)^([A-Z][A-Z0-9]*):=', HEADER) + list(EXPRESSIONS)
    assert len(names) == len({n.casefold() for n in names})
    assert r['expressions'] == EXPRESSIONS and r['native_header'] == HEADER
    assert r['native_core_gate'] == parent.GATE == 'CZN>=2000'
    proof = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'),
                 all_484_dates_population_variances_mapping_validity_and_encodings_rebuilt=True,
                 all_original_keys_and_48_values_unchanged=True,
                 effective_input_intersection_unchanged=bool(np.array_equal(final, old.formula_input_valid)),
                 member_and_minute_source_proofs_reused=True, new_group_outcomes_read=False,
                 new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', proof)
    return proof


def native():
    checked_sources()
    v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    old = json.loads((parent.ROOT / 'native_input_verification.json').read_text())
    assert old['passed'] and len(old['samples']) == 32
    members = pd.read_parquet(parent.equal.members_source.ROOT / 'members.parquet', columns=['date', 'code', 'p49', 'p20', 'pc'])
    f = pd.read_parquet(ROOT / 'features.parquet', columns=['date', 'code', 'CD01', 'CD02']).set_index(['date', 'code'])
    cases = []
    for sample in old['samples']:
        d = members.loc[members.date.eq(sample['date'])]
        vals = []
        for col in ['pc', 'p20']:
            atoms = [100 * (p49 / den - 1) for p49, den in zip(d.p49, d[col])]
            mean = math.fsum(atoms) / len(atoms)
            vals.append(math.sqrt(max(math.fsum(a * a for a in atoms) / len(atoms) - mean * mean, 0)))
        row = f.loc[(sample['date'], sample['code'])]
        np.testing.assert_allclose(vals, row.to_numpy(float), rtol=0, atol=2e-11)
        np.testing.assert_array_equal(np.floor(100 * np.array(vals) + 10000 + .000001),
                                      np.floor(100 * row.to_numpy(float) + 10000 + .000001))
        cases.append(dict(date=sample['date'], code=sample['code'], members=len(d), CD01=vals[0], CD02=vals[1]))
    proof = dict(passed=True, protocol_sha256=sha(PROTOCOL), feature_report_sha256=sha(ROOT / 'feature_report.json'),
                 feature_verification_sha256=sha(ROOT / 'feature_verification.json'),
                 parent_native_proof_sha256=sha(parent.ROOT / 'native_input_verification.json'), cases=cases,
                 all_native_aggregate_mappings_and_encodings_rebuilt=True, original_raw_minutes_not_reextracted=True,
                 full_client_member_and_history_guards_verified=False, software_compilation_verified=False,
                 native_source_parity_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'native_input_verification.json', proof)
    return {k: v for k, v in proof.items() if k != 'cases'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'verify_features', 'native'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
