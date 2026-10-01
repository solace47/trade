"""Audit broad, visible market members before testing sentiment transitions."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from . import tail_formula_baseline as prior
from . import tail_formula_additive as base
from .research_io import check_runtime, check_sources, save_json, sha

STEM = 'tail_formula_emotion_transition'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
SOURCE_PROTOCOL = Path('config') / (STEM + '_source_protocol.json')
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
META, CONTROL = prior.META, prior.CONTROL


def transition_atoms(p20, p49, pc):
    """Strict sign crossings; flat prices are neither repair nor fade."""
    a, b, c = [np.asarray(x, float) for x in [p20, p49, pc]]
    assert a.shape == b.shape == c.shape
    good = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & (a > 0) & (b > 0) & (c > 0)
    cents = [np.floor(x * 100 + .5) for x in [a, b, c]]
    good &= np.logical_and.reduce([abs(x * 100 - y) <= .01 for x, y in zip([a, b, c], cents)])
    x, y, z = cents
    return np.column_stack([np.where(good, (x < z) & (y > z), np.nan),
                            np.where(good, (x > z) & (y < z), np.nan)])


def source_checked():
    check_runtime()
    p = json.loads(SOURCE_PROTOCOL.read_text())
    assert subprocess.check_output(['git', 'show', f'HEAD:{SOURCE_PROTOCOL}']) == SOURCE_PROTOCOL.read_bytes()
    assert p['stage'] == 'source_gate_only' and p['no_new_fits'] and p['no_group_economics']
    assert p['dates'] == ['2023-01-01', '2026-01-01'] and p['minimum_members'] == 2000
    assert p['listing_age_sessions_minimum'] == 60
    assert sha(Path(p['source_inventory'])) == p['source_inventory_sha256']
    inventory = json.loads(Path(p['source_inventory']).read_text())
    check_sources(inventory['source_hashes'])
    p.update(inventory)
    return p


def source_audit():
    p = source_checked()
    report = ROOT / 'source_gate.json'
    members_file = ROOT / 'source_members.parquet'
    assert not report.exists() and not members_file.exists(), 'Do not replace audited market sources'
    ROOT.mkdir(parents=True, exist_ok=True)
    c = base.conn()
    c.read_parquet(list(p['daily_sources'])).create_view('daily')
    c.read_parquet(list(p['historical_prefix'])).create_view('old_prefix')
    c.read_parquet(list(p['prefix_sources'])).create_view('recent_prefix')
    c.execute("""CREATE TABLE states AS WITH traded AS (
        SELECT date,code,preclose,isST,tradestatus,
        row_number() OVER(PARTITION BY code ORDER BY date)-1 AS age
        FROM daily WHERE date<'2026-01-01' AND tradestatus=1), all_states AS (
        SELECT d.date,d.code,d.preclose,d.isST,d.tradestatus,t.age
        FROM daily d LEFT JOIN traded t USING(date,code)
        WHERE d.date>='2023-01-01' AND d.date<'2026-01-01')
        SELECT * FROM all_states WHERE starts_with(code,'sh.60') OR starts_with(code,'sz.00')""")
    cols = ','.join(p['prefix_fields'])
    c.execute(f"""CREATE TABLE prefixes AS SELECT {cols} FROM old_prefix
        WHERE date>='2023-01-01' AND date<'2024-01-01'
        UNION ALL SELECT {cols} FROM recent_prefix WHERE date>='2024-01-01' AND date<'2026-01-01'""")
    for table in ['states', 'prefixes']:
        assert c.sql(f'SELECT count(*) FROM {table}').fetchone()[0] == c.sql(
            f'SELECT count(*) FROM (SELECT DISTINCT date,code FROM {table})').fetchone()[0], table
    d = c.sql("""WITH joined AS (SELECT coalesce(s.date,p.date) AS date,coalesce(s.code,p.code) AS code,
        s.date IS NOT NULL AS state_present,p.date IS NOT NULL AS prefix_present,
        s.preclose,s.isST,s.tradestatus,s.age,p.price_1420,p.price_1449,p.volume_1449,p.amount_1449
        FROM states s FULL OUTER JOIN prefixes p USING(date,code)) SELECT * FROM joined
        WHERE starts_with(code,'sh.60') OR starts_with(code,'sz.00') ORDER BY date,code""").df()
    static = d.state_present & d.isST.isin([0, 1]) & d.tradestatus.isin([0, 1])
    active = d.tradestatus.eq(1)
    static &= ~active | (np.isfinite(d.age) & d.age.ge(0))
    candidate = static & d.isST.eq(0) & active & d.age.ge(60)
    va = d[['volume_1449', 'amount_1449']].to_numpy(float)
    good_va = np.isfinite(va).all(axis=1) & (va >= 0).all(axis=1)
    good_va &= d.volume_1449.eq(np.floor(d.volume_1449))
    positive = (va > 0).all(axis=1)
    zero = (va == 0).all(axis=1)
    known_volume = d.prefix_present & good_va & (positive | zero)
    eligible = candidate & known_volume & positive
    prices = d[['price_1420', 'price_1449', 'preclose']].to_numpy(float)
    good_price = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
    good_price &= (abs(prices * 100 - np.floor(prices * 100 + .5)) <= .01).all(axis=1)
    d['unknown_qualification'] = (~static & d.prefix_present) | (candidate & ~known_volume)
    d['eligible'] = eligible
    d['bad_price'] = eligible & ~good_price
    daily = d.groupby('date').agg(members=('eligible', 'sum'),
        unknown_qualification=('unknown_qualification', 'sum'),bad_price=('bad_price', 'sum')).reset_index()
    # Independent SQL reconstructs eligibility and price quality without dropping bad members.
    c.register('joined', d)
    expected = c.sql("""WITH a AS (SELECT *,
        coalesce(state_present AND isST IN (0,1) AND tradestatus IN (0,1)
          AND (tradestatus<>1 OR (isfinite(age) AND age>=0)),false) AS s,
        coalesce(prefix_present AND isfinite(volume_1449) AND isfinite(amount_1449)
          AND volume_1449>=0 AND amount_1449>=0 AND volume_1449=floor(volume_1449)
          AND ((volume_1449>0 AND amount_1449>0) OR (volume_1449=0 AND amount_1449=0)),false) AS v,
        coalesce(isfinite(price_1420) AND isfinite(price_1449) AND isfinite(preclose)
          AND price_1420>0 AND price_1449>0 AND preclose>0
          AND abs(price_1420*100-round(price_1420*100))<=.01
          AND abs(price_1449*100-round(price_1449*100))<=.01
          AND abs(preclose*100-round(preclose*100))<=.01,false) AS q FROM joined),
        b AS (SELECT *,s AND isST=0 AND tradestatus=1 AND age>=60 AS candidate FROM a),
        z AS (SELECT *,coalesce(candidate AND v AND volume_1449>0 AND amount_1449>0,false) AS e FROM b)
        SELECT date,sum(e::INT) AS members,
        sum(((NOT s AND prefix_present) OR coalesce(candidate AND NOT v,false))::INT) AS unknown_qualification,
        sum((e AND NOT q)::INT) AS bad_price FROM z GROUP BY date ORDER BY date""").df()
    pd.testing.assert_frame_equal(daily, expected, check_exact=True, check_dtype=False)
    keys = prior.original()[META]
    daily['source_valid'] = daily.members.ge(2000) & daily.unknown_qualification.eq(0) & daily.bad_price.eq(0)
    mapped = keys.merge(daily, on='date', how='left', validate='many_to_one')
    new_invalid = keys.formula_input_valid & ~mapped.source_valid.fillna(False)
    d.to_parquet(members_file, index=False, compression='zstd')
    daily.to_parquet(ROOT / 'source_dates.parquet', index=False, compression='zstd')
    mapped.loc[new_invalid, ['date','code']].to_parquet(ROOT / 'newly_invalid_keys.parquet', index=False, compression='zstd')
    c.close()
    result = dict(passed=True, source_protocol_sha256=sha(SOURCE_PROTOCOL),implementation_sha256=sha(Path(__file__)),
        source_members_sha256=sha(members_file),source_dates_sha256=sha(ROOT / 'source_dates.parquet'),
        newly_invalid_keys_sha256=sha(ROOT / 'newly_invalid_keys.parquet'),dates=len(daily),
        source_valid_dates=int(daily.source_valid.sum()),minimum_known_members=int(daily.members.min()),
        unknown_qualification_rows=int(d.unknown_qualification.sum()),bad_price_rows=int(d.bad_price.sum()),
        original_keys=len(keys),original_valid=int(keys.formula_input_valid.sum()),newly_invalid=int(new_invalid.sum()),
        input_gate_passed=bool(not new_invalid.any()),all_eligibility_counts_independent_SQL_equal=True,
        no_buyability_or_later_prices_in_membership=True,primary_daily_states_not_1450_snapshot_membership=True,
        no_transition_values_fits_or_new_group_economics=True,new_2026_prices_read=False,no_exit_rules=True,
        software_compilation_verified=False,native_source_parity_verified=False)
    save_json(report,result)
    return result
