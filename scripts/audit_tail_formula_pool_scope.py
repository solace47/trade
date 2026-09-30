"""Audit only visible pool coverage after removing an added amount cutoff.

Does not read execution, future prices, outcome labels or strategy scores.
File presence is not evidence that the required minute windows are complete.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research.corporate_cash import MINUTES, save_json, sha

PROTOCOL = Path('config/tail_formula_pool_scope_audit_protocol.json')
INTENT = Path('config/tail_formula_pool_scope_intent.json')
ROOT = Path('data/research/tail_formula_pool_scope')
VISIBLE = Path('data/research/next_day_winner/visible_base.parquet')
ORIGINAL = Path('data/research/tail_formula_1000/universe.parquet')
META = ['date','code','half','board','decision_shares']
FIELDS = [*META,'price_1449','upper_limit','amount_1449','reference_gap','known_delisting','listing_age_sessions','isST','tradestatus']


def run():
    p = json.loads(PROTOCOL.read_text()); assert p['intent_sha256']==sha(INTENT)
    for file,digest in p['source_hashes'].items():
        assert sha(Path(file))==digest
    r = json.loads(VISIBLE.with_name('base_report.json').read_text()); assert r['base_sha256']==sha(VISIBLE)
    assert r['source_manifest_sha256']==sha(VISIBLE.with_name('source_manifest.json'))
    assert not (ROOT/'coverage_verification.json').exists(); ROOT.mkdir(parents=True,exist_ok=True)
    f = pd.read_parquet(VISIBLE,columns=FIELDS,filters=[('date','>=','2024-01-01'),('date','<','2026-01-01')])
    assert not f[['date','code']].duplicated().any()
    assert not f[['reference_gap','known_delisting']].isna().any().any()
    pressure = f.price_1449+np.maximum(f.price_1449*.0015,.005)
    eligible = f.board.eq('main') & f.listing_age_sessions.ge(60) & f.price_1449.le(200)
    eligible &= pressure.lt(f.upper_limit-.005) & f.decision_shares.gt(0) & ~f.reference_gap & ~f.known_delisting
    eligible &= pd.to_numeric(f.isST).eq(0) & pd.to_numeric(f.tradestatus).eq(1)
    original = pd.read_parquet(ORIGINAL,columns=META)
    retained = f.loc[eligible & f.amount_1449.ge(3e7),META].reset_index(drop=True)
    pd.testing.assert_frame_equal(retained,original,check_exact=True)
    broad = f.loc[eligible,META].reset_index(drop=True)
    broad['original_pool'] = f.loc[eligible,'amount_1449'].ge(3e7).to_numpy()
    extra = broad.loc[~broad.original_pool].reset_index(drop=True)
    c = base.conn()
    expected = c.sql(f'''SELECT date,code,half,board,decision_shares,amount_1449>=30000000 AS original_pool
        FROM read_parquet('{VISIBLE}') WHERE date>='2024-01-01' AND date<'2026-01-01'
        AND board='main' AND listing_age_sessions>=60 AND price_1449<=200 AND decision_shares>0
        AND price_1449+greatest(price_1449*.0015,.005)<upper_limit-.005
        AND NOT reference_gap AND NOT known_delisting AND cast(isST AS INT)=0 AND cast(tradestatus AS INT)=1
        ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(broad,expected,check_exact=True)
    assert broad.date.between('2024-01-01','2025-12-30').all()
    assert not broad.code.str.startswith(('sh.68','sz.30','bj.92')).any()
    broad.to_parquet(ROOT/'visible_pool.parquet',index=False,compression='zstd')
    coverage = []
    for item in p['input_caches']:
        file = Path(item['path']); assert sha(file)==item['sha256']
        keys = pd.read_parquet(file,columns=['date','code'],filters=[('date','>=','2024-01-01')])
        assert not keys.duplicated().any()
        matched = extra[META[:2]].merge(keys.assign(cached=True),on=['date','code'],how='left',validate='one_to_one')
        coverage.append(dict(name=item['name'],extra_rows=len(extra),covered_extra_rows=int(matched.cached.eq(True).sum()),
                             window_or_formula_validity_not_inferred_from_key_presence=True))
    minute = json.loads(Path(p['minute_manifest']).read_text())['source_sha256']; sources = []
    for code in sorted(extra.code.unique()):
        file = MINUTES/code[:2].upper()/(code[3:]+'.parquet')
        sources.append(dict(code=code,source=str(file),in_frozen_manifest=str(file) in minute,local_file_exists=file.exists()))
    halves = broad.groupby('half').agg(rows=('code','size'),old_rows=('original_pool','sum'),days=('date','nunique'),codes=('code','nunique')).reset_index()
    halves['extra_rows'] = halves.rows-halves.old_rows
    months = broad.assign(month=broad.date.str[:7]).groupby('month').agg(rows=('code','size'),old_rows=('original_pool','sum')).reset_index()
    months['extra_rows'] = months.rows-months.old_rows
    proof = dict(passed=True,protocol_sha256=sha(PROTOCOL),intent_sha256=sha(INTENT),
        visible_pool_sha256=sha(ROOT/'visible_pool.parquet'),original_universe_sha256=sha(ORIGINAL),
        original_rows=len(original),broader_rows=len(broad),extra_rows=len(extra),extra_codes=extra.code.nunique(),
        all_original_metadata_and_keys_retained=True,all_broad_rows_and_pressure_conditions_independent_sql_equal=True,
        by_half=halves.to_dict('records'),by_month=months.to_dict('records'),input_cache_coverage=coverage,
        extra_code_source_inventory=sources,raw_window_completeness_not_inferred_from_file_presence=True,
        no_extra_execution_or_morning_price_or_outcome_or_score_fields_read=True,
        no_new_raw_extraction_or_labels_or_fits=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'coverage_verification.json',proof)
    return {k:v for k,v in proof.items() if k not in ['by_month','extra_code_source_inventory']}


if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False,indent=2))
