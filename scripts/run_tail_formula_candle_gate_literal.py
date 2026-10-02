"""Evaluate the source-fixed bullish doji/hammer gate using visible 14:49 bars."""
import argparse
import json
from functools import lru_cache
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd

from trade_research import tail_formula_baseline as baseline
from trade_research import tail_formula_additive as numeric
from trade_research.research_io import check_runtime, check_sources, save_json, sha
from tail_formula_reports import checked_selection
import finish_tail_formula_rule_search as shared

ROOT = Path('data/research/tail_formula_candle_gate_literal')
PROTOCOL = Path('config/tail_formula_candle_gate_literal_protocol.json')
KEYS = shared.KEYS
PRICES = ['price_1449', 'daily_open', 'high_1449', 'low_1449', 'preclose']


@lru_cache(maxsize=1)
def checked():
    check_runtime()
    assert subprocess.check_output(['git', 'show', f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    assert sha(PROTOCOL) in subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    p = json.loads(PROTOCOL.read_text()); check_sources(p['source_hashes'])
    assert p['selector_fits'] == p['new_tree_fits'] == 0
    assert not p['parameter_search'] and not p['window_search'] and not p['new_2026_prices_allowed']
    assert p['thresholds'] == dict(min_price=3, max_price=100, min_amount=100000000,
        max_abs_change=.095, min_range=.03, doji_body_max=.02, doji_shadow_min=.45,
        doji_shadow_imbalance_max=.06, hammer_body_min_exclusive=.03,
        hammer_body_max=.30, hammer_lower_body_min=2, hammer_upper_body_max=.50,
        hammer_lower_range_min=.60)
    registry = json.loads(Path(p['prior_registry']).read_text())
    return p, dict(p, input_protocol_sha256=sha(PROTOCOL),
        prior_analysis_roots=registry['prior_analysis_roots'],
        prior_completion_manifests=registry['prior_completion_manifests'])


def freeze():
    p, e = checked(); destination = ROOT/'joint_selection_freeze.json'
    assert not destination.exists()
    f = baseline.original().loc[lambda x: x.date.ge('2024-01-01'),
        [*KEYS, 'formula_input_valid']].reset_index(drop=True)
    assert len(f) == 1258085 and f.date.lt('2026-01-01').all()
    a = pd.read_parquet(p['visible_cache'], columns=[*KEYS, 'formula_input_valid', *PRICES, 'amount_1449'])
    pd.testing.assert_frame_equal(f, a[[*KEYS, 'formula_input_valid']], check_exact=True)
    report = json.loads(Path(p['visible_report']).read_text())
    proof = json.loads(Path(p['visible_proof']).read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(Path(p['visible_report']))
    assert report['features_sha256'] == sha(Path(p['visible_cache']))
    valid = f.formula_input_valid
    good = np.isfinite(a[PRICES + ['amount_1449']]).all(axis=1)
    good &= a[PRICES + ['amount_1449']].gt(0).all(axis=1)
    good &= (abs(a[PRICES]*100 - np.floor(a[PRICES]*100+.5))<=.01).all(axis=1)
    good &= a.low_1449.le(a.price_1449) & a.high_1449.ge(a.price_1449)
    assert good.loc[valid].all(), 'Input domain changed: fix matched controls before evaluation'
    cents = np.floor(a[PRICES].fillna(0)*100+.5).astype(np.int64)
    q, o, h, l, prev = [cents[x] for x in PRICES]
    # These MAX/MIN adjustments are explicitly present in the original source.
    high = np.maximum(np.maximum(h, o), q); low = np.minimum(np.minimum(l, o), q)
    r, b, u, d = high-low, q-o, high-q, o-low
    doji = (50*b<=r) & (20*u>=9*r) & (20*d>=9*r) & (50*abs(u-d)<=3*r)
    hammer = (100*b>3*r) & (10*b<=3*r) & (d>=2*b) & (2*u<=b) & (5*d>=3*r)
    scan = q.between(300,10000) & a.amount_1449.ge(100000000) & (1000*abs(q-prev)<=95*prev)
    common = scan & (b>0) & (r>0) & (100*r>=3*o)
    flags = valid & common & (doji | hammer)
    sql_input = a[['date','code','formula_input_valid',*PRICES,'amount_1449']]
    c = numeric.conn(); c.register('visible', sql_input)
    expected = c.sql('''WITH raw AS (
        SELECT *,round(price_1449*100)::BIGINT AS q,round(daily_open*100)::BIGINT AS o,
        round(high_1449*100)::BIGINT AS h,round(low_1449*100)::BIGINT AS l,
        round(preclose*100)::BIGINT AS prev FROM visible), shadows AS (
        SELECT *, greatest(h,o,q)-least(l,o,q) AS r, q-o AS b,
        greatest(h,o,q)-q AS u, o-least(l,o,q) AS d FROM raw)
        SELECT date,code,coalesce(formula_input_valid AND q BETWEEN 300 AND 10000
        AND amount_1449>=100000000 AND 1000*abs(q-prev)<=95*prev AND b>0 AND r>0
        AND 100*r>=3*o AND ((50*b<=r AND 20*u>=9*r AND 20*d>=9*r AND 50*abs(u-d)<=3*r)
        OR (100*b>3*r AND 10*b<=3*r AND d>=2*b AND 2*u<=b AND 5*d>=3*r)),false) AS selected,
        coalesce(isfinite(price_1449) AND isfinite(daily_open) AND isfinite(high_1449)
        AND isfinite(low_1449) AND isfinite(preclose) AND isfinite(amount_1449)
        AND price_1449>0 AND daily_open>0 AND high_1449>0 AND low_1449>0 AND preclose>0
        AND amount_1449>0 AND abs(price_1449*100-round(price_1449*100))<=.01
        AND abs(daily_open*100-round(daily_open*100))<=.01
        AND abs(high_1449*100-round(high_1449*100))<=.01
        AND abs(low_1449*100-round(low_1449*100))<=.01
        AND abs(preclose*100-round(preclose*100))<=.01
        AND low_1449<=price_1449 AND high_1449>=price_1449,false) AS input_good
        FROM shadows ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(f[['date','code']], expected[['date','code']], check_exact=True)
    np.testing.assert_array_equal(flags, expected.selected)
    np.testing.assert_array_equal(good, expected.input_good)
    save_json(ROOT/'input_verification.json', dict(passed=True, rows=len(f), valid=int(valid.sum()),
        newly_invalid=0, all_cents_shadows_thresholds_qualification_flags_SQL_rebuilt=True,
        source_max_min_normalization_retained=True,
        original_keys_qualification_and_metadata_exactly_preserved=True,
        new_outcomes_read=False, native_client_parity_verified=False, new_2026_prices_read=False))
    receipts = dict(p['source_hashes'], **{str(PROTOCOL):sha(PROTOCOL)})
    lists, lookup = [], []
    for year in ['2024','2025']:
        root = ROOT/('rule'+year); assert not root.exists(); root.mkdir()
        out = f[KEYS].copy(); out['selected'] = flags & f.date.str.startswith(year)
        out.to_parquet(root/'selection.parquet',index=False,compression='zstd')
        chosen = out.loc[out.selected]; sizes = chosen.groupby('date').size()
        equivalent = []
        for old in e['prior_analysis_roots']:
            old = Path(old); record = json.loads((old/'selection_report.json').read_text())
            if record['selected']==len(chosen) and checked_selection(old).equals(out):
                equivalent.append(str(old))
        lookup.append(dict(year=year, prior_selection_roots_examined=len(e['prior_analysis_roots']),
            exact_equivalent_complete_lists=equivalent, no_economic_statistics_read=True))
        save_json(root/'selection_report.json', dict(protocol_sha256=sha(PROTOCOL),
            selection_sha256=sha(root/'selection.parquet'), rows=len(out), selected=len(chosen),days=len(sizes),
            half_counts=chosen.groupby('half').agg(rows=('code','size'),days=('date','nunique')).reset_index().to_dict('records'),
            median_daily=float(sizes.median()) if len(sizes) else None,
            max_daily=int(sizes.max()) if len(sizes) else None,
            largest_day_fraction=float(sizes.max()/len(chosen)) if len(chosen) else None,
            condition='source_fixed_scan AND bullish_large_range AND (perfect_doji OR hammer)',
            shape_input_counts=dict(doji=int((out.selected & doji).sum()),hammer=int((out.selected & hammer).sum())),
            no_outcome_or_fill_filter=True, new_2026_prices_read=False,no_exit_rules=True))
        save_json(root/'selection_verification.json', dict(passed=True,
            selection_report_sha256=sha(root/'selection_report.json'),
            all_conditions_qualification_scopes_flags_and_metadata_SQL_rebuilt=True,
            input_verification_sha256=sha(ROOT/'input_verification.json'),native_client_parity_verified=False))
        lists.append(dict(group='rule'+year,root=str(root),selected=len(chosen),days=len(sizes)))
        control=Path(e['controls'][year]); old=checked_selection(control)
        pd.testing.assert_frame_equal(f[KEYS],old[KEYS],check_exact=True)
        lists.append(dict(group='control'+year,root=str(control),original_unchanged=True))
        for path in [root,control]:
            for name in ['selection.parquet','selection_report.json','selection_verification.json']:
                receipts[str(path/name)]=sha(path/name)
    save_json(ROOT/'selection_equivalence_lookup.json',dict(passed=True,records=lookup,
        keyword_absence_not_used_as_novelty_proof=True,new_model_fits=0))
    for name in ['input_verification.json','selection_equivalence_lookup.json']:
        receipts[str(ROOT/name)]=sha(ROOT/name)
    save_json(destination,dict(passed=True,input_protocol_sha256=sha(PROTOCOL),execution_protocol_sha256=sha(PROTOCOL),
        source_hashes=receipts,selections=lists,fits_completed=0,
        all_two_year_complete_lists_fixed_together_before_economics=True,
        new_economic_outcomes_read=False,new_2026_prices_read=False))
    return dict(joint_sha256=sha(destination),selections=lists,equivalence=lookup)


shared.ROOT=ROOT
shared.EXECUTION=PROTOCOL
shared.fit=SimpleNamespace(PROTOCOL=PROTOCOL,EXECUTION=PROTOCOL)
shared.checked=checked

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','analyze','finish'])
    stage=parser.parse_args().stage
    print(json.dumps(freeze() if stage=='freeze' else getattr(shared,stage)(),ensure_ascii=False),flush=True)
