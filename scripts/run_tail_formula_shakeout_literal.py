"""Test four source-fixed price/volume conditions using the 14:49 prefix."""
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

ROOT = Path('data/research/tail_formula_shakeout_literal')
PROTOCOL = Path('config/tail_formula_shakeout_literal_protocol.json')
KEYS = shared.KEYS
VISIBLE = ['price_1449', 'low_1449', 'daily_open', 'volume_1449',
           'float_source_date', 'float_prior_volume', 'float_prior_adjustflag']
PRIOR = ['py_prev_date', 'py_comp_date', 'py_close', 'py_comp_close',
         'py_prev_adjustflag', 'py_comp_adjustflag']


@lru_cache(maxsize=1)
def checked():
    check_runtime()
    assert subprocess.check_output(['git', 'show', f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    assert sha(PROTOCOL) in subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    p = json.loads(PROTOCOL.read_text()); check_sources(p['source_hashes'])
    assert p['selector_fits'] == p['new_tree_fits'] == 0
    assert p['condition'] == 'py_close>=1.095*py_comp_close AND price_1449<daily_open AND volume_1449>2*float_prior_volume AND low_1449>=py_close'
    assert p['require_unchanged_original_input_domain'] and not p['new_2026_prices_allowed']
    registry = json.loads(Path(p['prior_registry']).read_text())
    e = dict(p, input_protocol_sha256=sha(PROTOCOL),
             prior_analysis_roots=registry['prior_analysis_roots'],
             prior_completion_manifests=registry['prior_completion_manifests'])
    return p, e


def freeze():
    p, e = checked(); destination = ROOT/'joint_selection_freeze.json'
    assert not destination.exists()
    f = baseline.original().loc[lambda x: x.date.ge('2024-01-01'), [*KEYS, 'formula_input_valid']].reset_index(drop=True)
    assert len(f) == 1258085 and f.date.lt('2026-01-01').all()
    a = pd.read_parquet(p['visible_cache'], columns=[*KEYS, 'formula_input_valid', *VISIBLE])
    b = pd.read_parquet(p['prior_cache'], columns=['date', 'code', *PRIOR])
    pd.testing.assert_frame_equal(f, a[[*KEYS, 'formula_input_valid']], check_exact=True)
    pd.testing.assert_frame_equal(f[['date', 'code']], b[['date', 'code']], check_exact=True)
    for kind, report_path, proof_path in p['input_proofs']:
        report, proof = json.loads(Path(report_path).read_text()), json.loads(Path(proof_path).read_text())
        assert proof['passed'] and proof['feature_report_sha256'] == sha(Path(report_path))
        assert report['features_sha256'] == sha(Path(p['visible_cache'] if kind=='visible' else p['prior_features']))
        if kind=='prior':
            assert report['primitives_sha256'] == sha(Path(p['prior_cache']))
    f = f.merge(a[['date', 'code', *VISIBLE]], on=['date', 'code'], validate='one_to_one')
    f = f.merge(b, on=['date', 'code'], validate='one_to_one')
    valid = f.formula_input_valid
    good = f.py_comp_date.lt(f.py_prev_date) & f.py_prev_date.lt(f.date)
    good &= f.float_source_date.eq(f.py_prev_date)
    good &= f[['float_prior_adjustflag', 'py_prev_adjustflag', 'py_comp_adjustflag']].eq(3).all(axis=1)
    prices = ['price_1449', 'low_1449', 'daily_open', 'py_close', 'py_comp_close']
    amounts = ['volume_1449', 'float_prior_volume']
    good &= np.isfinite(f[prices + amounts]).all(axis=1) & f[prices + amounts].gt(0).all(axis=1)
    good &= (abs(f[prices]*100 - np.floor(f[prices]*100+.5))<=.01).all(axis=1)
    good &= (abs(f[amounts]-np.floor(f[amounts]+.5))<=.0001).all(axis=1)
    good &= f.low_1449.le(f.price_1449)
    # An input-domain change is a separate study; never silently filter old controls.
    assert good.loc[valid].all(), 'Input intersection changed: fix matched controls before evaluation'
    flags = valid & f.py_close.ge(1.095*f.py_comp_close) & f.price_1449.lt(f.daily_open)
    flags &= f.volume_1449.gt(2*f.float_prior_volume) & f.low_1449.ge(f.py_close)
    c = numeric.conn(); c.register('visible', f)
    expected = c.sql('''SELECT date,code,coalesce(formula_input_valid
        AND py_close>=1.095*py_comp_close AND price_1449<daily_open
        AND volume_1449>2*float_prior_volume AND low_1449>=py_close,false) AS selected,
        coalesce(py_comp_date<py_prev_date AND py_prev_date<date
        AND float_source_date=py_prev_date AND float_prior_adjustflag=3
        AND py_prev_adjustflag=3 AND py_comp_adjustflag=3
        AND isfinite(price_1449) AND isfinite(low_1449) AND isfinite(daily_open)
        AND isfinite(py_close) AND isfinite(py_comp_close)
        AND price_1449>0 AND low_1449>0 AND daily_open>0 AND py_close>0 AND py_comp_close>0
        AND abs(price_1449*100-round(price_1449*100))<=.01
        AND abs(low_1449*100-round(low_1449*100))<=.01
        AND abs(daily_open*100-round(daily_open*100))<=.01
        AND abs(py_close*100-round(py_close*100))<=.01
        AND abs(py_comp_close*100-round(py_comp_close*100))<=.01
        AND isfinite(volume_1449) AND isfinite(float_prior_volume)
        AND volume_1449>0 AND float_prior_volume>0
        AND abs(volume_1449-round(volume_1449))<=.0001
        AND abs(float_prior_volume-round(float_prior_volume))<=.0001
        AND low_1449<=price_1449,false) AS input_good
        FROM visible ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(f[['date', 'code']], expected[['date', 'code']], check_exact=True)
    np.testing.assert_array_equal(flags, expected.selected)
    np.testing.assert_array_equal(good, expected.input_good)
    keys = f[KEYS].copy(); receipts = dict(p['source_hashes'], **{str(PROTOCOL):sha(PROTOCOL)})
    save_json(ROOT/'input_verification.json', dict(passed=True, rows=len(f), valid=int(valid.sum()),
        newly_invalid=0, input_dates_price_volume_units_and_cents_SQL_rebuilt=True,
        original_keys_qualification_and_metadata_exactly_preserved=True,
        prior_cache_verified_lags_and_current_prefix_reused=True,
        current_full_day_close_low_volume_not_used=True, input_latest_stock_label='14:49',
        new_outcomes_read=False, native_client_parity_verified=False, new_2026_prices_read=False))
    lists, lookup = [], []
    for year in ['2024', '2025']:
        root = ROOT/('rule'+year); assert not root.exists(); root.mkdir()
        out = keys.copy(); out['selected'] = flags & f.date.str.startswith(year)
        out.to_parquet(root/'selection.parquet', index=False, compression='zstd')
        chosen = out.loc[out.selected]; sizes = chosen.groupby('date').size()
        equivalent = []
        for old in e['prior_analysis_roots']:
            old = Path(old); record = json.loads((old/'selection_report.json').read_text())
            if record['selected'] == len(chosen) and checked_selection(old).equals(out):
                equivalent.append(str(old))
        lookup.append(dict(year=year, prior_selection_roots_examined=len(e['prior_analysis_roots']),
            exact_equivalent_complete_lists=equivalent, no_economic_statistics_read=True))
        save_json(root/'selection_report.json', dict(protocol_sha256=sha(PROTOCOL),
            selection_sha256=sha(root/'selection.parquet'), rows=len(out), selected=len(chosen), days=len(sizes),
            half_counts=chosen.groupby('half').agg(rows=('code','size'), days=('date','nunique')).reset_index().to_dict('records'),
            median_daily=float(sizes.median()) if len(sizes) else None, max_daily=int(sizes.max()) if len(sizes) else None,
            largest_day_fraction=float(sizes.max()/len(chosen)) if len(chosen) else None,
            condition=p['condition'], no_outcome_or_fill_filter=True, new_2026_prices_read=False, no_exit_rules=True))
        save_json(root/'selection_verification.json', dict(passed=True,
            selection_report_sha256=sha(root/'selection_report.json'),
            all_four_conditions_dates_qualification_flags_and_metadata_SQL_rebuilt=True,
            input_verification_sha256=sha(ROOT/'input_verification.json'), native_client_parity_verified=False))
        lists.append(dict(group='rule'+year, root=str(root), selected=len(chosen), days=len(sizes)))
        control = Path(e['controls'][year]); old = checked_selection(control)
        pd.testing.assert_frame_equal(keys, old[KEYS], check_exact=True)
        lists.append(dict(group='control'+year, root=str(control), original_unchanged=True))
        for path in [root, control]:
            for name in ['selection.parquet','selection_report.json','selection_verification.json']:
                receipts[str(path/name)] = sha(path/name)
    save_json(ROOT/'selection_equivalence_lookup.json', dict(passed=True, records=lookup,
        keyword_absence_not_used_as_novelty_proof=True, new_model_fits=0))
    for name in ['input_verification.json', 'selection_equivalence_lookup.json']:
        receipts[str(ROOT/name)] = sha(ROOT/name)
    save_json(destination, dict(passed=True, input_protocol_sha256=sha(PROTOCOL), execution_protocol_sha256=sha(PROTOCOL),
        source_hashes=receipts, selections=lists, fits_completed=0,
        all_two_year_complete_lists_fixed_together_before_economics=True,
        new_economic_outcomes_read=False, new_2026_prices_read=False))
    return dict(joint_sha256=sha(destination), selections=lists, equivalence=lookup)


shared.ROOT = ROOT
shared.EXECUTION = PROTOCOL
shared.fit = SimpleNamespace(PROTOCOL=PROTOCOL, EXECUTION=PROTOCOL)
shared.checked = checked

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'analyze', 'finish'])
    stage = parser.parse_args().stage
    print(json.dumps(freeze() if stage=='freeze' else getattr(shared, stage)(), ensure_ascii=False), flush=True)
