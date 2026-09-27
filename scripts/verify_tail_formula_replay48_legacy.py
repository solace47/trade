"""Check the bounded forward adapter against fixed legacy raw-source cases."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_float as original
from trade_research import tail_formula_replay48 as replay
from trade_research.corporate_cash import DAILY, MINUTES, save_json, sha
from trade_research.tail_formula_forward import ROOT, PROTOCOL


def main():
    out = ROOT / 'legacy_replay_verification.json'
    if out.exists():
        raise ValueError('Do not replace the forward adapter legacy proof')
    names = list(original.EXPRESSIONS)
    report = json.loads((original.ROOT / 'feature_report.json').read_text())
    assert sha(original.ROOT / 'feature_report.json') == 'c8b2b2bf0f762fa8ce5fca811de5cdb110d307f00a260ba3ddc72e2f4e5c12e4'
    assert report['features_sha256'] == sha(original.ROOT / 'features.parquet')
    old = pd.read_parquet(original.ROOT / 'features.parquet', columns=['date','code','half','formula_input_valid',*names])
    old['sample_hash'] = [hashlib.sha256(('forward48-legacy-v1|'+d+'|'+c).encode()).hexdigest() for d,c in zip(old.date,old.code)]
    sample = old.sort_values('sample_hash').groupby(['half','formula_input_valid'], sort=True).head(32)
    sample = sample.sort_values(['date','code']).drop(columns='sample_hash').reset_index(drop=True)
    source = json.loads((base.SOURCE / 'feature_report.json').read_text())
    assert source['output_sha256']['universe.parquet'] == sha(base.SOURCE / 'universe.parquet')
    universe = sample[['date','code']].merge(pd.read_parquet(base.SOURCE / 'universe.parquet'), on=['date','code'], validate='one_to_one')
    codes = sorted(universe.code.unique())
    files = [MINUTES/c[:2].upper()/(c[3:]+'.parquet') for c in codes]
    m = Path('data/research/economic_winner/input_manifest.json')
    hashes = json.loads(m.read_text())['source_sha256']
    for p in files:
        assert sha(p) == hashes[str(p)]
    daily_files = [DAILY/(c.replace('.','_')+'.parquet') for c in codes]
    for p in daily_files:
        assert sha(p) == source['source_sha256'][str(p)]
    agg = replay.afternoon_aggregates(universe, files, '2024-01-01', '2025-12-30')
    daily = replay.read_daily(daily_files, '2023-06-01', '2025-12-30')
    market = original.context.market.ROOT
    indices = pd.read_parquet(market / 'indices.parquet')
    assert json.loads((market/'index_source_report.json').read_text())['indices_sha256'] == sha(market/'indices.parquet')
    context = original.context.ROOT
    points = pd.read_parquet(context / 'index_points.parquet')
    assert json.loads((context/'feature_report.json').read_text())['index_points_sha256'] == sha(context/'index_points.parquet')
    actual = replay.combine(universe, agg, daily, indices, points)
    pd.testing.assert_frame_equal(actual[['date','code']], sample[['date','code']], check_exact=True)
    np.testing.assert_allclose(actual[names], sample[names], atol=3e-9, rtol=2e-13, equal_nan=True)
    pd.testing.assert_series_equal(actual.formula_input_valid, sample.formula_input_valid, check_exact=True)
    base.EXPRESSIONS = original.EXPRESSIONS
    np.testing.assert_array_equal(base.encode(actual.loc[actual.formula_input_valid]), base.encode(sample.loc[sample.formula_input_valid]))
    r = dict(passed=True, protocol_sha256=sha(PROTOCOL), adapter_sha256=sha(Path(replay.__file__)),
        original_feature_report_sha256=sha(original.ROOT/'feature_report.json'), raw_manifest_sha256=sha(m),
        sample_salt='forward48-legacy-v1', cases=len(sample), fields=48, all_validity_flags_match=True,
        all_valid_integer_inputs_match=True, cases_per_half_and_validity=sample.groupby(['half','formula_input_valid']).size().reset_index(name='rows').to_dict('records'),
        sample_keys=sample[['date','code']].to_dict('records'), new_2026_prices_read=False,
        next_morning_outcomes_read=False, native_source_parity_verified=False)
    save_json(out, r)
    print(json.dumps({k:v for k,v in r.items() if k!='sample_keys'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
