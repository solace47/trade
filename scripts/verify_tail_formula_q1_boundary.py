"""Rebuild each Q1 29-minute observation with explicit timestamp-indexed arrays."""
import argparse
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd

from trade_research import tail_formula_q1_candidate_boundary as study
from trade_research import tail_formula_q1_candidates as coordinator
from trade_research.corporate_cash import save_json, sha
from verify_tail_formula_before1000 import array_observations


def observations():
    p, manifest, keys, raw = study.checked()
    report = json.loads((study.ROOT/'observation_report.json').read_text())
    assert report['raw_report_sha256'] == sha(study.OLD/'raw_report.json')
    assert report['observations_sha256'] == sha(study.ROOT/'observations.parquet')
    frames = []
    for file in raw['parts_sha256']:
        q = pd.read_parquet(file)
        q = q.loc[q.kind.eq('morning') & q.clock.between('09:31','09:59')].copy()
        if q.empty:
            continue
        a = q[['open','high','low','close','volume','amount']].to_numpy(float)
        o,h,l,cl,v,amt = a.T
        with np.errstate(divide='ignore', invalid='ignore'):
            price = amt/v
        q['valid'] = (np.isfinite(a).all(axis=1) & (a[:,:4]>0).all(axis=1)
            & (h+.0001>=a[:,:4].max(axis=1)) & (l-.0001<=np.minimum(o,cl))
            & (np.abs(a[:,:4]-np.rint(a[:,:4]*100)/100)<=.0001).all(axis=1)
            & (v>=0) & (amt>=0) & ((v==0)==(amt==0))
            & ((v==0) | ((price>=l-.0101)&(price<=h+.0101)))
            & q.timestamp.eq(q.timestamp.dt.floor('min')))
        q['active'] = q.valid & q.volume.gt(0)
        q['pos'] = q.timestamp.dt.hour*60+q.timestamp.dt.minute-571
        assert not q.duplicated(['date','code','pos']).any() and q.pos.between(0,28).all()
        groups = q.groupby(['date','code'], sort=True)
        out = groups.agg(bars_0959=('timestamp','size'),labels_0959=('clock','nunique'),valid_bars_0959=('valid','sum')).reset_index()
        matrices = [q.pivot(index=['date','code'], columns='pos', values=name).reindex(columns=range(29)) for name in ['close','low','active']]
        assert matrices[0].index.equals(pd.MultiIndex.from_frame(out[['date','code']]))
        x, low = [matrix.to_numpy(float) for matrix in matrices[:2]]
        active = matrices[2].eq(True).to_numpy(bool)
        values = array_observations(x,low,active,29)
        for name, value in values.items():
            out['price_0959' if name == 'price' else name+'_0959'] = value
        frames.append(out)
    expected = keys[['date','code','next_date']].merge(pd.concat(frames,ignore_index=True), on=['date','code'], how='left', validate='one_to_one')
    expected['source_valid_0959'] = expected.bars_0959.eq(29)&expected.labels_0959.eq(29)&expected.valid_bars_0959.eq(29)
    actual = pd.read_parquet(study.ROOT/'observations.parquet')
    pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False, check_exact=True)
    old = pd.read_parquet(study.OLD/'morning_windows.parquet')
    pd.testing.assert_frame_equal(old[['date','code','next_date']], actual[['date','code','next_date']], check_exact=True)
    assert actual.loc[old.source_valid, 'source_valid_0959'].all()
    result = dict(passed=True, observation_report_sha256=sha(study.ROOT/'observation_report.json'), rows=len(actual),
        all_29_timestamp_slots_and_activity_arrays_rebuilt=True, old_complete=int(old.source_valid.sum()),
        complete_29=int(actual.source_valid_0959.sum()), old_unknowns_not_recovered=True,
        inherited_raw_sample_proof_sha256=sha(study.OLD/'window_verification.json'),
        new_2026_prices_read=True, no_q2_signal_prices_read=True, no_exit_rules=True)
    save_json(study.ROOT/'observation_verification.json', result); return result


def labels():
    import verify_tail_formula_before1000 as verifier
    verifier.study = SimpleNamespace(ROOT=study.ROOT, PROTOCOL=coordinator.PROTOCOL,
        original=SimpleNamespace(ROOT=study.OLD), KEYS=['date','code','next_date'])
    return verifier.labels()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['observations','labels'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
