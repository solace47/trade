"""Conservative 09:59 Q1 marks, preserving the original 30-bar unknown states."""
import argparse
import json

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_before1000 as boundary
from . import tail_formula_q1_candidate_observations as observations
from . import tail_formula_q1_candidates as study
from .corporate_cash import save_json, sha

ROOT = study.ROOT/'before1000'
OLD = observations.LABELS


def checked():
    p, manifest, keys = observations.checked_keys()
    r = json.loads((OLD/'raw_report.json').read_text())
    proof = json.loads((OLD/'window_verification.json').read_text())
    windows = json.loads((OLD/'window_report.json').read_text())
    assert proof['passed'] and proof['window_report_sha256'] == sha(OLD/'window_report.json')
    assert windows['raw_report_sha256'] == sha(OLD/'raw_report.json')
    for file, digest in r['parts_sha256'].items():
        from pathlib import Path
        assert sha(Path(file)) == digest
    return p, manifest, keys, r


def build():
    p, manifest, keys, raw = checked(); assert not (ROOT/'observation_report.json').exists()
    c = base.conn(); c.read_parquet(list(raw['parts_sha256'])).create_view('raw'); c.register('keys', keys[boundary.KEYS])
    f = c.sql('''WITH b AS(SELECT *,coalesce(timestamp=date_trunc('minute',timestamp)
        AND isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close)
        AND isfinite(volume) AND isfinite(amount) AND least(open,high,low,close)>0
        AND high+.0001>=greatest(open,close,low) AND low-.0001<=least(open,close)
        AND abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
        AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001
        AND volume>=0 AND amount>=0 AND(volume=0)=(amount=0)
        AND(volume=0 OR amount/volume BETWEEN low-.0101 AND high+.0101),false) AS valid
        FROM raw WHERE kind='morning' AND clock BETWEEN '09:31' AND '09:59'),
        rolling AS(SELECT *,min(close) OVER w AS low_three,count(*) OVER w AS n_three,
        count(*) FILTER(WHERE valid AND volume>0) OVER w AS active_three,min(timestamp) OVER w AS first_three
        FROM b WINDOW w AS(PARTITION BY date,code ORDER BY timestamp ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)),
        g AS(SELECT date,code,count(*) AS bars_0959,count(DISTINCT clock) AS labels_0959,
        count(*) FILTER(WHERE valid) AS valid_bars_0959,count(*) FILTER(WHERE valid AND volume>0) AS active_minutes_0959,
        max(close) FILTER(WHERE valid AND volume>0) AS max_close_0959,
        max(low_three) FILTER(WHERE n_three=3 AND active_three=3 AND timestamp-first_three=INTERVAL 2 MINUTE) AS sustained_close_0959,
        min(low) FILTER(WHERE valid AND volume>0) AS min_low_0959,
        max(close) FILTER(WHERE clock='09:59' AND valid AND volume>0) AS price_0959
        FROM rolling GROUP BY date,code)
        SELECT k.*,g.* EXCLUDE(date,code),coalesce(bars_0959=29 AND labels_0959=29 AND valid_bars_0959=29,false) AS source_valid_0959
        FROM keys k LEFT JOIN g USING(date,code) ORDER BY date,code''').df(); c.close()
    ROOT.mkdir(parents=True, exist_ok=True); f.to_parquet(ROOT/'observations.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(study.PROTOCOL), input_manifest_sha256=sha(OLD/'input_manifest.json'),
        raw_report_sha256=sha(OLD/'raw_report.json'), old_window_verification_sha256=sha(OLD/'window_verification.json'),
        observations_sha256=sha(ROOT/'observations.parquet'), rows=len(f), window_labels=['09:31','09:59'],
        original_unknowns_not_recovered=True, new_2026_prices_read=True, no_q2_signal_prices_read=True, no_exit_rules=True)
    save_json(ROOT/'observation_report.json', r); return r


def labels():
    p, manifest, keys, raw = checked(); assert not (ROOT/'full_label_report.json').exists()
    proof = json.loads((ROOT/'observation_verification.json').read_text())
    old_proof = json.loads((OLD/'full_label_verification.json').read_text())
    assert proof['passed'] and proof['observation_report_sha256'] == sha(ROOT/'observation_report.json')
    assert old_proof['passed'] and old_proof['label_report_sha256'] == sha(OLD/'full_label_report.json')
    assert json.loads((OLD/'full_label_report.json').read_text())['labels_sha256'] == sha(OLD/'full_labels.parquet')
    obs = pd.read_parquet(ROOT/'observations.parquet'); old = pd.read_parquet(OLD/'full_labels.parquet')
    pd.testing.assert_frame_equal(obs[boundary.KEYS], old[boundary.KEYS], check_exact=True)
    r = old.copy()
    for new, source in [('active_minutes','active_minutes_0959'),('max_close','max_close_0959'),
                        ('sustained_close','sustained_close_0959'),('min_low','min_low_0959')]:
        r[new] = obs[source]
    r = r.drop(columns=['price_1000','mark_1000_return5','mark_1000_return15'])
    r['price_0959'] = obs.price_0959; r['source_valid_0959'] = obs.source_valid_0959
    for bps in [5,15]:
        known = old[f'known{bps}']; assert r.loc[known, 'source_valid_0959'].all()
        for name, price in [('sustained',r.sustained_close),('any_close',r.max_close),('mark_0959',r.price_0959),('adverse',r.min_low)]:
            r[f'{name}_return{bps}'] = boundary.net_mark(price, r.decision_shares, r[f'buy_cash{bps}'], bps).where(known)
        r[f'opportunity{bps}'] = r[f'sustained_return{bps}'].gt(0).astype(float).where(known)
        r[f'any_opportunity{bps}'] = r[f'any_close_return{bps}'].gt(0).astype(float).where(known)
        r[f'one_percent{bps}'] = r[f'sustained_return{bps}'].ge(.01).astype(float).where(known)
        assert r.loc[known,f'opportunity{bps}'].le(old.loc[known,f'opportunity{bps}']).all()
    r.to_parquet(ROOT/'full_labels.parquet', index=False, compression='zstd')
    report = dict(protocol_sha256=sha(study.PROTOCOL), rows=len(r), labels_sha256=sha(ROOT/'full_labels.parquet'),
        old_label_report_sha256=sha(OLD/'full_label_report.json'), old_label_verification_sha256=sha(OLD/'full_label_verification.json'),
        observation_verification_sha256=sha(ROOT/'observation_verification.json'), window_labels=['09:31','09:59'], window_bars=29,
        mark_reference_label='09:59', all_original_statuses_preserved=True, no_reclassification_of_old_unknowns=True,
        new_2026_prices_read=True, no_q2_signal_prices_read=True, no_exit_rules=True)
    save_json(ROOT/'full_label_report.json', report); return report


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['build','labels'])
    print(json.dumps(globals()[p.parse_args().stage](), ensure_ascii=False, indent=2))
