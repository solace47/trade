"""Training-only net price area over all 29 scheduled next-morning bars."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_before1000 as source
from .corporate_cash import save_json, sha

ROOT = Path('data/research/tail_formula_path_area_labels')
PROTOCOL = Path('config/tail_formula_path_area_protocol.json')


def price_area(marks, active):
    assert marks.shape == active.shape and marks.ndim == 2 and marks.shape[1] == 29
    assert active.dtype == bool and np.isfinite(marks[active]).all()
    return np.where(active, marks, 0.).sum(axis=1) / 29


def source_keys():
    p = json.loads(PROTOCOL.read_text())
    assert p['target_window'] == ['09:31', '09:59'] and p['target_window_bars'] == 29
    assert p['training_first'] == '2024-01-01' and p['training_observation_end_exclusive'] == '2025-07-01'
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    report = json.loads((source.ROOT / 'full_label_report.json').read_text())
    proof = json.loads((source.ROOT / 'full_label_verification.json').read_text())
    assert proof['passed'] and proof['label_report_sha256'] == sha(source.ROOT / 'full_label_report.json')
    assert report['labels_sha256'] == sha(source.ROOT / 'full_labels.parquet')
    c = base.conn()
    keys = c.execute('''SELECT date,code,next_date,half,known15,known_no_trade,decision_shares,buy_cash15,
        opportunity15,adverse_return15 FROM read_parquet(?)
        WHERE date>='2024-01-01' AND next_date<'2025-07-01' AND known15 ORDER BY date,code''',
        [str(source.ROOT / 'full_labels.parquet')]).df(); c.close()
    assert keys.known15.all() and not keys.known_no_trade.any()
    return keys


def target():
    keys = source_keys(); assert not (ROOT / 'full_label_report.json').exists()
    report = json.loads((source.ROOT / 'observation_report.json').read_text()); parts = []
    for file, digest in report['parts_sha256'].items():
        assert sha(Path(file)) == digest
        f = pd.read_parquet(file, columns=['date','code','next_date','source_valid','close_values','active_mask'],
            filters=[('date','>=','2024-01-01'),('next_date','<','2025-07-01')])
        f = keys.merge(f, on=['date','code','next_date'], validate='one_to_one')
        if not len(f): continue
        assert f.source_valid.all()
        closes = np.stack(f.close_values)[:,:29]
        active = (f.active_mask.to_numpy('int64')[:,None] & (1 << np.arange(29))) != 0
        marks = source.net_mark(closes, f.decision_shares.to_numpy()[:,None], f.buy_cash15.to_numpy()[:,None], 15)
        positive = active & (marks > 0)
        success = (positive[:,:-2] & positive[:,1:-1] & positive[:,2:]).any(axis=1)
        np.testing.assert_array_equal(success, f.opportunity15.eq(1))
        out = f[keys.columns].copy()
        out['active_bars29'] = active.sum(axis=1)
        out['path_area15'] = price_area(marks, active)
        out['active_mean_mark15'] = out.path_area15 * 29 / out.active_bars29.replace(0, np.nan)
        parts.append(out)
    out = pd.concat(parts, ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(out[keys.columns], keys, check_exact=True)
    assert np.isfinite(out.path_area15).all()
    ROOT.mkdir(parents=True, exist_ok=True); out.to_parquet(ROOT / 'full_labels.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), labels_sha256=sha(ROOT / 'full_labels.parquet'),
        source_label_report_sha256=sha(source.ROOT / 'full_label_report.json'),
        source_observation_report_sha256=sha(source.ROOT / 'observation_report.json'), rows=len(out),
        first_signal=out.date.min(), last_observation=out.next_date.max(), target_field='path_area15',
        training_scale=100, scheduled_bars=29, scope='training_target_only', not_for_evaluation=True,
        by_half=out.groupby('half').agg(rows=('code','size'),
            positive_area=('path_area15', lambda v:int(v.gt(0).sum())),
            zero_active=('active_bars29',lambda v:int(v.eq(0).sum()))).reset_index().to_dict('records'),
        original_known_keys_and_positive_opportunity_reproduced=True,
        inactive_contribution_is_neutral_training_utility_not_realized_zero=True,
        path_area_is_not_realized_return=True,new_2025H2_path_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT / 'full_label_report.json', r); return r


def verify():
    keys = source_keys(); r = json.loads((ROOT / 'full_label_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL) and r['labels_sha256'] == sha(ROOT / 'full_labels.parquet')
    assert r['target_field'] == 'path_area15' and r['training_scale'] == 100 and r['scheduled_bars'] == 29
    got = pd.read_parquet(ROOT / 'full_labels.parquet')
    pd.testing.assert_frame_equal(got[keys.columns], keys, check_exact=True)
    parts = json.loads((source.ROOT / 'observation_report.json').read_text())['parts_sha256']
    for file, digest in parts.items(): assert sha(Path(file)) == digest
    c = base.conn(); c.register('keys', keys); c.read_parquet(list(parts)).create_view('raw_parts')
    ex = c.sql('''WITH bars AS(SELECT k.*,pos,list_extract(p.close_values,pos+1) AS close,
        (p.active_mask&(1::BIGINT<<pos))<>0 AS active
        FROM keys k JOIN raw_parts p USING(date,code,next_date) CROSS JOIN range(29) t(pos)
        WHERE p.date>='2024-01-01' AND p.next_date<'2025-07-01'),
        cash AS(SELECT *,decision_shares*(close-greatest(.005,close*.0015)) AS value FROM bars),
        marks AS(SELECT *, (value-greatest(5,value*.0003)-value*.00051)/buy_cash15-1 AS mark FROM cash),
        windows AS(SELECT *,sum((active AND mark>0)::INT) OVER w AS npositive,count(*) OVER w AS n
            FROM marks WINDOW w AS(PARTITION BY date,code ORDER BY pos ROWS BETWEEN 2 PRECEDING AND CURRENT ROW))
        SELECT date,code,count(*) AS bars,sum(active::INT) AS active_bars29,
        avg(CASE WHEN active THEN mark ELSE 0 END) AS path_area15,
        avg(mark) FILTER(WHERE active) AS active_mean_mark15,
        max((npositive=3 AND n=3)::INT) AS opportunity15
        FROM windows GROUP BY date,code ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(ex[['date','code']],got[['date','code']],check_exact=True)
    assert ex.bars.eq(29).all()
    for name in ['active_bars29','opportunity15']:
        np.testing.assert_array_equal(got[name],ex[name])
    for name in ['path_area15','active_mean_mark15']:
        np.testing.assert_allclose(got[name],ex[name],rtol=0,atol=2e-12,equal_nan=True)
    for row in r['by_half']:
        q = got.loc[got.half.eq(row['half'])]
        assert row['rows']==len(q) and row['positive_area']==q.path_area15.gt(0).sum()
        assert row['zero_active']==q.active_bars29.eq(0).sum()
    assert len(got)==r['rows'] and got.next_date.max()==r['last_observation']<'2025-07-01'
    proof = dict(passed=True,label_report_sha256=sha(ROOT / 'full_label_report.json'), rows=len(got),
        minute_marks=len(got)*29, all_cost_marks_active_counts_area_and_original_opportunity_rebuilt=True,
        all_original_known_keys_and_buy_cash_unchanged=True,scope='training_target_only',not_for_evaluation=True,
        inactive_contribution_is_neutral_training_utility_not_realized_zero=True,
        path_area_is_not_realized_return=True,new_2025H2_path_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT / 'full_label_verification.json',proof); return proof


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['target','verify'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
