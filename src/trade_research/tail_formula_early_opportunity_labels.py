"""Training-only opportunity target for the first fifteen opening labels."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_before1000 as source
from .corporate_cash import save_json, sha

ROOT = Path('data/research/tail_formula_early_opportunity_labels')
PROTOCOL = Path('config/tail_formula_early_opportunity_protocol.json')


def source_keys():
    p = json.loads(PROTOCOL.read_text())
    for path, digest in p['source_hashes'].items():
        assert sha(Path(path)) == digest
    assert p['target_window_bars'] == 15 and p['target_window'] == ['09:31', '09:45']
    c = base.conn()
    keys = c.execute('''SELECT date,code,next_date,half,known15,known_no_trade,
        decision_shares,buy_cash15,opportunity15 AS original_opportunity15,adverse_return15
        FROM read_parquet(?) WHERE date>='2024-01-01' AND next_date<'2025-07-01' AND known15 ORDER BY date,code''',
        [str(source.ROOT / 'full_labels.parquet')]).df(); c.close()
    assert keys.next_date.gt(keys.date).all() and keys.next_date.lt('2025-07-01').all()
    assert keys.known15.all() and not keys.known_no_trade.any()
    return keys


def target():
    if (ROOT / 'full_label_report.json').exists():
        raise ValueError('Do not replace the fixed early training target')
    keys = source_keys(); reports = json.loads((source.ROOT / 'observation_report.json').read_text())
    parts = []
    for path, digest in reports['parts_sha256'].items():
        assert sha(Path(path)) == digest
        f = pd.read_parquet(path, filters=[('date', '>=', '2024-01-01'), ('next_date', '<', '2025-07-01')],
                            columns=['date', 'code', 'next_date', 'source_valid', 'close_values', 'active_mask'])
        f = keys.merge(f, on=['date', 'code', 'next_date'], validate='one_to_one')
        if not len(f):
            continue
        assert f.source_valid.all()
        values = np.stack(f.close_values)[:, :15]
        active = (f.active_mask.to_numpy(dtype='int64')[:, None] & (1 << np.arange(15))) != 0
        triples = np.minimum(np.minimum(values[:, :-2], values[:, 1:-1]), values[:, 2:])
        valid = active[:, :-2] & active[:, 1:-1] & active[:, 2:]
        best = np.where(valid, triples, -np.inf).max(axis=1); best[~np.isfinite(best)] = np.nan
        mark = source.net_mark(best, f.decision_shares.to_numpy(), f.buy_cash15.to_numpy(), 15)
        out = f[keys.columns].copy(); out['early_sustained_close'] = best
        out['early_sustained_return15'] = mark; out['opportunity15'] = (mark > 0).astype(float)
        assert out.opportunity15.le(out.original_opportunity15).all()
        parts.append(out)
    out = pd.concat(parts, ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(out[keys.columns], keys, check_exact=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    # Legacy filename is only the model adapter interface, never an evaluation label file.
    out.to_parquet(ROOT / 'full_labels.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), labels_sha256=sha(ROOT / 'full_labels.parquet'),
        source_label_report_sha256=sha(source.ROOT / 'full_label_report.json'),
        source_observation_report_sha256=sha(source.ROOT / 'observation_report.json'),
        extractor_sha256=sha(Path(__file__)), rows=len(out), first_signal=out.date.min(), last_observation=out.next_date.max(),
        scope='training_target_only', not_for_evaluation=True, target_window=['09:31', '09:45'],
        by_half=out.groupby('half').agg(rows=('code', 'size'), original_success=('original_opportunity15', 'sum'),
            early_success=('opportunity15', 'sum')).reset_index().to_dict('records'),
        original_unknowns_not_recovered=True, new_2025H2_early_groups_read=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'full_label_report.json', r); return r


def verify():
    keys = source_keys(); report = json.loads((ROOT / 'full_label_report.json').read_text())
    assert report['protocol_sha256'] == sha(PROTOCOL) and report['scope'] == 'training_target_only'
    assert report['labels_sha256'] == sha(ROOT / 'full_labels.parquet')
    got = pd.read_parquet(ROOT / 'full_labels.parquet')
    pd.testing.assert_frame_equal(got[keys.columns], keys, check_exact=True)
    parts = json.loads((source.ROOT / 'observation_report.json').read_text())['parts_sha256']
    c = base.conn(); c.register('keys', keys); c.read_parquet(list(parts)).create_view('raw_parts')
    ex = c.sql('''WITH bars AS(
        SELECT k.*,pos,list_extract(p.close_values,pos+1) AS close,
            (p.active_mask & (1::BIGINT << pos))<>0 AS active
        FROM keys k JOIN raw_parts p USING(date,code,next_date) CROSS JOIN range(15) t(pos)
        WHERE p.date>='2024-01-01' AND p.next_date<'2025-07-01'),
        marks AS(SELECT *,decision_shares*(close-greatest(.005,close*.0015)) AS value FROM bars),
        positive AS(SELECT *,active AND
            (value-greatest(5,value*.0003)-value*.00051)/buy_cash15-1>0 AS positive FROM marks),
        windows AS(SELECT *,sum(positive::INT) OVER w AS positive_three,
            sum(active::INT) OVER w AS active_three,count(*) OVER w AS bars_three,
            min(close) OVER w AS close_three FROM positive
            WINDOW w AS(PARTITION BY date,code ORDER BY pos ROWS BETWEEN 2 PRECEDING AND CURRENT ROW))
        SELECT date,code,count(*) AS observed_labels,
            max(close_three) FILTER(WHERE active_three=3 AND bars_three=3) AS early_close,
            max((positive_three=3 AND bars_three=3)::INT)::DOUBLE AS early_opportunity
        FROM windows GROUP BY date,code ORDER BY date,code''').df(); c.close()
    pd.testing.assert_frame_equal(got[['date', 'code']], ex[['date', 'code']], check_exact=True)
    assert ex.observed_labels.eq(15).all()
    np.testing.assert_allclose(got.early_sustained_close, ex.early_close, rtol=0, atol=0, equal_nan=True)
    np.testing.assert_array_equal(got.opportunity15, ex.early_opportunity)
    assert got.opportunity15.le(keys.original_opportunity15).all()
    for row in report['by_half']:
        q = got.loc[got.half.eq(row['half'])]
        assert row['rows'] == len(q) and row['early_success'] == q.opportunity15.sum()
        assert row['original_success'] == q.original_opportunity15.sum()
    r = dict(passed=True, label_report_sha256=sha(ROOT / 'full_label_report.json'), rows=len(got),
        training_minute_labels=len(got) * 15, all_positive_minute_cost_flags_and_rolling_triples_independently_rebuilt=True,
        all_buy_cash_and_original_known_keys_unchanged=True, scope='training_target_only', not_for_evaluation=True,
        target_window_end='09:45', evaluation_window_end_remains='09:59',
        new_2025H2_early_groups_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'full_label_verification.json', r); return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['target', 'verify'])
    print(json.dumps(globals()[parser.parse_args().stage](), ensure_ascii=False, indent=2))
