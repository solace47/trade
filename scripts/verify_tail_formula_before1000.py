"""Independent SQL rebuild of the frozen next-morning analysis."""
import argparse
import json
from pathlib import Path
import duckdb
import numpy as np
import pandas as pd
from trade_research.research_io import save_json, sha
from tail_formula_statistics import eq, interval

def analysis(root):
    report = json.loads((root / 'analysis_report.json').read_text())
    assert report['reference_label'] == report['window_end'] == '09:59'
    assert report['daily_summary_sha256'] == sha(root / 'daily_summary.parquet')
    assert report['selection_report_sha256'] == sha(root / 'selection_report.json')
    lr = json.loads((root / 'full_label_report.json').read_text())
    lp = json.loads((root / 'full_label_verification.json').read_text())
    assert lp['passed'] and lp['label_report_sha256'] == sha(root / 'full_label_report.json')
    assert lr['labels_sha256'] == sha(root / 'full_labels.parquet')
    c = duckdb.connect(); c.execute('SET threads=4')
    c.read_parquet(str(root / 'full_labels.parquet')).create_view('labels')
    c.read_parquet(str(root / 'selection.parquet')).create_view('selection')
    c.execute('''CREATE VIEW all_rows AS SELECT l.*,s.selected FROM labels l JOIN selection s USING(date,code)
        WHERE l.date IN (SELECT DISTINCT date FROM selection WHERE selected)''')
    frames = []
    for bps in [5, 15]:
        for sensitive in [False, True]:
            known = f'sensitive_known{bps}' if sensitive else f'known{bps}'
            for arm, clause in [('formula', 'WHERE selected'), ('base_same_dates', '')]:
                d = c.sql(f'''WITH d AS(SELECT date,half,count(*) AS rows,
                    count(*) FILTER(WHERE {known}) AS known,
                    count(*) FILTER(WHERE {known} AND opportunity{bps}=1) AS success,
                    count(*) FILTER(WHERE NOT {known} AND NOT known_no_trade) AS unknown,
                    count(*) FILTER(WHERE known_no_trade) AS no_trade,
                    count(*) FILTER(WHERE {known} AND one_percent{bps}=1) AS one_percent,
                    count(*) FILTER(WHERE {known} AND any_opportunity{bps}=1) AS any_success,
                    avg(mark_0959_return{bps}) FILTER(WHERE {known}) AS mean_reference,
                    avg((mark_0959_return{bps}<0)::INT) FILTER(WHERE {known}) AS negative_reference,
                    avg(adverse_return{bps}) FILTER(WHERE {known}) AS adverse_mean,
                    avg((adverse_return{bps}<=-.03)::INT) FILTER(WHERE {known}) AS bad3
                    FROM all_rows {clause} GROUP BY date,half)
                    SELECT *,success/nullif(known,0) AS rate,success/rows AS lower,(success+unknown)/rows AS upper,
                    one_percent/nullif(known,0) AS one_percent_rate,any_success/nullif(known,0) AS any_rate
                    FROM d ORDER BY date''').df()
                d['arm'] = arm; d['bps'] = bps; d['sensitive'] = sensitive; frames.append(d)
    c.close()
    expected = pd.concat(frames, ignore_index=True); got = pd.read_parquet(root / 'daily_summary.parquet')
    pd.testing.assert_frame_equal(got[expected.columns], expected, check_dtype=False, atol=2e-10, rtol=0)
    checks = 0
    for s in report['summaries']:
        d = expected.loc[expected.bps.eq(s['bps']) & expected.sensitive.eq(s['sensitive'])]
        p = s['period']
        if len(p) == 6 and p[4] == 'Q' and p[5] in '1234':
            first_month = 3*int(p[5])-2
            d = d.loc[d.date.str[:4].eq(p[:4]) & d.date.str[5:7].astype(int).between(first_month,first_month+2)]
        else:
            d = d.loc[d.half.eq(p)] if 'H' in p else d.loc[d.date.str.startswith(p)]
        if s['arm'] == 'same_day_difference':
            a = d.loc[d.arm.eq('formula')].set_index('date'); b = d.loc[d.arm.eq('base_same_dates')].set_index('date')
            assert a.index.equals(b.index)
            q = pd.DataFrame(dict(rate_delta=a.rate - b.rate, lower_delta=a.lower - b.upper,
                                 upper_delta=a.upper - b.lower, reference_delta=a.mean_reference - b.mean_reference))
            eq(s['days'], len(q), 'days')
            for k in q:
                v = q[k].mean(); eq(s[k], float(v) if pd.notna(v) else None, k); checks += 1
                if k != 'reference_delta':
                    eq(s[k + '_ci'], interval(q.reset_index(), k), k + '_ci'); checks += 1
        else:
            q = d.loc[d.arm.eq(s['arm'])]
            eq(s['days'], len(q), 'days')
            for k in ['rows', 'known', 'success', 'unknown', 'no_trade']:
                eq(s[k], q[k].sum(), k); checks += 1
            eq(s['pooled_rate'], q.success.sum() / q.known.sum() if q.known.sum() else None, 'pooled_rate'); checks += 1
            for k in ['rate', 'lower', 'upper', 'one_percent_rate', 'any_rate',
                      'mean_reference', 'negative_reference', 'adverse_mean', 'bad3']:
                v = q[k].mean(); eq(s[k], float(v) if pd.notna(v) else None, k); checks += 1
                if k in ['rate', 'lower', 'upper']:
                    eq(s[k + '_ci'], interval(q, k), k + '_ci'); checks += 1
    r = dict(passed=True, analysis_report_sha256=sha(root / 'analysis_report.json'),
             daily_rows=len(expected), summary_checks=checks, all_daily_statistics_rebuilt=True,
             reference_label='09:59', new_2026_prices_read=bool(expected.date.ge('2026-01-01').any()), no_exit_rules=True)
    save_json(root / 'analysis_verification.json', r)
    return r


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['analysis'])
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analysis(args.root), ensure_ascii=False, indent=2))
