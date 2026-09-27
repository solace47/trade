"""Compare two already-verified selections on exactly the same signal dates."""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha
from trade_research.reference_gain_accounting import weekly_interval
from verify_tick_flow_analysis import eq, interval


def compare(left, right, output):
    if output.exists():
        raise ValueError('Do not replace a recorded paired comparison')
    inputs = {}
    frames = []
    for name, path in [('left', left), ('right', right)]:
        report = json.loads((path / 'analysis_report.json').read_text())
        proof = json.loads((path / 'analysis_verification.json').read_text())
        assert proof['passed'] and proof['analysis_report_sha256'] == sha(path / 'analysis_report.json')
        assert report['daily_summary_sha256'] == sha(path / 'daily_summary.parquet')
        assert report['selection_report_sha256'] == sha(path / 'selection_report.json')
        inputs[name] = dict(root=str(path), analysis_report_sha256=sha(path / 'analysis_report.json'),
                            daily_summary_sha256=sha(path / 'daily_summary.parquet'))
        f = pd.read_parquet(path / 'daily_summary.parquet')
        frames.append(f.loc[f.arm.eq('formula')].sort_values(['bps', 'sensitive', 'date']).reset_index(drop=True))
    keys = ['bps', 'sensitive', 'date', 'half']
    pd.testing.assert_frame_equal(frames[0][keys], frames[1][keys], check_exact=True)
    a, b = frames
    result = a[keys].copy()
    for name in ['rate', 'one_percent_rate', 'mean1000', 'negative1000', 'bad3']:
        result[name + '_delta'] = a[name] - b[name]
    result['lower_delta'] = a.lower - b.upper
    result['upper_delta'] = a.upper - b.lower
    c = duckdb.connect(); c.register('a', a); c.register('b', b)
    fields = [f'a.{k}-b.{k} AS {k}_delta' for k in ['rate', 'one_percent_rate', 'mean1000', 'negative1000', 'bad3']]
    sql = ('SELECT a.bps,a.sensitive,a.date,a.half,' + ','.join(fields) +
           ',a.lower-b.upper AS lower_delta,a.upper-b.lower AS upper_delta FROM a JOIN b USING(bps,sensitive,date,half) ORDER BY bps,sensitive,date')
    expected = c.sql(sql).df(); c.close()
    pd.testing.assert_frame_equal(result, expected, check_exact=True)
    summaries = []
    checks = 0
    for (bps, sensitive), group in result.groupby(['bps', 'sensitive']):
        for period in ['2025H1', '2025H2', '2025']:
            q = group if period == '2025' else group.loc[group.half.eq(period)]
            s = dict(bps=int(bps), sensitive=bool(sensitive), period=period, days=len(q))
            for name in result.columns[len(keys):]:
                values = q.set_index('date')[name]
                value = float(values.mean()) if values.notna().any() else None
                s[name] = value
                s[name + '_ci'] = weekly_interval(values)
                eq(s[name + '_ci'], interval(q, name), name)
                numeric = q[name].dropna().to_numpy()
                eq(value, float(np.sum(numeric) / len(numeric)) if len(numeric) else None, name)
                checks += 2
            summaries.append(s)
    r = dict(inputs=inputs, summaries=summaries, summary_checks=checks, passed=True,
             all_daily_differences_independently_rebuilt=True, all_intervals_independently_rebuilt=True,
             same_signal_dates_required=True, lower_delta_is_left_lower_minus_right_upper=True,
             post_result_comparison_not_new_selection=True, year_2025_is_exploratory=True,
             new_2026_prices_read=False, no_exit_rules=True)
    save_json(output, r)
    return r


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--left', type=Path, required=True)
    p.add_argument('--right', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(compare(a.left, a.right, a.output), ensure_ascii=False, indent=2))
