"""Compare unchanged lists across the two predeclared observation windows."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research import tail_formula_before1000 as source
from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research.corporate_cash import save_json, sha
from verify_tick_flow_analysis import eq, interval


def compare():
    output = source.ROOT / 'boundary_comparison.json'
    if output.exists():
        raise ValueError('Do not overwrite a completed boundary comparison')
    p = source.checked_protocol()
    old = pd.read_parquet(source.original.ROOT / 'full_labels.parquet')
    old = old.rename(columns={f'mark_1000_return{bps}': f'mark_0959_return{bps}' for bps in [5, 15]})
    # Temporary names only reuse arithmetic. Reports explicitly identify both different reference labels.
    c = duckdb.connect(); c.execute('SET threads=4')
    c.read_parquet(str(source.original.ROOT / 'full_labels.parquet')).create_view('old_labels')
    c.read_parquet(str(source.ROOT / 'full_labels.parquet')).create_view('new_labels')
    changes = c.sql('''SELECT a.half,count(*) AS rows,
        count(*) FILTER(WHERE a.known15) AS known15,
        count(*) FILTER(WHERE a.known15 AND a.opportunity15>b.opportunity15) AS lost_opportunity15,
        count(*) FILTER(WHERE a.known15 AND a.one_percent15>b.one_percent15) AS lost_one_percent15,
        count(*) FILTER(WHERE a.known15 AND a.adverse_return15<=-.03 AND b.adverse_return15>-.03) AS lost_bad3_observation15,
        count(*) FILTER(WHERE a.known15<>b.known15 OR a.known_no_trade<>b.known_no_trade OR a.unknown15<>b.unknown15) AS changed_status
        FROM old_labels a JOIN new_labels b USING(date,code) GROUP BY a.half ORDER BY a.half''').df()
    assert changes.changed_status.eq(0).all()
    assert int(changes.rows.sum()) == len(old)
    reports = []; checks = 0
    for name in [*p['selections'], 'full_base']:
        root = source.ROOT / 'evaluation' / name
        proof = json.loads((root / 'analysis_verification.json').read_text())
        assert proof['passed'] and proof['analysis_report_sha256'] == sha(root / 'analysis_report.json')
        got = pd.read_parquet(root / 'daily_summary.parquet')
        chosen = pd.read_parquet(root / 'selection.parquet', columns=['date', 'code', 'selected'])
        selected = old.merge(chosen, on=['date', 'code'], validate='one_to_one')
        selected = selected.loc[selected.selected]
        c.register('selection', chosen)
        old_frames = []
        for bps in [5, 15]:
            for sensitive in [False, True]:
                known = f'sensitive_known{bps}' if sensitive else f'known{bps}'
                ex = c.sql(f'''SELECT date,half,count(*) AS rows,count(*) FILTER(WHERE {known}) AS known,
                    count(*) FILTER(WHERE NOT {known} AND NOT known_no_trade) AS unknown,
                    count(*) FILTER(WHERE known_no_trade) AS no_trade,
                    avg(opportunity{bps}) FILTER(WHERE {known}) AS rate,
                    count(*) FILTER(WHERE {known} AND opportunity{bps}=1)*1.0/count(*) AS lower,
                    (count(*) FILTER(WHERE {known} AND opportunity{bps}=1)
                        +count(*) FILTER(WHERE NOT {known} AND NOT known_no_trade))*1.0/count(*) AS upper,
                    avg(one_percent{bps}) FILTER(WHERE {known}) AS one_percent_rate,
                    avg(any_opportunity{bps}) FILTER(WHERE {known}) AS any_rate,
                    avg(mark_1000_return{bps}) FILTER(WHERE {known}) AS mean_reference,
                    avg((mark_1000_return{bps}<0)::INT) FILTER(WHERE {known}) AS negative_reference,
                    avg(adverse_return{bps}) FILTER(WHERE {known}) AS adverse_mean,
                    avg((adverse_return{bps}<=-.03)::INT) FILTER(WHERE {known}) AS bad3
                    FROM old_labels JOIN selection USING(date,code) WHERE selected GROUP BY date,half ORDER BY date''').df()
                d = evaluation.daily_summary(selected, bps, sensitive)
                pd.testing.assert_frame_equal(d[ex.columns], ex, check_dtype=False, atol=2e-10, rtol=0)
                d['bps'] = bps; d['sensitive'] = sensitive; old_frames.append(d)
        previous = pd.concat(old_frames, ignore_index=True)
        current = got.loc[got.arm.eq('formula')].reset_index(drop=True)
        keys = ['date', 'half', 'bps', 'sensitive']
        pd.testing.assert_frame_equal(previous[keys + ['rows', 'known', 'unknown', 'no_trade']],
                                      current[keys + ['rows', 'known', 'unknown', 'no_trade']], check_exact=True)
        delta = current[keys].copy()
        for metric in evaluation.METRICS:
            delta[metric + '_delta'] = current[metric] - previous[metric]
        summaries = []
        for (bps, sensitive), group in delta.groupby(['bps', 'sensitive']):
            for period in evaluation.PERIODS:
                q = evaluation.period(group, period)
                s = dict(bps=int(bps), sensitive=bool(sensitive), period=period, days=len(q))
                for metric in evaluation.METRICS:
                    k = metric + '_delta'; s[k] = evaluation.number(q[k].mean())
                    if k in ['rate_delta', 'lower_delta', 'upper_delta']:
                        s[k + '_ci'] = interval(q, k)
                    values = q[k].dropna().to_numpy()
                    eq(s[k], float(np.sum(values) / len(values)) if len(values) else None, k); checks += 1
                summaries.append(s)
        reports.append(dict(name=name, analysis_report_sha256=sha(root / 'analysis_report.json'),
                            rows=int(chosen.selected.sum()), signal_days=selected.date.nunique(), summaries=summaries))
    c.close()
    r = dict(protocol_sha256=sha(source.PROTOCOL), old_label_report_sha256=sha(source.original.ROOT / 'full_label_report.json'),
             new_label_report_sha256=sha(source.ROOT / 'full_label_report.json'), all_old_daily_statistics_independently_rebuilt=True,
             new_statistics_independently_verified=True, scalar_summary_checks=checks,
             all_selection_dates_statuses_and_denominators_unchanged=True,
             original_reference_label='10:00', conservative_reference_label='09:59',
             reference_and_adverse_differences_are_measurement_changes_not_strategy_improvements=True,
             half_label_changes=changes.to_dict('records'), reports=reports,
             year_2025_is_exploratory=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(output, r)
    return {k: v for k, v in r.items() if k != 'reports'}


if __name__ == '__main__':
    print(json.dumps(compare(), ensure_ascii=False, indent=2))
