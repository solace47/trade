"""Bound full-list opportunity differences with shared unknown outcomes coupled."""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha
from trade_research.reference_gain_accounting import weekly_interval
from verify_tick_flow_analysis import interval


def daily_bounds(rows):
    """Each union stock-day has one latent opportunity, including in both lists."""
    assert not rows.duplicated(['date', 'code']).any()
    assert rows[['left', 'right', 'known', 'no_trade']].isin([False, True]).all().all()
    assert (rows.left | rows.right).all() and not (rows.known & rows.no_trade).any()
    assert rows.loc[rows.known, 'opportunity'].isin([0, 1]).all()
    r = rows.copy()
    counts = r.groupby('date')[['left', 'right']].transform('sum')
    assert (counts > 0).all().all()
    weight = r.left.astype(float)/counts.left-r.right.astype(float)/counts.right
    unknown = ~r.known & ~r.no_trade
    observed = weight * r.opportunity.where(r.known, 0)
    r['lower'] = observed + weight.clip(upper=0).where(unknown, 0)
    r['upper'] = observed + weight.clip(lower=0).where(unknown, 0)
    r['unknown_weight'] = weight.abs().where(unknown, 0)
    r['shared_unknown'] = (r.left & r.right & unknown).astype(int)
    result = r.groupby(['date', 'half'], sort=True).agg(
        left_rows=('left', 'sum'), right_rows=('right', 'sum'),
        lower=('lower', 'sum'), upper=('upper', 'sum'),
        unknown_weight=('unknown_weight', 'sum'), shared_unknown=('shared_unknown', 'sum')).reset_index()
    np.testing.assert_allclose(result.upper-result.lower, result.unknown_weight, rtol=0, atol=2e-12)
    return result


def check_signal_dates(dates, signal_range=None):
    if signal_range is None:
        assert not dates.ge('2026-01-01').any()
    else:
        assert signal_range in [['2024-01-01', '2024-12-31'], ['2026-01-01', '2026-03-31']]
        assert dates.between(*signal_range).all()


def checked_selection(root, expected_analysis_sha, signal_range=None):
    report = json.loads((root/'analysis_report.json').read_text())
    proof = json.loads((root/'analysis_verification.json').read_text())
    assert sha(root/'analysis_report.json') == expected_analysis_sha
    assert proof['passed'] and proof['analysis_report_sha256'] == expected_analysis_sha
    assert report['reference_label'] == '09:59'
    selection = json.loads((root/'selection_report.json').read_text())
    selected_proof = json.loads((root/'selection_verification.json').read_text())
    assert selected_proof['passed']
    assert selected_proof['selection_report_sha256'] == report['selection_report_sha256'] == sha(root/'selection_report.json')
    assert selection['selection_sha256'] == sha(root/'selection.parquet')
    frame = pd.read_parquet(root/'selection.parquet')
    assert not frame.duplicated(['date', 'code']).any()
    check_signal_dates(frame.loc[frame.selected, 'date'], signal_range)
    return frame.loc[frame.selected, ['date', 'code', 'half', 'decision_shares']].copy()


def compare(spec, protocol):
    left, right, output = [Path(spec[k]) for k in ['left', 'right', 'output']]
    assert not output.exists(), 'Do not replace a recorded diagnostic'
    a = checked_selection(left, spec['left_analysis_sha256'], protocol.get('signal_range'))
    b = checked_selection(right, spec['right_analysis_sha256'], protocol.get('signal_range'))
    assert sha(left/'full_labels.parquet') == sha(right/'full_labels.parquet') == protocol['labels_sha256']
    original_dates = [set(f.date) for f in [a, b]]; common = original_dates[0] & original_dates[1]
    a = a[a.date.isin(common)].rename(columns={'decision_shares': 'left_shares'})
    b = b[b.date.isin(common)].rename(columns={'decision_shares': 'right_shares'})
    a['left'] = True; b['right'] = True
    merged = a.merge(b, on=['date', 'code', 'half'], how='outer', validate='one_to_one')
    for side in ['left', 'right']:
        merged[side] = merged[side].eq(True)
    labels = pd.read_parquet(left/'full_labels.parquet', columns=['date', 'code', 'half', 'decision_shares',
        'known_no_trade', 'known5', 'known15', 'sensitive_known5', 'sensitive_known15', 'opportunity5', 'opportunity15'])
    r = merged.merge(labels, on=['date', 'code', 'half'], validate='one_to_one')
    assert len(r) == len(merged)
    for side in ['left', 'right']:
        assert r.loc[r[side], side+'_shares'].eq(r.loc[r[side], 'decision_shares']).all()
    summaries = []; comparisons = 0; daily_records = []
    for bps in [5, 15]:
        for sensitive in [False, True]:
            rows = r[['date', 'code', 'half', 'left', 'right']].copy()
            rows['known'] = r[f'sensitive_known{bps}' if sensitive else f'known{bps}']
            rows['no_trade'] = r.known_no_trade
            rows['opportunity'] = r[f'opportunity{bps}']
            d = daily_bounds(rows)
            c = duckdb.connect(); c.register('records', rows)
            expected = c.sql('''WITH w AS(SELECT *,
                sum("left"::INT) OVER(PARTITION BY date) AS nl,
                sum("right"::INT) OVER(PARTITION BY date) AS nr FROM records),
                weights AS(SELECT *,"left"::DOUBLE/nl-"right"::DOUBLE/nr AS coefficient FROM w)
                SELECT date,half,sum("left"::INT) AS left_rows,sum("right"::INT) AS right_rows,
                sum(CASE WHEN known THEN coefficient*opportunity WHEN no_trade THEN 0 ELSE least(coefficient,0) END) AS lower,
                sum(CASE WHEN known THEN coefficient*opportunity WHEN no_trade THEN 0 ELSE greatest(coefficient,0) END) AS upper,
                sum(CASE WHEN NOT known AND NOT no_trade THEN abs(coefficient) ELSE 0 END) AS unknown_weight,
                sum(("left" AND "right" AND NOT known AND NOT no_trade)::INT) AS shared_unknown
                FROM weights GROUP BY date,half ORDER BY date''').df(); c.close()
            pd.testing.assert_frame_equal(d, expected, check_dtype=False, rtol=0, atol=2e-12)
            # Old arm-wise envelopes remain valid outer bounds, but permit
            # incompatible outcomes for the same shared unknown stock-day.
            d['bps'] = bps; d['sensitive'] = sensitive; daily_records.extend(d.to_dict('records'))
            for period in protocol.get('periods', ['2025H1', '2025H2', '2025']):
                if len(period) == 6 and period[4] == 'Q' and period[5] in '1234':
                    dates = pd.to_datetime(d.date)
                    q = d.loc[dates.dt.year.eq(int(period[:4])) & dates.dt.quarter.eq(int(period[5]))]
                else:
                    q = d.loc[d.half.eq(period) if 'H' in period else d.date.str.startswith(period)]
                out = dict(bps=bps, sensitive=sensitive, period=period, days=len(q),
                    shared_unknown_stock_days=int(q.shared_unknown.sum()))
                for column in ['lower', 'upper', 'unknown_weight']:
                    out[column] = float(q[column].mean()) if len(q) else None
                    out[column+'_ci'] = weekly_interval(q.set_index('date')[column])
                    independent = interval(q, column)
                    if independent is None:
                        assert out[column+'_ci'] is None
                    else:
                        np.testing.assert_allclose(out[column+'_ci'], independent, rtol=0, atol=2e-12)
                    comparisons += 1
                summaries.append(out)
    report = dict(protocol_sha256=protocol['protocol_sha256'], comparison=spec,
        date_coverage=dict(left_signal_days=len(original_dates[0]), right_signal_days=len(original_dates[1]),
            common_signal_days=len(common), left_only_dates=sorted(original_dates[0]-common),
            right_only_dates=sorted(original_dates[1]-common)),
        labels_sha256=protocol['labels_sha256'], summaries=summaries, daily=daily_records,
        passed=True, independent_daily_rows=len(daily_records), independent_interval_checks=comparisons,
        same_stock_day_same_unknown_opportunity=True, denominator_includes_unknown_and_no_trade=True,
        no_trade_is_zero_opportunity_not_realized_return=True,
        not_difference_of_known_conditional_rates=True, post_result_diagnostic_not_new_selection=True,
        original_reports_unchanged=True, year_2025_is_exploratory=True,
        new_2026_prices_read=bool(r.date.ge('2026-01-01').any()), no_exit_rules=True)
    output.parent.mkdir(parents=True, exist_ok=True); save_json(output, report)
    return {k:v for k,v in report.items() if k != 'daily'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True); args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text()); protocol['protocol_sha256'] = sha(args.protocol)
    for spec in protocol['comparisons']:
        print(json.dumps(compare(spec, protocol), ensure_ascii=False, indent=2))
