"""Diagnose completed selections without changing models, lists, or exit rules."""
import json
from pathlib import Path
import subprocess

import duckdb
import numpy as np
import pandas as pd

from trade_research.research_io import check_sources, save_json, sha
from trade_research.tail_formula_boundary_evaluation import daily_summary
from tail_formula_reports import checked_analysis


PROTOCOL = Path('config/tail_formula_failure_review_protocol.json')


def finite(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def diagnose(rows, bps, sensitive):
    r = rows[['date', 'code', 'known_no_trade', f'opportunity{bps}',
              f'one_percent{bps}', f'mark_0959_return{bps}', f'adverse_return{bps}',
              f'sensitive_known{bps}' if sensitive else f'known{bps}']].copy()
    r.columns = ['date', 'code', 'no_trade', 'opportunity', 'one_percent',
                 'reference', 'adverse', 'known']
    r['reference_known'] = r.known & r.reference.notna()
    r['adverse_known'] = r.known & r.adverse.notna()
    r['success'] = r.known & r.opportunity.eq(1)
    r['negative'] = r.reference_known & r.reference.lt(0)
    r['bad'] = r.adverse_known & r.adverse.le(-.03)
    r['unknown'] = ~r.known & ~r.no_trade
    r['success_reference_known'] = r.success & r.reference_known
    r['success_negative'] = r.success & r.negative
    r['success_bad'] = r.success & r.bad
    r['space_bad'] = r.known & r.one_percent.eq(1) & r.bad
    r['missing_reference'] = r.known & ~r.reference_known
    fields = ['known', 'no_trade', 'unknown', 'reference_known', 'adverse_known',
              'success', 'success_reference_known', 'success_negative',
              'success_bad', 'space_bad', 'missing_reference']
    daily = r.groupby('date')[fields].sum().reset_index()
    daily['rows'] = r.groupby('date').size().to_numpy()
    con = duckdb.connect()
    con.register('r', r)
    expected = con.sql('''SELECT date, count(*) AS rows,
        sum(CAST(known AS BIGINT)) AS known, sum(CAST(no_trade AS BIGINT)) AS no_trade,
        sum(CAST(NOT known AND NOT no_trade AS BIGINT)) AS unknown,
        sum(CAST(known AND reference IS NOT NULL AS BIGINT)) AS reference_known,
        sum(CAST(known AND adverse IS NOT NULL AS BIGINT)) AS adverse_known,
        sum(CAST(known AND opportunity=1 AS BIGINT)) AS success,
        sum(CAST(known AND opportunity=1 AND reference IS NOT NULL AS BIGINT)) AS success_reference_known,
        sum(CAST(known AND opportunity=1 AND reference<0 AS BIGINT)) AS success_negative,
        sum(CAST(known AND opportunity=1 AND adverse<=-.03 AS BIGINT)) AS success_bad,
        sum(CAST(known AND one_percent=1 AND adverse<=-.03 AS BIGINT)) AS space_bad,
        sum(CAST(known AND reference IS NULL AS BIGINT)) AS missing_reference
        FROM r GROUP BY date ORDER BY date''').df().fillna(0)
    con.close()
    pd.testing.assert_frame_equal(daily[['date', 'rows'] + fields],
                                  expected[['date', 'rows'] + fields], check_dtype=False)
    fractions = {'success_negative_of_reference_known': ('success_negative', 'reference_known'),
                 'success_bad_of_adverse_known': ('success_bad', 'adverse_known'),
                 'space_bad_of_adverse_known': ('space_bad', 'adverse_known'),
                 'negative_among_success_with_reference': ('success_negative', 'success_reference_known')}
    for name, (num, den) in fractions.items():
        daily[name] = daily[num] / daily[den].replace(0, np.nan)
    daily['bps'] = bps
    daily['sensitive'] = sensitive
    output = []
    for year in ['2024', '2025']:
        d = daily.loc[daily.date.str.startswith(year)]
        total = int(d.rows.sum())
        summary = dict(year=year, bps=bps, sensitive=sensitive, days=len(d), rows=total,
                       median_daily_choices=finite(d.rows.median()),
                       max_daily_choices=int(d.rows.max()) if len(d) else None,
                       largest_day_share=finite(d.rows.max()/total) if total else None)
        summary.update({name: int(d[name].sum()) for name in fields})
        summary.update({name: finite(d[name].mean()) for name in fractions})
        output.append(summary)
    return daily, output


def review():
    p = json.loads(PROTOCOL.read_text())
    assert subprocess.check_output(['git', 'show', f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    assert p['new_model_fits'] == p['new_selection_frames'] == 0
    assert not p['new_raw_prices_allowed'] and not p['new_2026_prices_allowed']
    check_sources(p['source_hashes'])
    root = Path(p['output_root'])
    assert not root.exists(), 'Keep completed diagnostics immutable'
    records = []
    frames = []
    receipts = dict(p['source_hashes'])
    for name in p['roots']:
        source = Path('data/research') / name
        selection, report = checked_analysis(source)
        labels = pd.read_parquet(source / 'full_labels.parquet', columns=[
            'date', 'code', 'half', 'known_no_trade', *[
                field + str(bps) for bps in p['cost_sides_bps'] for field in
                ['known', 'sensitive_known', 'opportunity', 'one_percent',
                 'any_opportunity', 'mark_0959_return', 'adverse_return']]])
        rows = labels.merge(selection[['date', 'code', 'selected']],
                            on=['date', 'code'], validate='one_to_one')
        assert len(rows) == len(selection) and rows.date.lt(p['evaluation_end']).all()
        chosen = rows.loc[rows.selected]
        assert chosen.date.ge(p['evaluation_start']).all()
        existing_daily = pd.read_parquet(source / 'daily_summary.parquet')
        for bps in p['cost_sides_bps']:
            for sensitive in p['source_sensitive_views']:
                original = existing_daily.loc[existing_daily.arm.eq('formula') &
                    existing_daily.bps.eq(bps) & existing_daily.sensitive.eq(sensitive)]
                rebuilt = daily_summary(chosen, bps, sensitive)
                pd.testing.assert_frame_equal(rebuilt.reset_index(drop=True),
                    original[rebuilt.columns].reset_index(drop=True), check_exact=True)
                daily, summaries = diagnose(chosen, bps, sensitive)
                daily['source'] = name
                frames.append(daily)
                records.extend([dict(source=name, **v) for v in summaries])
        for file in ['selection.parquet', 'full_labels.parquet', 'daily_summary.parquet']:
            receipts[str(source / file)] = sha(source / file)
    root.mkdir(parents=True)
    daily_file = root / 'diagnostic_daily.parquet'
    pd.concat(frames, ignore_index=True).to_parquet(daily_file, index=False, compression='zstd')
    result = dict(passed=True, protocol_sha256=sha(PROTOCOL),
        implementation_sha256=sha(Path(__file__)), source_hashes=receipts,
        diagnostic_daily_sha256=sha(daily_file), summaries=records,
        all_pandas_counts_independently_rebuilt_in_sql=True,
        all_original_daily_metrics_exactly_rebuilt=True,
        new_models=0, new_selections=0, new_raw_prices_read=False,
        new_2026_prices_read=False, retrospective_diagnostic_only=True,
        time_order_of_opportunity_and_adverse_not_inferred=True, no_exit_rules=True)
    save_json(root / 'review.json', result)
    return dict(passed=True, roots=len(p['roots']), review_sha256=sha(root / 'review.json'))


if __name__ == '__main__':
    print(json.dumps(review(), ensure_ascii=False))
