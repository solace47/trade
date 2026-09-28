"""Audit reference availability and positive marks without changing selections."""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha


def audit(root):
    output = root / 'reference_coverage_verification.json'
    assert not output.exists(), 'Do not replace a recorded audit'
    for kind in ['analysis', 'selection']:
        report = json.loads((root / f'{kind}_report.json').read_text())
        proof = json.loads((root / f'{kind}_verification.json').read_text())
        assert proof['passed'] and proof[f'{kind}_report_sha256'] == sha(root / f'{kind}_report.json')
    ar = json.loads((root / 'analysis_report.json').read_text())
    sr = json.loads((root / 'selection_report.json').read_text())
    lr = json.loads((root / 'full_label_report.json').read_text())
    lv = json.loads((root / 'full_label_verification.json').read_text())
    assert ar['reference_label'] == '09:59'
    assert ar['selection_report_sha256'] == sha(root / 'selection_report.json')
    assert ar['label_report_sha256'] == sha(root / 'full_label_report.json')
    assert lv['passed'] and lv['label_report_sha256'] == sha(root / 'full_label_report.json')
    assert lr['labels_sha256'] == sha(root / 'full_labels.parquet')
    assert sr['selection_sha256'] == sha(root / 'selection.parquet')
    assert ar['daily_summary_sha256'] == sha(root / 'daily_summary.parquet')
    selection = pd.read_parquet(root / 'selection.parquet', columns=['date', 'code', 'selected'])
    selected = selection.loc[selection.selected, ['date', 'code']]
    assert selected.date.between('2025-01-01', '2025-12-31').all()
    columns = ['date', 'code', 'half'] + [
        name for bps in [5, 15] for name in
        [f'known{bps}', f'sensitive_known{bps}', f'mark_0959_return{bps}']]
    labels = pd.read_parquet(root / 'full_labels.parquet', columns=columns)
    rows = selected.merge(labels, on=['date', 'code'], validate='one_to_one')
    assert len(rows) == len(selected)
    original_daily = pd.read_parquet(root / 'daily_summary.parquet')
    c = duckdb.connect()
    c.read_parquet(str(root / 'selection.parquet')).create_view('selection')
    c.read_parquet(str(root / 'full_labels.parquet')).create_view('labels')
    reports, daily, missing = [], [], []
    for bps in [5, 15]:
        for sensitive in [False, True]:
            known = f'sensitive_known{bps}' if sensitive else f'known{bps}'
            mark = f'mark_0959_return{bps}'
            valid = rows[known] & np.isfinite(rows[mark])
            r = rows[['date', 'code', 'half']].copy()
            r['known'] = rows[known].astype(int)
            r['reference_available'] = valid.astype(int)
            r['reference_missing'] = (rows[known] & ~valid).astype(int)
            r['positive_reference'] = (valid & rows[mark].gt(0)).astype(int)
            r['mean_reference'] = rows[mark].where(valid)
            got = r.groupby(['date', 'half'], sort=True).agg(
                selected=('code', 'size'), known=('known', 'sum'),
                reference_available=('reference_available', 'sum'),
                reference_missing=('reference_missing', 'sum'),
                positive_reference=('positive_reference', 'sum'),
                mean_reference=('mean_reference', 'mean')).reset_index()
            expected = c.sql(f'''SELECT l.date,l.half,count(*) AS selected,
                count(*) FILTER(WHERE {known}) AS known,
                count(*) FILTER(WHERE {known} AND isfinite({mark})) AS reference_available,
                count(*) FILTER(WHERE {known} AND NOT coalesce(isfinite({mark}),false)) AS reference_missing,
                count(*) FILTER(WHERE {known} AND isfinite({mark}) AND {mark}>0) AS positive_reference,
                avg({mark}) FILTER(WHERE {known} AND isfinite({mark})) AS mean_reference
                FROM labels l JOIN selection s USING(date,code) WHERE s.selected
                GROUP BY l.date,l.half ORDER BY l.date''').df()
            pd.testing.assert_frame_equal(got, expected, check_dtype=False, rtol=0, atol=2e-12)
            old = original_daily.loc[original_daily.arm.eq('formula') & original_daily.bps.eq(bps)
                                     & original_daily.sensitive.eq(sensitive)].sort_values('date')
            assert old.date.tolist() == got.date.tolist()
            np.testing.assert_allclose(old.mean_reference, got.mean_reference, rtol=0, atol=2e-12)
            np.testing.assert_array_equal(old.known, got.known)
            assert (got.known == got.reference_available + got.reference_missing).all()
            got['positive_reference_rate'] = got.positive_reference / got.reference_available.replace(0, np.nan)
            got['bps'] = bps
            got['sensitive'] = sensitive
            # Preserve dates without a known reference as JSON null, never zero.
            daily.extend(got.astype(object).where(got.notna(), None).to_dict('records'))
            keys = rows.loc[rows[known] & ~valid, ['date', 'code']].sort_values(['date', 'code']).reset_index(drop=True)
            sql_keys = c.sql(f'''SELECT l.date,l.code FROM labels l JOIN selection s USING(date,code)
                WHERE s.selected AND {known} AND NOT coalesce(isfinite({mark}),false)
                ORDER BY l.date,l.code''').df()
            pd.testing.assert_frame_equal(keys, sql_keys, check_exact=True, check_dtype=not keys.empty)
            missing.extend(dict(bps=bps, sensitive=sensitive, **x) for x in keys.to_dict('records'))
            for period in ['2025H1', '2025H2', '2025']:
                g = got.loc[got.half.eq(period)] if 'H' in period else got
                reports.append(dict(bps=bps, sensitive=sensitive, period=period,
                    days=len(g), reference_available_days=int(g.reference_available.gt(0).sum()),
                    selected=int(g.selected.sum()), known=int(g.known.sum()),
                    reference_available=int(g.reference_available.sum()),
                    reference_missing=int(g.reference_missing.sum()),
                    positive_reference=int(g.positive_reference.sum()),
                    date_equal_positive_reference_rate=float(g.positive_reference_rate.mean())
                        if g.positive_reference_rate.notna().any() else None,
                    date_equal_mean_reference=float(g.mean_reference.mean())
                        if g.mean_reference.notna().any() else None))
    c.close()
    result = dict(passed=True, analysis_report_sha256=sha(root / 'analysis_report.json'),
        selection_report_sha256=sha(root / 'selection_report.json'),
        label_report_sha256=sha(root / 'full_label_report.json'),
        auditor_sha256=sha(Path(__file__)), summaries=reports, daily=daily,
        reference_missing_cases=missing, all_counts_keys_and_means_independently_rebuilt=True,
        positive_reference_is_strictly_above_zero=True,
        reference_rates_conditional_on_available_marks=True,
        known_binary_outcomes_retained=True, no_selection_changes_or_zero_imputation=True,
        post_result_diagnostic_not_new_selection=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(output, result)
    return {k: v for k, v in result.items() if k not in ['daily', 'reference_missing_cases']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    print(json.dumps(audit(p.parse_args().root), ensure_ascii=False, indent=2))
