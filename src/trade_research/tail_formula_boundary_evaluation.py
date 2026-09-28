"""Date-weighted opportunity reports with an explicitly named reference minute."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_before1000 as source
from .corporate_cash import save_json, sha
from .reference_gain_accounting import weekly_interval

METRICS = ['rate', 'lower', 'upper', 'one_percent_rate', 'any_rate',
           'mean_reference', 'negative_reference', 'adverse_mean', 'bad3']
PERIODS = ['2024H1', '2024H2', '2025H1', '2025H2', '2024', '2025']


def number(x):
    return float(x) if pd.notna(x) and np.isfinite(x) else None


def period(f, name):
    if len(name) == 6 and name[4] == 'Q' and name[5] in '1234':
        dates = pd.to_datetime(f.date)
        return f.loc[dates.dt.year.eq(int(name[:4])) & dates.dt.quarter.eq(int(name[5]))]
    return f.loc[f.half.eq(name)] if 'H' in name else f.loc[f.date.str.startswith(name)]


def attach_labels(root):
    proof = json.loads((source.ROOT / 'full_label_verification.json').read_text())
    assert proof['passed'] and proof['label_report_sha256'] == sha(source.ROOT / 'full_label_report.json')
    for name in ['full_labels.parquet', 'full_label_report.json', 'full_label_verification.json']:
        path = root / name
        if not path.exists():
            path.symlink_to((source.ROOT / name).resolve())
        assert sha(path) == sha(source.ROOT / name)


def prepare():
    p = source.checked_protocol()
    roots = []
    for name, receipt in p['selections'].items():
        root = source.ROOT / 'evaluation' / name; root.mkdir(parents=True, exist_ok=True)
        original = Path('data/research') / name
        for file in ['selection.parquet', 'selection_report.json', 'selection_verification.json']:
            path = root / file
            if not path.exists():
                path.symlink_to((original / file).resolve())
            assert sha(path) == sha(original / file)
        attach_labels(root); roots.append(str(root))
    root = source.ROOT / 'evaluation' / 'full_base'; root.mkdir(parents=True, exist_ok=True)
    if not (root / 'selection_report.json').exists():
        keys = pd.read_parquet(source.original.ROOT / 'observation_keys.parquet')
        out = keys[['date', 'code', 'half', 'decision_shares']].copy()
        out['board'] = 'main'; out['selected'] = True
        out.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
        save_json(root / 'selection_report.json', dict(protocol_sha256=sha(source.PROTOCOL),
            selection_sha256=sha(root / 'selection.parquet'), selected=len(out), all_original_keys=True))
    expected = pd.read_parquet(source.original.ROOT / 'observation_keys.parquet', columns=['date', 'code'])
    got = pd.read_parquet(root / 'selection.parquet')
    pd.testing.assert_frame_equal(got[['date', 'code']], expected, check_exact=True)
    assert got.selected.all()
    save_json(root / 'selection_verification.json', dict(passed=True,
        selection_report_sha256=sha(root / 'selection_report.json'), all_original_keys_verified=True,
        no_outcome_based_selection=True))
    attach_labels(root); roots.append(str(root))
    save_json(source.ROOT / 'evaluation_manifest.json', dict(protocol_sha256=sha(source.PROTOCOL), roots=roots,
        selections_unchanged=True, original_unknowns_unchanged=True, outcomes_not_used_for_selection=True))
    return roots


def daily_summary(rows, bps, sensitive):
    r = rows.copy()
    r['known'] = r[f'sensitive_known{bps}' if sensitive else f'known{bps}']
    r['unknown'] = ~r.known & ~r.known_no_trade
    for name, field in [('success', 'opportunity'), ('one_percent', 'one_percent'), ('any_success', 'any_opportunity')]:
        r[name] = r.known & r[f'{field}{bps}'].eq(1)
    r['reference'] = r[f'mark_0959_return{bps}'].where(r.known)
    r['adverse'] = r[f'adverse_return{bps}'].where(r.known)
    r['negative'] = r.reference.lt(0).astype(float).where(r.reference.notna())
    r['bad'] = r.adverse.le(-.03).astype(float).where(r.adverse.notna())
    d = r.groupby(['date', 'half'], sort=True).agg(rows=('code', 'size'), known=('known', 'sum'),
        success=('success', 'sum'), unknown=('unknown', 'sum'), no_trade=('known_no_trade', 'sum'),
        one_percent=('one_percent', 'sum'), any_success=('any_success', 'sum'),
        mean_reference=('reference', 'mean'), negative_reference=('negative', 'mean'),
        adverse_mean=('adverse', 'mean'), bad3=('bad', 'mean')).reset_index()
    for name, numerator in [('rate', 'success'), ('one_percent_rate', 'one_percent'), ('any_rate', 'any_success')]:
        d[name] = d[numerator] / d.known.replace(0, np.nan)
    d['lower'] = d.success / d.rows; d['upper'] = (d.success + d.unknown) / d.rows
    return d


def analyze(root, protocol=source.PROTOCOL):
    if (root / 'analysis_report.json').exists():
        raise ValueError('Do not replace boundary-window statistics')
    attach_labels(root)
    sr = json.loads((root / 'selection_report.json').read_text())
    sp = json.loads((root / 'selection_verification.json').read_text())
    assert sp['passed'] and sp['selection_report_sha256'] == sha(root / 'selection_report.json')
    assert sr['selection_sha256'] == sha(root / 'selection.parquet')
    selection = pd.read_parquet(root / 'selection.parquet', columns=['date', 'code', 'selected'])
    labels = pd.read_parquet(root / 'full_labels.parquet')
    rows = labels.merge(selection, on=['date', 'code'], validate='one_to_one')
    chosen = rows.loc[rows.selected]; base = rows.loc[rows.date.isin(chosen.date.unique())]
    daily = []; summaries = []
    for bps in [5, 15]:
        for sensitive in [False, True]:
            frames = []
            for arm, frame in [('formula', chosen), ('base_same_dates', base)]:
                d = daily_summary(frame, bps, sensitive)
                d['arm'] = arm; d['bps'] = bps; d['sensitive'] = sensitive
                daily.append(d); frames.append(d)
                for p in PERIODS:
                    q = period(d, p).set_index('date')
                    s = dict(arm=arm, bps=bps, sensitive=sensitive, period=p, days=len(q))
                    s.update({k: int(q[k].sum()) for k in ['rows', 'known', 'success', 'unknown', 'no_trade']})
                    s.update({k: number(q[k].mean()) for k in METRICS})
                    s['pooled_rate'] = number(q.success.sum() / q.known.sum()) if q.known.sum() else None
                    for k in ['rate', 'lower', 'upper']:
                        s[k + '_ci'] = weekly_interval(q[k])
                    summaries.append(s)
            a, b = frames
            pd.testing.assert_frame_equal(a[['date', 'half']], b[['date', 'half']], check_exact=True)
            delta = a[['date', 'half']].copy()
            for name, x, y in [('rate_delta', 'rate', 'rate'), ('lower_delta', 'lower', 'upper'),
                               ('upper_delta', 'upper', 'lower'), ('reference_delta', 'mean_reference', 'mean_reference')]:
                delta[name] = a[x] - b[y]
            for p in PERIODS:
                q = period(delta, p).set_index('date')
                s = dict(arm='same_day_difference', bps=bps, sensitive=sensitive, period=p, days=len(q))
                for k in ['rate_delta', 'lower_delta', 'upper_delta', 'reference_delta']:
                    s[k] = number(q[k].mean())
                    if k != 'reference_delta':
                        s[k + '_ci'] = weekly_interval(q[k])
                summaries.append(s)
    pd.concat(daily, ignore_index=True).to_parquet(root / 'daily_summary.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(protocol), selection_report_sha256=sha(root / 'selection_report.json'),
             label_report_sha256=sha(root / 'full_label_report.json'), daily_summary_sha256=sha(root / 'daily_summary.parquet'),
             reference_label='09:59', window_start='09:31', window_end='09:59', summaries=summaries,
             year_2025_is_exploratory=True, opportunity_is_not_realized_profit=True,
             new_2026_prices_read=bool(rows.date.ge('2026-01-01').any()), no_exit_rules=True)
    save_json(root / 'analysis_report.json', r)
    return {k: v for k, v in r.items() if k != 'summaries'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'analyze'])
    p.add_argument('--root', type=Path)
    a = p.parse_args()
    print(json.dumps(prepare() if a.stage == 'prepare' else analyze(a.root), ensure_ascii=False, indent=2))
