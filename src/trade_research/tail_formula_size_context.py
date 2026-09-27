"""CSI 1000 versus CSI 300 context, preserving the original 48 stock inputs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_prior_day as adapter
from .corporate_cash import save_json, sha
from .turnover_reference import CALENDAR

STEM = 'tail_formula_size_context'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
NEW_EXPRESSIONS = {'SZ01': '100*(SM48/SMP-LG48/LGP)', 'SZ02': '100*(SM48/SM20-LG48/LG20)'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER
for name, symbol in [('SM', 'SH000852'), ('LG', 'SH000300')]:
    for minute in [48, 20]:
        HEADER += f'{name}{minute}:=VALUEWHEN(TIME=14{minute},"{symbol}$CLOSE");\n'
    HEADER += f'{name}P:=REF("{symbol}$CLOSE",B0);\n'


def checked_sources():
    p = json.loads(PROTOCOL.read_text()); r = json.loads((ROOT / 'source_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL)
    assert p['previous_feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert p['daily_feature_report_sha256'] == sha(base.SOURCE / 'feature_report.json')
    assert p['calendar_sha256'] == sha(CALENDAR)
    old = json.loads((previous.ROOT / 'feature_report.json').read_text())
    proof = json.loads((previous.ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert old['features_sha256'] == r['source_feature_sha256'] == sha(previous.ROOT / 'features.parquet')
    assert r['indices_sha256'] == sha(ROOT / 'indices.parquet')
    return previous.context.market.stock.source_files()


def prefix_points(rows):
    prefix = rows[:228]
    valid = len(prefix) == 228 and all(r['sequence'] == i and r['price_raw'] > 0 for i, r in enumerate(prefix))
    return dict(p48=rows[227]['price_raw'] / 100 if valid else np.nan,
        p20=rows[199]['price_raw'] / 100 if valid else np.nan, prefix_valid=valid)


def features():
    assert not (ROOT / 'feature_report.json').exists(), 'Do not replace frozen size context'
    files = checked_sources(); source = json.loads((ROOT / 'source_report.json').read_text())
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    assert {(s['symbol'], s['date']) for s in source['sessions']} == {
        (code, day) for code in ['sh.000852', 'sh.000300'] for day in old.date.unique()}
    indices = pd.read_parquet(ROOT / 'indices.parquet').set_index(['code', 'date'])
    points = []; audits = []
    for s in source['sessions']:
        if 'path' in s:
            path = Path(s['path']); assert sha(path) == s['sha256']; rows = json.loads(path.read_text())
        else:
            rows = []
        points.append(dict(date=s['date'], index_code=s['symbol'], **prefix_points(rows)))
        day = indices.loc[(s['symbol'], s['date'])]; prices = np.array([x['price_raw'] for x in rows]) / 100
        full = bool(len(prices) == 240 and (prices > 0).all() and
            (prices >= day.low - .0051).all() and (prices <= day.high + .0051).all() and abs(prices[-1] - day.close) <= .0051)
        audits.append(dict(date=s['date'], index_code=s['symbol'], rows=len(rows), full_day_source_valid=full,
            last_minus_daily_close=float(prices[-1] - day.close) if len(prices) else None))
    points = pd.DataFrame(points).sort_values(['date', 'index_code']).reset_index(drop=True)
    audit = pd.DataFrame(audits).sort_values(['date', 'index_code']).reset_index(drop=True)
    c = base.conn(); c.read_parquet(files).create_view('daily'); c.register('indices', indices.reset_index())
    prior = c.sql('''WITH traded AS (
        SELECT date,code,lag(date) OVER(PARTITION BY code ORDER BY date) AS sc_prior_date
        FROM daily WHERE tradestatus=1 AND date BETWEEN '2023-06-01' AND '2025-12-30')
        SELECT t.date,t.code,t.sc_prior_date,s.close AS sc_small_prior,l.close AS sc_large_prior
        FROM traded t LEFT JOIN indices s ON s.code='sh.000852' AND s.date=t.sc_prior_date
        LEFT JOIN indices l ON l.code='sh.000300' AND l.date=t.sc_prior_date
        WHERE t.date>='2024-01-01' ORDER BY t.date,t.code''').df(); c.close()
    f = old.merge(prior, on=['date', 'code'], how='left', validate='one_to_one')
    for name, code in [('small', 'sh.000852'), ('large', 'sh.000300')]:
        q = points.loc[points.index_code.eq(code)].drop(columns='index_code').rename(
            columns={k: f'sc_{name}_{k}' for k in ['p48', 'p20', 'prefix_valid']})
        f = f.merge(q, on='date', how='left', validate='many_to_one')
    values = f[['sc_small_prior', 'sc_large_prior', 'sc_small_p48', 'sc_small_p20', 'sc_large_p48', 'sc_large_p20']]
    valid = (f.sc_prior_date.lt(f.date) & values.gt(0).all(axis=1) & np.isfinite(values).all(axis=1)
        & f.sc_small_prefix_valid.fillna(False) & f.sc_large_prefix_valid.fillna(False))
    f['SZ01'] = (100 * (f.sc_small_p48 / f.sc_small_prior - f.sc_large_p48 / f.sc_large_prior)).where(valid)
    f['SZ02'] = (100 * (f.sc_small_p48 / f.sc_small_p20 - f.sc_large_p48 / f.sc_large_p20)).where(valid)
    f['size_context_valid'] = valid; f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    cal = pd.read_parquet(CALENDAR)
    days = sorted(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between('2023-06-01', '2025-12-30'), 'calendar_date'])
    ranks = {day: index for index, day in enumerate(days)}
    f['sc_prior_gap'] = f.date.map(ranks) - f.sc_prior_date.map(ranks)
    f = f.sort_values(['date', 'code']).reset_index(drop=True)
    points.to_parquet(ROOT / 'index_points.parquet', index=False, compression='zstd')
    audit.to_parquet(ROOT / 'source_audit.parquet', index=False, compression='zstd')
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), previous_feature_report_sha256=sha(previous.ROOT / 'feature_report.json'),
        daily_feature_report_sha256=sha(base.SOURCE / 'feature_report.json'), source_report_sha256=sha(ROOT / 'source_report.json'),
        calendar_sha256=sha(CALENDAR), index_points_sha256=sha(ROOT / 'index_points.parquet'),
        source_audit_sha256=sha(ROOT / 'source_audit.parquet'), features_sha256=sha(ROOT / 'features.parquet'),
        rows=len(f), valid=int(f.formula_input_valid.sum()), previous_valid=int(f.prior_formula_input_valid.sum()),
        newly_invalid=int((f.prior_formula_input_valid & ~f.formula_input_valid).sum()),
        valid_with_prior_stock_day_gaps=int((f.formula_input_valid & f.sc_prior_gap.gt(1)).sum()),
        index_sessions=len(points), invalid_prefix_sessions=int((~points.prefix_valid).sum()),
        full_day_source_conflicts=int((~audit.full_day_source_valid).sum()),
        expressions=EXPRESSIONS, native_header=HEADER, full_day_source_audit_used_for_selection=False,
        native_source_price_parity_verified=False, software_compilation_verified=False,
        all_previous_48_inputs_retained=True, new_selection_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['expressions', 'native_header']}


def configure():
    adapter.STEM = STEM; adapter.ROOT = ROOT; adapter.PROTOCOL = PROTOCOL
    adapter.COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')
    adapter.CONTROL = Path('data/research') / (STEM + '_control')
    adapter.EXPRESSIONS = EXPRESSIONS; adapter.HEADER = HEADER
    v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert v['native_expression_arithmetic_verified']
    assert v['after_cutoff_perturbation_checks'] == r['index_sessions'] - r['invalid_prefix_sessions']
    for fold in ['2024', 'recent', 'combined']:
        p = json.loads((Path('config') / (STEM + '_' + fold + '_protocol.json')).read_text())
        assert p['inputs_protocol_sha256'] == sha(PROTOCOL)


if __name__ == '__main__':
    from . import tail_formula_context_2024 as linkage
    from . import tail_formula_recent as study
    from . import tail_formula_relative as relative
    from .tail_formula_offset_logit48 import verify_scores
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    p.add_argument('--fold', choices=['2024', 'recent', 'combined', 'control'], default='2024')
    a = p.parse_args()
    if a.stage == 'features':
        result = features()
    else:
        configure()
        if a.fold == 'control':
            assert a.stage in ['freeze', 'verify', 'analyze']
            result = (linkage.common_analysis(adapter.CONTROL, adapter.COMBINED_PROTOCOL) if a.stage == 'analyze'
                else adapter.control(a.stage + '_control'))
        else:
            adapter.setup(a.fold)
            if a.stage == 'analyze':
                assert (Path('data/research') / (STEM + '_2025') / 'selection_verification.json').exists()
                assert (adapter.CONTROL / 'selection_verification.json').exists()
            if a.fold == 'combined':
                assert a.stage in ['freeze', 'verify', 'analyze']
                result = (linkage.common_analysis(linkage.COMBINED, linkage.PROTOCOL) if a.stage == 'analyze'
                    else getattr(linkage, 'combine' if a.stage == 'freeze' else 'verify_combined')())
            elif a.stage in ['model', 'verify_model']:
                result = getattr(relative, a.stage)('relative')
            elif a.stage == 'verify_scores':
                result = verify_scores()
            elif a.stage in ['freeze', 'verify']:
                result = getattr(study, a.stage)()
            else:
                result = getattr(base, a.stage)()
    print(json.dumps(result, ensure_ascii=False, indent=2))
