"""Strictly previous completed-day directional volume and its recent change."""
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

STEM = 'tail_formula_history_direction'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
NEW_EXPRESSIONS = {
    'DN01': '100*('+ '+'.join(f'HDS{i:02d}' for i in range(1,21))+')/('+ '+'.join(f'HDV{i:02d}' for i in range(1,21))+')',
    'DN02': '50*('+ '('+ '+'.join(f'HDS{i:02d}' for i in range(1,6))+')/('+ '+'.join(f'HDV{i:02d}' for i in range(1,6))+')-('+ '+'.join(f'HDS{i:02d}' for i in range(1,21))+')/('+ '+'.join(f'HDV{i:02d}' for i in range(1,21))+'))'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER + 'HDV:=SUM(V,B0);\nHDS:=HDV*IF(C>REF(C,B0),1,IF(C<REF(C,B0),-1,0));\n'
HEADER += ''.join(f'HDV{i:02d}:=REF(HDV,B{i-1});\nHDS{i:02d}:=REF(HDS,B{i-1});\n' for i in range(1,21))


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['previous_feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert p['daily_feature_report_sha256'] == sha(base.SOURCE / 'feature_report.json')
    r = json.loads((previous.ROOT / 'feature_report.json').read_text())
    v = json.loads((previous.ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert r['features_sha256'] == sha(previous.ROOT / 'features.parquet')
    assert p['history_first'] == '2023-06-01' and p['history_stock_days'] == 20
    return previous.context.market.stock.source_files()


def features():
    if (ROOT / 'feature_report.json').exists():
        raise ValueError('Do not replace the frozen historical directional-volume inputs')
    files = checked_sources(); c = base.conn(); c.read_parquet(files).create_view('daily')
    hist = c.sql('''WITH active AS (
        SELECT date,code,close::DOUBLE AS raw_close,round(close::DOUBLE,2) AS close,
            preclose::DOUBLE AS preclose,volume::DOUBLE AS volume,adjustflag::DOUBLE AS adjustflag
        FROM daily WHERE tradestatus=1 AND date BETWEEN '2023-06-01' AND '2025-12-30'),
        lagged AS (SELECT *,lag(date) OVER w AS previous_date,lag(close) OVER w AS previous_close,
            lag(raw_close) OVER w AS previous_raw_close,lag(adjustflag) OVER w AS previous_adjustflag
            FROM active WINDOW w AS(PARTITION BY code ORDER BY date)),
        atoms AS (SELECT *,coalesce(isfinite(raw_close) AND raw_close>0 AND abs(raw_close-close)<=.0001
            AND isfinite(previous_raw_close) AND previous_raw_close>0 AND abs(previous_raw_close-previous_close)<=.0001
            AND adjustflag=3 AND previous_adjustflag=3 AND isfinite(volume) AND volume>=0 AND volume=floor(volume),false) AS good,
            volume*CASE WHEN close>previous_close THEN 1 WHEN close<previous_close THEN -1 ELSE 0 END AS signed_volume,
            coalesce(abs(round(preclose,2)-previous_close)>.005,false) AS reference_break FROM lagged)
        SELECT date,code,count(*) OVER w20 AS hd_rows20,count(*) OVER w5 AS hd_rows5,
            sum(good::INT) OVER w20 AS hd_good20,
            min(date) OVER w20 AS hd_first_date,max(date) OVER w20 AS hd_last_date,
            first_value(previous_date) OVER w20 AS hd_reference_date,
            sum(volume) OVER w20 AS hd_volume20,sum(volume) OVER w5 AS hd_volume5,
            sum(signed_volume) OVER w20 AS hd_signed20,sum(signed_volume) OVER w5 AS hd_signed5,
            sum(reference_break::INT) OVER w20 AS hd_reference_breaks
        FROM atoms WINDOW w20 AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
            w5 AS(PARTITION BY code ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING)
        ORDER BY date,code''').df(); c.close()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = old.merge(hist, on=['date','code'], how='left', validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
    valid = (f.hd_rows20.eq(20) & f.hd_rows5.eq(5) & f.hd_good20.eq(20)
        & f.hd_volume20.gt(0) & f.hd_volume5.gt(0)
        & f.hd_reference_date.lt(f.hd_first_date) & f.hd_last_date.lt(f.date))
    f['history_direction_valid'] = valid
    f['DN01'] = (100*f.hd_signed20/f.hd_volume20).where(valid)
    f['DN02'] = (50*(f.hd_signed5/f.hd_volume5-f.hd_signed20/f.hd_volume20)).where(valid)
    f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    cal = pd.read_parquet(CALENDAR)
    dates = sorted(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between('2023-06-01','2025-12-30'),'calendar_date'])
    ranks = {d:i for i,d in enumerate(dates)}
    f['hd_market_span'] = f.hd_last_date.map(ranks)-f.hd_first_date.map(ranks)+1
    f['hd_last_gap'] = f.date.map(ranks)-f.hd_last_date.map(ranks)
    ROOT.mkdir(parents=True, exist_ok=True)
    hist.to_parquet(ROOT / 'history.parquet', index=False, compression='zstd')
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), previous_feature_report_sha256=sha(previous.ROOT / 'feature_report.json'),
        daily_feature_report_sha256=sha(base.SOURCE / 'feature_report.json'), calendar_sha256=sha(CALENDAR),
        history_sha256=sha(ROOT / 'history.parquet'), features_sha256=sha(ROOT / 'features.parquet'),
        rows=len(f), valid=int(f.formula_input_valid.sum()), previous_valid=int(f.prior_formula_input_valid.sum()),
        newly_invalid=int((f.prior_formula_input_valid & ~f.formula_input_valid).sum()),
        valid_with_reference_breaks=int((f.formula_input_valid & f.hd_reference_breaks.gt(0)).sum()),
        valid_with_market_gaps=int((f.formula_input_valid & (f.hd_market_span.gt(20)|f.hd_last_gap.gt(1))).sum()),
        first_reference_date=f.hd_reference_date.dropna().min(), last_history_date=f.hd_last_date.dropna().max(),
        expressions=EXPRESSIONS, native_header=HEADER, all_previous_48_inputs_retained=True,
        new_selection_outcomes_read=False, native_source_parity_verified=False, software_compilation_verified=False,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k:v for k,v in r.items() if k not in ['expressions','native_header']}


def configure():
    adapter.STEM = STEM; adapter.ROOT = ROOT; adapter.PROTOCOL = PROTOCOL
    adapter.COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')
    adapter.CONTROL = Path('data/research') / (STEM + '_control')
    adapter.EXPRESSIONS = EXPRESSIONS; adapter.HEADER = HEADER
    r = json.loads((ROOT / 'native_input_verification.json').read_text())
    assert r['passed'] and r['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    assert r['feature_verification_sha256'] == sha(ROOT / 'feature_verification.json')
    for fold in ['2024','recent','combined']:
        p = json.loads((Path('config') / (STEM + '_' + fold + '_protocol.json')).read_text())
        assert p['inputs_protocol_sha256'] == sha(PROTOCOL)


if __name__ == '__main__':
    from . import tail_formula_context_2024 as linkage
    from . import tail_formula_recent as study
    from . import tail_formula_relative as relative
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['features','model','verify_model','scores','freeze','verify','analyze'])
    p.add_argument('--fold', choices=['2024','recent','combined','control'], default='2024')
    a = p.parse_args()
    if a.stage == 'features':
        result = features()
    else:
        configure()
        if a.fold == 'control':
            assert a.stage in ['freeze','verify','analyze']
            result = (linkage.common_analysis(adapter.CONTROL, adapter.COMBINED_PROTOCOL) if a.stage == 'analyze'
                      else adapter.control(a.stage + '_control'))
        else:
            adapter.setup(a.fold)
            if a.stage == 'analyze':
                assert (Path('data/research') / (STEM + '_2025') / 'selection_verification.json').exists()
                assert (adapter.CONTROL / 'selection_verification.json').exists()
            if a.fold == 'combined':
                assert a.stage in ['freeze','verify','analyze']
                result = (linkage.common_analysis(linkage.COMBINED, linkage.PROTOCOL) if a.stage == 'analyze'
                          else getattr(linkage, 'combine' if a.stage == 'freeze' else 'verify_combined')())
            elif a.stage in ['model','verify_model']:
                result = getattr(relative, a.stage)('relative')
            elif a.stage in ['freeze','verify']:
                result = getattr(study, a.stage)()
            else:
                result = getattr(base, a.stage)()
    print(json.dumps(result, ensure_ascii=False, indent=2))
