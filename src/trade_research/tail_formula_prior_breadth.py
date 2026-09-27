"""Prior stock-day index breadth as two additional native formula inputs."""
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

STEM = 'tail_formula_prior_breadth'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
NEW_EXPRESSIONS = {'BR01': 'BRD1', 'BR02': 'BRD1-(BRD1+BRD2+BRD3+BRD4+BRD5)/5'}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}
HEADER = previous.HEADER + 'BRR:=IF(INDEXADV+INDEXDEC>0,100*INDEXADV/(INDEXADV+INDEXDEC),DRAWNULL);\n'
HEADER += ''.join(f'BRD{i}:=REF(BRR,B{i-1});\n' for i in range(1, 6))


def checked_sources():
    p = json.loads(PROTOCOL.read_text()); r = json.loads((ROOT / 'source_report.json').read_text())
    assert r['passed'] and r['protocol_sha256'] == sha(PROTOCOL)
    assert r['counts_sha256'] == sha(ROOT / 'counts.parquet')
    assert p['previous_feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert p['daily_feature_report_sha256'] == sha(base.SOURCE / 'feature_report.json')
    old = json.loads((previous.ROOT / 'feature_report.json').read_text())
    proof = json.loads((previous.ROOT / 'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256'] == sha(previous.ROOT / 'feature_report.json')
    assert old['features_sha256'] == sha(previous.ROOT / 'features.parquet')
    return previous.context.market.stock.source_files()


def features():
    assert not (ROOT / 'feature_report.json').exists(), 'Do not replace frozen breadth inputs'
    files = checked_sources(); c = base.conn(); c.read_parquet(files).create_view('daily')
    c.read_parquet(str(ROOT / 'counts.parquet')).create_view('breadth')
    expressions = ','.join(f'lag({field},{i}) OVER w AS br_{name}_{i}'
        for i in range(1, 6) for field, name in [('date', 'date'), ('breadth_value', 'value')])
    hist = c.sql(f'''WITH active AS(SELECT date,code FROM daily
        WHERE tradestatus=1 AND date BETWEEN '2023-06-01' AND '2025-12-30'),
        joined AS(SELECT a.date,a.code,100.0*b.up_count/(b.up_count+b.down_count) AS breadth_value
        FROM active a LEFT JOIN breadth b ON b.date=a.date
        AND b.code=CASE WHEN starts_with(a.code,'sh.') THEN 'sh.000001' ELSE 'sz.399001' END),
        h AS(SELECT date,code,{expressions} FROM joined WINDOW w AS(PARTITION BY code ORDER BY date))
        SELECT * FROM h WHERE date>='2024-01-01' ORDER BY date,code''').df(); c.close()
    old = pd.read_parquet(previous.ROOT / 'features.parquet')
    f = old.merge(hist, on=['date', 'code'], how='left', validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)
    values = f[[f'br_value_{i}' for i in range(1, 6)]]
    valid = np.isfinite(values).all(axis=1) & values.ge(0).all(axis=1) & values.le(100).all(axis=1)
    for i in range(1, 6):
        valid &= f[f'br_date_{i}'].lt(f.date)
    f['BR01'] = f.br_value_1.where(valid)
    f['BR02'] = (f.br_value_1 - values.mean(axis=1)).where(valid)
    f['breadth_valid'] = valid; f['prior_formula_input_valid'] = f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    calendar = pd.read_parquet(CALENDAR)
    days = sorted(calendar.loc[calendar.is_trading_day.eq('1') & calendar.calendar_date.between('2023-06-01', '2025-12-30'), 'calendar_date'])
    ranks = {day: i for i, day in enumerate(days)}
    f['br_prior_gap'] = f.date.map(ranks) - f.br_date_1.map(ranks)
    f['br_history_span'] = f.br_date_1.map(ranks) - f.br_date_5.map(ranks) + 1
    f.to_parquet(ROOT / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), source_report_sha256=sha(ROOT / 'source_report.json'),
        previous_feature_report_sha256=sha(previous.ROOT / 'feature_report.json'),
        daily_feature_report_sha256=sha(base.SOURCE / 'feature_report.json'), calendar_sha256=sha(CALENDAR),
        features_sha256=sha(ROOT / 'features.parquet'), counts_sha256=sha(ROOT / 'counts.parquet'),
        rows=len(f), valid=int(f.formula_input_valid.sum()), previous_valid=int(f.prior_formula_input_valid.sum()),
        newly_invalid=int((f.prior_formula_input_valid & ~f.formula_input_valid).sum()),
        valid_with_prior_stock_day_gaps=int((f.formula_input_valid & f.br_prior_gap.gt(1)).sum()),
        valid_with_five_day_gaps=int((f.formula_input_valid & f.br_history_span.gt(5)).sum()),
        first_history_date=f.br_date_5.min(), last_history_date=f.br_date_1.max(),
        expressions=EXPRESSIONS, native_header=HEADER, all_previous_48_inputs_retained=True,
        native_source_parity_verified=False, software_compilation_verified=False,
        new_selection_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_report.json', r)
    return {k: v for k, v in r.items() if k not in ['expressions', 'native_header']}


def configure():
    adapter.STEM = STEM; adapter.ROOT = ROOT; adapter.PROTOCOL = PROTOCOL
    adapter.COMBINED_PROTOCOL = Path('config') / (STEM + '_combined_protocol.json')
    adapter.CONTROL = Path('data/research') / (STEM + '_control')
    adapter.EXPRESSIONS = EXPRESSIONS; adapter.HEADER = HEADER
    v = json.loads((ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(ROOT / 'feature_report.json')
    assert v['native_expression_arithmetic_verified'] and v['after_cutoff_perturbation_checks'] >= 484
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
