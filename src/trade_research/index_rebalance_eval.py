"""Four independent rebalance dates, fixed top-five and full-cohort diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json, sha
from .index_rebalance import BATCHES, ROOT


def complete_average(values: pd.Series) -> float | None:
    return float(values.mean()) if len(values) and values.notna().all() else None


def summarize(rows: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    cells = []
    for cohort in ('primary_top5', 'full_eligible'):
        current = rows.loc[rows.primary_top5] if cohort == 'primary_top5' else rows
        for horizon in (1, 5):
            for slip in (5, 15):
                fields = {k: f'catalog_scenario_{k}{slip}' for k in ('return','lower','upper')}
                for date in BATCHES.values():
                    part = current.loc[current.date.eq(date) & current.horizon.eq(horizon)]
                    high, low = part.loc[part.arm.eq('high')], part.loc[part.arm.eq('low')]
                    pairs = high.merge(low, on=['date','pair_id'], suffixes=('_high','_low'), validate='one_to_one')
                    if len(pairs) != len(low):
                        raise ValueError('A fixed control lost its candidate')
                    cell = {'cohort':cohort,'date':date,'horizon':horizon,'slip':slip,
                        'candidates':len(high),'controls':len(low),'unmatched':len(high)-len(low),
                        'bought':int(high.entry_status.eq('filled').sum()),
                        'verified_unknown_candidates':int(high.known_return5.isna().sum()),
                        'source_invalid_candidates':int((~high.execution_source_valid).sum()),
                        'source_invalid_controls':int((~low.execution_source_valid).sum()),
                        'catalogue_actions_candidates':int(high.catalog_action_applied.sum()),
                        'catalogue_actions_controls':int(low.catalog_action_applied.sum()),
                        'unresolved_candidates':int((high.entry_status.eq('filled') & high.exit_price.isna()).sum()),
                        'unresolved_controls':int((low.entry_status.eq('filled') & low.exit_price.isna()).sum())}
                    for name, field in fields.items():
                        cell['own_'+name] = complete_average(high[field])
                        cell['control_'+name] = complete_average(low[field])
                    cell['matched_own_return'] = complete_average(pairs[fields['return']+'_high'])
                    for name, left, right in [('return','return','return'),('lower','lower','upper'),('upper','upper','lower')]:
                        cell['edge_'+name] = complete_average(pairs[fields[left]+'_high']-pairs[fields[right]+'_low'])
                    cells.append(cell)
    summary=[]
    frame=pd.DataFrame(cells)
    means=['own_return','own_lower','own_upper','control_return','matched_own_return','edge_return','edge_lower','edge_upper']
    for (cohort,horizon,slip), part in frame.groupby(['cohort','horizon','slip']):
        if set(part.date) != set(BATCHES.values()):
            raise ValueError('Every frozen batch must be retained')
        summary.append({'cohort':cohort,'horizon':int(horizon),'slip':int(slip),'batches':len(part),
            **{name:complete_average(part[name]) for name in means}})
    return cells,summary


def evaluate(output: Path = ROOT) -> dict:
    inputs=json.loads((output/'input_report.json').read_text())
    accounting=json.loads((output/'catalog_scenario_report.json').read_text())
    if sha(output/'signals.parquet') != inputs['signals_sha256'] or sha(output/'catalog_scenario.parquet') != accounting['output_sha256']:
        raise ValueError('Frozen index inputs or economic outputs changed')
    if accounting['bootstrap_requested'] or any(row['weekly_interval'] is not None for row in accounting['annual']):
        raise ValueError('Four batches do not justify the generic bootstrap')
    signals=pd.read_parquet(output/'signals.parquet')
    rows=pd.read_parquet(output/'catalog_scenario.parquet')
    rows=rows.merge(signals[['date','code','daily_rank','primary_top5','decision_shares']],on=['date','code'],validate='many_to_one')
    if len(rows) != len(signals)*2 or not np.array_equal(rows.loc[rows.entry_status.eq('filled'),'shares'],rows.loc[rows.entry_status.eq('filled'),'decision_shares']):
        raise ValueError('The fixed trade grid or decision share count changed')
    cells,summary=summarize(rows)
    rows.to_parquet(output/'batch_accounted.parquet',index=False,compression='zstd')
    result={'interpretation':'conditional_catalogue_and_recorded_execution_signal_day_means_not_capital_portfolio',
        'by_batch':cells,'batch_equal_summary':summary,'independent_batches':4,
        'bootstrap_intervals':None,'source_invalid_rows':accounting['source_invalid_rows'],
        'catalogue_action_rows':accounting['catalogue_action_rows'],
        'unresolved_terminal_rows':accounting['unresolved_terminal_rows'],
        'execution_rows':len(rows),'holdout_read':False,
        'accounted_sha256':sha(output/'batch_accounted.parquet')}
    save_json(output/'batch_report.json',result)
    return result


if __name__ == '__main__':
    report=evaluate()
    print(json.dumps({'primary_stress_T5':[r for r in report['by_batch'] if r['cohort']=='primary_top5' and r['horizon']==5 and r['slip']==15],
        'summaries':report['batch_equal_summary']},ensure_ascii=False,indent=2))
