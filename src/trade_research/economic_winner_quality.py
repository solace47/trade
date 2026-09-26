"""Preserve inspected cash labels and add the existing period quality flags."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corporate_cash import save_json,sha
from .economic_winner import ROOT as SOURCE

ROOT=SOURCE/'period_quality'
PROTOCOL=Path('config/economic_winner_quality.json')
ISSUES=Path('data/research/market_issues_ci')
PERIOD=Path('data/research/quality_period_2024_2025.json')


def evaluate():
    if (ROOT/'label_report.json').exists():raise ValueError('Do not overwrite period-quality sensitivity')
    source_report=json.loads((SOURCE/'label_report.json').read_text())
    check=json.loads((SOURCE/'label_verification.json').read_text())
    assert check['passed'] and check['label_report_sha256']==sha(SOURCE/'label_report.json')
    assert sha(SOURCE/'labels.parquet')==source_report['labels_sha256']
    protocol=json.loads(PROTOCOL.read_text())
    paths=sorted(ISSUES.glob('shard_*.csv'));assert len(paths)==20
    issues=pd.concat([pd.read_csv(path,dtype=str) for path in paths],ignore_index=True)
    bad=issues.loc[issues.kind.isin(protocol['bad_day_kinds'])&issues.date.between('2024-01-01','2025-12-31'),
        ['date','code']].drop_duplicates()
    period=json.loads(PERIOD.read_text())
    assert period['first_date']<='2024-01-01' and period['last_date']>='2025-12-31'
    row=pd.read_parquet(SOURCE/'labels.parquet')
    bad_keys=pd.MultiIndex.from_frame(bad)
    row['period_entry_bad_day']=pd.MultiIndex.from_frame(row[['date','code']]).isin(bad_keys)
    row['period_exit_bad_day']=pd.MultiIndex.from_frame(row[['next_date','code']]).isin(bad_keys)
    row['period_bad_symbol']=row.code.isin(period['period_bad_symbols'])
    entry_bad=row.period_entry_bad_day|row.period_bad_symbol
    exit_bad=row.period_exit_bad_day|row.period_bad_symbol
    # A valid known non-entry has no exit whose provenance could affect P&L.
    source_unknown=row.necessary_tradeable&(entry_bad|(exit_bad&~row.label15.eq('no_trade')))
    row['period_source_unknown']=source_unknown
    row['original_base_status']=row.base_status
    row.loc[source_unknown&row.base_status.isin(['ordinary_t1','not_bought']),'base_status']='period_source_unknown'
    for bps in [5,15]:
        row[f'original_label{bps}']=row[f'label{bps}']
        row.loc[source_unknown,f'label{bps}']='unknown'
        row.loc[source_unknown,f'known_profit{bps}']=False
        row.loc[source_unknown,[f'buy_cash{bps}',f'sell_cash{bps}',f'net_return{bps}']]=np.nan
    ROOT.mkdir(exist_ok=True)
    row.to_parquet(ROOT/'labels.parquet',index=False,compression='zstd')
    result=dict(source_report,labels_sha256=sha(ROOT/'labels.parquet'),
        interpretation='existing_period_quality_sensitivity_after_original_analysis',
        original_label_report_sha256=sha(SOURCE/'label_report.json'),
        original_label_verification_sha256=sha(SOURCE/'label_verification.json'),
        original_analysis_report_sha256=sha(SOURCE/'analysis_report.json'),
        protocol_sha256=sha(PROTOCOL),period_report_sha256=sha(PERIOD),
        issue_files_sha256={str(path):sha(path) for path in paths},
        changed_counts={str(bps):int(row[f'label{bps}'].ne(row[f'original_label{bps}']).sum()) for bps in [5,15]},
        label_counts={str(bps):row[f'label{bps}'].value_counts().to_dict() for bps in [5,15]},
        source_flag_counts={col:int(row[col].sum()) for col in ['period_entry_bad_day','period_exit_bad_day','period_bad_symbol','period_source_unknown']},
        base_statuses=row.groupby(['half','board','base_status']).size().rename('rows').reset_index().to_dict('records'))
    save_json(ROOT/'label_report.json',result)
    return {k:result[k] for k in ['labels_sha256','changed_counts','label_counts','source_flag_counts']}


if __name__=='__main__':print(json.dumps(evaluate(),ensure_ascii=False,indent=2))
