"""Audit mature unknown-label coverage without reading opportunity values."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_morning_range as prior
from trade_research.corporate_cash import save_json,sha
from trade_research.turnover_reference import CALENDAR
from find_existing_tail_formula_models import find

ROOT=Path('data/research/tail_formula_confirmed_success')
INTENT=Path('config/tail_formula_confirmed_success_intent.json')
PROTOCOL=Path('config/tail_formula_confirmed_success_audit_protocol.json')
META=prior.META
LABEL_META=['date','code','next_date','known15','known_no_trade']


def mature(frame,spec):
    """Missing, current or future observation dates cannot become negatives."""
    return (frame.date.ge(spec['training_start']) & frame.date.lt(spec['training_end'])
            & frame.next_date.notna() & frame.next_date.gt(frame.date)
            & frame.next_date.lt(spec['training_end']))


def checked():
    p=json.loads(PROTOCOL.read_text()); i=json.loads(INTENT.read_text())
    assert p['intent_sha256']==sha(INTENT) and p['label_columns']==LABEL_META
    assert p['input_columns']==META and not p['new_2026_prices_allowed']
    for file,digest in p['source_hashes'].items(): assert sha(Path(file))==digest,file
    gate=json.loads(Path(i['conditional_gate']).read_text())
    assert gate['passed'] and not gate['supports_further_validation']
    committed=subprocess.run(['git','show','HEAD:docs/selection-formula.md'],text=True,capture_output=True,check=True).stdout
    assert sha(INTENT) in committed and sha(PROTOCOL) in committed
    return p


def run():
    p=checked(); assert not (ROOT/'audit_verification.json').exists(); ROOT.mkdir(parents=True,exist_ok=True)
    c=base.conn(); columns=','.join(META)
    old=c.sql(f"SELECT {columns} FROM read_parquet('{p['feature_files'][0]}') ORDER BY date,code").df()
    current=c.sql(f"SELECT {columns} FROM read_parquet('{p['feature_files'][1]}') ORDER BY date,code").df()
    pd.testing.assert_frame_equal(old.loc[old.date.ge('2024-01-01')].reset_index(drop=True),
        current.loc[current.date.lt('2025-01-01')].reset_index(drop=True),check_exact=True)
    features=pd.concat([old.loc[old.date.lt('2024-01-01')],current],ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    assert len(features)==1815129 and not features.duplicated(['date','code']).any()
    statuses=c.sql(f"SELECT {','.join(LABEL_META)} FROM read_parquet('{p['label_file']}') WHERE date<'2025-07-01' ORDER BY date,code").df()
    assert not statuses.duplicated(['date','code']).any()
    assert statuses[['known15','known_no_trade']].isin([False,True]).all().all()
    assert not (statuses.known15 & statuses.known_no_trade).any()
    scope=features.loc[features.date.lt('2025-07-01')].merge(statuses,on=['date','code'],how='left',validate='one_to_one',indicator=True)
    assert scope['_merge'].eq('both').all(); scope=scope.drop(columns='_merge')
    scope['unknown']=~scope.known15 & ~scope.known_no_trade
    dates=c.sql(f"SELECT calendar_date FROM read_parquet('{CALENDAR}') WHERE is_trading_day='1' AND calendar_date BETWEEN '2023-01-01' AND '2025-07-01' ORDER BY calendar_date").df().calendar_date.tolist()
    scheduled=dict(zip(dates[:-1],dates[1:])); scope['scheduled_next_market_date']=scope.date.map(scheduled)
    assert scope.scheduled_next_market_date.notna().all()
    scope['next_date_matches_calendar']=scope.next_date.eq(scope.scheduled_next_market_date)
    assert scope.loc[scope.known15,'next_date_matches_calendar'].all(), 'Known target dates must be strict next market dates'
    scope.to_parquet(ROOT/'status_scope.parquet',index=False,compression='zstd')
    c.register('features',features); c.register('statuses',statuses)
    records=[]; receipts={}
    for fold,spec in p['folds'].items():
        selected=scope.loc[mature(scope,spec) & scope.formula_input_valid].sort_values(['date','code']).reset_index(drop=True)
        independent=c.sql(f'''SELECT f.*,s.next_date,s.known15,s.known_no_trade FROM features f JOIN statuses s USING(date,code)
            WHERE formula_input_valid AND f.date>='{spec['training_start']}' AND f.date<'{spec['training_end']}'
              AND next_date>f.date AND next_date<'{spec['training_end']}' ORDER BY date,code''').df()
        pd.testing.assert_frame_equal(selected[[*META,'next_date','known15','known_no_trade']],independent,check_exact=True)
        candidates=scope.loc[scope.date.ge(spec['training_start']) & scope.date.lt(spec['training_end']) & scope.formula_input_valid]
        filename=ROOT/(fold+'_mature_metadata.parquet'); selected.to_parquet(filename,index=False,compression='zstd'); receipts[str(filename)]=sha(filename)
        proposal=ROOT/(fold+'_metadata_lookup.json')
        q=dict(parameters=p['parameters'],feature_names=list(prior.EXPRESSIONS),
            training_start=spec['training_start'],training_end=spec['training_end'],
            expected_training_rows=len(selected),expected_training_days=selected.date.nunique())
        save_json(proposal,q); lookup=find(proposal,'relative')
        record=dict(fold=fold,rows=len(selected),days=selected.date.nunique(),last_observation=selected.next_date.max(),
            known_rows=int(selected.known15.sum()),confirmed_no_trade_rows=int(selected.known_no_trade.sum()),
            newly_included_unknown_rows=int(selected.unknown.sum()),unknown_ratio=float(selected.unknown.mean()),
            old_known_or_no_trade_rows=int((selected.known15|selected.known_no_trade).sum()),
            missing_next_date_valid_rows=int(candidates.next_date.isna().sum()),
            delayed_next_date_valid_rows=int((candidates.next_date.notna() & ~candidates.next_date_matches_calendar).sum()),
            next_date_not_after_signal_valid_rows=int((candidates.next_date.notna() & candidates.next_date.le(candidates.date)).sum()),
            mature_unknown_delayed_rows=int((selected.unknown & ~selected.next_date_matches_calendar).sum()),
            existing_model_metadata_matches=lookup['matches'],metadata_sha256=sha(filename))
        assert record['last_observation']<spec['evaluation_start']
        records.append(record)
    c.close()
    report=dict(protocol_sha256=sha(PROTOCOL),intent_sha256=sha(INTENT),source_hashes=p['source_hashes'],
        original_metadata_keys=len(features),audited_status_keys=len(scope),records=records,
        status_scope_sha256=sha(ROOT/'status_scope.parquet'),all_mature_metadata_sha256=receipts,
        opportunity_returns_scores_and_selected_groups_not_read=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'audit_report.json',report)
    proof=dict(passed=True,protocol_sha256=sha(PROTOCOL),audit_report_sha256=sha(ROOT/'audit_report.json'),
        all_original_metadata_keys_preserved=True,all_four_mature_domains_independent_sql_equal=True,
        known_targets_match_strict_next_market_calendar=True,missing_dates_not_assigned_negative=True,
        supports_complete_model_protocol=all(r['newly_included_unknown_rows']>0 and r['days']>=100
            and r['next_date_not_after_signal_valid_rows']==0 for r in records),
        requires_explicit_delayed_and_missing_date_policy=any(r['delayed_next_date_valid_rows'] or r['missing_next_date_valid_rows'] for r in records),
        no_new_fit_or_label_numeric_projection=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'audit_verification.json',proof)
    return dict(report_sha256=sha(ROOT/'audit_report.json'),proof_sha256=sha(ROOT/'audit_verification.json'),records=records,
        supports_complete_model_protocol=proof['supports_complete_model_protocol'],
        requires_date_policy=proof['requires_explicit_delayed_and_missing_date_policy'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__); parser.parse_args()
    print(json.dumps(run(),ensure_ascii=False,indent=2),flush=True)
