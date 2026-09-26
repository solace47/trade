"""One expanded training history; the existing absolute target and test folds stay fixed."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .absolute_ridge import FEATURES,PERIODS,score,choose,match_controls
from .corporate_cash import save_json,sha
from .downside_ridge_1449 import fit_score
from .downside_ridge_inputs import ROOT as RECENT
from .long_history_inputs import ROOT,RULE_COMMIT
from .quote_precision import quote_cents
from .turnover_reference import CALENDAR


def freeze(output: Path = ROOT) -> dict:
    if (output/'long/repriced.parquet').exists():
        raise ValueError('Cannot replace a model list after its returns exist')
    feature_report=json.loads((output/'feature_report.json').read_text())
    historical_report=json.loads((output/'historical_label_report.json').read_text())
    recent_report=json.loads((RECENT/'label_report.json').read_text())
    original_report=json.loads((RECENT/'input_report.json').read_text())
    for path,expected in [(output/'features.parquet',feature_report['features_sha256']),
        (output/'historical_training_labels.parquet',historical_report['labels_sha256']),
        (RECENT/'features.parquet',recent_report['features_sha256']),
        (RECENT/'training_labels.parquet',recent_report['labels_sha256']),
        (RECENT/'downside/signals.parquet',original_report['models']['downside']['signals_sha256'])]:
        if sha(path)!=expected:
            raise ValueError('Frozen historical or comparison inputs changed')
    features=pd.read_parquet(output/'features.parquet')
    labels=pd.concat([pd.read_parquet(output/'historical_training_labels.parquet'),pd.read_parquet(RECENT/'training_labels.parquet')],ignore_index=True)
    old_signals=pd.read_parquet(RECENT/'downside/signals.parquet')
    if labels.duplicated(['date','code']).any() or features.duplicated(['date','code']).any():
        raise ValueError('Historical and recent inputs must not overlap')
    eligible=features.loc[features.date.ge('2024-01-01')].copy()
    eligible['price_1449']=eligible.price_1449.map(lambda p:quote_cents(p)/100)
    eligible['price_signal']=eligible.price_1449
    table=pd.read_parquet(CALENDAR)
    calendar=sorted(table.loc[table.is_trading_day.eq('1') & table.calendar_date.between('2022-01-01','2025-12-31'),'calendar_date'])
    index={d:i for i,d in enumerate(calendar)}
    picks,scores,audits=[],[],[]
    with threadpool_limits(limits=4):
        for last,first,end,period in PERIODS:
            train=features.loc[features.date.le(last)].merge(labels[['date','code','downside_score','score_origin','target_exit_date','exit_date']],on=['date','code'],validate='one_to_one').sort_values(['date','code'])
            last_possible=calendar[index[train.date.max()]+10]
            if len(train)!=features.date.le(last).sum() or train.target_exit_date.ge(first).any() or train.exit_date.dropna().ge(first).any() or last_possible>=first:
                raise ValueError('The expanded training set has missing or overlapping labels')
            test=eligible.loc[eligible.date.between(first,end)]
            recent=train.loc[train.date.ge('2024-01-01')]
            original,audit_original=fit_score(recent,recent.downside_score)
            expected=next(a for a in original_report['training'] if a['model']=='downside' and a['period']==period)
            error=max([abs(audit_original['coefficients'][f]-expected['coefficients'][f]) for f in FEATURES]+[abs(audit_original['intercept']-expected['intercept'])])
            if error>1e-12:
                raise ValueError('The unchanged original fitting no longer reproduces')
            reproduced=choose(score(test,original),calendar)
            original_picks=old_signals.loc[old_signals.arm.eq('high') & old_signals.date.between(first,end),['date','code','daily_rank','score']]
            pd.testing.assert_frame_equal(reproduced.sort_values(['date','code']).reset_index(drop=True),original_picks.sort_values(['date','code']).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=0)
            model,audit=fit_score(train,train.downside_score)
            ranked=score(test,model);selected=choose(ranked,calendar)
            if selected.empty:
                selected=pd.DataFrame(columns=['date','code','daily_rank','score'])
            picks.append(selected);scores.append(ranked[['date','code','score']])
            audits.append({'period':period,'train_first':train.date.min(),'train_last':last,'test_first':first,'test_last':end,
                'last_possible_label_day':last_possible,'original_fit_max_difference':error,'old_candidates_reproduced':len(reproduced),
                'new_candidates':len(selected),'positive_test_predictions':int(ranked.score.gt(0).sum()),
                'training_rows_by_year':train.groupby(train.date.str[:4]).size().to_dict(),
                'score_origins':train.score_origin.value_counts().to_dict(),**audit})
    chosen=pd.concat(picks,ignore_index=True);controls=match_controls(chosen,eligible)
    chosen['arm'],chosen['pair_id']='high',chosen.code
    controls['arm']='low'
    members=pd.concat([chosen,controls],ignore_index=True);members['pair_id']=members.date+':'+members.pair_id
    signals=members.merge(eligible,on=['date','code'],validate='one_to_one')
    signals['isST'],signals['reference_gap'],signals['listing_age_sessions']=0,False,20
    signals['half']=signals.date.str[:4]+np.where(signals.date.str[5:7].astype(int).le(6),'H1','H2')
    if len(signals)!=len(members) or signals.duplicated(['date','code']).any():
        raise ValueError('The frozen model/control identities changed')
    folder=output/'long';folder.mkdir(exist_ok=True)
    signals.to_parquet(folder/'signals.parquet',index=False,compression='zstd')
    pd.concat(scores,ignore_index=True).to_parquet(folder/'all_scores.parquet',index=False,compression='zstd')
    model_report={'candidates':len(chosen),'controls':len(controls),'signals_sha256':sha(folder/'signals.parquet'),
        'by_half':signals.groupby(['half','arm']).agg(rows=('code','size'),days=('date','nunique')).reset_index().to_dict('records')}
    save_json(folder/'input_report.json',model_report)
    result={'rule_commit':RULE_COMMIT,'training':audits,'models':{'long':model_report,'recent':original_report['models']['downside']},
        'historical_label_report_sha256':sha(output/'historical_label_report.json'),'feature_report_sha256':sha(output/'feature_report.json'),
        'source_recent_label_report_sha256':sha(RECENT/'label_report.json'),'new_test_returns_read':False,'holdout_read':False}
    save_json(output/'input_report.json',result)
    return result


if __name__=='__main__':
    report=freeze()
    print(json.dumps({'training':report['training'],'long_model':report['models']['long']},ensure_ascii=False,indent=2))
