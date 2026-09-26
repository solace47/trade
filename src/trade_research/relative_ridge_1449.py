"""Single daily-demeaned training target, holding all other model rules fixed."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .absolute_ridge import FEATURES, PERIODS, score, choose, match_controls
from .corporate_cash import save_json, sha
from .downside_ridge_1449 import fit_score
from .downside_ridge_inputs import ROOT as SOURCE
from .quote_precision import quote_cents
from .turnover_reference import CALENDAR

ROOT=Path('data/research/relative_ridge_1449')
RULE_COMMIT='6cc751b'


def relative_targets(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    if frame.empty or frame[['date','code','downside_score']].isna().any().any() or frame.duplicated(['date','code']).any():
        raise ValueError('Daily training targets require unique complete identities')
    if not np.isfinite(frame.downside_score).all():
        raise ValueError('Training scenarios must be finite; actual unknown P&L is separate')
    clipped=frame.downside_score.clip(-.15,.15)
    center=clipped.groupby(frame.date).transform('mean')
    relative=clipped-center
    if relative.groupby(frame.date).mean().abs().max()>1e-12:
        raise ValueError('The complete daily training universe did not center')
    return relative,center


def freeze(output: Path = ROOT) -> dict:
    if (output/'relative/repriced.parquet').exists():
        raise ValueError('Cannot replace new lists after their execution has been read')
    source=json.loads((SOURCE/'label_report.json').read_text())
    old_report=json.loads((SOURCE/'input_report.json').read_text())
    for name,key in [('features','features_sha256'),('training_labels','labels_sha256')]:
        if sha(SOURCE/(name+'.parquet'))!=source[key]:
            raise ValueError('Existing absolute-model inputs changed')
    if sha(SOURCE/'downside/signals.parquet')!=old_report['models']['downside']['signals_sha256']:
        raise ValueError('The predeclared absolute comparator changed')
    features=pd.read_parquet(SOURCE/'features.parquet')
    labels=pd.read_parquet(SOURCE/'training_labels.parquet')
    old_signals=pd.read_parquet(SOURCE/'downside/signals.parquet')
    eligible=features.copy()
    eligible['price_1449']=eligible.price_1449.map(lambda p:quote_cents(p)/100)
    eligible['price_signal']=eligible.price_1449
    raw_calendar=pd.read_parquet(CALENDAR)
    calendar=sorted(raw_calendar.loc[raw_calendar.is_trading_day.eq('1') & raw_calendar.calendar_date.between('2024-01-01','2025-12-31'),'calendar_date'])
    positions={d:i for i,d in enumerate(calendar)}
    picks,scores,training,audits=[],[],[],[]
    with threadpool_limits(limits=4):
        for last,first,end,period in PERIODS:
            train=features.loc[features.date.le(last)].merge(labels[['date','code','downside_score','target_exit_date','exit_date']],on=['date','code'],validate='one_to_one').sort_values(['date','code'])
            last_possible=calendar[positions[train.date.max()]+10]
            if len(train)!=features.date.le(last).sum() or train.target_exit_date.ge(first).any() or train.exit_date.dropna().ge(first).any() or last_possible>=first:
                raise ValueError('Training labels are incomplete or overlap testing')
            test=eligible.loc[eligible.date.between(first,end)]
            original,original_audit=fit_score(train,train.downside_score)
            expected=next(a for a in old_report['training'] if a['model']=='downside' and a['period']==period)
            errors=[abs(original_audit['coefficients'][name]-expected['coefficients'][name]) for name in FEATURES]
            errors.append(abs(original_audit['intercept']-expected['intercept']))
            if max(errors)>1e-12:
                raise ValueError('The absolute comparator no longer reproduces')
            original_picks=choose(score(test,original),calendar)
            expected_picks=old_signals.loc[old_signals.arm.eq('high') & old_signals.date.between(first,end),['date','code','daily_rank','score']]
            pd.testing.assert_frame_equal(original_picks.sort_values(['date','code']).reset_index(drop=True),expected_picks.sort_values(['date','code']).reset_index(drop=True),check_dtype=False,atol=1e-12,rtol=0)
            target,center=relative_targets(train)
            model,audit=fit_score(train,target,clip_target=False)
            ranked=score(test,model)
            chosen=choose(ranked,calendar)
            if chosen.empty:
                chosen=pd.DataFrame(columns=['date','code','daily_rank','score'])
            picks.append(chosen);scores.append(ranked[['date','code','score']])
            training.append(train[['date','code','downside_score']].assign(period=period,daily_center=center,relative_target=target))
            audits.append({'period':period,'train_last':last,'test_first':first,'test_last':end,
                'last_possible_training_day':last_possible,'days':train.date.nunique(),
                'max_abs_daily_target_mean':float(target.groupby(train.date).mean().abs().max()),
                'min_relative_target':float(target.min()),'max_relative_target':float(target.max()),
                'absolute_fit_max_difference':max(errors),'absolute_candidates_reproduced':len(original_picks),
                'relative_candidates':len(chosen),'positive_test_predictions':int(ranked.score.gt(0).sum()),**audit})
    chosen=pd.concat(picks,ignore_index=True)
    controls=match_controls(chosen,eligible)
    chosen['arm'],chosen['pair_id']='high',chosen.code
    controls['arm']='low'
    members=pd.concat([chosen,controls],ignore_index=True)
    members['pair_id']=members.date+':'+members.pair_id
    signals=members.merge(eligible,on=['date','code'],validate='one_to_one')
    signals['isST'],signals['reference_gap'],signals['listing_age_sessions']=0,False,20
    signals['half']=signals.date.str[:4]+np.where(signals.date.str[5:7].astype(int).le(6),'H1','H2')
    if len(signals)!=len(members) or signals.duplicated(['date','code']).any():
        raise ValueError('A selected key changed')
    folder=output/'relative';folder.mkdir(parents=True,exist_ok=True)
    signals.to_parquet(folder/'signals.parquet',index=False,compression='zstd')
    pd.concat(scores,ignore_index=True).to_parquet(folder/'all_scores.parquet',index=False,compression='zstd')
    pd.concat(training,ignore_index=True).to_parquet(output/'training_targets.parquet',index=False,compression='zstd')
    model_report={'candidates':len(chosen),'controls':len(controls),'signals_sha256':sha(folder/'signals.parquet'),
        'by_half':signals.groupby(['half','arm']).agg(rows=('code','size'),days=('date','nunique')).reset_index().to_dict('records')}
    save_json(folder/'input_report.json',model_report)
    result={'rule_commit':RULE_COMMIT,'training':audits,'models':{'relative':model_report,'absolute':old_report['models']['downside']},
        'source_labels_sha256':source['labels_sha256'],'source_features_sha256':source['features_sha256'],
        'training_targets_sha256':sha(output/'training_targets.parquet'),
        'new_selected_test_returns_read':False,'holdout_read':False,
        'positive_relative_score_is_not_positive_expected_profit':True}
    save_json(output/'input_report.json',result)
    return result


if __name__=='__main__':
    report=freeze()
    print(json.dumps({'training':report['training'],'relative':report['models']['relative']},ensure_ascii=False,indent=2))
