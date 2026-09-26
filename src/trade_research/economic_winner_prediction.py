"""Fit one cash-class predictor, then calibrate it chronologically and abstain."""
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from .corporate_cash import save_json,sha
from .next_day_winner import calendar

ROOT=Path('data/research/economic_winner_prediction')
SOURCE=Path('data/research/economic_winner/period_quality')
FEATURE_SOURCE=Path('data/research/winner_direction')
PROTOCOL=Path('config/economic_winner_prediction.json')
ORDINARY=['economic_loser','economic_winner','middle']


def select(scores,dates,policy):
    position={date:i for i,date in enumerate(dates)};last={};out=[]
    primary=policy=='calibrated'
    eligible=scores.loc[scores['calibrated_score' if primary else 'raw_score'].gt(0)].copy()
    if primary:eligible=eligible.loc[eligible.in_calibration_support]
    order=['date','calibrated_score','raw_score','code'] if primary else ['date','raw_score','code']
    ascending=[True,False,False,True] if primary else [True,False,True]
    for date,g in eligible.sort_values(order,ascending=ascending).groupby('date',sort=True):
        rank=0
        for row in g.itertuples():
            # Match the existing policy: skip the five sessions after selection.
            if row.code in last and position[date]-last[row.code]<=5:continue
            rank+=1;last[row.code]=position[date]
            out.append({'date':date,'code':row.code,'daily_rank':rank,'policy':policy})
            if rank==5:break
    return pd.DataFrame(out,columns=['date','code','daily_rank','policy'])


def freeze():
    if (ROOT/'input_report.json').exists():raise ValueError('Do not overwrite frozen predictions')
    source_report=json.loads((SOURCE/'label_report.json').read_text())
    check=json.loads((SOURCE/'label_verification.json').read_text())
    assert check['passed'] and check['label_report_sha256']==sha(SOURCE/'label_report.json')
    assert source_report['labels_sha256']==sha(SOURCE/'labels.parquet')
    source_input=json.loads((FEATURE_SOURCE/'input_report.json').read_text())
    for name in ['model_features.parquet','visible_pool.parquet']:
        assert sha(FEATURE_SOURCE/name)==source_input['outputs_sha256'][name]
    ROOT.mkdir(exist_ok=True)
    encoded=pd.read_parquet(FEATURE_SOURCE/'model_features.parquet')
    pool=pd.read_parquet(FEATURE_SOURCE/'visible_pool.parquet',columns=['date','code','half','board','necessary_tradeable','decision_shares','price_1449'])
    pd.testing.assert_frame_equal(encoded[['date','code']],pool[['date','code']])
    assert pool.necessary_tradeable.all() and not pool.duplicated(['date','code']).any()
    x=encoded.drop(columns=['date','code']);assert x.shape[1]==39
    old=pool.date.lt('2025-01-01')
    historic=pd.read_parquet(SOURCE/'labels.parquet',filters=[('date','<','2025-01-01')],
        columns=['date','code','next_date','label15','net_return15'])
    historic=pool.loc[old,['date','code']].merge(historic,on=['date','code'],validate='one_to_one')
    pd.testing.assert_frame_equal(historic[['date','code']].reset_index(drop=True),pool.loc[old,['date','code']].reset_index(drop=True))
    historic.index=pool.index[old]
    train=pd.Series(False,index=pool.index)
    calibration=pd.Series(False,index=pool.index)
    train.loc[historic.index]=historic.date.lt('2024-07-01')&historic.next_date.lt('2024-07-01')
    calibration.loc[historic.index]=historic.date.ge('2024-07-01')&historic.next_date.lt('2025-01-01')
    test=pool.date.ge('2025-01-01')
    training=historic.loc[train.loc[historic.index]].copy()
    medians=x.loc[train].median();assert medians.notna().all()
    filled=x.fillna(medians);mean=filled.loc[train].mean();scale=filled.loc[train].std(ddof=0).clip(lower=1e-12)
    z=((filled-mean)/scale).to_numpy(dtype='float64');assert np.isfinite(z).all()
    with warnings.catch_warnings(),threadpool_limits(limits=4):
        warnings.simplefilter('error',ConvergenceWarning)
        fitted=LogisticRegression(C=1.,solver='newton-cholesky',tol=1e-8,max_iter=100,random_state=20260927)
        fitted.fit(z[train],training.label15)
        probability=fitted.predict_proba(z)
    print(json.dumps({'stage':'fit_complete','train_rows':len(training),'iterations':fitted.n_iter_.tolist()}),flush=True)
    classes=list(fitted.classes_);assert set(classes)==set(ORDINARY+['unknown','no_trade'])
    class_returns=training.groupby('label15').net_return15.mean().reindex(ORDINARY)
    assert class_returns.notna().all()
    ordinary_probability=probability[:,[classes.index(name) for name in ORDINARY]]
    raw=(ordinary_probability@class_returns.to_numpy())/ordinary_probability.sum(axis=1)
    known=historic.loc[calibration.loc[historic.index]&historic.net_return15.notna()].copy()
    known['raw_score']=raw[known.index]
    weight=1/known.groupby('date').code.transform('size').to_numpy(dtype='float64')
    weight/=weight.sum()
    sx=known.raw_score.to_numpy();y=known.net_return15.to_numpy()
    xmean=float(weight@sx);ymean=float(weight@y)
    variance=float(weight@((sx-xmean)**2));covariance=float(weight@((sx-xmean)*(y-ymean)))
    slope=max(0.,covariance/variance) if variance>0 else 0.
    intercept=ymean-slope*xmean
    cal_scores=raw[calibration];bounds=[float(cal_scores.min()),float(cal_scores.max())]
    edges=np.quantile(cal_scores,np.arange(.1,1,.1))
    assert np.all(np.diff(edges)>0)
    scores=pool[['date','code','half','board']].copy()
    for i,name in enumerate(classes):scores['p_'+name]=probability[:,i]
    scores['ordinary_probability']=ordinary_probability.sum(axis=1)
    scores['raw_score']=raw;scores['calibrated_score']=intercept+slope*raw
    scores['in_calibration_support']=scores.raw_score.between(*bounds)
    scores['score_bin']=np.searchsorted(edges,raw,side='right')+1
    scores['role']=np.select([train,calibration,test],['fit','calibration','test'],default='purged_boundary')
    scores.to_parquet(ROOT/'scores.parquet',index=False,compression='zstd')
    historic.to_parquet(ROOT/'historic_labels.parquet',index=False,compression='zstd')
    model={'feature_names':list(x),'median':medians.tolist(),'mean':mean.tolist(),'scale':scale.tolist(),
        'classes':classes,'coefficients':fitted.coef_.tolist(),'intercepts':fitted.intercept_.tolist(),
        'iterations':fitted.n_iter_.tolist(),'class_returns':class_returns.to_dict(),
        'C':1.,'solver':'newton-cholesky','tol':1e-8,'max_iter':100,
        'calibration':{'intercept':intercept,'slope':slope,'raw_score_bounds':bounds,'raw_score_deciles':edges.tolist(),
            'known_rows':len(known),'known_dates':int(known.date.nunique()),'weighted_score_mean':xmean,
            'weighted_net_mean':ymean,'weighted_covariance':covariance,'weighted_score_variance':variance}}
    save_json(ROOT/'model.json',model)
    selected=pd.concat([select(scores.loc[test],calendar(),name) for name in ['calibrated','raw']],ignore_index=True)
    selected=selected.merge(scores,on=['date','code'],how='left',validate='many_to_one')
    selected=selected.merge(pool[['date','code','decision_shares','price_1449']],on=['date','code'],validate='many_to_one')
    selected.to_parquet(ROOT/'signals.parquet',index=False,compression='zstd')
    training_end=training.next_date.max()
    calibration_end=historic.loc[calibration.loc[historic.index],'next_date'].max()
    report={'protocol_sha256':sha(PROTOCOL),'source_label_report_sha256':sha(SOURCE/'label_report.json'),
        'source_label_verification_sha256':sha(SOURCE/'label_verification.json'),
        'source_input_report_sha256':sha(FEATURE_SOURCE/'input_report.json'),
        'training_rows':len(training),'training_counts':training.label15.value_counts().to_dict(),
        'training_last_label':training_end,'calibration_last_label':calibration_end,
        'role_counts':scores.role.value_counts().to_dict(),'feature_count':len(x.columns),'iterations':model['iterations'],
        'test_counts':{name:int(selected.policy.eq(name).sum()) for name in ['calibrated','raw']},
        'test_positive_raw':int(scores.loc[test,'raw_score'].gt(0).sum()),
        'test_positive_calibrated_supported':int((scores.loc[test,'calibrated_score'].gt(0)&scores.loc[test,'in_calibration_support']).sum()),
        'calibration':model['calibration'],
        'outputs_sha256':{name:sha(ROOT/(name+suffix)) for name,suffix in [('model','.json'),('historic_labels','.parquet'),('scores','.parquet'),('signals','.parquet')]},
        'test_outcomes_used_to_fit_or_select':False,'new_selected_test_returns_joined':False,'new_2026_prices_read':False}
    save_json(ROOT/'input_report.json',report)
    return report


if __name__=='__main__':print(json.dumps(freeze(),ensure_ascii=False,indent=2))
