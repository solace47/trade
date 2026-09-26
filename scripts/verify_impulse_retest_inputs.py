"""Independently replay fixed hash-sampled minute paths and all input-only pairs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.spatial.distance import cdist

from trade_research.corporate_cash import MINUTES, save_json, sha

root=Path('data/research/impulse_retest')
base_path=Path('data/research/next_day_winner/visible_base.parquet')
manifest=json.loads((root/'manifest.json').read_text())
report=json.loads((root/'input_report.json').read_text())
raw=json.loads((root/'raw_report.json').read_text())
assert report['manifest_sha256']==sha(root/'manifest.json') and report['raw_report_sha256']==sha(root/'raw_report.json')
assert sha(base_path)==manifest['base_sha256'] and sha(Path('config/impulse_retest_protocol.json'))==manifest['protocol_sha256']
for name in ['features','pairs']:assert report[name+'_sha256']==sha(root/(name+'.parquet'))
for path,digest in raw['parts_sha256'].items():assert sha(Path(path))==digest
if raw['extractor_repair_sha256']:
    assert sha(root/'extractor_repair.json')==raw['extractor_repair_sha256']
    repair=json.loads((root/'extractor_repair.json').read_text())
    assert repair['fixed_code_sha256']==raw['extractor_code_sha256'] and repair['classified_outputs_before_repair']==0
features=pd.read_parquet(root/'features.parquet').set_index(['date','code']).sort_index()
base=pd.read_parquet(base_path).set_index(['date','code']).sort_index()
pd.testing.assert_frame_equal(features[base.columns],base,atol=0,rtol=0)
assert len(features)==report['rows']==manifest['stock_days'] and not features.index.duplicated().any()
assert features.category.notna().all()
assert np.array_equal(features.last_cents.to_numpy(),np.rint(base.price_1449.to_numpy()*100))
expected_quality=features.bars.eq(229)&features.labels.eq(229)&features.valid_bars.fillna(False)
assert features.category.eq('unknown_source').eq(~expected_quality).all()
for label,left,right in [('pulse',605,860),('retest',606,889),('recovery',607,889)]:
    times=features[label+'_minute'].dropna();assert times.between(left,right).all()
assert (features.retest_minute.dropna()>features.loc[features.retest_minute.notna(),'pulse_minute']).all()
assert (features.recovery_minute.dropna()>features.loc[features.recovery_minute.notna(),'retest_minute']).all()
assert features.loc[features.pulse_minute.notna(),'pulse_minute'].map(lambda v:605<=v<=660 or 815<=v<=860).all()

# Four deterministic samples from every observed board/half/category, independent of labels.
sample=features.reset_index()
sample['identity_hash']=[hashlib.sha256(('impulse-retest-input-v1|'+d+'|'+c).encode()).hexdigest() for d,c in zip(sample.date,sample.code)]
sample=sample.sort_values('identity_hash').groupby(['board','half','category'],sort=True).head(4)
sample=sample.sort_values(['code','date']);checked=0
sample_records=[]
expected_labels=np.r_[np.arange(571,691),np.arange(781,890)]
for code,part in sample.groupby('code',sort=True):
    path=MINUTES/code[:2].upper()/(code[3:]+'.parquet')
    assert sha(path)==manifest['source_sha256'][str(path)]
    for row in part.itertuples():
        start=pd.Timestamp(row.date);end=start+pd.Timedelta(days=1)
        p=pq.read_table(path,columns=['timestamp','open','high','low','close','volume','turnover'],
            filters=[('timestamp','>=',start.to_pydatetime()),('timestamp','<',end.to_pydatetime())]).to_pandas()
        p=p.sort_values('timestamp');minute=p.timestamp.dt.hour*60+p.timestamp.dt.minute
        p=p.loc[minute.isin(expected_labels)].copy();minute=(p.timestamp.dt.hour*60+p.timestamp.dt.minute).to_numpy()
        values=p[['open','high','low','close','volume','turnover']].to_numpy(dtype=float)
        o,h,l,close,v,amount=values.T
        ratio=np.divide(amount,v,out=np.zeros(len(v)),where=v>0)
        valid=(np.isfinite(values).all(axis=1)&(values[:,:4].min(axis=1)>0)&(h+.0001>=np.maximum.reduce([o,l,close]))
            &(l-.0001<=np.minimum(o,close))&(np.abs(close-np.round(close,2))<=.0001)&(v>=0)&(amount>=0)
            &((v==0)==(amount==0))&((v==0)|((ratio>=l-.0101)&(ratio<=h+.0101))))
        valid &= (p.timestamp==p.timestamp.dt.floor('min')).to_numpy()
        quality=len(p)==229 and len(np.unique(minute))==229 and bool(valid.all())
        assert row.bars==len(p) and row.labels==len(np.unique(minute)) and row.valid_bars==bool(valid.all())
        np.testing.assert_allclose([row.prefix_volume,row.prefix_amount],[v.sum(),amount.sum()],atol=.001,rtol=1e-12)
        c=np.rint(close*100).astype('int64')
        expected={};category='no_impulse';confirmed=False
        pulses=[]
        for i,e in enumerate(minute):
            if not(605<=e<=660 or 815<=e<=860) or i<34:continue
            v5=v[i-4:i+1].sum();v30=v[i-34:i-4].sum()
            if c[i-5]>0 and c[i]*100>=c[i-5]*101 and v30>0 and v5*6>=v30*2:pulses.append((i,v5,v30))
        if pulses:
            i,v5,v30=pulses[0]
            expected={'pulse_minute':minute[i],'pulse_trading_minute':minute[i]-(90 if minute[i]>=781 else 0),
                'pulse_ordinal':i+1,'start_cents':c[i-5],'end_cents':c[i],'pulse_volume':v5,'prior_volume':v30,
                'pulse_cumulative':v[:i+1].sum(),'pulse_gain':c[i]/c[i-5]-1}
            drops=np.flatnonzero(c[i+1:]*2<=c[i-5]+c[i])+i+1
            category='impulse_without_half_retest'
            if len(drops):
                j=int(drops[0]);down_volume=v[i+1:j+1].mean()
                expected.update(retest_minute=minute[j],retest_ordinal=j+1,retest_cents=c[j],
                    retest_cumulative=v[:j+1].sum(),retest_mean_volume=down_volume)
                recover=np.flatnonzero(c[j+1:]>=c[i])+j+1;category='retested_not_recovered'
                if len(recover):
                    k=int(recover[0]);up_volume=v[j+1:k+1].mean()
                    expected.update(recovery_minute=minute[k],recovery_ordinal=k+1,recovery_cents=c[k],
                        recovery_cumulative=v[:k+1].sum(),recovery_mean_volume=up_volume)
                    confirmed=down_volume<v5/5 and up_volume>down_volume
                    category='recovered_but_lost' if c[-1]<c[i] else ('held_with_volume_confirmation' if confirmed else 'held_without_volume_confirmation')
        for field in ['pulse_minute','pulse_trading_minute','pulse_ordinal','start_cents','end_cents','pulse_volume','prior_volume',
            'pulse_cumulative','pulse_gain','retest_minute','retest_ordinal','retest_cents','retest_cumulative','retest_mean_volume',
            'recovery_minute','recovery_ordinal','recovery_cents','recovery_cumulative','recovery_mean_volume']:
            if field in expected:np.testing.assert_allclose(getattr(row,field),expected[field],atol=1e-10,rtol=1e-12)
            else:assert pd.isna(getattr(row,field)),(row.date,code,field)
        assert row.volume_confirmed==confirmed
        if not quality:category='unknown_source'
        assert row.category==category,(row.date,code,row.category,category)
        sample_records.append({'date':row.date,'code':code,'category':category,'raw_minutes':len(p),'identity_hash':row.identity_hash})
        checked+=len(p)

f=features.reset_index();pairs=pd.read_parquet(root/'pairs.parquet').set_index(['date','code']).sort_index()
reference_fields=['return_1449','return20_prior_adjusted','day_range','price_1449','amount_1449','pulse_gain','pulse_trading_minute']
assert np.allclose(f.day_range,(f.high_1449-f.low_1449)/f.preclose,atol=0,rtol=0)
available=f.necessary_tradeable&f.return20_prior_adjusted.notna()
reconstructed=[];max_distance_error=0.
def coords(rows):
    p=rows[reference_fields].to_numpy(dtype=float).copy()
    p[:,0]/=.01;p[:,1]/=.1;p[:,2]/=.02;p[:,3:5]=np.log2(p[:,3:5]);p[:,5]/=.01;p[:,6]/=60
    return p
for (date,board),p in f.loc[available].groupby(['date','board'],sort=True):
    a=p.loc[p.category.eq('held_with_volume_confirmation')].sort_values('code')
    b=p.loc[p.category.eq('held_without_volume_confirmation')].sort_values('code')
    if a.empty or b.empty:continue
    distances=cdist(coords(a),coords(b),metric='cityblock')
    for i,row in enumerate(a.itertuples()):
        j=int(np.argmin(distances[i]));control=b.iloc[j];stored=pairs.loc[(date,row.code)]
        assert stored.board==board and stored.half==row.half and stored.control_code==control.code
        error=abs(stored.distance-distances[i,j]);assert error<2e-12;max_distance_error=max(max_distance_error,error)
        for col in reference_fields:np.testing.assert_allclose(stored[col+'_difference'],getattr(row,col)-control[col],atol=2e-12,rtol=0)
        reconstructed.append((date,row.code))
assert set(reconstructed)==set(pairs.index) and len(reconstructed)==len(pairs)==report['pairs']
result={'passed':True,'cohort_rows':len(f),'complete_original_columns':len(base.columns),
    'sampled_stock_days':len(sample),'sampled_raw_minutes':checked,'samples':sample_records,
    'full_input_pairs':len(pairs),'maximum_distance_error':max_distance_error,
    'input_report_sha256':sha(root/'input_report.json'),'outcome_labels_accessed':False,'new_2026_prices_read':False}
save_json(root/'input_verification.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='samples'},ensure_ascii=False,indent=2))
