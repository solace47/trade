"""Recover old opportunity labels from minute masks and audit raw cash signs."""
import hashlib
import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_duration_labels as study
from trade_research.corporate_cash import MINUTES,save_json,sha


def main():
    root=study.ROOT;r=json.loads((root/'duration_report.json').read_text());keys,hashes=study.source_keys()
    for key,path in [('protocol_sha256',study.PROTOCOL),('full_label_report_sha256',study.base.SOURCE/'full_label_report.json'),
        ('full_label_verification_sha256',study.base.SOURCE/'full_label_verification.json'),
        ('manifest_sha256',study.MANIFEST),('extractor_sha256',Path(study.__file__)),('labels_sha256',root/'duration_labels.parquet')]:
        assert r[key]==sha(path)
    for path,digest in r['parts_sha256'].items():
        assert sha(Path(path))==digest
    f=pd.read_parquet(root/'duration_labels.parquet')
    pd.testing.assert_frame_equal(f[['date','code','next_date']],keys[['date','code','next_date']],check_exact=True)
    assert len(f)==r['rows'] and f.bars.sum()==r['raw_minutes']
    for name in ['bars','labels','minute_labels']:
        assert f[name].eq(30).all()
    assert f.first_pos.eq(0).all() and f.last_pos.eq(29).all()
    assert f.profitable_mask15.between(0,(1<<30)-1).all()
    masks=f.profitable_mask15.to_numpy(dtype='uint64')
    # Bit-by-bit accumulation independently checks counts and three adjacent observed prices.
    counts=np.zeros(len(f),dtype='int16');three=np.zeros(len(f),dtype=bool)
    for i in range(30):
        counts+=((masks>>i)&1).astype('int16')
        if i<28:
            three|=((masks>>i)&7)==7
    np.testing.assert_array_equal(counts,f.profitable_minutes15)
    np.testing.assert_array_equal(three,keys.opportunity15.astype(bool))
    np.testing.assert_array_equal(counts>0,keys.any_opportunity15.astype(bool))
    np.testing.assert_allclose(f.profitable_fraction15,counts/30,rtol=0,atol=0)
    assert f.active_minutes.ge(f.profitable_minutes15).all() and f.active_minutes.le(30).all()
    np.testing.assert_allclose(f.mark_1000_return15_rebuilt,keys.mark_1000_return15,rtol=0,atol=2e-12,equal_nan=True)
    assert f.date.ge(study.START).all() and f.next_date.lt(study.END).all()
    assert f.date.min()==r['signal_start'] and f.date.max()==r['signal_end']
    assert f.next_date.min()==r['observation_start'] and f.next_date.max()==r['observation_end']
    candidates=keys.copy()
    candidates['key_hash']=[hashlib.sha256(('duration-v1/'+d+'/'+code).encode()).hexdigest()
        for d,code in zip(candidates.date,candidates.code)]
    sample=candidates.sort_values('key_hash').groupby('half',sort=True).head(8)
    assert set(sample.half)=={'2024H1','2024H2','2025H1'} and len(sample)==24
    c=study.base.conn();cases=[];checked=set();lookup=f.set_index(['date','code'])
    for row in sample.sort_values(['date','code']).itertuples():
        path=MINUTES/row.code[:2].upper()/(row.code[3:]+'.parquet')
        if path not in checked:
            assert sha(path)==hashes[str(path)];checked.add(path)
        bars=c.execute('''SELECT timestamp,close,volume FROM read_parquet(?)
            WHERE timestamp>=?::TIMESTAMP AND timestamp<=?::TIMESTAMP ORDER BY timestamp''',
            [str(path),row.next_date+' 09:31:00',row.next_date+' 10:00:00']).df()
        assert bars.timestamp.tolist()==list(pd.date_range(row.next_date+' 09:31',periods=30,freq='min'))
        positive=[];returns=[]
        for bar in bars.itertuples():
            price=Decimal(str(float(bar.close)));shares=Decimal(int(row.decision_shares))
            marked=shares*(price-max(price*Decimal('.0015'),Decimal('.005')))
            net=marked-max(marked*Decimal('.0003'),Decimal(5))-marked*Decimal('.00051')
            gain=net/Decimal(str(float(row.buy_cash15)))-1
            positive.append(bool(bar.volume>0 and gain>0));returns.append(float(gain))
        mask=sum((1<<i) for i,value in enumerate(positive) if value)
        cached=lookup.loc[(row.date,row.code)]
        assert mask==cached.profitable_mask15 and sum(positive)==cached.profitable_minutes15
        if bars.volume.iloc[-1]>0:
            np.testing.assert_allclose(returns[-1],cached.mark_1000_return15_rebuilt,rtol=0,atol=2e-12)
        cases.append(dict(date=row.date,code=row.code,next_date=row.next_date,raw_minutes=30,
            profitable_minutes15=sum(positive),decimal_mask_equal=True))
    c.close()
    proof=dict(passed=True,duration_report_sha256=sha(root/'duration_report.json'),rows=len(f),
        all_original_known_keys_and_time_limits_rebuilt=True,all_masks_counts_and_fractions_rebuilt=True,
        all_original_any_and_three_consecutive_opportunities_recovered=True,all_original_1000_marks_recovered=True,
        original_decimal_cases=len(cases),original_decimal_minutes=30*len(cases),original_cases=cases,
        no_unknowns_filled=True,new_2025H2_duration_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'duration_verification.json',proof)
    print(json.dumps({k:v for k,v in proof.items() if k!='original_cases'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
