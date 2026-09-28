"""Training-only order of positive opportunity and preceding adverse excursion."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_before1000 as source
from .corporate_cash import save_json,sha

ROOT=Path('data/research/tail_formula_path_order_labels')
PROTOCOL=Path('config/tail_formula_path_order_protocol.json')


def event_positions(positive,adverse):
    """Indices denote completed bars; a same-bar adverse event has priority."""
    assert positive.shape==adverse.shape and positive.ndim==2 and positive.shape[1]>=3
    size=positive.shape[1]
    triples=positive[:,:-2]&positive[:,1:-1]&positive[:,2:]
    first_positive=np.where(triples,np.arange(2,size),size).min(axis=1)
    first_adverse=np.where(adverse,np.arange(size),size).min(axis=1)
    return first_positive,first_adverse,(first_positive<first_adverse).astype(float)


def source_keys():
    p=json.loads(PROTOCOL.read_text())
    for file,digest in p['source_hashes'].items():
        assert sha(Path(file))==digest
    assert p['target_window_bars']==29 and p['target_window']==['09:31','09:59']
    assert p['positive_threshold']==0 and p['adverse_threshold']==-.03 and p['consecutive_active_minutes']==3
    report=json.loads((source.ROOT/'full_label_report.json').read_text())
    proof=json.loads((source.ROOT/'full_label_verification.json').read_text())
    assert proof['passed'] and proof['label_report_sha256']==sha(source.ROOT/'full_label_report.json')
    assert report['labels_sha256']==sha(source.ROOT/'full_labels.parquet')
    c=base.conn()
    keys=c.execute('''SELECT date,code,next_date,half,known15,known_no_trade,decision_shares,buy_cash15,
        opportunity15 AS original_opportunity15,adverse_return15 FROM read_parquet(?)
        WHERE date>='2024-01-01' AND next_date<'2025-07-01' AND known15 ORDER BY date,code''',
        [str(source.ROOT/'full_labels.parquet')]).df();c.close()
    assert keys.known15.all() and not keys.known_no_trade.any() and keys.next_date.lt('2025-07-01').all()
    return keys


def target():
    keys=source_keys();assert not (ROOT/'full_label_report.json').exists()
    report=json.loads((source.ROOT/'observation_report.json').read_text());parts=[]
    for file,digest in report['parts_sha256'].items():
        assert sha(Path(file))==digest
        f=pd.read_parquet(file,columns=['date','code','next_date','source_valid','close_values','low_values','active_mask'],
            filters=[('date','>=','2024-01-01'),('next_date','<','2025-07-01')])
        f=keys.merge(f,on=['date','code','next_date'],validate='one_to_one')
        if not len(f):
            continue
        assert f.source_valid.all()
        close=np.stack(f.close_values)[:,:29];low=np.stack(f.low_values)[:,:29]
        active=(f.active_mask.to_numpy('int64')[:,None]&(1<<np.arange(29)))!=0
        shares=f.decision_shares.to_numpy()[:,None];cash=f.buy_cash15.to_numpy()[:,None]
        positive=active&(source.net_mark(close,shares,cash,15)>0)
        adverse=active&(source.net_mark(low,shares,cash,15)<=-.03)
        first,risk,y=event_positions(positive,adverse)
        np.testing.assert_array_equal(first<29,f.original_opportunity15.eq(1))
        np.testing.assert_array_equal(risk<29,f.adverse_return15.le(-.03))
        out=f[keys.columns].copy();out['first_positive_end']=first;out['first_adverse']=risk;out['opportunity15']=y
        assert out.opportunity15.le(out.original_opportunity15).all()
        parts.append(out)
    out=pd.concat(parts,ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(out[keys.columns],keys,check_exact=True)
    ROOT.mkdir(parents=True,exist_ok=True);out.to_parquet(ROOT/'full_labels.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),labels_sha256=sha(ROOT/'full_labels.parquet'),
        source_label_report_sha256=sha(source.ROOT/'full_label_report.json'),source_observation_report_sha256=sha(source.ROOT/'observation_report.json'),
        rows=len(out),first_signal=out.date.min(),last_observation=out.next_date.max(),scope='training_target_only',not_for_evaluation=True,
        by_half=out.groupby('half').agg(rows=('code','size'),original_success=('original_opportunity15','sum'),
            ordered_success=('opportunity15','sum')).reset_index().to_dict('records'),
        original_known_keys_and_full_window_events_reproduced=True,same_bar_adverse_has_priority=True,
        no_first_hit_sale_assumed=True,new_2025H2_path_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'full_label_report.json',r);return r


def verify():
    keys=source_keys();r=json.loads((ROOT/'full_label_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['labels_sha256']==sha(ROOT/'full_labels.parquet')
    got=pd.read_parquet(ROOT/'full_labels.parquet');pd.testing.assert_frame_equal(got[keys.columns],keys,check_exact=True)
    parts=json.loads((source.ROOT/'observation_report.json').read_text())['parts_sha256']
    for file,digest in parts.items():
        assert sha(Path(file))==digest
    c=base.conn();c.register('keys',keys);c.read_parquet(list(parts)).create_view('raw_parts')
    ex=c.sql('''WITH bars AS(SELECT k.*,pos,list_extract(p.close_values,pos+1) AS close,
        list_extract(p.low_values,pos+1) AS low,(p.active_mask&(1::BIGINT<<pos))<>0 AS active
        FROM keys k JOIN raw_parts p USING(date,code,next_date) CROSS JOIN range(29) t(pos)
        WHERE p.date>='2024-01-01' AND p.next_date<'2025-07-01'),
        cash AS(SELECT *,decision_shares*(close-greatest(.005,close*.0015)) AS cl_value,
            decision_shares*(low-greatest(.005,low*.0015)) AS lo_value FROM bars),
        marks AS(SELECT *,active AND(cl_value-greatest(5,cl_value*.0003)-cl_value*.00051)>buy_cash15 AS positive,
            active AND(lo_value-greatest(5,lo_value*.0003)-lo_value*.00051)/buy_cash15-1<=-.03 AS adverse FROM cash),
        windows AS(SELECT *,sum(positive::INT) OVER w AS three,count(*) OVER w AS n FROM marks
            WINDOW w AS(PARTITION BY date,code ORDER BY pos ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)),
        firsts AS(SELECT date,code,count(*) AS bars,coalesce(min(pos) FILTER(WHERE three=3 AND n=3),29) AS first_positive_end,
            coalesce(min(pos) FILTER(WHERE adverse),29) AS first_adverse FROM windows GROUP BY date,code)
        SELECT *, (first_positive_end<first_adverse)::DOUBLE AS opportunity15 FROM firsts ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(got[['date','code']],ex[['date','code']],check_exact=True)
    assert ex.bars.eq(29).all()
    for name in ['first_positive_end','first_adverse','opportunity15']:
        np.testing.assert_array_equal(got[name],ex[name])
    np.testing.assert_array_equal(ex.first_positive_end.lt(29),keys.original_opportunity15.eq(1))
    np.testing.assert_array_equal(ex.first_adverse.lt(29),keys.adverse_return15.le(-.03))
    for row in r['by_half']:
        q=got.loc[got.half.eq(row['half'])]
        assert row['rows']==len(q) and row['original_success']==q.original_opportunity15.sum() and row['ordered_success']==q.opportunity15.sum()
    proof=dict(passed=True,label_report_sha256=sha(ROOT/'full_label_report.json'),rows=len(got),minute_pairs=len(got)*29,
        all_cost_marks_rolling_triples_and_first_event_positions_independently_rebuilt=True,
        all_original_known_keys_and_buy_cash_unchanged=True,original_full_window_opportunity_and_adverse_events_reproduced=True,
        scope='training_target_only',not_for_evaluation=True,no_first_hit_sale_assumed=True,
        new_2025H2_path_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'full_label_verification.json',proof);return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['target','verify'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
