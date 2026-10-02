"""One symmetric gross direction label; original economic labels are untouched."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research.research_io import check_runtime, check_sources, sha, save_json
from trade_research.tail_formula_additive import conn

ROOT=Path('data/research/tail_formula_direction_target')
PROTOCOL=Path('config/tail_formula_direction_target_input.json')
KEYS=['date','code','next_date']
META=KEYS+['half','board','decision_shares','known15','known_no_trade']


def direction(closes, active, entry):
    closes=np.asarray(closes,dtype=float);active=np.asarray(active,dtype=bool);entry=np.asarray(entry,dtype=float)
    assert closes.ndim==2 and closes.shape==active.shape and closes.shape[1]>=3
    assert len(entry)==len(closes) and np.isfinite(closes).all() and np.isfinite(entry).all() and (entry>0).all()
    ends=[]
    for flags in [active&(closes*100>=entry[:,None]*101),active&(closes*100<=entry[:,None]*99)]:
        triple=flags[:,:-2]&flags[:,1:-1]&flags[:,2:]
        ends.append(np.where(triple.any(axis=1),triple.argmax(axis=1)+2,-1))
    up,down=ends;assert not ((up>=0)&(up==down)).any()
    utility=((up>=0)&((down<0)|(up<down))).astype('int64')-((down>=0)&((up<0)|(down<up))).astype('int64')
    return up,down,utility


def derive(frame):
    closes=np.stack(frame.close_values)[:,:29]
    active=(frame.active_mask.to_numpy(dtype='int64')[:,None]&(1<<np.arange(29)))!=0
    up,down,utility=direction(closes,active,frame.entry_vwap.to_numpy())
    out=frame[KEYS].copy();out['first_gross_up_end']=up;out['first_gross_down_end']=down;out['opportunity15']=utility.astype(float)
    c=conn();c.register('source_windows',frame)
    expected=c.sql('''WITH bars AS(SELECT date,code,next_date,pos,
        (active_mask&(1::BIGINT<<pos))<>0 AS active,
        list_extract(close_values,pos+1) AS price,entry_vwap FROM source_windows CROSS JOIN range(29) t(pos)),
        counted AS(SELECT *,count(*) OVER w AS n,
        count(*) FILTER(WHERE active AND price*100>=entry_vwap*101) OVER w AS up,
        count(*) FILTER(WHERE active AND price*100<=entry_vwap*99) OVER w AS down FROM bars
        WINDOW w AS(PARTITION BY date,code ORDER BY pos ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)),
        events AS(SELECT date,code,next_date,coalesce(min(pos) FILTER(WHERE n=3 AND up=3),-1) AS u,
        coalesce(min(pos) FILTER(WHERE n=3 AND down=3),-1) AS d FROM counted GROUP BY date,code,next_date)
        SELECT date,code,next_date,u AS first_gross_up_end,d AS first_gross_down_end,
        CASE WHEN u>=0 AND (d<0 OR u<d) THEN 1. WHEN d>=0 AND (u<0 OR d<u) THEN -1. ELSE 0. END AS opportunity15
        FROM events ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(out,expected,check_dtype=False,rtol=0,atol=0)
    return out


def main():
    check_runtime();p=json.loads(PROTOCOL.read_text());check_sources(p['source_hashes'])
    assert subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}'])==PROTOCOL.read_bytes()
    assert sha(PROTOCOL) in subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    assert p['maximum_new_fits']==0 and not ROOT.exists();ROOT.mkdir()
    parent=pd.read_parquet(p['prior_targets']);frames=[]
    for year,path in [(2023,p['historical_labels']),(2024,p['current_labels'])]:
        f=pd.read_parquet(path,columns=META+['entry_vwap'],filters=[('date','>=',f'{year}-01-01'),('date','<','2024-01-01' if year==2023 else '2026-01-01')]);frames.append(f)
    cash=pd.concat(frames,ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(parent[META],cash[META],check_exact=True)
    assert len(parent)==1815129 and parent.date.lt('2026-01-01').all()
    eligible=cash.loc[cash.known15].copy();assert np.isfinite(eligible.entry_vwap).all() and eligible.entry_vwap.gt(0).all()
    derived=[];parts=[]
    for path in p['historical_raw_parts']:
        entry=pd.read_parquet(path,columns=['date','code','source_date','clock','volume','amount'],filters=[('date','>=','2023-01-01'),('date','<','2024-01-01'),('clock','>=','14:52'),('clock','<=','14:55')])
        buy=eligible.loc[eligible.date.lt('2024-01-01')].merge(entry,on=['date','code'],validate='one_to_many').sort_values(['date','code','clock'])
        if not len(buy):continue
        assert buy.source_date.eq(buy.date).all() and buy.groupby(KEYS).size().eq(4).all() and buy.groupby(KEYS).clock.nunique().eq(4).all()
        eb=buy.groupby(KEYS).agg(volume=('volume','sum'),amount=('amount','sum'),entry_vwap=('entry_vwap','first')).reset_index()
        assert eb.volume.gt(0).all();np.testing.assert_allclose(eb.amount/eb.volume,eb.entry_vwap,rtol=0,atol=2e-12)
        raw=pd.read_parquet(path,columns=['date','code','source_date','clock','timestamp','close','volume'],filters=[('date','>=','2023-01-01'),('date','<','2024-01-01'),('kind','=','morning'),('clock','>=','09:31'),('clock','<=','09:59')])
        q=eligible.loc[eligible.date.lt('2024-01-01')].merge(raw,on=['date','code'],validate='one_to_many').sort_values(['date','code','clock']).reset_index(drop=True)
        assert not q.duplicated(['date','code','timestamp']).any() and q.source_date.eq(q.next_date).all()
        assert q.clock.eq(q.timestamp.dt.strftime('%H:%M')).all() and q.groupby(KEYS).size().eq(29).all()
        assert np.array_equal(q.clock.to_numpy().reshape(-1,29),np.tile([f'09:{i:02d}' for i in range(31,60)],(len(q)//29,1)))
        keys=q.drop_duplicates(KEYS).reset_index(drop=True);pd.testing.assert_frame_equal(keys[KEYS],eb[KEYS],check_exact=True)
        values=q.close.to_numpy().reshape(-1,29);active=q.volume.to_numpy().reshape(-1,29)>0
        packed=keys[KEYS+['entry_vwap']].copy();packed['close_values']=list(values);packed['active_mask']=(active*(1<<np.arange(29))).sum(axis=1)
        out=derive(packed);derived.append(out);parts.append(dict(path=path,rows=len(out),raw_four_minute_entry_vwap_verified=True,all_direction_events_SQL_verified=True))
        print(json.dumps(dict(history_parts=len(parts),known_history_rows=sum(a['rows'] for a in parts))),flush=True)
    index=json.loads(Path(p['observation_index']).read_text());check_sources(index['source_hashes'])
    for path in index['observation_parts']:
        obs=pd.read_parquet(path,columns=KEYS+['source_valid_0959','valid_mask','active_mask','close_values'],filters=[('date','>=','2024-01-01'),('next_date','<','2026-01-01')])
        q=eligible.loc[eligible.date.ge('2024-01-01')].merge(obs,on=KEYS,validate='one_to_one').sort_values(KEYS).reset_index(drop=True)
        if not len(q):continue
        assert q.source_valid_0959.all() and np.all((q.valid_mask.to_numpy(dtype='int64')&((1<<29)-1))==((1<<29)-1))
        out=derive(q);derived.append(out);parts.append(dict(path=path,rows=len(out),all_direction_events_SQL_verified=True))
        print(json.dumps(dict(total_parts=len(parts),known_rows=sum(a['rows'] for a in parts))),flush=True)
    events=pd.concat(derived,ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(eligible[KEYS].reset_index(drop=True),events[KEYS],check_exact=True)
    old=parent.loc[parent.known15,['date','code','first_space_end15']].reset_index(drop=True)
    assert ((old.first_space_end15<0)|((events.first_gross_up_end>=0)&(events.first_gross_up_end<=old.first_space_end15))).all()
    out=parent[META].merge(events,on=KEYS,how='left',validate='one_to_one')
    assert out.loc[~out.known15,['opportunity15','first_gross_up_end','first_gross_down_end']].isna().all().all()
    folder=ROOT/'training_labels';folder.mkdir();out.to_parquet(folder/'full_labels.parquet',index=False,compression='zstd')
    save_json(folder/'full_label_report.json',dict(labels_sha256=sha(folder/'full_labels.parquet'),rows=len(out),training_only_gross_direction_alias_not_economic_opportunity=True,utility_values=[-1,0,1],original_states_and_metadata_unchanged=True))
    save_json(folder/'full_label_verification.json',dict(passed=True,label_report_sha256=sha(folder/'full_label_report.json'),all_event_positions_and_utilities_SQL_verified=True))
    sources=dict(p['source_hashes']);sources.update(index['source_hashes'])
    for path in folder.iterdir():sources[str(path)]=sha(path)
    save_json(ROOT/'source_gate.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),source_hashes=sources,rows=len(out),known_rows=len(events),parts=parts,full_keys_and_unknown_states_preserved=True,raw_entry_vwap_history_and_cash_definition_current_previously_verified=True,all_direction_events_SQL_verified=True,gross_up_precedes_existing_net_space_when_present=True,new_fits=0,new_selection_lists=0,new_2026_prices_read=False,no_exit_rules=True,not_economic_profit=True))
    print(json.dumps(dict(source_gate_sha256=sha(ROOT/'source_gate.json'),rows=len(out),known_rows=len(events))),flush=True)


if __name__=='__main__':main()
