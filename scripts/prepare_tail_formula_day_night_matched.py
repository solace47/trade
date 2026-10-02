"""Rebuild twenty prior stock days for a matched, strict-window supplement."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research import tail_formula_baseline as baseline
from trade_research import tail_formula_additive as numeric
from trade_research.research_io import check_runtime, check_sources, sha, save_json

ROOT = Path('data/research/tail_formula_day_night_matched')
PROTOCOL = Path('config/tail_formula_day_night_matched_protocol.json')
NEW = ['G01', 'G02', 'G03']


def checked():
    check_runtime()
    assert subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    assert sha(PROTOCOL) in subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    p = json.loads(PROTOCOL.read_text()); check_sources(p['source_hashes'])
    assert p['history_days']==20 and p['history_initialization_start']=='2022-01-01'
    assert not p['new_2026_prices_allowed'] and p['source_gate_only_first']
    return p


def prepare():
    p=checked();assert not (ROOT/'input_verification.json').exists()
    m=json.loads(Path(p['daily_manifest']).read_text())
    f=baseline.original();assert len(f)==1815129 and int(f.formula_input_valid.sum())==1602413
    assert set(f.code.unique())==set(m['codes'])
    check_sources({r['path']:r['sha256'] for r in m['daily_sources'].values()})
    blocks, raw = [], []
    for code, group in f.groupby('code',sort=True):
        d=pd.read_parquet(m['daily_sources'][code]['path'],
            columns=['date','code','open','close','preclose','adjustflag','tradestatus'],
            filters=[('date','>=',p['history_initialization_start']),('date','<','2026-01-01')])
        assert d.code.eq(code).all() and not d.duplicated(['date','code']).any()
        assert d.date.ge(p['history_initialization_start']).all() and d.date.lt('2026-01-01').all()
        for n in ['open','close','preclose','adjustflag','tradestatus']:
            d[n]=pd.to_numeric(d[n],errors='coerce')
        d=d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True)
        raw.append(d)
        dates=d.date.to_numpy();o=d.open.to_numpy();cl=d.close.to_numpy();adj=d.adjustflag.to_numpy()
        pc=np.r_[np.nan,cl[:-1]];pa=np.r_[np.nan,adj[:-1]]
        good=np.isfinite(np.column_stack([o,cl,pc])).all(axis=1)
        good &= (o>0)&(cl>0)&(pc>0)&(adj==3)&(pa==3)
        with np.errstate(divide='ignore',invalid='ignore'):
            gap=o/pc-1;day=cl/o-1
        reversal=(o>pc)&(cl<o)
        pos=np.searchsorted(dates,group.date.to_numpy(),side='left')
        start=np.maximum(pos-20,0)
        def total(a):
            cs=np.r_[0,np.cumsum(a)];return cs[pos]-cs[start]
        counts=total(good.astype(int));rows=pos-start
        out=group[['date','code']].copy()
        out['history_rows']=rows;out['history_good']=counts
        for name,index in [('history_start',start),('history_end',pos-1),('history_reference_start',start-1)]:
            values=pd.Series(pd.NA,index=out.index,dtype='object');valid=(index>=0)&(index<len(dates))&(rows>0)
            values.loc[valid]=dates[index[valid]];out[name]=values
        quality=(rows==20)&(counts==20)&(pos>=21)
        quality &= np.isfinite(group.V01.to_numpy()) & (group.V01.to_numpy()>0)
        out['history_source_valid']=quality
        out['G01']=np.where(quality,100*total(np.where(good,gap,0))/20/group.V01.to_numpy(),np.nan)
        out['G02']=np.where(quality,100*total(np.where(good,day,0))/20/group.V01.to_numpy(),np.nan)
        out['G03']=np.where(quality,5*total((good&reversal).astype(int)),np.nan)
        out['history_reference_breaks']=total((abs(d.preclose.to_numpy()-pc)>.005).astype(int))
        blocks.append(out)
    atoms=pd.concat(blocks,ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(f[['date','code']],atoms[['date','code']],check_exact=True)
    c=numeric.conn();c.register('daily',pd.concat(raw,ignore_index=True));c.register('keys',f[['date','code','V01']])
    sql=c.sql('''WITH lagged AS(
        SELECT *,lag(close) OVER w AS pc,lag(adjustflag) OVER w AS padj,
        lag(date) OVER w AS reference_date FROM daily WINDOW w AS(PARTITION BY code ORDER BY date)),
        primitives AS(SELECT *,coalesce(open>0 AND close>0 AND pc>0 AND adjustflag=3 AND padj=3
        AND isfinite(open) AND isfinite(close) AND isfinite(pc),false) AS good,
        open/pc-1 AS gap,close/open-1 AS daytime,
        CASE WHEN open>pc AND close<open THEN 1 ELSE 0 END AS reversal,
        CASE WHEN abs(preclose-pc)>.005 THEN 1 ELSE 0 END AS reference_break FROM lagged),
        history AS(SELECT date AS history_end,code,count(*) OVER w AS history_rows,
        sum(good::INT) OVER w AS history_good,min(date) OVER w AS history_start,
        first_value(reference_date) OVER w AS history_reference_start,
        avg(CASE WHEN good THEN gap END) OVER w AS gap_mean,
        avg(CASE WHEN good THEN daytime END) OVER w AS day_mean,
        sum(CASE WHEN good THEN reversal ELSE 0 END) OVER w AS reversal_count,
        sum(reference_break) OVER w AS history_reference_breaks FROM primitives
        WINDOW w AS(PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW))
        SELECT k.date,k.code,h.history_rows,h.history_good,h.history_start,h.history_end,h.history_reference_start,
        coalesce(h.history_rows=20 AND h.history_good=20 AND h.history_reference_start<h.history_start
        AND h.history_end<k.date AND isfinite(k.V01) AND k.V01>0,false) AS history_source_valid,
        CASE WHEN history_source_valid THEN 100*h.gap_mean/k.V01 END AS G01,
        CASE WHEN history_source_valid THEN 100*h.day_mean/k.V01 END AS G02,
        CASE WHEN history_source_valid THEN 5*h.reversal_count END AS G03,h.history_reference_breaks
        FROM keys k ASOF LEFT JOIN history h ON k.code=h.code AND k.date>h.history_end ORDER BY k.date,k.code''').df();c.close()
    for name in ['history_rows','history_good','history_reference_breaks']:
        atoms[name]=atoms[name].where(atoms.history_rows.gt(0))
    pd.testing.assert_frame_equal(atoms.drop(columns=NEW),sql.drop(columns=NEW),check_exact=True,check_dtype=False)
    np.testing.assert_allclose(atoms[NEW],sql[NEW],rtol=0,atol=2e-12,equal_nan=True)
    a,b=atoms.loc[atoms.history_source_valid,NEW],sql.loc[sql.history_source_valid,NEW]
    np.testing.assert_array_equal(np.floor(np.clip(100*a+10000+.000001,0,999999)),
                                 np.floor(np.clip(100*b+10000+.000001,0,999999)))
    out=f.copy();out['prior_formula_input_valid']=out.formula_input_valid
    out=out.merge(atoms,on=['date','code'],validate='one_to_one')
    out['formula_input_valid'] &= out.history_source_valid & np.isfinite(out[NEW]).all(axis=1)
    out.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    atoms.to_parquet(ROOT/'atoms.parquet',index=False,compression='zstd')
    v=dict(passed=True,protocol_sha256=sha(PROTOCOL),features_sha256=sha(ROOT/'features.parquet'),
        atoms_sha256=sha(ROOT/'atoms.parquet'),keys=len(out),original_valid=1602413,valid=int(out.formula_input_valid.sum()),
        newly_invalid=int((out.prior_formula_input_valid&~out.formula_input_valid).sum()),
        metadata_and_fifty_values_preserved=True,all_lags_dates_units_adjustflags_means_counts_encodings_SQL_rebuilt=True,
        bad_active_rows_not_skipped=True,fits_performed=0,new_economic_groups_read=False,
        new_2026_prices_read=False,no_exit_rules=True,native_client_parity_verified=False)
    save_json(ROOT/'input_verification.json',v);return v


if __name__=='__main__':
    print(json.dumps(prepare(),ensure_ascii=False),flush=True)
