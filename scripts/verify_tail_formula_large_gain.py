"""Independently reconstruct all prior 20 events from stock-ordered raw arrays."""
import json
import re

import numpy as np
import pandas as pd

from trade_research import tail_formula_large_gain as study
from trade_research.corporate_cash import save_json, sha


def main():
    files=study.checked_sources();root=study.ROOT;r=json.loads((root/'feature_report.json').read_text())
    for key,path in [('protocol_sha256',study.PROTOCOL),('features_sha256',root/'features.parquet'),('history_sha256',root/'history.parquet')]:
        assert r[key]==sha(path)
    old=pd.read_parquet(study.previous.ROOT/'features.parquet');actual=pd.read_parquet(root/'features.parquet')
    pd.testing.assert_frame_equal(actual[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    assert actual.prior_formula_input_valid.equals(old.formula_input_valid)
    c=study.base.conn();c.read_parquet(files).create_view('daily')
    d=c.sql('''SELECT date,code,close::DOUBLE AS close,preclose::DOUBLE AS preclose,adjustflag::DOUBLE AS adjustflag
        FROM daily WHERE tradestatus=1 AND date BETWEEN '2023-06-01' AND '2025-12-30' ORDER BY code,date''').df();c.close()
    index=pd.MultiIndex.from_frame(d[['date','code']]);assert index.is_unique
    pos=index.get_indexer(pd.MultiIndex.from_frame(old[['date','code']]));assert (pos>=0).all()
    prices=d.close.to_numpy();dates=d.date.to_numpy();codes=d.code.to_numpy();adj=d.adjustflag.to_numpy()
    cents=np.rint(prices*100);n=len(old);counts=np.zeros(n,dtype=int);age=np.full(n,21,dtype=int)
    good=np.zeros(n,dtype=int);rows=np.zeros(n,dtype=int);breaks=np.zeros(n,dtype=int);variables={}
    for k in range(1,21):
        now=pos-k;before=now-1;safe=np.maximum(now,0);prior=np.maximum(before,0)
        present=(now>=0)&(codes[safe]==old.code.to_numpy())
        pair=present&(before>=0)&(codes[prior]==old.code.to_numpy())
        event=pair&(100*cents[safe]>=109*cents[prior])
        valid=pair&np.isfinite(prices[safe])&np.isfinite(prices[prior])&(cents[safe]>0)&(cents[prior]>0)
        valid &= (np.abs(prices[safe]-cents[safe]/100)<=.0001)&(np.abs(prices[prior]-cents[prior]/100)<=.0001)
        valid &= (adj[safe]==3)&(adj[prior]==3)
        counts+=event;age=np.minimum(age,np.where(event,k,21));good+=valid;rows+=present
        breaks+=pair&(np.abs(d.preclose.to_numpy()[safe]-prices[prior])>.005)
        variables[f'DCP{k}']=prices[safe];variables[f'DCP{k+1}']=prices[prior]
        assert (now[pos>=k]<pos[pos>=k]).all()
    # These base rows have 21 actual stock days, including the event reference close.
    assert (pos>=21).all() and (codes[pos-21]==old.code.to_numpy()).all()
    assert (dates[pos-1]<old.date.to_numpy()).all() and (dates[pos-21]<dates[pos-20]).all()
    expected=pd.DataFrame(dict(hg_rows=rows,hg_good=good,hg_count=counts,hg_age=age,
        hg_first_date=dates[pos-20],hg_last_date=dates[pos-1],hg_reference_date=dates[pos-21],hg_reference_breaks=breaks))
    pd.testing.assert_frame_equal(actual[expected.columns],expected,check_dtype=False,rtol=0,atol=0)
    valid=(rows==20)&(good==20)
    np.testing.assert_array_equal(actual.large_gain_history_valid,valid)
    final=old.formula_input_valid.to_numpy()&valid
    np.testing.assert_array_equal(actual.formula_input_valid,final)
    np.testing.assert_allclose(actual[['HG01','HG02']],np.where(valid[:,None],np.column_stack([counts,age]),np.nan),rtol=0,atol=0,equal_nan=True)
    namespace={'IF':np.where,'INTPART':np.trunc,'MIN':np.minimum,**variables}
    for k in range(1,21):
        line=f'HGF{k}:=IF(100*INTPART(DCP{k}*100+0.5)>=109*INTPART(DCP{k+1}*100+0.5),1,0);'
        assert line+'\n' in study.HEADER
        namespace[f'HGF{k}']=eval(line.split(':=',1)[1][:-1],{'__builtins__':{}},namespace)
    for name,expression in study.NEW_EXPRESSIONS.items():
        value=eval(expression,{'__builtins__':{}},namespace)
        np.testing.assert_array_equal(value[valid],actual.loc[valid,name])
        np.testing.assert_array_equal(np.floor(100*value[final]+10000+.000001),np.floor(100*actual.loc[final,name]+10000+.000001))
    assert np.all((counts==0)==(age==21)) and (counts>=0).all() and (counts<=20).all()
    assert r['valid']==int(final.sum()) and r['newly_invalid']==int((old.formula_input_valid&~final).sum())
    assert r['valid_with_historical_reference_breaks']==int((final&(breaks>0)).sum())
    assert r['expressions']==study.EXPRESSIONS and r['native_header']==study.HEADER
    names=re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',study.HEADER)+list(study.EXPRESSIONS)
    assert len(names)==len(set(names))
    proof=dict(passed=True,feature_report_sha256=sha(root/'feature_report.json'),rows=len(old),valid=int(final.sum()),
        raw_event_comparisons=len(old)*20,all_window_counts_dates_recency_and_reference_breaks_rebuilt=True,
        all_history_positions_strictly_precede_signal=True,native_20_event_and_distance_expressions_rebuilt=True,
        all_new_integer_encodings_and_validity_verified=True,original_48_inputs_unchanged=True,
        event_is_not_verified_limit_up_or_adjusted_return=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'feature_verification.json',proof);return proof


if __name__=='__main__':
    print(json.dumps(main(),ensure_ascii=False,indent=2))
