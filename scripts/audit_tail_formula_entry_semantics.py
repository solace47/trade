"""Reconcile stored cash arithmetic; do not create new selection-group results."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as numeric
from trade_research.research_io import check_runtime,check_sources,save_json,sha

ROOT=Path('data/research/tail_formula_entry_semantics_reflection')
PROTOCOL=ROOT/'protocol.json'


def audit():
    check_runtime();p=json.loads(PROTOCOL.read_text());check_sources(p['source_hashes'])
    assert sha(PROTOCOL) in subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    assert p['strict_labels_sha256']==sha(Path(p['strict_labels']))
    cols=['date','code','decision_shares','entry_vwap','entry_high','price_0959','sustained_close','min_low',
          'known5','known15','buy_cash5','buy_cash15','mark_0959_return5','mark_0959_return15',
          'sustained_return5','sustained_return15','adverse_return5','adverse_return15']
    f=pd.read_parquet(p['strict_labels'],columns=cols)
    assert len(f)==1258085 and f.date.ge('2024-01-01').all() and f.date.lt('2026-01-01').all()
    assert not f.duplicated(['date','code']).any()
    c=numeric.conn();c.register('f',f);checks=[]
    for bps in [5,15]:
        price=f.entry_vwap+np.maximum(f.entry_vwap*bps/10000,.005)
        value=f.decision_shares*price;cash=value+np.maximum(5,value*.0003)+value*.00001
        sql=c.sql(f'''WITH a AS(SELECT *,decision_shares*(entry_vwap+greatest(entry_vwap*{bps}/10000.,.005)) AS value FROM f)
            SELECT date,code,value+greatest(5,value*.0003)+value*.00001 AS cash FROM a ORDER BY date,code''').df()
        pd.testing.assert_frame_equal(f[['date','code']],sql[['date','code']],check_exact=True)
        np.testing.assert_allclose(cash,sql.cash,rtol=0,atol=2e-10,equal_nan=True)
        np.testing.assert_allclose(f['buy_cash'+str(bps)],cash,rtol=0,atol=2e-10,equal_nan=True)
        errors={'buy_cash':float(np.nanmax(abs(f['buy_cash'+str(bps)]-cash)))}
        for name,field in [('mark_0959','price_0959'),('sustained','sustained_close'),('adverse','min_low')]:
            sale=f[field]-np.maximum(f[field]*bps/10000,.005);v=f.decision_shares*sale
            result=((v-np.maximum(5,v*.0003)-v*.00051)/cash-1).where(f['known'+str(bps)])
            expected=c.sql(f'''WITH a AS(SELECT *,decision_shares*({field}-greatest({field}*{bps}/10000.,.005)) AS value FROM f)
                SELECT CASE WHEN known{bps} THEN (value-greatest(5,value*.0003)-value*.00051)/buy_cash{bps}-1 END AS mark
                FROM a ORDER BY date,code''').df()
            np.testing.assert_allclose(result,expected.mark,rtol=0,atol=2e-10,equal_nan=True)
            np.testing.assert_allclose(f[name+'_return'+str(bps)],result,rtol=0,atol=2e-10,equal_nan=True)
            errors[name]=float(np.nanmax(abs(f[name+'_return'+str(bps)]-result)))
        checks.append(dict(bps=bps,maximum_absolute_errors=errors))
    c.close()
    assert sha(Path(p['strict_labels']))==p['strict_labels_sha256']
    sources=dict(p['source_hashes'],**{str(PROTOCOL):sha(PROTOCOL),str(Path(__file__).relative_to(Path.cwd())):sha(Path(__file__))})
    save_json(ROOT/'verification.json',dict(passed=True,rows=len(f),checks=checks,source_hashes=sources,
        primary_entry_price='entry_vwap + max(entry_vwap*bps/10000,0.005)',
        primary_capacity='source classifier uses 10% of summed four-minute volume; exact lot-by-minute fills not proved',
        history_proxy_entry_price='entry_high + max(entry_high*15/10000,0.005)',
        history_proxy_capacity='at least one of four minutes has 10% volume covering fixed shares',
        all_original_labels_qualifications_unknowns_and_cash_preserved=True,
        no_new_selection_or_fits=True,no_new_group_economic_aggregation=True,new_2026_prices_read=False,no_exit_rules=True,
        implication='主评价不是最高价买入；与五日代理口径不同，不能把已失败主评价归因于最高价，也不能把均价当已证明真实成交。'))
    return dict(verification_sha256=sha(ROOT/'verification.json'),rows=len(f),checks=checks)


if __name__=='__main__':print(json.dumps(audit(),ensure_ascii=False),flush=True)
