"""Frozen ranking and target-centering diagnosis; no new fits or selections."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from trade_research import tail_formula_additive as numeric
from trade_research import tail_formula_baseline as original
from trade_research.research_io import check_runtime, check_sources, sha, save_json
from tail_formula_reports import checked_selection
from tail_formula_statistics import interval
from tail_formula_rank_statistics import daily_metrics, verify_metrics
from verify_tail_formula_additive import tree_sql

ROOT=Path('data/research/tail_formula_direction_target/reflection')
PROTOCOL=Path('config/tail_formula_direction_target_review.json')


def main():
    check_runtime();p=json.loads(PROTOCOL.read_text());check_sources(p['source_hashes'])
    complete=json.loads(Path(p['completed_receipt']).read_text());assert complete['passed'];check_sources(complete['source_hashes'])
    assert sha(PROTOCOL) in subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    assert subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}'])==PROTOCOL.read_bytes()
    assert not ROOT.exists(),'Do not repeat a completed diagnosis';ROOT.mkdir()
    numeric.EXPRESSIONS=original.EXPRESSIONS
    f=pd.read_parquet(p['features']);t=pd.read_parquet(p['targets'])
    u=pd.read_parquet(p['direction_labels'],columns=['date','code','opportunity15']).rename(columns={'opportunity15':'direction'})
    assert len(f)==1815129 and int(f.formula_input_valid.sum())==1602413 and f.date.lt('2026-01-01').all()
    pd.testing.assert_frame_equal(f[['date','code']],t[['date','code']],check_exact=True)
    t=t.merge(u,on=['date','code'],validate='one_to_one');assert t.loc[t.known15,'direction'].isin([-1,0,1]).all()
    for field,event in [('positive_before','first_positive_end15'),('space_before','first_space_end15')]:
        t[field]=(t[event].ge(0)&(t.first_bad315.lt(0)|t[event].lt(t.first_bad315))).astype('int64')
    columns=['date','code','next_date','known15','known_no_trade','direction','positive_before','space_before']
    valid=f.loc[f.formula_input_valid].merge(t[columns],on=['date','code'],validate='one_to_one')
    frames={y:checked_selection(Path(root)) for y,root in p['selections'].items()}
    ranks=[];classes=[];rank_summaries=[];class_summaries=[];gates=[];sources=dict(p['source_hashes'])
    for fold in p['folds']:
        m=json.loads((Path(fold['model_root'])/'model_report.json').read_text())
        assert m['variant']=='relative' and m['training_allowed_utility_values']==[-1,0,1]
        assert m['last_observation']<fold['evaluation_start'] and len(m['thresholds'])==1 and m['thresholds'][0]['training_quantile']==.995
        for scope in ['training','evaluation']:
            start,end=fold[scope+'_start'],fold[scope+'_end'];q=valid.loc[valid.date.ge(start)&valid.date.lt(end)].copy()
            population=t.loc[t.known15&t.date.ge(start)&t.date.lt(end)].copy()
            if scope=='training':
                q=q.loc[q.next_date.lt(end)];population=population.loc[population.next_date.lt(end)]
            means=population.groupby('date').direction.mean()
            q['future_source_mean']=q.date.map(means)
            q['model_score']=numeric.predict(numeric.encode(q),m)
            c=numeric.conn();c.register('visible',q[['date','code',*original.EXPRESSIONS]])
            encoded=','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS X{i:02d}' for i,n in enumerate(original.EXPRESSIONS,1))
            c.sql('SELECT date,code,'+encoded+' FROM visible').create_view('encoded')
            equation=format(m['bias'],'.17e')+'+'+'+'.join(tree_sql(tree) for tree in m['trees'])
            ex=c.sql('SELECT date,code,'+equation+' AS score FROM encoded ORDER BY date,code').df();c.close()
            pd.testing.assert_frame_equal(q[['date','code']].reset_index(drop=True),ex[['date','code']],check_exact=True)
            np.testing.assert_allclose(q.model_score,ex.score,rtol=0,atol=2e-11)
            q['chosen']=q.model_score.gt(m['thresholds'][0]['threshold'])
            if scope=='evaluation':
                flags=frames[start[:4]].loc[lambda z:z.date.ge(start)&z.date.lt(end),['date','code','selected']]
                merged=q[['date','code','chosen']].merge(flags,on=['date','code'],validate='one_to_one')
                assert len(merged)==len(q) and merged.chosen.equals(merged.selected)
            elif scope=='training':
                assert int(q.known15.sum())==m['rows']
                np.testing.assert_allclose(np.quantile(q.loc[q.known15,'model_score'],.995),m['thresholds'][0]['threshold'],rtol=0,atol=2e-10)
            gates.append(dict(fold=fold['id'],scope=scope,rows=len(q),known=int(q.known15.sum()),unknown=int((~q.known15&~q.known_no_trade).sum()),no_trade=int(q.known_no_trade.sum()),chosen=int(q.chosen.sum()),chosen_known=int((q.chosen&q.known15).sum()),chosen_unknown=int((q.chosen&~q.known15&~q.known_no_trade).sum()),chosen_no_trade=int((q.chosen&q.known_no_trade).sum())))
            observed=q.loc[q.known15].copy();assert observed.future_source_mean.notna().all()
            for scorer in ['model','V01','negative_V01']:
                part=observed[['date','code','positive_before','space_before']].copy()
                # Strictly increasing ordinal alias for the already tested pair counter.
                # This -3 is never a training penalty or a new payoff.
                part['utility']=observed.direction.map({-1:-3,0:0,1:1})
                part['score']=observed.model_score if scorer=='model' else observed.V01*(-1 if scorer=='negative_V01' else 1)
                d=daily_metrics(part);verify_metrics(part,d);d['fold']=fold['id'];d['scope']=scope;d['scorer']=scorer;ranks.append(d)
                for target in ['positive_before','space_before','utility']:
                    field=target+'_auc';rank_summaries.append(dict(fold=fold['id'],scope=scope,scorer=scorer,target=target,days=len(d),defined_days=int(d[field].notna().sum()),date_equal_auc_defined_only=float(d[field].mean()) if d[field].notna().any() else None,weekly_ci=interval(d,field)))
            for arm,a in [('all_known_valid',observed),('selected_known',observed.loc[observed.chosen])]:
                a=a.copy();a['up']=a.direction.eq(1);a['down']=a.direction.eq(-1);a['none']=a.direction.eq(0)
                a['positive_centered_none']=a['none']&a.future_source_mean.lt(0)
                a['centered_direction']=a.direction-a.future_source_mean
                d=a.groupby('date').agg(rows=('code','size'),up=('up','sum'),down=('down','sum'),none=('none','sum'),positive_centered_none=('positive_centered_none','sum'),mean_direction=('direction','mean'),future_source_mean=('future_source_mean','mean'),mean_centered_direction=('centered_direction','mean'),mean_V01=('V01','mean'),mean_S02=('S02','mean')).reset_index()
                c=numeric.conn();c.register('class_rows',a)
                ex=c.sql('''SELECT date,count(*) AS rows,sum((direction=1)::INT) AS up,sum((direction=-1)::INT) AS down,sum((direction=0)::INT) AS none,sum((direction=0 AND future_source_mean<0)::INT) AS positive_centered_none,avg(direction) AS mean_direction,avg(future_source_mean) AS future_source_mean,avg(direction-future_source_mean) AS mean_centered_direction,avg(V01) AS mean_V01,avg(S02) AS mean_S02 FROM class_rows GROUP BY date ORDER BY date''').df();c.close()
                pd.testing.assert_frame_equal(d,ex,check_dtype=False,rtol=0,atol=2e-11)
                for n in ['up','down','none','positive_centered_none']:d[n+'_rate']=d[n]/d.rows
                class_summaries.append(dict(fold=fold['id'],scope=scope,arm=arm,days=len(d),rows=int(d.rows.sum()),**{n:float(d[n].mean()) if len(d) else None for n in ['up_rate','down_rate','none_rate','positive_centered_none_rate','mean_direction','future_source_mean','mean_centered_direction','mean_V01','mean_S02']}))
                d['fold']=fold['id'];d['scope']=scope;d['arm']=arm;classes.append(d)
        print(json.dumps(dict(diagnosis_verified=fold['id'])),flush=True)
    for name,parts in [('rank_daily',ranks),('class_daily',classes)]:
        path=ROOT/(name+'.parquet');pd.concat(parts,ignore_index=True).to_parquet(path,index=False,compression='zstd');sources[str(path)]=sha(path)
    save_json(ROOT/'report.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),source_hashes=sources,rank_summaries=rank_summaries,class_summaries=class_summaries,gate_summaries=gates,all_scores_and_eval_flags_SQL_verified=True,all_rank_pairs_ties_and_class_counts_SQL_verified=True,future_source_means_retrospective_only_not_selection_inputs=True,ordinal_alias_not_training_utility=True,new_fits=0,new_selection_lists=0,new_2026_prices_read=False,no_exit_rules=True,not_economic_profit_or_formula_validation=True))
    print(json.dumps(dict(report_sha256=sha(ROOT/'report.json'),rank_groups=len(rank_summaries),class_groups=len(class_summaries))),flush=True)


if __name__=='__main__':main()
