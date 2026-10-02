"""Frozen score diagnostics; no fitting, new selection, or threshold search."""
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
from verify_tail_formula_additive import tree_sql

PROTOCOL = Path('config/tail_formula_order_risk_review.json')
ROOT = Path('data/research/tail_formula_order_risk/reflection')


def binary_daily(frame, field):
    p = frame[['date', 'score', field]].copy()
    assert p[field].isin([0, 1]).all() and np.isfinite(p.score).all()
    p['rank'] = p.groupby('date').score.rank(method='average')
    p['positive_rank'] = p['rank'].where(p[field].eq(1), 0.)
    d = p.groupby('date').agg(rows=(field, 'size'), positives=(field, 'sum'), positive_ranks=('positive_rank', 'sum')).reset_index()
    d['pairs'] = d.positives * (d.rows - d.positives)
    d['numerator2'] = (2*d.positive_ranks - d.positives*(d.positives+1)).astype('int64')
    d['auc'] = d.numerator2 / (2*d.pairs.replace(0, np.nan))
    return d[['date', 'pairs', 'numerator2', 'auc']]


def daily_metrics(frame):
    """Higher score should precede the higher label; score ties count one half."""
    d = frame.groupby('date').size().rename('rows').reset_index()
    for field in ['positive_before', 'space_before']:
        part = binary_daily(frame, field).rename(columns={n:field+'_'+n for n in ['pairs','numerator2','auc']})
        d = d.merge(part, on='date', validate='one_to_one')
    d['utility_pairs'] = 0
    d['utility_numerator2'] = 0
    for high, low in [(0, -3), (1, -3), (1, 0)]:
        part = frame.loc[frame.utility.isin([high, low]), ['date','score','utility']].copy()
        part['higher'] = part.utility.eq(high).astype('int64')
        b = binary_daily(part, 'higher').set_index('date')
        for field in ['pairs', 'numerator2']:
            d['utility_'+field] += d.date.map(b[field]).fillna(0).astype('int64')
    d['utility_auc'] = d.utility_numerator2 / (2*d.utility_pairs.replace(0, np.nan))
    return d


def verify_metrics(frame, actual):
    c = numeric.conn(); c.register('diagnostic_rows', frame)
    expected = c.sql('SELECT date,count(*) AS rows FROM diagnostic_rows GROUP BY date ORDER BY date').df()
    def one(where, label):
        return c.sql(f'''WITH ranked AS (SELECT date,{label} AS positive,
            rank() OVER(PARTITION BY date ORDER BY score) +
            (count(*) OVER(PARTITION BY date,score)-1)/2. AS average_rank
            FROM diagnostic_rows {where}), counts AS (
            SELECT date,count(*) AS n,sum(positive::INT) AS m,
            sum(CASE WHEN positive THEN average_rank ELSE 0. END) AS total
            FROM ranked GROUP BY date)
            SELECT date,(m*(n-m))::BIGINT AS pairs,(2*total-m*(m+1))::BIGINT AS numerator2 FROM counts ORDER BY date''').df()
    for field in ['positive_before','space_before']:
        b=one('', field+'=1').rename(columns={n:field+'_'+n for n in ['pairs','numerator2']})
        expected=expected.merge(b,on='date',validate='one_to_one')
    expected['utility_pairs']=0; expected['utility_numerator2']=0
    for high,low in [(0,-3),(1,-3),(1,0)]:
        b=one(f'WHERE utility IN ({high},{low})',f'utility={high}').set_index('date')
        for field in ['pairs','numerator2']:
            expected['utility_'+field]+=expected.date.map(b[field]).fillna(0).astype('int64')
    for field in ['positive_before','space_before','utility']:
        expected[field+'_auc']=expected[field+'_numerator2']/(2*expected[field+'_pairs'].replace(0,np.nan))
    c.close()
    pd.testing.assert_frame_equal(actual[expected.columns],expected,check_dtype=False,rtol=0,atol=2e-12)


def main():
    check_runtime(); p=json.loads(PROTOCOL.read_text());check_sources(p['source_hashes'])
    prior=json.loads(Path(p['completed_receipt']).read_text());assert prior['passed'];check_sources(prior['source_hashes'])
    head=subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    assert sha(PROTOCOL) in head and subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}'])==PROTOCOL.read_bytes()
    assert not ROOT.exists(),'Do not repeat a completed diagnostic'
    ROOT.mkdir()
    numeric.EXPRESSIONS = original.EXPRESSIONS
    f=pd.read_parquet(p['features']);t=pd.read_parquet(p['targets']);u=pd.read_parquet(p['utilities'],columns=['date','code','opportunity15'])
    assert len(f)==1815129 and int(f.formula_input_valid.sum())==1602413 and f.date.lt('2026-01-01').all()
    pd.testing.assert_frame_equal(f[['date','code']],t[['date','code']],check_exact=True)
    t=t.merge(u.rename(columns={'opportunity15':'utility'}),on=['date','code'],validate='one_to_one')
    for field,event in [('positive_before','first_positive_end15'),('space_before','first_space_end15')]:
        t[field]=(t[event].ge(0)&(t.first_bad315.lt(0)|t[event].lt(t.first_bad315))).astype('int64')
    labels=t[['date','code','next_date','known15','known_no_trade','utility','positive_before','space_before']]
    valid=f.loc[f.formula_input_valid].merge(labels,on=['date','code'],validate='one_to_one')
    records=[];gates=[];summaries=[];gate_reports=[];sources=dict(p['source_hashes'])
    for fold in p['folds']:
        mpath=Path(fold['model_root'])/'model_report.json';m=json.loads(mpath.read_text());assert m['variant']=='relative' and m['training_allowed_utility_values']==[-3,0,1]
        assert m['last_observation']<fold['evaluation_start'] and len(m['thresholds'])==1
        assert m['feature_names']==list(original.EXPRESSIONS) and m['thresholds'][0]['training_quantile']==.995
        for scope in ['training','evaluation']:
            start,end=fold[scope+'_start'],fold[scope+'_end'];q=valid.loc[valid.date.ge(start)&valid.date.lt(end)].copy()
            if scope=='training':q=q.loc[q.next_date.lt(end)]
            scores=numeric.predict(numeric.encode(q),m);q['model_score']=scores
            c=numeric.conn();c.register('visible',q[['date','code',*original.EXPRESSIONS]])
            encoded=','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS X{i:02d}' for i,n in enumerate(original.EXPRESSIONS,1))
            c.sql('SELECT date,code,'+encoded+' FROM visible').create_view('encoded')
            equation=format(m['bias'],'.17e')+'+'+'+'.join(tree_sql(tree) for tree in m['trees'])
            rebuilt=c.sql('SELECT date,code,'+equation+' AS score FROM encoded ORDER BY date,code').df();c.close()
            pd.testing.assert_frame_equal(q[['date','code']].reset_index(drop=True),rebuilt[['date','code']],check_exact=True)
            np.testing.assert_allclose(scores,rebuilt.score,rtol=0,atol=2e-11)
            chosen=q.model_score.gt(m['thresholds'][0]['threshold'])
            gate=q[['date']].assign(score=q.model_score,chosen=chosen)
            gd=gate.groupby('date').agg(rows=('score','size'),selected=('chosen','sum'),mean_score=('score','mean'),median_score=('score','median')).reset_index()
            gd['fold']=fold['id'];gd['scope']=scope;gates.append(gd)
            if scope=='training':
                assert int(q.known15.sum())==m['rows']
                np.testing.assert_allclose(np.quantile(q.loc[q.known15,'model_score'],.995),m['thresholds'][0]['threshold'],rtol=0,atol=2e-10)
            gate_reports.append(dict(fold=fold['id'],scope=scope,days=len(gd),rows=len(q),known=int(q.known15.sum()),unknown=int((~q.known15&~q.known_no_trade).sum()),no_trade=int(q.known_no_trade.sum()),threshold=m['thresholds'][0]['threshold'],days_above_threshold=int(gd.selected.gt(0).sum()),selected=int(chosen.sum()),score_quantiles=[float(q.model_score.quantile(v)) for v in [.5,.995]]))
            observed=q.loc[q.known15].copy();assert observed.utility.isin([-3,0,1]).all()
            for scorer in ['model','V01','negative_V01']:
                part=observed[['date','code','utility','positive_before','space_before']].copy()
                part['score']=observed.model_score if scorer=='model' else observed.V01*(-1 if scorer=='negative_V01' else 1)
                d=daily_metrics(part);verify_metrics(part,d);d['fold']=fold['id'];d['scope']=scope;d['scorer']=scorer;records.append(d)
                for target in ['positive_before','space_before','utility']:
                    field=target+'_auc';summaries.append(dict(fold=fold['id'],scope=scope,scorer=scorer,target=target,days=len(d),defined_days=int(d[field].notna().sum()),rows=int(d.rows.sum()),date_equal_auc=float(d[field].mean()) if d[field].notna().any() else None,weekly_ci=interval(d,field)))
        print(json.dumps(dict(diagnostic_fold_verified=fold['id'])),flush=True)
    desc=[];descriptor_summaries=[]
    visible=f.loc[f.formula_input_valid,['date','code','half','V01','NA01','A12','R01']].copy();visible['visible_return']=visible.V01*visible.NA01
    for year,folder in p['selections'].items():
        selected=checked_selection(Path(folder));joined=visible.merge(selected[['date','code','selected']],on=['date','code'],validate='one_to_one');pick=joined.loc[joined.selected];pool=joined.loc[joined.date.isin(pick.date.unique())]
        for arm,q in [('formula',pick),('same_date_valid_pool',pool)]:
            d=q.groupby(['date','half']).agg(rows=('code','size'),**{n:(n,'mean') for n in ['V01','visible_return','A12','R01']}).reset_index()
            c=numeric.conn();c.register('descriptors',q)
            expected=c.sql('SELECT date,half,count(*) AS rows,avg(V01) AS V01,avg(NA01*V01) AS visible_return,avg(A12) AS A12,avg(R01) AS R01 FROM descriptors GROUP BY date,half ORDER BY date,half').df();c.close()
            pd.testing.assert_frame_equal(d,expected,check_dtype=False,rtol=0,atol=2e-10)
            d['year']=year;d['arm']=arm;desc.append(d)
            for period in [year+'H1',year+'H2',year]:
                z=d if period==year else d.loc[d.half.eq(period)];descriptor_summaries.append(dict(year=year,arm=arm,period=period,days=len(z),rows=int(z.rows.sum()),**{n:float(z[n].mean()) if len(z) else None for n in ['V01','visible_return','A12','R01']}))
    for name,parts in [('daily_metrics',records),('daily_gates',gates),('descriptor_daily',desc)]:
        path=ROOT/(name+'.parquet');pd.concat(parts,ignore_index=True).to_parquet(path,index=False,compression='zstd');sources[str(path)]=sha(path)
    save_json(ROOT/'report.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),source_hashes=sources,summaries=summaries,score_gate_summaries=gate_reports,descriptor_summaries=descriptor_summaries,all_scores_SQL_verified=True,all_rank_pair_counts_and_ties_SQL_verified=True,all_descriptors_SQL_verified=True,new_fits=0,new_selection_lists=0,not_economic_profit_or_formula_validation=True,new_2026_prices_read=False,no_exit_rules=True))
    print(json.dumps(dict(report_sha256=sha(ROOT/'report.json'),summaries=len(summaries),gate_groups=len(gate_reports),descriptor_groups=len(descriptor_summaries))),flush=True)


if __name__=='__main__':main()
