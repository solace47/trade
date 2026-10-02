"""Evaluate fixed quote-order lists and coupled unknowns, with no exit rules."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

import freeze_tail_formula_order_target as freeze
from trade_research.tail_formula_additive import conn
from trade_research.tail_formula_boundary_evaluation import period, number
from trade_research.research_io import check_sources, save_json, sha
from tail_formula_reports import checked_selection
from tail_formula_statistics import weekly_interval, interval
from compare_tail_formula_shared_unknowns import daily_bounds

ROOT=freeze.ROOT
COUNTS=['rows','known','unknown','no_trade','positive','space','positive_before','space_before','risk_known','bad_first','bad3']
METRICS=['positive_rate','space_rate','positive_before_rate','space_before_rate','bad_first_rate','bad3_rate','lower','upper','mean_reference']


def aggregate(frame,bps,sensitive):
    f=frame.copy();f['known']=f[f'sensitive_known{bps}' if sensitive else f'known{bps}']
    f['unknown']=~f.known & ~f.known_no_trade;f['no_trade']=f.known_no_trade
    f['positive']=f.known & f[f'opportunity{bps}'].eq(1);f['space']=f.known & f[f'one_percent{bps}'].eq(1)
    f['positive_before']=f.known & f[f'positive_before_bad3{bps}'].eq(1)
    f['space_before']=f.known & f[f'space_before_bad3{bps}'].eq(1)
    f['risk_known']=f.known & f[f'risk_observed{bps}'].eq(1)
    z=f[f'first_bad3{bps}'];a=f[f'first_positive_end{bps}']
    f['bad_first']=f.risk_known & z.ge(0) & (a.lt(0)|z.le(a));f['bad3']=f.risk_known & z.ge(0)
    f['reference']=f[f'mark_0959_return{bps}'].where(f.known)
    d=f.groupby(['date','half']).agg(rows=('code','size'),**{n:(n,'sum') for n in COUNTS[1:]},mean_reference=('reference','mean')).reset_index()
    for name in ['positive','space','positive_before','space_before']:d[name+'_rate']=d[name]/d.known.replace(0,np.nan)
    for name in ['bad_first','bad3']:d[name+'_rate']=d[name]/d.risk_known.replace(0,np.nan)
    d['lower']=d.positive_before/d.rows;d['upper']=(d.positive_before+d.unknown)/d.rows
    c=conn();c.register('label_rows',f)
    terms=','.join(f'count(*) FILTER(WHERE "{n}") AS "{n}"' for n in COUNTS[1:])
    c.sql('SELECT date,half,count(*) AS rows,'+terms+',avg(reference) AS mean_reference FROM label_rows GROUP BY date,half').create_view('counts')
    ex=c.sql('''SELECT *,positive::DOUBLE/nullif(known,0) AS positive_rate,space::DOUBLE/nullif(known,0) AS space_rate,
        positive_before::DOUBLE/nullif(known,0) AS positive_before_rate,space_before::DOUBLE/nullif(known,0) AS space_before_rate,
        bad_first::DOUBLE/nullif(risk_known,0) AS bad_first_rate,bad3::DOUBLE/nullif(risk_known,0) AS bad3_rate,
        positive_before::DOUBLE/rows AS lower,(positive_before+unknown)::DOUBLE/rows AS upper FROM counts ORDER BY date,half''').df();c.close()
    pd.testing.assert_frame_equal(d[['date','half',*COUNTS]],ex[['date','half',*COUNTS]],check_dtype=False,check_exact=True)
    np.testing.assert_allclose(d[METRICS].to_numpy(dtype=float),ex[METRICS].to_numpy(dtype=float,na_value=np.nan),rtol=0,atol=2e-12,equal_nan=True)
    return d


def summarize(d,year,group,arm,bps,sensitive):
    result=[]
    for name in [year+'H1',year+'H2',year]:
        q=period(d,name).set_index('date');s=dict(group=group,arm=arm,period=name,bps=bps,sensitive=sensitive,days=len(q))
        s.update({n:int(q[n].sum()) for n in COUNTS});s.update({n:number(q[n].mean()) for n in METRICS})
        for n in ['positive_before_rate','space_before_rate','lower','upper']:s[n+'_ci']=weekly_interval(q[n])
        result.append(s)
    return result


def paired(left,right,labels,year,left_name,right_name):
    dates=[set(f.loc[f.selected,'date']) for f in [left,right]];common=dates[0]&dates[1]
    pieces=[]
    for side,frame in [('left',left),('right',right)]:
        q=frame.loc[frame.selected & frame.date.isin(common),freeze.KEYS].copy();q[side]=True;pieces.append(q)
    merged=pieces[0].merge(pieces[1],on=freeze.KEYS,how='outer',validate='one_to_one')
    for side in ['left','right']:merged[side]=merged[side].eq(True)
    r=merged.merge(labels,on=freeze.KEYS,validate='one_to_one');assert len(r)==len(merged)
    summaries=[];parts=[]
    for bps in [5,15]:
        for sensitive in [False,True]:
            for target,field in [('positive_before','positive_before_bad3'),('space_before','space_before_bad3')]:
                rows=r[['date','code','half','left','right']].copy()
                rows['known']=r[f'sensitive_known{bps}' if sensitive else f'known{bps}'];rows['no_trade']=r.known_no_trade
                rows['opportunity']=r[field+str(bps)];d=daily_bounds(rows)
                c=conn();c.register('comparison_rows',rows)
                ex=c.sql('''WITH weights AS(SELECT *,"left"::DOUBLE/sum("left"::INT) OVER(PARTITION BY date)
                    -"right"::DOUBLE/sum("right"::INT) OVER(PARTITION BY date) AS w FROM comparison_rows)
                    SELECT date,half,sum("left"::INT) AS left_rows,sum("right"::INT) AS right_rows,
                    sum(CASE WHEN known THEN w*opportunity WHEN no_trade THEN 0 ELSE least(w,0) END) AS lower,
                    sum(CASE WHEN known THEN w*opportunity WHEN no_trade THEN 0 ELSE greatest(w,0) END) AS upper,
                    sum(CASE WHEN NOT known AND NOT no_trade THEN abs(w) ELSE 0 END) AS unknown_weight,
                    sum(("left" AND "right" AND NOT known AND NOT no_trade)::INT) AS shared_unknown
                    FROM weights GROUP BY date,half ORDER BY date''').df();c.close()
                pd.testing.assert_frame_equal(d,ex,check_dtype=False,rtol=0,atol=2e-12)
                for name in [year+'H1',year+'H2',year]:
                    q=period(d,name);s=dict(left=left_name,right=right_name,period=name,bps=bps,sensitive=sensitive,target=target,days=len(q))
                    for n in ['lower','upper','unknown_weight']:
                        s[n]=number(q[n].mean());s[n+'_ci']=weekly_interval(q.set_index('date')[n]);other=interval(q,n)
                        if other is None:assert s[n+'_ci'] is None
                        else:np.testing.assert_allclose(s[n+'_ci'],other,rtol=0,atol=2e-12)
                    summaries.append(s)
                d['bps']=bps;d['sensitive']=sensitive;d['target']=target;d['left']=left_name;d['right']=right_name;parts.append(d)
    return dict(left=left_name,right=right_name,year=year,left_days=len(dates[0]),right_days=len(dates[1]),common_days=len(common),
        left_only_dates=sorted(dates[0]-common),right_only_dates=sorted(dates[1]-common),complete_frames_identical=left.equals(right),summaries=summaries),parts


def main():
    _,execution,e=freeze.checked();assert not (ROOT/'economic_report.json').exists()
    joint=json.loads((ROOT/'joint_selection_freeze.json').read_text());assert joint['passed']
    assert joint['evaluation_protocol_sha256']==sha(freeze.EVALUATION);check_sources(joint['source_hashes'])
    assert sha(ROOT/'joint_selection_freeze.json') in subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    labels=pd.read_parquet(e['ordered_labels']);refs=pd.read_parquet(e['original_labels'],columns=['date','code','mark_0959_return5','mark_0959_return15'])
    labels=labels.merge(refs,on=['date','code'],validate='one_to_one');assert len(labels)==1258085 and labels.date.lt('2026-01-01').all()
    items=list(joint['selections'])+[dict(group='original50'+year,arm='original50',year=year,root=path) for year,path in e['original_controls'].items()]
    frames={};daily=[];summaries=[];reuse=[];seen=[]
    for item in items:
        group=item['group'];frame=checked_selection(Path(item['root']));frames[group]=frame
        assert frame.loc[frame.selected,'date'].str.startswith(item['year']).all()
        parent=next((old for old in seen if old['frame'].equals(frame) and old['year']==item['year']),None)
        if parent:
            parts=[d.copy() for d in parent['daily']]
            for d in parts:d['group']=group
            daily.extend(parts);summaries.extend([{**s,'group':group} for s in parent['summaries']]);reuse.append(dict(group=group,source=parent['group'],complete_selection_and_labels_exact=True));continue
        q=labels.merge(frame,on=freeze.KEYS,validate='one_to_one');assert len(q)==len(frame)==len(labels)
        chosen=q.loc[q.selected];base=q.loc[q.date.isin(chosen.date.unique())];local=[];stats=[]
        for arm,f in [('formula',chosen),('base_same_dates',base)]:
            for bps in [5,15]:
                for sensitive in [False,True]:
                    d=aggregate(f,bps,sensitive);s=summarize(d,item['year'],group,arm,bps,sensitive)
                    d['group']=group;d['arm']=arm;d['bps']=bps;d['sensitive']=sensitive;local.append(d);stats.extend(s)
        daily.extend(local);summaries.extend(stats);seen.append(dict(group=group,year=item['year'],frame=frame,daily=local,summaries=stats))
        print(json.dumps(dict(group_completed=group)),flush=True)
    pd.concat(daily,ignore_index=True).to_parquet(ROOT/'economic_daily.parquet',index=False,compression='zstd')
    comparisons=[];paired_days=[]
    for year in ['2024','2025']:
        for right in ['whole_window','original50']:
            result,parts=paired(frames['ordered'+year],frames[right+year],labels,year,'ordered'+year,right+year)
            comparisons.append(result);paired_days.extend(parts)
    pd.concat(paired_days,ignore_index=True).to_parquet(ROOT/'paired_daily.parquet',index=False,compression='zstd')
    def get(group,name):
        return next(s for s in summaries if s['group']==group and s['arm']=='formula' and s['period']==name and s['bps']==15 and not s['sensitive'])
    def greater(a,b):return a is not None and b is not None and a>b
    quality=[]
    for year in ['2024','2025']:
        a=get('ordered'+year,year);controls=[get(v+year,year) for v in ['whole_window','original50']]
        quality.append(dict(year=year,
            both_primary_and_space_point_rates_above_both_controls=all(greater(a[n],b[n]) for b in controls for n in ['positive_before_rate','space_before_rate']),
            prior_bad_rate_not_higher_than_either_control=all(a['bad_first_rate'] is not None and b['bad_first_rate'] is not None and a['bad_first_rate']<=b['bad_first_rate'] for b in controls),
            coupled_primary_lower_intervals_strictly_positive=all(s['lower_ci'] is not None and s['lower_ci'][0]>0 for c in comparisons if c['year']==year for s in c['summaries'] if s['period']==year and s['bps']==15 and not s['sensitive'] and s['target']=='positive_before')))
    half_coverage=all(get('ordered'+year,year+h)['days']>=20 and get('ordered'+year,year+h)['known']>=100 for year in ['2024','2025'] for h in ['H1','H2'])
    size_checks=[]
    for year in ['2024','2025']:
        r=json.loads((ROOT/('ordered'+year)/'selection_report.json').read_text());size_checks.append(r['median_daily'] is not None and r['median_daily']<=5)
    criteria=dict(four_half_coverage=half_coverage,both_year_rates_improved=all(q['both_primary_and_space_point_rates_above_both_controls'] for q in quality),
        both_year_coupled_intervals_positive=all(q['coupled_primary_lower_intervals_strictly_positive'] for q in quality),
        both_year_prior_risk_not_worse=all(q['prior_bad_rate_not_higher_than_either_control'] for q in quality),both_year_daily_median_at_most_five=all(size_checks))
    save_json(ROOT/'economic_report.json',dict(passed=True,evaluation_protocol_sha256=sha(freeze.EVALUATION),joint_sha256=sha(ROOT/'joint_selection_freeze.json'),
        economic_daily_sha256=sha(ROOT/'economic_daily.parquet'),paired_daily_sha256=sha(ROOT/'paired_daily.parquet'),summaries=summaries,comparisons=comparisons,
        quality_checks=quality,criteria=criteria,exact_complete_frame_analysis_reuse=reuse,all_new_daily_counts_rates_and_coupled_bounds_SQL_verified=True,
        weekly_intervals_independently_verified=True,not_realized_profit=True,no_native_formula_publication_claim=True,new_2026_prices_read=False,no_exit_rules=True))
    print(json.dumps(dict(report_sha256=sha(ROOT/'economic_report.json'),criteria=criteria,exact_reuse=reuse)),flush=True)


if __name__=='__main__':
    main()
