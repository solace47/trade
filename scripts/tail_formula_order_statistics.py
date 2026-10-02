"""Shared frozen quote-order statistics and coupled unknown comparisons."""
import numpy as np
import pandas as pd

from trade_research.tail_formula_additive import conn
from trade_research.tail_formula_boundary_evaluation import period, number
from tail_formula_statistics import weekly_interval, interval
from compare_tail_formula_shared_unknowns import daily_bounds

KEYS=['date','code','half','board','decision_shares']
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
        q=frame.loc[frame.selected & frame.date.isin(common),KEYS].copy();q[side]=True;pieces.append(q)
    merged=pieces[0].merge(pieces[1],on=KEYS,how='outer',validate='one_to_one')
    for side in ['left','right']:merged[side]=merged[side].eq(True)
    r=merged.merge(labels,on=KEYS,validate='one_to_one');assert len(r)==len(merged)
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

