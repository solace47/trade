"""Audit within-date information in four frozen 50-input scores, without refits."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research.corporate_cash import save_json, sha
from trade_research.reference_gain_accounting import weekly_interval
from freeze_tail_formula_bipower_gap import checked_selection

ROOT=Path('data/research/tail_formula_daily_order')
PROTOCOL=Path('config/tail_formula_daily_order_protocol.json')
META=['date','code','half','board','decision_shares','formula_input_valid']
PERIODS=['2024H1','2024H2','2025H1','2025H2','2024','2025']


def checked():
    p=json.loads(PROTOCOL.read_text())
    assert p['bins']==10 and p['score_integer_scale']==1000000 and not p['new_model_fit_allowed']
    assert p['diagnostic_only'] and p['periods']==PERIODS and not p['new_2026_prices_allowed']
    for file,digest in p['source_hashes'].items():assert sha(Path(file))==digest,file
    return p


def score_bins(frame):
    """Visible-pool tie midranks; integer division avoids boundary roundoff."""
    assert not frame.duplicated(['date','code']).any()
    out=frame[META].copy();out['score_integer']=0;out['bucket']=-1
    valid=frame.formula_input_valid
    values=frame.loc[valid,'score'].to_numpy(float)
    assert np.isfinite(values).all() and np.abs(values).max()<=1
    out.loc[valid,'score_integer']=np.floor((values+1)*1000000+.5).astype('int64')
    subset=out.loc[valid]
    ranks=subset.groupby('date').score_integer.rank(method='average')
    counts=subset.groupby('date').code.transform('size')
    numerator=(2*ranks-1).astype('int64')*5
    out.loc[valid,'bucket']=(numerator//counts).astype('int64')
    assert out.loc[valid,'bucket'].between(0,9).all()
    return out


def freeze():
    p=checked();assert not (ROOT/'ranking_report.json').exists()
    features=pd.read_parquet(p['features'],columns=META)
    frames=[]
    for fold in p['folds']:
        root=Path(fold['root'])
        m=json.loads((root/'model_report.json').read_text());s=json.loads((root/'score_report.json').read_text())
        for kind in ['model','score']:
            v=json.loads((root/(kind+'_verification.json')).read_text())
            assert v['passed'] and v[kind+'_report_sha256']==sha(root/(kind+'_report.json'))
        assert s['scores_sha256']==sha(root/'scores.parquet') and s['model_report_sha256']==sha(root/'model_report.json')
        assert m['last_observation']<fold['start'] and m['feature_names']==p['feature_names']
        d=pd.read_parquet(root/'scores.parquet',filters=[('date','>=',fold['start']),('date','<',fold['end'])])
        original=features.loc[features.date.ge(fold['start']) & features.date.lt(fold['end'])].reset_index(drop=True)
        pd.testing.assert_frame_equal(d[META],original,check_exact=True);frames.append(d)
    combined=pd.concat(frames,ignore_index=True)
    pd.testing.assert_frame_equal(combined[META],features,check_exact=True)
    ranked=score_bins(combined);c=base.conn();c.register('visible',combined)
    expected=c.sql('''WITH q AS(SELECT *,floor((score+1)*1000000+.5)::BIGINT AS si
        FROM visible WHERE formula_input_valid),r AS(SELECT *,rank() OVER(PARTITION BY date ORDER BY si) AS first,
        count(*) OVER(PARTITION BY date,si) AS tied,count(*) OVER(PARTITION BY date) AS n FROM q)
        SELECT f.date,f.code,f.half,f.board,f.decision_shares,f.formula_input_valid,
        coalesce(si,0)::BIGINT AS score_integer,coalesce((5*(2*first+tied-2))//n,-1)::BIGINT AS bucket
        FROM visible f LEFT JOIN r USING(date,code) ORDER BY f.date,f.code''').df();c.close()
    pd.testing.assert_frame_equal(ranked,expected,check_exact=True)
    selected=[checked_selection(Path(r)) for r in p['original_selection_roots']]
    for d in selected:pd.testing.assert_frame_equal(d[META[:-1]],features[META[:-1]],check_exact=True)
    assert not (selected[0].selected & selected[1].selected).any()
    ranked['original_selected']=selected[0].selected | selected[1].selected
    assert not (ranked.original_selected & ~ranked.formula_input_valid).any()
    ranked.to_parquet(ROOT/'ranking.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),ranking_sha256=sha(ROOT/'ranking.parquet'),rows=len(ranked),
        valid=int(ranked.formula_input_valid.sum()),source_hashes=p['source_hashes'],
        no_label_or_execution_values_read_to_form_memberships=True,zero_new_model_fit_or_prediction=True,
        all_ties_use_visible_pool_midrank=True,not_ten_candidate_formulas=True,diagnostic_only=True,
        software_compilation_verified=False,native_ranking_parity_verified=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'ranking_report.json',r)
    save_json(ROOT/'ranking_verification.json',dict(passed=True,ranking_report_sha256=sha(ROOT/'ranking_report.json'),
        all_full_keys_metadata_integer_scores_visible_midrank_and_bins_independently_sql_verified=True,
        original_two_annual_flags_preserved=True,no_economic_groups_read=True,new_2026_prices_read=False))
    return dict(ranking_report_sha256=sha(ROOT/'ranking_report.json'),verification_sha256=sha(ROOT/'ranking_verification.json'),
                rows=len(ranked),valid=r['valid'])


def pandas_auc(rows):
    records=[]
    for date,d in rows.groupby('date',sort=True):
        k=d.loc[d.known15];positive=k.opportunity15.eq(1);n1=int(positive.sum());n0=len(k)-n1
        ranks=k.score_integer.rank(method='average')
        auc=(ranks.loc[positive].sum()-n1*(n1+1)/2)/(n0*n1) if n0 and n1 else np.nan
        records.append(dict(date=date,visible=len(d),known=len(k),positive=n1,negative=n0,auc=auc))
    return pd.DataFrame(records)


def analyze():
    p=checked();assert not (ROOT/'analysis_report.json').exists()
    rr=json.loads((ROOT/'ranking_report.json').read_text());rv=json.loads((ROOT/'ranking_verification.json').read_text())
    assert rv['passed'] and rv['ranking_report_sha256']==sha(ROOT/'ranking_report.json')
    assert rr['protocol_sha256']==sha(PROTOCOL) and rr['ranking_sha256']==sha(ROOT/'ranking.parquet')
    committed=subprocess.run(['git','show','HEAD:docs/selection-formula.md'],capture_output=True,text=True,check=True).stdout
    assert sha(ROOT/'ranking_report.json') in committed
    root=Path(p['labels']);lr=json.loads((root/'full_label_report.json').read_text());lv=json.loads((root/'full_label_verification.json').read_text())
    assert lv['passed'] and lv['label_report_sha256']==sha(root/'full_label_report.json') and lr['labels_sha256']==sha(root/'full_labels.parquet')
    fields=['date','code','half','decision_shares','known_no_trade']
    for bps in [5,15]:fields += [f'{name}{bps}' for name in ['known','sensitive_known','opportunity','one_percent','any_opportunity','mark_0959_return','adverse_return']]
    f=pd.read_parquet(ROOT/'ranking.parquet');labels=pd.read_parquet(root/'full_labels.parquet',columns=fields)
    rows=f.merge(labels.drop(columns=['half','decision_shares']),on=['date','code'],validate='one_to_one')
    pd.testing.assert_frame_equal(f[['date','code','half','decision_shares']],labels[['date','code','half','decision_shares']],check_exact=True)
    assert len(rows)==1258085 and rows.date.between('2024-01-01','2025-12-31').all()
    rows=rows.loc[rows.formula_input_valid].copy()
    c=base.conn();c.register('records',rows);daily=[];summaries=[];checks=0
    for bps in [5,15]:
        for sensitive in [False,True]:
            names=[]
            for bucket in range(10):
                d=evaluation.daily_summary(rows.loc[rows.bucket.eq(bucket)],bps,sensitive)
                d['bucket']=bucket;names.append(d)
            actual=pd.concat(names,ignore_index=True).sort_values(['date','bucket']).reset_index(drop=True)
            known=f'sensitive_known{bps}' if sensitive else f'known{bps}'
            expected=c.sql(f'''WITH d AS(SELECT date,half,bucket,count(*) AS rows,
                count(*) FILTER(WHERE {known}) AS known,count(*) FILTER(WHERE {known} AND opportunity{bps}=1) AS success,
                count(*) FILTER(WHERE NOT {known} AND NOT known_no_trade) AS unknown,
                count(*) FILTER(WHERE known_no_trade) AS no_trade,
                count(*) FILTER(WHERE {known} AND one_percent{bps}=1) AS one_percent,
                count(*) FILTER(WHERE {known} AND any_opportunity{bps}=1) AS any_success,
                avg(mark_0959_return{bps}) FILTER(WHERE {known}) AS mean_reference,
                avg((mark_0959_return{bps}<0)::INT) FILTER(WHERE {known}) AS negative_reference,
                avg(adverse_return{bps}) FILTER(WHERE {known}) AS adverse_mean,
                avg((adverse_return{bps}<=-.03)::INT) FILTER(WHERE {known}) AS bad3
                FROM records GROUP BY date,half,bucket)
                SELECT *,success/nullif(known,0) AS rate,success/rows AS lower,(success+unknown)/rows AS upper,
                one_percent/nullif(known,0) AS one_percent_rate,any_success/nullif(known,0) AS any_rate
                FROM d ORDER BY date,bucket''').df()
            pd.testing.assert_frame_equal(actual[expected.columns],expected,check_dtype=False,atol=2e-12,rtol=0)
            coverage=c.sql(f'''SELECT date,bucket,count(*) FILTER(WHERE {known} AND isfinite(mark_0959_return{bps})) AS reference_known
                FROM records GROUP BY date,bucket ORDER BY date,bucket''').df()
            actual=actual.merge(coverage,on=['date','bucket'],validate='one_to_one')
            actual['bps']=bps;actual['sensitive']=sensitive;daily.append(actual);checks+=len(actual)
            for bucket in range(10):
                for period in PERIODS:
                    d=evaluation.period(actual.loc[actual.bucket.eq(bucket)],period).set_index('date')
                    calendar=evaluation.period(f,period).date.nunique()
                    s=dict(bucket=bucket,bps=bps,sensitive=sensitive,period=period,calendar_days=calendar,active_bucket_days=len(d),
                        empty_bucket_days=calendar-len(d),all_unknown_days=int((d.known+d.no_trade).eq(0).sum()))
                    s.update({k:int(d[k].sum()) for k in ['rows','known','success','unknown','no_trade','reference_known']})
                    s.update({k:evaluation.number(d[k].mean()) for k in evaluation.METRICS})
                    for field in ['rate','lower','upper']:s[field+'_ci']=weekly_interval(d[field])
                    summaries.append(s)
    auc=pandas_auc(rows)
    expected_auc=c.sql('''WITH hist AS(SELECT date,score_integer,
        count(*) FILTER(WHERE opportunity15=1) AS pos,count(*) FILTER(WHERE opportunity15=0) AS neg
        FROM records WHERE known15 GROUP BY date,score_integer),ordered AS(SELECT *,
        coalesce(sum(neg) OVER(PARTITION BY date ORDER BY score_integer ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING),0) AS preceding
        FROM hist),a AS(SELECT date,sum(pos) AS positive,sum(neg) AS negative,
        sum(pos*(preceding+.5*neg))/nullif(sum(pos)*sum(neg),0) AS auc FROM ordered GROUP BY date),
        n AS(SELECT date,count(*) AS visible,count(*) FILTER(WHERE known15) AS known FROM records GROUP BY date)
        SELECT n.*,coalesce(a.positive,0) AS positive,coalesce(a.negative,0) AS negative,auc
        FROM n LEFT JOIN a USING(date) ORDER BY date''').df();c.close()
    pd.testing.assert_frame_equal(auc[expected_auc.columns],expected_auc,check_dtype=False,rtol=0,atol=2e-12)
    auc_summaries=[]
    for period in PERIODS:
        d=auc.loc[auc.date.str.startswith(period[:4])]
        if 'H' in period:d=d.loc[d.date.str[5:7].astype(int).le(6).eq(period.endswith('H1'))]
        auc_summaries.append(dict(period=period,calendar_days=len(d),estimable_days=int(d.auc.notna().sum()),
            undefined_auc_days=int(d.auc.isna().sum()),mean_known_conditional_auc=evaluation.number(d.auc.mean()),auc_ci=weekly_interval(d.set_index('date').auc)))
    daily=pd.concat(daily,ignore_index=True)
    daily.to_parquet(ROOT/'daily_summary.parquet',index=False,compression='zstd');auc.to_parquet(ROOT/'auc_daily.parquet',index=False,compression='zstd')
    report=dict(protocol_sha256=sha(PROTOCOL),ranking_report_sha256=sha(ROOT/'ranking_report.json'),
        label_report_sha256=sha(root/'full_label_report.json'),daily_sha256=sha(ROOT/'daily_summary.parquet'),auc_daily_sha256=sha(ROOT/'auc_daily.parquet'),
        summaries=summaries,auc_summaries=auc_summaries,input_invalid_rows=int((~f.formula_input_valid).sum()),
        conditional_auc_is_not_profit_or_full_unknown_bound=True,ten_bins_not_candidate_formulas=True,
        no_binning_on_future_labels_or_fills=True,no_new_model_fit_or_prediction=True,diagnostic_only=True,
        year_2024_is_exploratory=True,year_2025_is_exploratory=True,no_publish_claim=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'analysis_report.json',report)
    save_json(ROOT/'analysis_verification.json',dict(passed=True,analysis_report_sha256=sha(ROOT/'analysis_report.json'),
        all_bucket_dates_costs_unknowns_reference_gaps_and_risk_statistics_independently_sql_verified=True,
        auc_independently_rebuilt_by_positive_negative_tie_pair_counts=True,daily_checks=checks,new_2026_prices_read=False))
    return dict(analysis_report_sha256=sha(ROOT/'analysis_report.json'),auc_summaries=auc_summaries,daily_checks=checks)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=['freeze','analyze'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
