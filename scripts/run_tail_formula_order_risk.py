"""One fixed risk-weighted opportunity study, with reusable frozen controls."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as numeric
from trade_research import tail_formula_relative as relative
from trade_research import tail_formula_baseline as original
from trade_research.research_io import check_runtime, check_sources, sha, save_json
from tail_formula_reports import checked_selection
from verify_tail_formula_additive import tree_sql
from find_existing_tail_formula_models import find, confirmed_target, verification_registry
from analyze_tail_formula_order_target import aggregate, summarize, paired

ROOT=Path('data/research/tail_formula_order_risk')
PROTOCOL=Path('config/tail_formula_order_risk.json')
KEYS=['date','code','half','board','decision_shares']


def checked():
    check_runtime();p=json.loads(PROTOCOL.read_text())
    assert subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}'])==PROTOCOL.read_bytes()
    assert sha(PROTOCOL) in subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    check_sources(p['source_hashes'])
    return p


def committed_receipt(name):
    path=ROOT/name;r=json.loads(path.read_text())
    assert r['passed'] and sha(path) in subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    check_sources(r['source_hashes'])
    return r


def centered(path,start,end):
    c=numeric.conn()
    out=c.execute('''SELECT date,code,opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target
        FROM read_parquet(?) WHERE known15 AND date>=? AND next_date<? ORDER BY date,code''',[str(path),start,end]).df()
    c.close();return out


def prepare(p):
    assert not (ROOT/'preparation.json').exists()
    t=pd.read_parquet(p['targets']);f=pd.read_parquet(Path(p['features_root'])/'features.parquet')
    pd.testing.assert_frame_equal(t[KEYS],f[KEYS],check_exact=True)
    assert len(f)==1815129 and int(f.formula_input_valid.sum())==1602413 and f.date.lt('2026-01-01').all()
    se=t.first_space_end15;be=t.first_bad315
    win=se.ge(0)&(be.lt(0)|se.lt(be));bad=be.ge(0)&(se.lt(0)|be.le(se))
    assert not (win&bad).any()
    utility=(win.astype(float)-3*bad.astype(float)).where(t.known15)
    out=t[[*KEYS,'next_date','known15','known_no_trade']].copy();out['opportunity15']=utility
    c=numeric.conn();c.register('events',t)
    ex=c.sql('''SELECT date,code,half,board,decision_shares,next_date,known15,known_no_trade,
        CASE WHEN NOT known15 THEN NULL WHEN first_space_end15>=0 AND (first_bad315<0 OR first_space_end15<first_bad315) THEN 1.
        WHEN first_bad315>=0 AND (first_space_end15<0 OR first_bad315<=first_space_end15) THEN -3. ELSE 0. END AS opportunity15
        FROM events ORDER BY date,code''').df()
    pd.testing.assert_frame_equal(out,ex,check_dtype=False,rtol=0,atol=0);c.close()
    folder=ROOT/'training_labels';folder.mkdir(exist_ok=False)
    out.to_parquet(folder/'full_labels.parquet',index=False,compression='zstd')
    save_json(folder/'full_label_report.json',dict(labels_sha256=sha(folder/'full_labels.parquet'),rows=len(out),
        training_only_utility_alias_not_economic_opportunity=True,utility_values=[-3,0,1],original_states_and_metadata_unchanged=True))
    save_json(folder/'full_label_verification.json',dict(passed=True,label_report_sha256=sha(folder/'full_label_report.json'),
        every_position_based_utility_and_unknown_SQL_verified=True,source_events_previously_raw_cash_verified=True))
    reports={};feature_reports={}
    files=subprocess.check_output(['rg','--files','--no-ignore','-g','full_label_report.json','-g','feature_report.json','data/research'],text=True).splitlines()
    for file in files:
        if any('2026' in part or 'q1' in part.lower() for part in Path(file).parts):continue
        mapping=feature_reports if Path(file).name=='feature_report.json' else reports
        mapping.setdefault(sha(Path(file)),[]).append(Path(file))
    registry=verification_registry();lookup=[];receipts=dict(p['source_hashes']);configs=[]
    cfgdir=ROOT/'model_configs';cfgdir.mkdir(exist_ok=False)
    for spec in p['folds']:
        config=json.loads((Path(p['prior_models'])/'model_configs'/('ordered_'+spec['id']+'.json')).read_text())
        config.update(master_input_protocol_sha256=sha(PROTOCOL),utility='space_first_minus_three_bad_first',
            training_event='one_percent_before_three_percent_else_prior_risk_penalty',training_allowed_utility_values=[-3,0,1],parameters=p['fit_parameters'])
        path=cfgdir/(spec['id']+'.json');save_json(path,config);configs.append(dict(fold=spec['id'],config=str(path)))
        new=centered(folder/'full_labels.parquet',spec['training_start'],spec['training_end'])
        new_training=f.loc[f.formula_input_valid,['date','code']].merge(new,on=['date','code'],validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
        query=dict(config,expected_training_rows=len(new_training))
        query_path=cfgdir/(spec['id']+'_lookup.json');save_json(query_path,query)
        result=find(query_path,variant='relative');comparisons=[]
        for item in result['matches']:
            model_root=Path(item['root']);variant=confirmed_target(model_root,registry,receipts)
            assert variant=='relative'
            candidates=reports.get(item['label_report_sha256'],[])
            source=next((q for q in candidates if (q.parent/'full_labels.parquet').exists()),None)
            assert source is not None,'Resolve the prior training label before fitting: '+str(model_root)
            report=json.loads(source.read_text());oldpath=source.parent/'full_labels.parquet'
            assert sha(oldpath)==report['labels_sha256']
            old=centered(oldpath,spec['training_start'],spec['training_end'])
            feature_source=next((q for q in feature_reports.get(item['feature_report_sha256'],[]) if (q.parent/'features.parquet').exists()),None)
            assert feature_source is not None,'Resolve the prior valid-input domain before fitting: '+str(model_root)
            feature_report=json.loads(feature_source.read_text());oldfeatures=feature_source.parent/'features.parquet'
            assert sha(oldfeatures)==feature_report['features_sha256']
            visible=pd.read_parquet(oldfeatures,columns=['date','code','formula_input_valid'],filters=[('date','>=',spec['training_start']),('date','<',spec['training_end'])])
            old_training=visible.loc[visible.formula_input_valid,['date','code']].merge(old,on=['date','code'],validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
            assert len(old_training)==item['rows']
            # Target inequality alone excludes exact fit reuse, regardless of X.
            same_keys=new_training[['date','code']].equals(old_training[['date','code']])
            same_target=same_keys and np.array_equal(new_training.target.to_numpy(),old_training.target.to_numpy())
            assert not same_target,'An exact target match needs input/weight comparison and model reuse'
            comparisons.append(dict(root=str(model_root),same_mature_source_keys=same_keys,centered_targets_equal=False))
            receipts[str(source)]=sha(source);receipts[str(oldpath)]=sha(oldpath)
            receipts[str(feature_source)]=sha(feature_source);receipts[str(oldfeatures)]=sha(oldfeatures)
        lookup.append(dict(fold=spec['id'],metadata_lookup=result,target_comparisons=comparisons))
    for path in list(folder.iterdir())+list(cfgdir.iterdir()):receipts[str(path)]=sha(path)
    save_json(ROOT/'model_lookup.json',dict(passed=True,records=lookup,source_hashes=receipts,no_new_fits=True))
    receipts[str(ROOT/'model_lookup.json')]=sha(ROOT/'model_lookup.json')
    save_json(ROOT/'preparation.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),source_hashes=receipts,models=configs,
        all_keys_and_original_validity_preserved=True,all_utilities_SQL_verified=True,maximum_new_fits=4,new_fits=0,new_2026_prices_read=False))
    print(json.dumps(dict(preparation_sha256=sha(ROOT/'preparation.json'),candidate_models=[len(x['target_comparisons']) for x in lookup])),flush=True)


def setup(item,p):
    numeric.FEATURES=Path(p['features_root']);numeric.SOURCE=ROOT/'training_labels';numeric.ROOT=ROOT/'models'/item['fold']
    numeric.PROTOCOL=relative.PROTOCOL=Path(item['config']);numeric.EXPRESSIONS=original.EXPRESSIONS;numeric.QUANTILES=[.995]


def fit(p):
    prep=committed_receipt('preparation.json');assert not (ROOT/'all_models_verified.json').exists()
    records=[];receipts=dict(prep['source_hashes']);fits=0
    for item in prep['models']:
        setup(item,p);folder=numeric.ROOT
        if (folder/'model_report.json').exists():
            r=json.loads((folder/'model_report.json').read_text());assert r['protocol_sha256']==sha(Path(item['config']))
            print(json.dumps(dict(resume_saved_model=item['fold'],no_refit=True)),flush=True)
        else:
            assert not folder.exists(),'Inspect the interrupted fit folder before proceeding'
            print(json.dumps(dict(fitting=item['fold'],completed_new_fits=fits)),flush=True)
            relative.model('relative');fits+=1
        relative.verify_model('relative')
        m=json.loads((folder/'model_report.json').read_text())
        assert m['training_allowed_utility_values']==[-3,0,1] and m['last_observation']<m['training_end']
        for name in ['model_report.json','model_verification.json']:receipts[str(folder/name)]=sha(folder/name)
        records.append(dict(fold=item['fold'],root=str(folder)))
        print(json.dumps(dict(model_verified=item['fold'],rows=m['rows'],last_observation=m['last_observation'])),flush=True)
    save_json(ROOT/'all_models_verified.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),models=records,source_hashes=receipts,fits_completed=4,new_fits_this_invocation=fits,new_economic_groups_read=False))
    print(json.dumps(dict(models_sha256=sha(ROOT/'all_models_verified.json'))),flush=True)


def freeze(p):
    models=committed_receipt('all_models_verified.json');assert not (ROOT/'joint_selection_freeze.json').exists()
    f=pd.read_parquet(Path(p['features_root'])/'features.parquet').loc[lambda z:z.date.ge('2024-01-01')].reset_index(drop=True)
    flags=np.zeros(len(f),dtype=bool);specs={s['id']:s for s in p['folds']};checks=[];receipts=dict(models['source_hashes'])
    for item in models['models']:
        m=json.loads((Path(item['root'])/'model_report.json').read_text());spec=specs[item['fold']]
        mask=f.date.ge(spec['evaluation_start'])&f.date.lt(spec['evaluation_end']);d=f.loc[mask].reset_index(drop=True);valid=d.formula_input_valid
        x=numeric.encode(d.loc[valid]);scores=np.full(len(d),np.nan);scores[valid]=numeric.predict(x,m);cut=m['thresholds'][0]
        assert len(m['thresholds'])==1 and cut['training_quantile']==.995 and m['last_observation']<spec['evaluation_start']
        c=numeric.conn();c.register('visible',d[['date','code','formula_input_valid',*original.EXPRESSIONS]])
        enc=','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS X{i:02d}' for i,n in enumerate(original.EXPRESSIONS,1))
        c.sql('SELECT date,code,'+enc+' FROM visible WHERE formula_input_valid').create_view('encoded')
        equation=format(m['bias'],'.17e')+'+'+'+'.join(tree_sql(t) for t in m['trees'])
        c.sql('SELECT date,code,'+equation+' AS score FROM encoded').create_view('rebuilt')
        ex=c.sql('SELECT v.date,v.code,r.score FROM visible v LEFT JOIN rebuilt r USING(date,code) ORDER BY v.date,v.code').df();c.close()
        pd.testing.assert_frame_equal(d[['date','code']],ex[['date','code']],check_exact=True)
        np.testing.assert_allclose(scores,ex.score,rtol=0,atol=2e-11,equal_nan=True)
        chosen=scores>cut['threshold'];np.testing.assert_array_equal(chosen,ex.score.gt(cut['threshold']));flags[mask]=chosen
        checks.append(dict(fold=item['fold'],all_scores_and_flags_SQL_verified=True,selected=int(chosen.sum())))
    registry=json.loads(Path(p['prior_registry']).read_text());check_sources(registry['source_hashes']);lookup=[];lists=[]
    previous=[v['root'] for v in registry['reports']]+[p['controls'][a][y] for a in ['ordered','whole_window'] for y in ['2024','2025']]
    for year in ['2024','2025']:
        folder=ROOT/('risk'+year);folder.mkdir(exist_ok=False)
        out=f[KEYS].copy();out['selected']=flags&f.date.str.startswith(year)
        out.to_parquet(folder/'selection.parquet',index=False,compression='zstd');chosen=out.loc[out.selected];sizes=chosen.groupby('date').size();aliases=[]
        for path in previous:
            report=json.loads((Path(path)/'selection_report.json').read_text())
            if report['selected']==len(chosen) and checked_selection(Path(path)).equals(out):aliases.append(path)
        save_json(folder/'selection_report.json',dict(protocol_sha256=sha(PROTOCOL),selection_sha256=sha(folder/'selection.parquet'),rows=len(out),selected=len(chosen),days=len(sizes),half_counts=chosen.groupby('half').agg(rows=('code','size'),days=('date','nunique')).reset_index().to_dict('records'),median_daily=float(sizes.median()) if len(sizes) else None,max_daily=int(sizes.max()) if len(sizes) else None,largest_day_fraction=float(sizes.max()/len(chosen)) if len(chosen) else None,no_outcome_or_fill_filter=True,new_2026_prices_read=False))
        save_json(folder/'selection_verification.json',dict(passed=True,selection_report_sha256=sha(folder/'selection_report.json'),all_scores_flags_scopes_and_metadata_SQL_verified=True,native_client_parity_verified=False))
        lookup.append(dict(year=year,equivalent_complete_lists=aliases,prior_complete_lists_examined=len(previous)))
        lists.append(dict(group='risk'+year,year=year,root=str(folder)))
        for path in folder.iterdir():receipts[str(path)]=sha(path)
    save_json(ROOT/'joint_selection_freeze.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),source_hashes=receipts,selections=lists,score_checks=checks,equivalence_lookup=lookup,all_models_and_both_annual_lists_before_economics=True,new_economic_groups_read=False,new_2026_prices_read=False))
    print(json.dumps(dict(joint_sha256=sha(ROOT/'joint_selection_freeze.json'),equivalence=lookup)),flush=True)


def analyze(p):
    joint=committed_receipt('joint_selection_freeze.json');assert not (ROOT/'economic_report.json').exists()
    labels=pd.read_parquet(p['ordered_labels']);refs=pd.read_parquet(p['original_labels'],columns=['date','code','mark_0959_return5','mark_0959_return15'])
    labels=labels.merge(refs,on=['date','code'],validate='one_to_one');assert len(labels)==1258085
    old=json.loads(Path(p['prior_economic_report']).read_text());old_daily=pd.read_parquet(Path(p['prior_economic_report']).parent/'economic_daily.parquet')
    frames={a+y:checked_selection(Path(p['controls'][a][y])) for a in p['controls'] for y in ['2024','2025']}
    summaries=[];daily=[];comparisons=[];pairdays=[];reuse=[]
    for item in joint['selections']:
        group=item['group'];year=item['year'];frame=checked_selection(Path(item['root']));frames[group]=frame
        alias=next((a+year for a in p['controls'] if frames[a+year].equals(frame)),None)
        if alias:
            summaries.extend([{**s,'group':group} for s in old['summaries'] if s['group']==alias]);d=old_daily.loc[old_daily.group.eq(alias)].copy();d['group']=group;daily.append(d);reuse.append(dict(group=group,source=alias,exact_complete_selection_and_labels=True))
        else:
            q=labels.merge(frame,on=KEYS,validate='one_to_one');chosen=q.loc[q.selected];base=q.loc[q.date.isin(chosen.date.unique())]
            for arm,part in [('formula',chosen),('base_same_dates',base)]:
                for bps in [5,15]:
                    for sensitive in [False,True]:
                        d=aggregate(part,bps,sensitive);summaries.extend(summarize(d,year,group,arm,bps,sensitive));d['group']=group;d['arm']=arm;d['bps']=bps;d['sensitive']=sensitive;daily.append(d)
        for control in p['controls']:
            prior=next((v for v in old['comparisons'] if alias==v['left'] and v['right']==control+year),None)
            if prior:
                result=json.loads(json.dumps(prior));result['left']=group
                for s in result['summaries']:s['left']=group
                comparisons.append(result);reuse.append(dict(left=group,right=control+year,prior_paired_result_exact=True))
            else:
                result,parts=paired(frame,frames[control+year],labels,year,group,control+year);comparisons.append(result);pairdays.extend(parts)
        print(json.dumps(dict(economic_group_completed=group)),flush=True)
    pd.concat(daily,ignore_index=True).to_parquet(ROOT/'economic_daily.parquet',index=False,compression='zstd')
    if pairdays:pd.concat(pairdays,ignore_index=True).to_parquet(ROOT/'paired_daily.parquet',index=False,compression='zstd')
    def get(group,period):
        records=summaries if group.startswith('risk') else old['summaries']
        return next(s for s in records if s['group']==group and s['period']==period and s['arm']=='formula' and s['bps']==15 and not s['sensitive'])
    quality=[]
    for year in ['2024','2025']:
        a=get('risk'+year,year);controls=[get(v+year,year) for v in p['controls']]
        quality.append(dict(year=year,rates_above_all_controls=all(a[n] is not None and b[n] is not None and a[n]>b[n] for b in controls for n in ['positive_before_rate','space_before_rate']),risk_not_above_any_control=all(a['bad_first_rate'] is not None and b['bad_first_rate'] is not None and a['bad_first_rate']<=b['bad_first_rate'] for b in controls),coupled_lower_intervals_positive=all(s['lower_ci'] is not None and s['lower_ci'][0]>0 for c in comparisons if c['year']==year for s in c['summaries'] if s['period']==year and s['bps']==15 and not s['sensitive'] and s['target']=='positive_before')))
    criteria=dict(four_half_coverage=all(get('risk'+y,y+h)['days']>=20 and get('risk'+y,y+h)['known']>=100 for y in ['2024','2025'] for h in ['H1','H2']),both_year_rates_improved=all(v['rates_above_all_controls'] for v in quality),both_year_coupled_intervals_positive=all(v['coupled_lower_intervals_positive'] for v in quality),both_year_prior_risk_not_worse=all(v['risk_not_above_any_control'] for v in quality),both_year_daily_median_at_most_five=all((lambda r:r['median_daily'] is not None and r['median_daily']<=5)(json.loads((ROOT/('risk'+y)/'selection_report.json').read_text())) for y in ['2024','2025']))
    sources=dict(joint['source_hashes'])
    for path in [ROOT/'economic_daily.parquet',ROOT/'paired_daily.parquet']:
        if path.exists():sources[str(path)]=sha(path)
    save_json(ROOT/'economic_report.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),source_hashes=sources,summaries=summaries,comparisons=comparisons,quality_checks=quality,criteria=criteria,exact_reuse=reuse,all_new_statistics_and_shared_unknown_bounds_SQL_verified=True,not_realized_profit=True,new_2026_prices_read=False,no_exit_rules=True))
    print(json.dumps(dict(report_sha256=sha(ROOT/'economic_report.json'),criteria=criteria)),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('stage',choices=['prepare','fit','freeze','analyze']);args=a.parse_args()
    globals()[args.stage](checked())
