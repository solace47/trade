"""Chronological 2024 falsification, preserving the fixed 2025 definitions."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from trade_research import tail_formula_stock_2024 as study
from trade_research.corporate_cash import save_json, sha
from find_existing_tail_formula_models import find
import run_tail_formula_paired_study as shared


def configure():
    shared.configure(study.STEM)
    shared.model.checked = study.checked
    shared.common.checked_analysis = checked_analysis
    shared.common.reference = reference


def checked_analysis(root):
    """Keep all original checks, accepting the predefined 2024 scope too."""
    f=shared.checked_selection(root)
    assert f.loc[f.selected,'date'].between('2024-01-01','2025-12-31').all()
    assert f.date.lt('2026-01-01').all()
    r=json.loads((root/'analysis_report.json').read_text())
    v=json.loads((root/'analysis_verification.json').read_text())
    assert v['passed'] and v['analysis_report_sha256']==sha(root/'analysis_report.json')
    assert r['selection_report_sha256']==sha(root/'selection_report.json')
    assert r['daily_summary_sha256']==sha(root/'daily_summary.parquet') and r['reference_label']=='09:59'
    lr=json.loads((root/'full_label_report.json').read_text())
    lv=json.loads((root/'full_label_verification.json').read_text())
    assert lv['passed'] and lv['label_report_sha256']==r['label_report_sha256']==sha(root/'full_label_report.json')
    assert lr['labels_sha256']==sha(root/'full_labels.parquet')
    return f,r


def reference(root):
    checked_analysis(root)
    path=root/'reference_coverage_verification.json'
    if path.exists():
        r=json.loads(path.read_text())
        assert r['passed'] and r['analysis_report_sha256']==sha(root/'analysis_report.json')
        assert r['selection_report_sha256']==sha(root/'selection_report.json')
        assert r['label_report_sha256']==sha(root/'full_label_report.json')
        return r
    return shared.common.audit(root,[2024])


def protocols():
    p=study.checked(); records=[]
    receipts={str(study.INPUTS/file):sha(study.INPUTS/file) for file in
              ['feature_report.json','feature_verification.json','native_input_verification.json','full_label_report.json','full_label_verification.json']}
    for fold, dates in p['folds'].items():
        for arm in study.ARMS:
            counts=p['expected_training'][fold][arm]
            q=dict(master_protocol_sha256=sha(study.MODEL_PROTOCOL),fold=fold,arm=arm,**dates,
                training_group=None if arm=='full' else int(arm[-1]),
                expected_training_rows=counts['rows'],expected_training_days=counts['days'],
                expected_last_observation=counts['last_observation'],expected_features=50,
                feature_names=list(study.EXPRESSIONS),parameters=p['parameters'],model_max_depth=3,threshold=.995,
                target='full_known_pool_day_centered' if arm=='full' else 'own_stock_group_day_centered_full_known_pool',
                projected_label_directory=str(study.model_root(fold,arm)/'labels'),input_receipts=receipts,
                no_training_period_selection=True,new_2026_prices_allowed=False,no_exit_rules=True)
            path=shared.model.fold_protocol(arm,fold);assert not path.exists()
            path.parent.mkdir(parents=True,exist_ok=True);save_json(path,q)
            lookup=find(path,variant='relative')
            assert not lookup['matches'], 'Audit and reuse exact historical models before fitting'
            records.append(dict(fold=fold,arm=arm,protocol_sha256=sha(path),lookup=lookup))
    r=dict(passed=True,master_protocol_sha256=sha(study.MODEL_PROTOCOL),records=records,
        all_six_protocols_and_exact_cache_checks_before_any_fit=True,no_2024_new_selection_evaluation=True,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(study.ROOT/'prefit_lookup_verification.json',r)
    return dict(prefit_sha256=sha(study.ROOT/'prefit_lookup_verification.json'),metadata_matches=0)


def setup(fold, arm):
    p=study.checked();root=study.model_root(fold,arm)
    shared.base.ROOT=root;shared.base.FEATURES=study.INPUTS;shared.base.SOURCE=root/'labels'
    shared.base.EXPRESSIONS=study.EXPRESSIONS;shared.base.HEADER=study.HEADER
    shared.base.PROTOCOL=shared.relative.PROTOCOL=shared.model.fold_protocol(arm,fold)
    q=json.loads(shared.base.PROTOCOL.read_text());pre=json.loads((study.ROOT/'prefit_lookup_verification.json').read_text())
    assert pre['passed'] and pre['master_protocol_sha256']==q['master_protocol_sha256']==sha(study.MODEL_PROTOCOL)
    assert any(r['fold']==fold and r['arm']==arm and r['protocol_sha256']==sha(shared.base.PROTOCOL) for r in pre['records'])
    assert q['parameters']==p['parameters'] and q['feature_names']==list(study.EXPRESSIONS)
    assert all(q[k]==v for k,v in p['folds'][fold].items())
    for file,digest in q['input_receipts'].items(): assert sha(Path(file))==digest,file
    return q


def fit():
    root=shared.base.ROOT;q=json.loads(shared.base.PROTOCOL.read_text())
    r=json.loads((shared.base.SOURCE/'full_label_report.json').read_text());v=json.loads((shared.base.SOURCE/'full_label_verification.json').read_text())
    assert v['passed'] and v['label_report_sha256']==sha(shared.base.SOURCE/'full_label_report.json')
    assert r['labels_sha256']==sha(shared.base.SOURCE/'full_labels.parquet')
    assert r['original_labels_sha256']==sha(study.INPUTS/'full_labels.parquet')
    shared.relative.model('relative')
    m=json.loads((root/'model_report.json').read_text())
    assert m['rows']==q['expected_training_rows'] and m['days']==q['expected_training_days']
    assert m['last_observation']==q['expected_last_observation']<q['evaluation_start']
    assert all(m['parameters'][k]==v for k,v in q['parameters'].items())
    m.update(arm=q['arm'],training_group=q['training_group'],historical_2024_falsification=True,
             uses_pre_2024_labels_for_training_only=True,new_2026_prices_read=False)
    save_json(root/'model_report.json',m)
    return {k:v for k,v in m.items() if k!='trees'}


def checked_scores(fold, arm):
    q=setup(fold,arm);root=shared.base.ROOT
    m=json.loads((root/'model_report.json').read_text());s=json.loads((root/'score_report.json').read_text())
    assert m['protocol_sha256']==s['protocol_sha256']==sha(shared.base.PROTOCOL)
    assert m['variant']=='relative' and m['arm']==arm and m['training_group']==q['training_group']
    assert m['feature_names']==q['feature_names'] and len(m['trees'])==64
    assert all(m['parameters'][k]==v for k,v in q['parameters'].items())
    assert m['feature_report_sha256']==s['feature_report_sha256']==sha(study.INPUTS/'feature_report.json')
    assert m['label_report_sha256']==sha(root/'labels/full_label_report.json')
    assert m['rows']==q['expected_training_rows'] and m['days']==q['expected_training_days']
    assert m['last_observation']==q['expected_last_observation']<q['evaluation_start']
    assert s['model_report_sha256']==sha(root/'model_report.json') and s['scores_sha256']==sha(root/'scores.parquet')
    for kind in ['model','score']:
        v=json.loads((root/(kind+'_verification.json')).read_text())
        assert v['passed'] and v[kind+'_report_sha256']==sha(root/(kind+'_report.json'))
    assert s['rows']==1144320 and s['valid']==1006916
    return q,m,s


def calibrate(fold):
    p=study.checked();dates=p['folds'][fold];target=study.ROOT/('calibration_'+fold+'.json')
    assert not target.exists();frames=[];receipts={};models=[]
    for arm in study.ARMS:
        q,m,s=checked_scores(fold,arm);root=study.model_root(fold,arm);models.append(m)
        d=pd.read_parquet(root/'scores.parquet')
        if arm=='full':
            full_cut=m['thresholds'][3]['threshold'];assert m['thresholds'][3]['training_quantile']==.995
        else:frames.append(d)
        for file in ['model_report.json','model_verification.json','score_report.json','score_verification.json','scores.parquet']:
            receipts[str(root/file)]=sha(root/file)
    pd.testing.assert_frame_equal(frames[0][study.META],frames[1][study.META],check_exact=True)
    mask=frames[0].date.ge(dates['training_start']) & frames[0].date.lt(dates['training_end']) & frames[0].formula_input_valid
    group=frames[0].code.str[-1].astype(int).ge(5).astype(int)
    cuts=[float(np.quantile(frames[i].loc[mask & group.eq(1-i),'score'],.995,method='linear')) for i in [0,1]]
    mean=(frames[0].score.to_numpy(float)+frames[1].score.to_numpy(float))/2
    mean_cut=float(np.quantile(mean[mask.to_numpy()],.995,method='linear'))
    c=shared.base.conn();paths=[study.model_root(fold,'group'+str(i))/'scores.parquet' for i in [0,1]]
    where=f"date>='{dates['training_start']}' AND date<'{dates['training_end']}' AND formula_input_valid"
    for i in [0,1]:
        expected=c.sql(f"SELECT quantile_cont(score,.995) AS q FROM read_parquet('{paths[i]}') WHERE {where} AND (cast(substr(code,9,1) AS INT)>=5)::INT={1-i}").df().q.iloc[0]
        np.testing.assert_allclose(cuts[i],expected,rtol=0,atol=2e-12)
    expected=c.sql(f"SELECT quantile_cont((a.score+b.score)/2,.995) AS q FROM read_parquet('{paths[0]}') a JOIN read_parquet('{paths[1]}') b USING(date,code) WHERE a.date>='{dates['training_start']}' AND a.date<'{dates['training_end']}' AND a.formula_input_valid").df().q.iloc[0];c.close()
    np.testing.assert_allclose(mean_cut,expected,rtol=0,atol=2e-12)
    out=dict(passed=True,protocol_sha256=sha(study.MODEL_PROTOCOL),fold=fold,source_hashes=receipts,
        full_training_q995=full_cut,opposite_stock_input_q995=cuts,all_visible_mean_q995=mean_cut,
        both_input_calibrations_independently_sql_rebuilt=True,no_input_calibration_label_values_read=True,
        mean_calibration_is_partly_in_sample=True,no_new_model_fit_or_prediction=True,
        no_new_group_evaluation=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(target,out);return out


def paired_native_body(models):
    cores=[shared.base.native_core(m,0,study.EXPRESSIONS,study.HEADER) for m in models]
    header=cores[0].split('T01:=',1)[0]
    assert cores[1].split('T01:=',1)[0]==header
    bodies=[]
    for group,core in enumerate(cores):
        body='T01:='+core.split('T01:=',1)[1].split('CORE:',1)[0]
        bodies.append(re.sub(r'\b(T\d{2}|SC)\b',lambda m:'H'+str(group)+m[0],body))
    return header+''.join(bodies)


def freeze():
    p=study.checked();joint_path=study.ROOT/'joint_selection_freeze.json'
    assert not joint_path.exists()
    f=pd.read_parquet(study.prior.INPUTS/'features.parquet',columns=study.META)
    keys=f[study.META[:-1]];assert len(keys)==1258085
    historical=pd.read_parquet(study.INPUTS/'features.parquet',columns=study.META)
    pd.testing.assert_frame_equal(historical.loc[historical.date.ge('2024-01-01')].reset_index(drop=True),
        f.loc[f.date.lt('2025-01-01')].reset_index(drop=True),check_exact=True)
    flags={name:[] for name in p['groups']};queries={name:[] for name in p['groups']}
    receipts={str(study.MODEL_PROTOCOL):sha(study.MODEL_PROTOCOL),
        str(study.ROOT/'prefit_lookup_verification.json'):sha(study.ROOT/'prefit_lookup_verification.json')}
    compositions=[]
    for fold,dates in p['folds'].items():
        cp=study.ROOT/('calibration_'+fold+'.json');cal=json.loads(cp.read_text())
        assert cal['passed'] and cal['protocol_sha256']==sha(study.MODEL_PROTOCOL)
        for file,digest in cal['source_hashes'].items():assert sha(Path(file))==digest,file
        receipts[str(cp)]=sha(cp);frames=[];models=[];paths=[]
        for arm in study.ARMS:
            q,m,s=checked_scores(fold,arm);root=study.model_root(fold,arm)
            frames.append(pd.read_parquet(root/'scores.parquet',filters=[('date','>=',dates['evaluation_start']),('date','<',dates['evaluation_end'])]))
            models.append(m);paths.append(root/'scores.parquet')
            for file in ['model_report.json','model_verification.json','score_report.json','score_verification.json','scores.parquet',
                         'labels/full_label_report.json','labels/full_label_verification.json']:
                receipts[str(root/file)]=sha(root/file)
            receipts[str(shared.model.fold_protocol(arm,fold))]=sha(shared.model.fold_protocol(arm,fold))
        for d in frames:
            pd.testing.assert_frame_equal(d[study.META],frames[0][study.META],check_exact=True)
        fc=cal['full_training_q995'];gc=cal['opposite_stock_input_q995'];mc=cal['all_visible_mean_q995']
        assert fc==models[0]['thresholds'][3]['threshold']
        values=[frames[0].score.gt(fc),frames[1].score.gt(gc[0]) & frames[2].score.gt(gc[1]),
                ((frames[1].score+frames[2].score)/2).gt(mc)]
        expressions=[f'a.score>{fc:.17e}',f'b.score>{gc[0]:.17e} AND c.score>{gc[1]:.17e}',
                     f'(b.score+c.score)/2>{mc:.17e}']
        body=paired_native_body(models[1:])
        native_texts=[shared.base.native_core(models[0],fc,study.EXPRESSIONS,study.HEADER).split('CORE:',1)[0]+
                      f'CORE:{study.CORE_GATE} AND SC>{fc:.17g};\n',
                      body+f'CORE:{study.CORE_GATE} AND H0SC>{gc[0]:.17g} AND H1SC>{gc[1]:.17g};\n',
                      body+f'MSC:=(H0SC+H1SC)/2;\nCORE:{study.CORE_GATE} AND MSC>{mc:.17g};\n']
        assert all('HG:=' not in t and 'SUBSTR(CODE' not in t for t in native_texts)
        for name,value,expr,txt in zip(p['groups'],values,expressions,native_texts):
            d=frames[0][['date','code']].copy();d['selected']=frames[0].formula_input_valid & value
            flags[name].append(d)
            queries[name].append(f"""SELECT a.date,a.code,coalesce(a.formula_input_valid AND ({expr}),false) AS selected
                FROM read_parquet('{paths[0]}') a JOIN read_parquet('{paths[1]}') b USING(date,code)
                JOIN read_parquet('{paths[2]}') c USING(date,code)
                WHERE a.date>='{dates['evaluation_start']}' AND a.date<'{dates['evaluation_end']}'""")
            core=study.ROOT/(fold+'_'+name+'_frozen_numeric_core.tdx');assert not core.exists()
            core.write_text(txt);receipts[str(core)]=sha(core)
        compositions.append(dict(fold=fold,full_cut=fc,group_cuts=gc,mean_cut=mc,
            exactly_three_existing_model_scores_used=True,zero_additional_mean_or_agreement_model_predictions=True,
            native_code_routing_removed=True,software_compilation_verified=False,native_source_parity_verified=False))
    c=shared.base.conn();c.register('keys',keys);selections=[];equality=[]
    for name in p['groups']:
        frame=keys.merge(pd.concat(flags[name],ignore_index=True),on=['date','code'],how='left',validate='one_to_one')
        frame['selected']=frame.selected.eq(True)
        query=' UNION ALL '.join(queries[name])
        expected=c.sql(f'SELECT k.*,coalesce(s.selected,false) AS selected FROM keys k LEFT JOIN ({query}) s USING(date,code) ORDER BY date,code').df()
        pd.testing.assert_frame_equal(frame,expected,check_exact=True)
        assert frame.loc[frame.selected,'date'].between('2024-01-01','2024-12-31').all()
        root=study.ROOT/name;root.mkdir(parents=True,exist_ok=True);assert not (root/'selection_report.json').exists()
        frame.to_parquet(root/'selection.parquet',index=False,compression='zstd')
        r=dict(protocol_sha256=sha(study.MODEL_PROTOCOL),group=name,rows=len(frame),selected=int(frame.selected.sum()),
            days=frame.loc[frame.selected,'date'].nunique(),selection_sha256=sha(root/'selection.parquet'),
            source_hashes=receipts.copy(),no_new_group_evaluation=True,new_2026_prices_read=False,no_exit_rules=True)
        save_json(root/'selection_report.json',r)
        save_json(root/'selection_verification.json',dict(passed=True,selection_report_sha256=sha(root/'selection_report.json'),
            all_full_original_keys_metadata_validity_and_fixed_flags_independently_sql_verified=True,
            no_additional_composition_prediction=True,software_compilation_verified=False,native_source_parity_verified=False,
            new_2026_prices_read=False,no_exit_rules=True))
        selections.append(dict(group=name,root=str(root),rows=len(frame),selected=r['selected'],days=r['days']))
        for other,source in p['controls'].items():
            equality.append(dict(left=name,right=other,full_frame_equal=frame.equals(shared.checked_selection(Path(source)))))
        for file in ['selection.parquet','selection_report.json','selection_verification.json']:receipts[str(root/file)]=sha(root/file)
    c.close()
    for i,a in enumerate(selections):
        for b in selections[i+1:]:
            equality.append(dict(left=a['group'],right=b['group'],full_frame_equal=shared.checked_selection(Path(a['root'])).equals(shared.checked_selection(Path(b['root'])))))
    joint=dict(passed=True,model_protocol_sha256=sha(study.MODEL_PROTOCOL),source_hashes=receipts,
        compositions=compositions,selections=selections,equality=equality,
        all_six_models_and_three_full_annual_lists_frozen_before_new_economics=True,
        old_2025_failure_gates_unchanged=True,year_2024_is_exploratory=True,
        no_new_group_evaluation=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(joint_path,joint)
    return dict(joint_sha256=sha(joint_path),compositions=compositions,selections=selections,equality=equality)


def finish():
    p,joint,ep=shared.checked_joint();roots={r['group']:Path(r['root']) for r in joint['selections']}
    for name,parent in p['controls'].items():roots[name]=shared.control_view(name,Path(parent))
    cache={};receipts={};prior_pairs=[]
    def get(root):
        root=Path(root)
        if root not in cache:cache[root]=checked_analysis(root)
        return cache[root]
    def equivalent(a,b):
        x,xr=get(a);y,yr=get(b)
        return x.equals(y) and all(xr[k]==yr[k] for k in ['summaries','daily_summary_sha256','label_report_sha256'])
    for name in ep['prior_completed_comparison_roots']:
        root=Path(name);path=root/'complete_results_manifest.json';complete=json.loads(path.read_text())
        assert complete['passed'] and ep['prior_comparison_completion_hashes'][str(path)]==sha(path)
        for file,digest in complete['source_hashes'].items():assert sha(Path(file))==digest,file
        receipts[str(path)]=sha(path)
        for pair in complete['comparisons']:
            assert all(json.loads(Path(file).read_text())['passed'] for file in pair['outputs'])
            prior_pairs.append(dict(**pair,left_root=root/pair['left'],
                right_root=root/pair['right'] if (root/pair['right']).exists() else root/'comparison_views'/pair['right']))
    for root in roots.values():
        frame,_=get(root);assert frame.loc[frame.selected,'date'].between('2024-01-01','2024-12-31').all()
        reference(root)
        for file in ['selection_report.json','selection_verification.json','analysis_report.json','analysis_verification.json',
                     'daily_summary.parquet','reference_coverage_verification.json','full_label_report.json','full_label_verification.json']:
            receipts[str(root/file)]=sha(root/file)
    done=[];aliases=[]
    for left,right in p['planned_comparisons']:
        a,b=roots[left],roots[right];pair=left+'_minus_'+right
        paths=[study.ROOT/('same_dates_'+pair+'.json'),study.ROOT/('shared_unknowns_'+pair+'.json')]
        old_pairs=[dict(**item,left_root=roots[item['left']],right_root=roots[item['right']]) for item in done]+prior_pairs
        alias=next((old for old in old_pairs if equivalent(a,old['left_root']) and equivalent(b,old['right_root'])),None)
        if alias:
            paths=[Path(f) for f in alias['outputs']]
            aliases.append(dict(left=left,right=right,reused_pair=[alias['left'],alias['right']],
                complete_frames_all_statistics_daily_and_labels_equal=True))
        else:
            if not paths[0].exists():shared.common.compare(a,b,paths[0],periods=ep['periods'],intersection_only=True)
            if not paths[1].exists():
                spec=dict(left=str(a),right=str(b),output=str(paths[1]),left_analysis_sha256=sha(a/'analysis_report.json'),
                          right_analysis_sha256=sha(b/'analysis_report.json'))
                receipt=dict(protocol_sha256=sha(shared.EVALUATION_PROTOCOL),labels_sha256=sha(shared.common.evaluation.source.ROOT/'full_labels.parquet'),
                             periods=ep['periods'],signal_range=['2024-01-01','2024-12-31'])
                shared.common.shared(spec,receipt)
        for file in paths:assert json.loads(file.read_text())['passed'];receipts[str(file)]=sha(file)
        done.append(dict(left=left,right=right,outputs=[str(f) for f in paths],reused=alias is not None))
        print(json.dumps(dict(compared=pair)),flush=True)
    def main(name,period='2024'):
        return next(s for s in get(roots[name])[1]['summaries'] if s['arm']=='formula' and s['bps']==15 and not s['sensitive'] and s['period']==period)
    gates={}
    for name in ['agreement2024','mean2024']:
        new,old=main(name),main('morning2024')
        pair=next(r for r in done if r['left']==name and r['right']=='morning2024')
        s=next(s for s in json.loads(Path(pair['outputs'][1]).read_text())['summaries'] if s['period']=='2024' and s['bps']==15 and not s['sensitive'])
        criteria=dict(both_half_reference_positive=all((main(name,h)['mean_reference'] or -1)>0 for h in ['2024H1','2024H2']),
            annual_opportunity_above_same_50=new['rate'] is not None and old['rate'] is not None and new['rate']>old['rate'],
            annual_bad3_not_above_same_50=new['bad3'] is not None and old['bad3'] is not None and new['bad3']<=old['bad3'],
            shared_unknown_lower_ci_strict_positive=s['lower_ci'] is not None and s['lower_ci'][0]>0)
        gates[name]=dict(criteria=criteria,survives_2024_falsification_screen=all(criteria.values()))
    gate=dict(passed=True,protocol_sha256=sha(shared.EVALUATION_PROTOCOL),selectors=gates,
        old_2025_failure_gates_unchanged=True,no_automatic_2026_evaluation=True,no_publish_claim=True,new_2026_prices_read=False,no_exit_rules=True)
    gatefile=study.ROOT/'stock_2024_gate.json';save_json(gatefile,gate);receipts[str(gatefile)]=sha(gatefile)
    dispatch=study.ROOT/'analysis_dispatch_verification.json';receipts[str(dispatch)]=sha(dispatch)
    out=dict(passed=True,protocol_sha256=sha(shared.EVALUATION_PROTOCOL),joint_sha256=sha(study.ROOT/'joint_selection_freeze.json'),
        source_hashes=receipts,comparisons=done,comparison_reuse=aliases,
        all_three_groups_five_predefined_comparisons_and_reference_gaps_preserved=True,
        no_duplicate_half_year_aggregation=True,year_2024_is_exploratory=True,old_2025_failure_gates_unchanged=True,
        no_new_2026_economics=True,no_publish_claim=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(study.ROOT/'complete_results_manifest.json',out)
    return dict(completion_sha256=sha(study.ROOT/'complete_results_manifest.json'),gate=gate,
                new_numerical_comparisons=sum(not r['reused'] for r in done))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['protocols','project','model','verify_model','scores','verify_scores','calibrate','freeze','analyze','finish'])
    parser.add_argument('--fold',choices=['2024h1','2024h2'])
    parser.add_argument('--arm',choices=list(study.ARMS))
    args=parser.parse_args();configure()
    if args.stage in ['protocols','freeze','finish']:result=globals()[args.stage]()
    elif args.stage=='analyze':result=shared.analyze()
    elif args.stage=='calibrate':assert args.fold;result=calibrate(args.fold)
    else:
        assert args.fold and args.arm;setup(args.fold,args.arm)
        if args.stage=='project':result=study.project_labels(args.fold,args.arm)
        elif args.stage=='model':result=fit()
        elif args.stage=='verify_model':result=shared.relative.verify_model('relative')
        elif args.stage=='scores':result=shared.base.scores()
        else:result=shared.verify_scores(expected_expressions=study.EXPRESSIONS)
    print(json.dumps(result,ensure_ascii=False,indent=2))
