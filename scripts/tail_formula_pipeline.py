"""Current four-fold pipeline with exact controls and complete-list result reuse."""
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from trade_research import tail_formula_intraday_scale as study
from trade_research import tail_formula_additive as base
from trade_research import tail_formula_relative as relative
from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research.research_io import save_json, sha, check_sources, RUNTIME
from verify_tail_formula_additive import verify_scores
from tail_formula_reports import checked_selection
from audit_tail_formula_reference_coverage import audit
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared_compare
from find_existing_tail_formula_models import find
import tail_formula_reports as market
import tail_formula_analysis_reuse as reuse_common

EXECUTION = Path('config') / (study.STEM + '_model_protocol.json')


ARMS = {'control': study.CONTROL, 'memory': study.EXPRESSIONS}


KEYS = study.META[:-1]


def changed_nodes(model):
    positions = {i for i,name in enumerate(model['feature_names']) if name not in study.CONTROL}
    return sum(v in positions for tree in model['trees'] for v in tree['feature'])


def checked():
    p = study.checked(); e = json.loads(EXECUTION.read_text())
    assert e['input_protocol_sha256'] == sha(study.PROTOCOL)
    check_sources(e['source_hashes'])
    patch=Path('config')/(study.STEM+'_verification_patch.json')
    if patch.exists():
        v=json.loads(patch.read_text())
        assert subprocess.check_output(['git','show',f'HEAD:{patch}'])==patch.read_bytes()
        assert v['input_protocol_sha256']==sha(study.PROTOCOL) and v['model_protocol_sha256']==sha(EXECUTION)
        check_sources(v['source_hashes']);check_sources(v['checkpoint_receipts'])
        assert v['no_feature_model_score_selection_or_economic_change'] and v['new_fits_performed']==0
    return p, e


def protocols():
    p, _ = checked(); assert not (study.ROOT / 'prefit_lookup_verification.json').exists()
    receipts = {str(study.INPUTS / f): sha(study.INPUTS / f) for f in
                ['feature_report.json','feature_verification.json','native_input_verification.json',
                 'full_label_report.json','full_label_verification.json']}
    for name in ['feature','native_input']:
        v = json.loads((study.INPUTS / (name+'_verification.json')).read_text())
        assert v['passed'] and v['feature_report_sha256'] == sha(study.INPUTS / 'feature_report.json')
    base.FEATURES = base.SOURCE = study.INPUTS
    counts = {}
    for fold, spec in p['folds'].items():
        t = base.training(start=spec['training_start'], end=spec['training_end'])
        assert t.next_date.max() < spec['evaluation_start']
        counts[fold] = dict(rows=len(t),days=t.date.nunique(),last_observation=t.next_date.max())
    records = []
    for arm, expressions in ARMS.items():
        for fold, spec in p['folds'].items():
            q = dict(master_protocol_sha256=sha(study.PROTOCOL), execution_protocol_sha256=sha(EXECUTION),
                arm=arm,fold=fold,**spec,expected_training_rows=counts[fold]['rows'],
                expected_training_days=counts[fold]['days'],expected_last_observation=counts[fold]['last_observation'],
                expected_features=len(expressions),feature_names=list(expressions),parameters=p['parameters'],
                model_max_depth=3,threshold=.995,target='relative',input_receipts=receipts,
                no_training_period_selection=True,new_2026_prices_allowed=False,no_exit_rules=True,cleanup_runtime_sha256=sha(RUNTIME))
            file = Path('config') / study.STEM / (arm+'_'+fold+'.json')
            assert not file.exists(); file.parent.mkdir(parents=True, exist_ok=True); save_json(file,q)
            records.append(dict(arm=arm,fold=fold,protocol_sha256=sha(file),lookup=find(file,'relative')))
            if arm == 'control':
                assert any(c['root'] == p['old_model_roots'][fold] for c in records[-1]['lookup']['matches'])
    assert not any(r['lookup']['matches'] for r in records if r['arm']=='memory'), 'Audit equivalent candidates before fitting'
    save_json(study.ROOT / 'prefit_lookup_verification.json',dict(passed=True,
        master_protocol_sha256=sha(study.PROTOCOL),execution_protocol_sha256=sha(EXECUTION),
        counts=counts,records=records,all_eight_protocols_before_any_fit=True,
        maximum_new_fits=4,exact_control_reuse_first=True,new_group_outcomes_read=False,
        new_2026_prices_read=False,no_exit_rules=True))
    return dict(counts=counts,existing_control_candidates={r['fold']:len(r['lookup']['matches'])
        for r in records if r['arm']=='control'},prefit_sha256=sha(study.ROOT / 'prefit_lookup_verification.json'))


def setup(arm,fold):
    p, e = checked(); assert arm in ARMS and fold in p['folds']
    base.ROOT = study.ROOT / arm / fold; base.FEATURES = base.SOURCE = study.INPUTS
    base.EXPRESSIONS = ARMS[arm]; base.HEADER = study.HEADER
    base.PROTOCOL = relative.PROTOCOL = Path('config') / study.STEM / (arm+'_'+fold+'.json')
    q = json.loads(base.PROTOCOL.read_text())
    assert q['master_protocol_sha256'] == sha(study.PROTOCOL) and q['execution_protocol_sha256'] == sha(EXECUTION)
    runtime=json.loads(RUNTIME.read_text())
    assert q['cleanup_runtime_sha256']==runtime.get('fixed_fit_runtime_sha256',sha(RUNTIME))
    assert q['arm']==arm and q['fold']==fold and q['feature_names']==list(ARMS[arm])
    assert q['parameters']==p['parameters'] and all(q[k]==v for k,v in p['folds'][fold].items())
    for file,digest in q['input_receipts'].items(): assert sha(Path(file))==digest,file
    pre = json.loads((study.ROOT / 'prefit_lookup_verification.json').read_text())
    assert pre['passed'] and pre['master_protocol_sha256']==sha(study.PROTOCOL)
    return p, q, next(r for r in pre['records'] if r['arm']==arm and r['fold']==fold)


def reuse_control(p,q,record):
    """Reuse only after full training values, target and day weights agree."""
    if not record['lookup']['matches']:
        return False
    parent = Path(p['old_model_roots'][q['fold']])
    assert any(r['root']==str(parent) for r in record['lookup']['matches']), 'Audit nonstandard equivalent controls'
    m = json.loads((parent / 'model_report.json').read_text())
    for kind in ['model','score']:
        v = json.loads((parent / (kind+'_verification.json')).read_text())
        assert v['passed'] and v[kind+'_report_sha256']==sha(parent / (kind+'_report.json'))
    old_input = Path('data/research/tail_formula_stock_2024/inputs') if q['fold'].startswith('2024') else study.prior.INPUTS
    old = pd.read_parquet(old_input / 'features.parquet',columns=[*study.META,*study.CONTROL])
    fresh = relative.training('relative')
    c = base.conn(); c.register('old_features',old)
    columns = ','.join('f.'+n for n in study.CONTROL)
    prior = c.sql(f'''WITH labels AS(SELECT date,code,next_date,opportunity15,
        opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target
        FROM read_parquet('{old_input}/full_labels.parquet') WHERE known15
          AND date>='{q['training_start']}' AND next_date<'{q['training_end']}')
        SELECT date,code,next_date,opportunity15,target,1./count(*) OVER(PARTITION BY date) AS w,{columns}
        FROM old_features f JOIN labels USING(date,code) WHERE formula_input_valid ORDER BY date,code''').df()
    c.close()
    names = ['date','code','next_date','opportunity15',*study.CONTROL]
    pd.testing.assert_frame_equal(fresh[names],prior[names],check_exact=True,check_dtype=False)
    np.testing.assert_allclose(fresh.target,prior.target,rtol=0,atol=2e-12)
    np.testing.assert_array_equal(1/fresh.groupby('date').code.transform('size'),prior.w)
    assert len(fresh)==m['rows']==q['expected_training_rows']
    assert fresh.date.nunique()==m['days']==q['expected_training_days']
    assert fresh.next_date.max()==m['last_observation']==q['expected_last_observation']
    for name in ['feature_names','training_start','training_end']:
        assert m[name]==q[name]
    assert all(m['parameters'][k]==v for k,v in p['parameters'].items())
    base.ROOT.mkdir(parents=True,exist_ok=True)
    updated = dict(m,protocol_sha256=sha(base.PROTOCOL),feature_report_sha256=sha(study.INPUTS / 'feature_report.json'),
        label_report_sha256=sha(study.INPUTS / 'full_label_report.json'),no_model_fit_performed=True,
        reused_source_root=str(parent),reused_model_report_sha256=sha(parent / 'model_report.json'))
    for name in ['trees','bias','parameters','thresholds']:
        assert updated[name]==m[name]
    save_json(base.ROOT / 'model_report.json',updated)
    save_json(base.ROOT / 'control_reuse_verification.json',dict(passed=True,
        source_hashes={str(parent/f):sha(parent/f) for f in ['model_report.json','model_verification.json']},
        all_training_values_target_and_date_weights_equal=True,no_fit_performed=True,
        evaluation_scores_not_used_to_decide_reuse=True,new_2026_prices_read=False))
    return True


def fit(arm,fold):
    p,q,record = setup(arm,fold)
    assert not (base.ROOT / 'model_report.json').exists()
    reused = arm=='control' and reuse_control(p,q,record)
    if arm == 'control':
        assert reused, 'The frozen protocol permits no new control fits'
    if not reused:
        assert not record['lookup']['matches'], 'Do not repeat an unaudited equivalent fit'
        relative.model('relative')
    m = json.loads((base.ROOT / 'model_report.json').read_text())
    assert m['rows']==q['expected_training_rows'] and m['days']==q['expected_training_days']
    assert m['last_observation']==q['expected_last_observation']<q['evaluation_start']
    assert m['feature_names']==list(ARMS[arm])
    assert all(m['parameters'][k]==v for k,v in p['parameters'].items())
    relative.verify_model('relative')
    base.scores(); verify_scores(expected_expressions=ARMS[arm],definition_protocol=study.PROTOCOL)
    return dict(arm=arm,fold=fold,reused=reused,rows=m['rows'],days=m['days'],last_observation=m['last_observation'],
        changed_input_nodes=changed_nodes(m) if arm=='memory' else 0)


def freeze():
    p,_ = checked(); joint = study.ROOT / 'joint_selection_freeze.json'; assert not joint.exists()
    f = pd.read_parquet(study.INPUTS / 'features.parquet',columns=study.META)
    keys = f.loc[f.date.ge('2024-01-01'),KEYS].reset_index(drop=True)
    assert len(keys)==1258085
    receipts = {str(EXECUTION):sha(EXECUTION)}; models = []; flags = {}
    for arm in ARMS:
        for year in ['2024','2025']: flags[(arm,year)] = np.zeros(len(keys),bool)
        for fold in p['folds']:
            _,q,_ = setup(arm,fold); root = base.ROOT
            m = json.loads((root / 'model_report.json').read_text())
            assert m['protocol_sha256']==sha(base.PROTOCOL) and m['last_observation']<q['evaluation_start']
            assert m['feature_names']==q['feature_names'] and m['rows']==q['expected_training_rows']
            for kind in ['model','score']:
                r = json.loads((root / (kind+'_report.json')).read_text())
                v = json.loads((root / (kind+'_verification.json')).read_text())
                assert v['passed'] and v[kind+'_report_sha256']==sha(root / (kind+'_report.json'))
                assert r['protocol_sha256']==sha(base.PROTOCOL)
            assert r['scores_sha256']==sha(root / 'scores.parquet')
            d = pd.read_parquet(root / 'scores.parquet',filters=[('date','>=',q['evaluation_start']),('date','<',q['evaluation_end'])])
            mask = keys.date.ge(q['evaluation_start']) & keys.date.lt(q['evaluation_end'])
            pd.testing.assert_frame_equal(d[KEYS].reset_index(drop=True),keys.loc[mask].reset_index(drop=True),check_exact=True)
            cut = m['thresholds'][3]; assert cut['training_quantile']==.995
            values = d.formula_input_valid & d.score.gt(cut['threshold'])
            c = base.conn(); c.register('d',d)
            expected = c.sql(f'SELECT coalesce(formula_input_valid AND score>{cut["threshold"]:.17e},false) AS flag FROM d').df()
            c.close(); np.testing.assert_array_equal(values,expected.flag)
            flags[(arm,fold[:4])][mask] = values.to_numpy()
            core = root / 'frozen_numeric_core.tdx'; assert not core.exists()
            text = base.native_core(m,cut['threshold'],ARMS[arm],study.HEADER)
            assert text.count('CORE:SC>')==1
            if arm == 'memory':
                text = text.replace('CORE:SC>', 'CORE:'+study.NATIVE_GATE+' AND SC>')
            core.write_text(text)
            models.append(dict(arm=arm,fold=fold,new_fit=not m.get('no_model_fit_performed',False),
                rows=m['rows'],days=m['days'],last_observation=m['last_observation'],
                selected=int(values.sum()),signal_days=d.loc[values,'date'].nunique(),
                changed_input_nodes=changed_nodes(m) if arm=='memory' else 0))
            for file in ['model_report.json','model_verification.json','score_report.json','score_verification.json','scores.parquet','frozen_numeric_core.tdx']:
                receipts[str(root / file)] = sha(root / file)
            receipts[str(base.PROTOCOL)] = sha(base.PROTOCOL)
    selections = []
    for (arm,year),flag in flags.items():
        group = arm+year; root = study.ROOT / group; root.mkdir(parents=True,exist_ok=True)
        out = keys.copy(); out['selected']=flag
        assert out.loc[out.selected,'date'].str.startswith(year).all()
        out.to_parquet(root / 'selection.parquet',index=False,compression='zstd')
        r = dict(protocol_sha256=sha(study.PROTOCOL),execution_protocol_sha256=sha(EXECUTION),
            selection_sha256=sha(root / 'selection.parquet'),group=group,rows=len(out),selected=int(flag.sum()),
            days=out.loc[out.selected,'date'].nunique(),no_new_group_outcomes_read=True,
            no_future_fill_or_label_filtering=True,new_2026_prices_read=False,no_exit_rules=True)
        save_json(root / 'selection_report.json',r)
        save_json(root / 'selection_verification.json',dict(passed=True,
            selection_report_sha256=sha(root / 'selection_report.json'),
            full_original_metadata_and_all_fold_flags_independent_sql_equal=True,
            all_invalid_unselected_and_other_year_keys_retained=True))
        selections.append(dict(group=group,root=str(root),rows=len(out),selected=r['selected'],days=r['days']))
        for file in ['selection.parquet','selection_report.json','selection_verification.json']:
            receipts[str(root / file)] = sha(root / file)
    assert sum(item['new_fit'] for item in models) == 4
    save_json(joint,dict(passed=True,protocol_sha256=sha(study.PROTOCOL),execution_protocol_sha256=sha(EXECUTION),
        source_hashes=receipts,selections=selections,models=models,
        all_four_half_models_each_arm_and_four_full_year_frames_fixed_together=True,
        no_new_group_outcomes_read=True,new_2026_prices_read=False,no_exit_rules=True,
        software_compilation_verified=False,native_source_parity_verified=False,
        cleanup_runtime_sha256=sha(RUNTIME),maximum_new_fits=4,no_new_control_fits=True,
        native_cross_security_helper_required=None))
    return dict(joint_sha256=sha(joint),selections=selections,models=models)


def checked_joint():
    p,e = checked(); path = study.ROOT / 'joint_selection_freeze.json'; joint=json.loads(path.read_text())
    assert joint['passed'] and joint['protocol_sha256']==sha(study.PROTOCOL)
    committed = subprocess.run(['git','show','HEAD:docs/selection-formula.md'],text=True,capture_output=True,check=True).stdout
    assert sha(path) in committed
    for file,digest in joint['source_hashes'].items(): assert sha(Path(file))==digest,file
    for item in joint['selections']: checked_selection(Path(item['root']))
    evaluation.source.ROOT=Path(p['evaluation_label_root'])
    return p,e,joint


def analyze():
    p,e,joint = checked_joint(); matches={}; receipts={}; scans=0
    for item in joint['selections']:
        frame=checked_selection(Path(item['root'])); same=[]
        for root in p['prior_analysis_roots']:
            root=Path(root); sr=json.loads((root / 'selection_report.json').read_text()); scans+=1
            if sr.get('selected')!=item['selected']: continue
            candidate,_=market.checked_analysis(root)
            if frame.equals(candidate):
                same.append(str(root))
                for file in ['selection_report.json','selection_verification.json','analysis_report.json','analysis_verification.json']:
                    receipts[str(root / file)]=sha(root / file)
        matches[item['group']]=same
    save_json(study.ROOT / 'analysis_reuse_lookup.json',dict(passed=True,
        joint_sha256=sha(study.ROOT / 'joint_selection_freeze.json'),full_frame_matches=matches,
        metadata_scans=scans,source_hashes=receipts,no_count_only_reuse=True,new_2026_prices_read=False))
    master=dict(original_control=p['original_selection_roots']['2024'],
        controls={'original2025':p['original_selection_roots']['2025']})
    master['controls'].update({f'cached{i}':r for i,r in enumerate(sorted({r for names in matches.values() for r in names}))})
    reuse_common.PROTOCOL=EXECUTION; reuse_common.inputs.ROOT=study.ROOT
    reuse_common.checked_joint=lambda:(master,joint)
    reuse_common.checked_analysis=market.checked_analysis
    return reuse_common.analyze()


def finish():
    p,e,joint=checked_joint(); path=study.ROOT / 'complete_results_manifest.json'; assert not path.exists()
    roots={r['group']:Path(r['root']) for r in joint['selections']}
    roots.update({f'original{y}':Path(r) for y,r in p['original_selection_roots'].items()})
    cache={}; receipts={}; done=[]
    for name,root in roots.items():
        cache[name]=market.checked_analysis(root)
        if not (root / 'reference_coverage_verification.json').exists(): audit(root,[2024,2025])
        ref=json.loads((root / 'reference_coverage_verification.json').read_text())
        assert ref['passed'] and ref['analysis_report_sha256']==sha(root / 'analysis_report.json')
        for file in ['selection.parquet','selection_report.json','selection_verification.json','analysis_report.json',
                     'analysis_verification.json','daily_summary.parquet','full_label_report.json',
                     'full_label_verification.json','reference_coverage_verification.json']:
            receipts[str(root / file)]=sha(root / file)
    def equivalent(a,b):
        af,ar=cache[a]; bf,br=cache[b]
        return af.equals(bf) and all(ar[k]==br[k] for k in ['summaries','daily_summary_sha256','label_report_sha256'])
    # A prior numerical pair can only match if both complete endpoint
    # frames and all applicable label/statistical receipts match first.
    prior_pairs=[]
    lookup=json.loads((study.ROOT / 'analysis_reuse_lookup.json').read_text())['full_frame_matches']
    if any(lookup.values()):
        for file,digest in e['prior_completion_manifests'].items():
            manifest=Path(file); assert sha(manifest)==digest
            old=json.loads(manifest.read_text())
            if not old.get('passed'): continue
            for pair in old.get('comparisons',[]):
                endpoints=[]
                for side in ['left','right']:
                    value=pair.get(side)
                    if not value: break
                    options=[manifest.parent/value,manifest.parent/'comparison_views'/value]
                    root=next((r for r in options if (r/'selection_report.json').exists()),None)
                    if root is None: break
                    endpoints.append(root)
                if len(endpoints)==2 and all(Path(f).exists() for f in pair.get('outputs',[])):
                    prior_pairs.append(dict(left_root=endpoints[0],right_root=endpoints[1],
                        outputs=pair['outputs'],manifest=str(manifest),manifest_sha256=digest))
    def equal_prior(name,root):
        frame,report=market.checked_analysis(root); own,own_report=cache[name]
        return own.equals(frame) and all(own_report[k]==report[k] for k in ['summaries','daily_summary_sha256','label_report_sha256'])
    for left,right in p['planned_comparisons']:
        pair=left+'_minus_'+right; year=left[-4:]
        assert year==right[-4:] and year in ['2024','2025']
        prior=next((d for d in done if equivalent(left,d['left']) and equivalent(right,d['right'])),None)
        files=[study.ROOT / ('same_dates_'+pair+'.json'),study.ROOT / ('shared_unknowns_'+pair+'.json')]
        periods=[year+'H1',year+'H2',year]
        cached=next((d for d in prior_pairs if equal_prior(left,d['left_root']) and equal_prior(right,d['right_root'])),None) if prior is None else None
        if prior: files=[Path(f) for f in prior['outputs']]
        elif cached:
            files=[Path(f) for f in cached['outputs']]
            receipts[cached['manifest']]=cached['manifest_sha256']
        else:
            if not files[0].exists(): compare(roots[left],roots[right],files[0],periods,intersection_only=True)
            if not files[1].exists():
                spec=dict(left=str(roots[left]),right=str(roots[right]),output=str(files[1]),
                    left_analysis_sha256=sha(roots[left] / 'analysis_report.json'),
                    right_analysis_sha256=sha(roots[right] / 'analysis_report.json'))
                receipt=dict(protocol_sha256=sha(EXECUTION),labels_sha256=sha(evaluation.source.ROOT / 'full_labels.parquet'),
                    periods=periods,signal_range=['2024-01-01','2024-12-31'] if year=='2024' else None)
                shared_compare(spec,receipt)
        for file in files: assert json.loads(file.read_text())['passed']; receipts[str(file)]=sha(file)
        done.append(dict(left=left,right=right,outputs=[str(f) for f in files],
            reused=prior is not None or cached is not None,prior_manifest=cached['manifest'] if cached else None))
        print(json.dumps(dict(compared=pair)),flush=True)
    def main(group,period):
        return next(s for s in cache[group][1]['summaries'] if s['arm']=='formula' and s['bps']==15
                    and not s['sensitive'] and s['period']==period)
    uncertainty=[]
    for year in ['2024','2025']:
        pair=next(d for d in done if d['left']=='memory'+year and d['right']=='control'+year)
        shared=json.loads(Path(pair['outputs'][1]).read_text())
        uncertainty.append(next(s for s in shared['summaries'] if s['bps']==15 and not s['sensitive'] and s['period']==year))
    criteria=dict(each_half_at_least_20_signal_days=all(main('memory'+y,y+h)['days']>=20 for y in ['2024','2025'] for h in ['H1','H2']),
        all_four_half_reference_means_positive=all((main('memory'+y,y+h)['mean_reference'] or -1)>0 for y in ['2024','2025'] for h in ['H1','H2']),
        both_year_opportunity_above_matched_control_and_bad3_not_above=all(
            main('memory'+y,y)['rate'] is not None and main('memory'+y,y)['bad3'] is not None
            and main('memory'+y,y)['rate']>main('control'+y,y)['rate']
            and main('memory'+y,y)['bad3']<=main('control'+y,y)['bad3'] for y in ['2024','2025']),
        both_year_shared_unknown_lower_ci_strict_positive=all(s['lower_ci'] is not None and s['lower_ci'][0]>0 for s in uncertainty))
    gate=study.ROOT / 'memory_gate.json'
    save_json(gate,dict(passed=True,criteria=criteria,supports_further_validation=all(criteria.values()),
        no_automatic_2026_evaluation=True,no_publish_claim=True,new_2026_prices_read=False,no_exit_rules=True))
    for file in [gate,study.ROOT / 'analysis_dispatch_verification.json',study.ROOT / 'analysis_reuse_lookup.json']:
        receipts[str(file)]=sha(file)
    save_json(path,dict(passed=True,protocol_sha256=sha(EXECUTION),input_protocol_sha256=sha(study.PROTOCOL),
        joint_sha256=sha(study.ROOT / 'joint_selection_freeze.json'),source_hashes=receipts,comparisons=done,
        all_four_full_year_groups_and_four_predefined_comparisons_complete=True,
        all_full_frames_labels_daily_statistics_equal_before_reuse=True,no_duplicate_half_year_aggregation=True,
        years_2024_and_2025_exploratory=True,no_publish_claim=True,new_2026_prices_read=False,no_exit_rules=True))
    return dict(completion_sha256=sha(path),criteria=criteria,
        annual={g:main(g,g[-4:]) for g in roots if not g.startswith('original')})
