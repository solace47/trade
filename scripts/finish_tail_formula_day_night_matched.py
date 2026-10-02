"""Freeze the matched component supplement and reuse its exact controls."""
import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd

import run_tail_formula_day_night_matched as fit
import finish_tail_formula_rule_search as shared
from trade_research import tail_formula_additive as numeric
from trade_research.research_io import check_sources,sha,save_json
from tail_formula_reports import checked_selection
from verify_tail_formula_additive import tree_sql

ROOT=fit.ROOT
EVALUATION=Path('config/tail_formula_day_night_matched_evaluation.json')
KEYS=shared.KEYS


def checked():
    p,_,_=fit.checked();e=json.loads(EVALUATION.read_text())
    assert subprocess.check_output(['git','show',f'HEAD:{EVALUATION}'])==EVALUATION.read_bytes()
    assert sha(EVALUATION) in subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    assert e['input_protocol_sha256']==sha(fit.PROTOCOL)
    check_sources(e['source_hashes'])
    models=json.loads((ROOT/'all_models_verified.json').read_text())
    assert models['passed'] and models['fits_completed']==4 and models['exact_controls_reused']==4
    check_sources(models['source_hashes'])
    registry=json.loads(Path(e['prior_registry']).read_text())
    return p,dict(e,prior_analysis_roots=registry['prior_analysis_roots'],
        prior_completion_manifests=registry['prior_completion_manifests'])


def freeze():
    p,e=checked();destination=ROOT/'joint_selection_freeze.json';assert not destination.exists()
    f=pd.read_parquet(ROOT/'features.parquet').loc[lambda x:x.date.ge('2024-01-01')].reset_index(drop=True)
    assert len(f)==1258085 and int(f.formula_input_valid.sum())==1117397
    models=json.loads((ROOT/'all_models_verified.json').read_text())
    flags={arm:np.zeros(len(f),dtype=bool) for arm in p['arms']}
    receipts=dict(e['source_hashes']);model_checks=[]
    for record in models['models']:
        root=Path(record['root']);m=json.loads((root/'model_report.json').read_text())
        v=json.loads((root/'model_verification.json').read_text())
        assert v['passed'] and v['model_report_sha256']==sha(root/'model_report.json')==record['model_report_sha256']
        assert sha(root/'model_verification.json')==record['model_verification_sha256']
        names=m['feature_names'];spec=p['folds'][record['fold']]
        assert m['training_start']==spec['training_start'] and m['training_end']==spec['training_end']
        assert m['last_observation']<spec['evaluation_start'] and len(m['thresholds'])==1
        cut=m['thresholds'][0];assert cut['training_quantile']==.995
        scope=f.date.ge(spec['evaluation_start']) & f.date.lt(spec['evaluation_end'])
        d=f.loc[scope].reset_index(drop=True);valid=d.formula_input_valid
        x=np.floor(np.clip(100*d.loc[valid,names].to_numpy()+10000+.000001,0,999999)).astype('int32')
        scores=np.full(len(d),np.nan);scores[valid]=numeric.predict(x,m)
        c=numeric.conn();c.register('visible',d[['date','code','formula_input_valid',*names]])
        encoding=','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS X{i:02d}' for i,n in enumerate(names,1))
        c.sql('SELECT date,code,'+encoding+' FROM visible WHERE formula_input_valid').create_view('encoded')
        equation=format(m['bias'],'.17e')+'+'+'+'.join(tree_sql(t) for t in m['trees'])
        c.sql('SELECT date,code,'+equation+' AS score FROM encoded').create_view('rebuilt')
        expected=c.sql('''SELECT v.date,v.code,r.score FROM visible v LEFT JOIN rebuilt r
            USING(date,code) ORDER BY v.date,v.code''').df();c.close()
        pd.testing.assert_frame_equal(d[['date','code']],expected[['date','code']],check_exact=True)
        np.testing.assert_allclose(scores,expected.score,rtol=0,atol=2e-11,equal_nan=True)
        selected=scores>cut['threshold']
        np.testing.assert_array_equal(selected,expected.score.gt(cut['threshold']))
        flags[record['arm']][scope]=selected
        model_checks.append(dict(arm=record['arm'],fold=record['fold'],rows=len(d),selected=int(selected.sum()),
            threshold=cut['threshold'],all_integer_inputs_tree_scores_and_flags_SQL_rebuilt=True,
            component_splits=sum(int(j>=50) for t in m['trees'] for j in t['feature'])))
        for name in ['model_report.json','model_verification.json']:receipts[str(root/name)]=sha(root/name)
    lists,lookup,aliases=[],[],[]
    for year in ['2024','2025']:
        control=Path(e['controls'][year]);old=checked_selection(control)
        expected=f[KEYS].copy();expected['selected']=flags['control50'] & f.date.str.startswith(year)
        pd.testing.assert_frame_equal(expected,old,check_exact=True)
        aliases.append(dict(year=year,computed_same_quality_control_exactly_original=True,
            original_root=str(control),no_additional_statistical_or_comparison_arm=True))
        root=ROOT/('rule'+year);assert not root.exists();root.mkdir()
        out=f[KEYS].copy();out['selected']=flags['components53'] & f.date.str.startswith(year)
        out.to_parquet(root/'selection.parquet',index=False,compression='zstd')
        chosen=out.loc[out.selected];sizes=chosen.groupby('date').size();equivalent=[]
        for prior in e['prior_analysis_roots']:
            prior=Path(prior);r=json.loads((prior/'selection_report.json').read_text())
            if r['selected']==len(chosen) and checked_selection(prior).equals(out):equivalent.append(str(prior))
        lookup.append(dict(year=year,previous_complete_lists_examined=len(e['prior_analysis_roots']),
            exact_equivalent_complete_lists=equivalent,new_economic_statistics_read=False))
        save_json(root/'selection_report.json',dict(protocol_sha256=sha(EVALUATION),selection_sha256=sha(root/'selection.parquet'),
            rows=len(out),selected=len(chosen),days=len(sizes),
            half_counts=chosen.groupby('half').agg(rows=('code','size'),days=('date','nunique')).reset_index().to_dict('records'),
            median_daily=float(sizes.median()) if len(sizes) else None,max_daily=int(sizes.max()) if len(sizes) else None,
            largest_day_fraction=float(sizes.max()/len(chosen)) if len(chosen) else None,
            all_models_verified_sha256=sha(ROOT/'all_models_verified.json'),
            no_outcome_or_fill_filter=True,new_2026_prices_read=False,no_exit_rules=True))
        save_json(root/'selection_verification.json',dict(passed=True,selection_report_sha256=sha(root/'selection_report.json'),
            all_scores_flags_scopes_qualification_and_metadata_SQL_rebuilt=True,
            native_client_parity_verified=False))
        lists.extend([dict(group='rule'+year,root=str(root),selected=len(chosen),days=len(sizes)),
            dict(group='control'+year,root=str(control),original_unchanged=True)])
        for path in [root,control]:
            for name in ['selection.parquet','selection_report.json','selection_verification.json']:receipts[str(path/name)]=sha(path/name)
    for name,value in [('score_verification.json',dict(passed=True,records=model_checks,new_economics_read=False)),
        ('selection_equivalence_lookup.json',dict(passed=True,records=lookup)),
        ('matched_control_aliases.json',dict(passed=True,records=aliases))]:
        save_json(ROOT/name,value);receipts[str(ROOT/name)]=sha(ROOT/name)
    save_json(destination,dict(passed=True,input_protocol_sha256=sha(fit.PROTOCOL),execution_protocol_sha256=sha(EVALUATION),
        source_hashes=receipts,selections=lists,fits_completed=4,controls_reused=4,
        all_models_and_complete_annual_lists_fixed_together_before_economics=True,
        new_economic_outcomes_read=False,new_2026_prices_read=False))
    return dict(joint_sha256=sha(destination),selections=lists,equivalence=lookup)


shared.ROOT=ROOT
shared.EXECUTION=EVALUATION
shared.fit=SimpleNamespace(PROTOCOL=fit.PROTOCOL,EXECUTION=fit.EXECUTION)
shared.checked=checked

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','analyze','finish'])
    stage=parser.parse_args().stage
    print(json.dumps(freeze() if stage=='freeze' else getattr(shared,stage)(),ensure_ascii=False),flush=True)
