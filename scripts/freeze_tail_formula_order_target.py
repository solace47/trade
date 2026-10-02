"""Freeze every matched list after independent tree replay, before economics."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

import run_tail_formula_order_target as fit
from trade_research import tail_formula_additive as numeric
from trade_research.research_io import check_sources, save_json, sha
from tail_formula_reports import checked_selection
from verify_tail_formula_additive import tree_sql

ROOT=fit.ROOT
EVALUATION=Path('config/tail_formula_order_target_evaluation.json')
KEYS=['date','code','half','board','decision_shares']


def checked():
    execution=fit.checked();p=json.loads(fit.INPUT.read_text());e=json.loads(EVALUATION.read_text())
    assert e['input_protocol_sha256']==sha(fit.INPUT) and e['execution_protocol_sha256']==sha(fit.EXECUTION)
    assert subprocess.check_output(['git','show',f'HEAD:{EVALUATION}'])==EVALUATION.read_bytes()
    assert sha(EVALUATION) in subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    check_sources(e['source_hashes'])
    return p,execution,e


def main():
    p,execution,e=checked();destination=ROOT/'joint_selection_freeze.json';assert not destination.exists()
    models=json.loads((ROOT/'all_models_verified.json').read_text())
    assert models['passed'] and models['fits_completed']==8 and len(models['models'])==8
    check_sources(models['source_hashes'])
    assert sha(ROOT/'all_models_verified.json') in subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    f=pd.read_parquet(ROOT/'features.parquet').loc[lambda z:z.date.ge('2024-01-01')].reset_index(drop=True)
    assert len(f)==1258085 and int(f.formula_input_valid.sum())==1117397 and f.date.lt('2026-01-01').all()
    flags={arm:np.zeros(len(f),dtype=bool) for arm in p['planned_arms']};checks=[];receipts=dict(e['source_hashes'])
    specs={a['id']:a for a in p['planned_folds']}
    for record in models['models']:
        folder=Path(record['root']);m=json.loads((folder/'model_report.json').read_text());v=json.loads((folder/'model_verification.json').read_text())
        assert v['passed'] and v['model_report_sha256']==sha(folder/'model_report.json')==record['model_report_sha256']
        assert record['model_verification_sha256']==sha(folder/'model_verification.json')
        spec=specs[record['fold']];assert m['training_start']==spec['training_start'] and m['training_end']==spec['training_end']
        assert m['last_observation']<spec['evaluation_start'] and len(m['thresholds'])==1
        cut=m['thresholds'][0];assert cut['training_quantile']==.995
        scope=f.date.ge(spec['evaluation_start']) & f.date.lt(spec['evaluation_end'])
        d=f.loc[scope].reset_index(drop=True);valid=d.formula_input_valid;names=m['feature_names']
        x=np.floor(np.clip(100*d.loc[valid,names].to_numpy()+10000+.000001,0,999999)).astype('int32')
        scores=np.full(len(d),np.nan);scores[valid]=numeric.predict(x,m)
        c=numeric.conn();c.register('visible',d[['date','code','formula_input_valid',*names]])
        encoding=','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS X{i:02d}' for i,n in enumerate(names,1))
        c.sql('SELECT date,code,'+encoding+' FROM visible WHERE formula_input_valid').create_view('encoded')
        equation=format(m['bias'],'.17e')+'+'+'+'.join(tree_sql(t) for t in m['trees'])
        c.sql('SELECT date,code,'+equation+' AS score FROM encoded').create_view('rebuilt')
        ex=c.sql('SELECT v.date,v.code,r.score FROM visible v LEFT JOIN rebuilt r USING(date,code) ORDER BY v.date,v.code').df();c.close()
        pd.testing.assert_frame_equal(d[['date','code']],ex[['date','code']],check_exact=True)
        np.testing.assert_allclose(scores,ex.score,rtol=0,atol=2e-11,equal_nan=True)
        selected=scores>cut['threshold'];np.testing.assert_array_equal(selected,ex.score.gt(cut['threshold']))
        flags[record['arm']][scope]=selected
        checks.append(dict(arm=record['arm'],fold=record['fold'],rows=len(d),selected=int(selected.sum()),
            threshold=cut['threshold'],all_integer_inputs_scores_and_flags_SQL_verified=True))
        for name in ['model_report.json','model_verification.json']:receipts[str(folder/name)]=sha(folder/name)
    registry=json.loads(Path(e['prior_registry']).read_text());check_sources(registry['source_hashes']);lookup=[];lists=[]
    for arm in p['planned_arms']:
        for year in ['2024','2025']:
            folder=ROOT/(arm+year);assert not folder.exists();folder.mkdir()
            out=f[KEYS].copy();out['selected']=flags[arm] & f.date.str.startswith(year)
            out.to_parquet(folder/'selection.parquet',index=False,compression='zstd')
            chosen=out.loc[out.selected];sizes=chosen.groupby('date').size();same=[]
            for prior in registry['reports']:
                path=Path(prior['root']);r=json.loads((path/'selection_report.json').read_text())
                if r['selected']==len(chosen) and checked_selection(path).equals(out):same.append(str(path))
            for item in lists:
                if checked_selection(Path(item['root'])).equals(out):same.append(item['root'])
            lookup.append(dict(arm=arm,year=year,previous_complete_lists_examined=len(registry['reports']),
                equivalent_complete_lists=same,no_economic_outcome_groups_read=True))
            save_json(folder/'selection_report.json',dict(protocol_sha256=sha(EVALUATION),selection_sha256=sha(folder/'selection.parquet'),
                rows=len(out),selected=len(chosen),days=len(sizes),half_counts=chosen.groupby('half').agg(rows=('code','size'),days=('date','nunique')).reset_index().to_dict('records'),
                median_daily=float(sizes.median()) if len(sizes) else None,max_daily=int(sizes.max()) if len(sizes) else None,
                largest_day_fraction=float(sizes.max()/len(chosen)) if len(chosen) else None,
                no_outcome_or_fill_filter=True,new_2026_prices_read=False,no_exit_rules=True))
            save_json(folder/'selection_verification.json',dict(passed=True,selection_report_sha256=sha(folder/'selection_report.json'),
                all_scores_flags_scopes_qualification_and_metadata_SQL_verified=True,native_client_parity_verified=False))
            lists.append(dict(group=arm+year,arm=arm,year=year,root=str(folder),selected=len(chosen),days=len(sizes)))
            for name in ['selection.parquet','selection_report.json','selection_verification.json']:receipts[str(folder/name)]=sha(folder/name)
    for name,data in [('score_verification.json',dict(passed=True,records=checks)),('selection_equivalence_lookup.json',dict(passed=True,records=lookup))]:
        save_json(ROOT/name,data);receipts[str(ROOT/name)]=sha(ROOT/name)
    receipts[str(ROOT/'all_models_verified.json')]=sha(ROOT/'all_models_verified.json')
    save_json(destination,dict(passed=True,input_protocol_sha256=sha(fit.INPUT),execution_protocol_sha256=sha(fit.EXECUTION),
        evaluation_protocol_sha256=sha(EVALUATION),source_hashes=receipts,selections=lists,fits_completed=8,
        all_models_and_all_four_complete_lists_fixed_before_economics=True,new_economic_outcomes_read=False,new_2026_prices_read=False))
    print(json.dumps(dict(joint_sha256=sha(destination),selections=lists,equivalence=lookup)),flush=True)


if __name__=='__main__':
    main()
