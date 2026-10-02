"""Apply four frozen original models to the audited listing expansion; no fit."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research.research_io import check_runtime, check_sources, sha, save_json
from trade_research.tail_formula_additive import conn, encode, predict
from trade_research.tail_formula_baseline import EXPRESSIONS
from tail_formula_order_statistics import KEYS
from tail_formula_reports import checked_selection
from verify_tail_formula_additive import tree_sql

PROTOCOL = Path('config/tail_formula_listing_selection.json')


def main():
    check_runtime(); p=json.loads(PROTOCOL.read_text()); check_sources(p['source_hashes'])
    assert subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    committed=subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    assert sha(PROTOCOL) in committed and sha(Path(p['input_receipt'])) in committed
    gate=json.loads(Path(p['input_receipt']).read_text()); assert gate['passed']; check_sources(gate['source_hashes'])
    root=Path(p['output_root']); root.mkdir(exist_ok=False)
    f=pd.read_parquet(p['replayed_inputs']); sources=dict(p['source_hashes'])
    flags=np.zeros(len(f),dtype=bool); all_scores=np.full(len(f),np.nan); checks=[]
    old={year:checked_selection(Path(folder)) for year,folder in p['original_lists'].items()}
    for fold in p['folds']:
        folder=Path(fold['model_root']); m=json.loads((folder/'model_report.json').read_text())
        v=json.loads((folder/'model_verification.json').read_text()); assert v['passed']
        assert v['model_report_sha256'] == sha(folder/'model_report.json')
        assert m['feature_names']==list(EXPRESSIONS) and m['learning_rate']==.05
        assert m['last_observation']<fold['evaluation_start'] and m['training_end']==fold['evaluation_start']
        cuts=[cut for cut in m['thresholds'] if cut['training_quantile']==p['training_quantile']]
        assert len(cuts)==1; threshold=cuts[0]['threshold']
        mask=f.date.ge(fold['evaluation_start']) & f.date.lt(fold['evaluation_end'])
        part=f.loc[mask].reset_index(drop=True); valid=part.formula_input_valid
        scores=np.full(len(part),np.nan); scores[valid]=predict(encode(part.loc[valid]),m)
        c=conn();c.register('visible',part[['date','code','formula_input_valid',*EXPRESSIONS]])
        enc=','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS X{i:02d}' for i,n in enumerate(EXPRESSIONS,1))
        c.sql('SELECT date,code,'+enc+' FROM visible WHERE formula_input_valid').create_view('encoded')
        equation=format(m['bias'],'.17e')+'+'+'+'.join(tree_sql(t) for t in m['trees'])
        c.sql('SELECT date,code,'+equation+' AS score FROM encoded').create_view('rebuilt')
        ex=c.sql('SELECT v.date,v.code,r.score FROM visible v LEFT JOIN rebuilt r USING(date,code) ORDER BY v.date,v.code').df();c.close()
        pd.testing.assert_frame_equal(part[['date','code']],ex[['date','code']],check_exact=True)
        np.testing.assert_allclose(scores,ex.score,rtol=0,atol=2e-11,equal_nan=True)
        selected=scores>threshold; np.testing.assert_array_equal(selected,ex.score.gt(threshold))
        prior=part.loc[part.original_pool,KEYS].merge(old[fold['evaluation_start'][:4]],on=KEYS,validate='one_to_one')
        np.testing.assert_array_equal(selected[part.original_pool],prior.selected)
        flags[mask]=selected;all_scores[mask]=scores
        checks.append(dict(fold=fold['id'],training_quantile=p['training_quantile'],threshold=threshold,
            all_scores_flags_SQL_verified=True,all_same_code_original_flags_match=True,
            new_selected=int((selected&part.extra_pool).sum()),new_valid=int((valid&part.extra_pool).sum())))
    out=f[[*KEYS,'original_pool','extra_pool','formula_input_valid']].copy();out['score']=all_scores;out['selected']=flags
    path=root/'scores.parquet';out.to_parquet(path,index=False,compression='zstd');sources[str(path)]=sha(path)
    pool=pd.read_parquet(p['visible_pool'],columns=[*KEYS,'original_pool','extra_pool']);lists=[]
    for year in ['2024','2025']:
        extra=out.loc[out.extra_pool,KEYS+['selected']].copy();extra['selected'] &= extra.date.str.startswith(year)
        expanded=pd.concat([old[year],extra],ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
        pd.testing.assert_frame_equal(expanded[KEYS],pool[KEYS],check_exact=True)
        pd.testing.assert_frame_equal(expanded.loc[pool.original_pool].reset_index(drop=True),old[year],check_exact=True)
        for name,selection in [('expanded',expanded),('extra',expanded.assign(selected=expanded.selected&pool.extra_pool))]:
            folder=root/(name+year);folder.mkdir()
            selection.to_parquet(folder/'selection.parquet',index=False,compression='zstd')
            chosen=selection.loc[selection.selected];sizes=chosen.groupby('date').size()
            save_json(folder/'selection_report.json',dict(protocol_sha256=sha(PROTOCOL),rows=len(selection),
                selection_sha256=sha(folder/'selection.parquet'),selected=len(chosen),days=len(sizes),
                half_counts=chosen.groupby('half').agg(rows=('code','size'),days=('date','nunique')).reset_index().to_dict('records'),
                median_daily=float(sizes.median()) if len(sizes) else None,max_daily=int(sizes.max()) if len(sizes) else None,
                no_outcome_or_fill_filter=True,original_flags_retained=True,new_fits=0,new_2026_prices_read=False))
            save_json(folder/'selection_verification.json',dict(passed=True,selection_report_sha256=sha(folder/'selection_report.json'),
                all_expanded_keys_and_metadata_match=True,old_flags_copied_exactly_and_same_code_controls_replayed=True,
                all_new_scores_flags_SQL_verified=True,native_client_parity_verified=False))
            lists.append(dict(group=name+year,root=str(folder),year=year))
            for file in folder.iterdir():sources[str(file)]=sha(file)
    save_json(root/'joint_selection_freeze.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),source_hashes=sources,
        selections=lists,score_checks=checks,original_model_training_and_thresholds_unchanged=True,
        all_four_complete_lists_fixed_before_new_labels=True,extra_only_diagnostic_not_native_listing_age_formula=True,
        new_fits=0,new_economic_groups_read=False,new_2026_prices_read=False,no_exit_rules=True))
    print(json.dumps(dict(joint_sha256=sha(root/'joint_selection_freeze.json'),score_checks=checks)),flush=True)


if __name__ == '__main__':
    main()
