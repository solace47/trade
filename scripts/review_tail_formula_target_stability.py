"""Compare two frozen binary models and their date support without new selection."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as numeric
from trade_research import tail_formula_baseline as original
from trade_research.tail_formula_boundary_evaluation import number
from trade_research.research_io import check_runtime,check_sources,sha,save_json
from tail_formula_reports import checked_selection
from tail_formula_rank_statistics import daily_metrics,verify_metrics
from tail_formula_statistics import interval
from verify_tail_formula_additive import tree_sql

PROTOCOL=Path('config/tail_formula_target_stability.json')


def cut(model):
    found=[x['threshold'] for x in model['thresholds'] if x['training_quantile']==.995]
    assert len(found)==1
    return found[0]


def resolve(receipts,digest,name):
    files=[Path(file) for file,value in receipts.items() if value==digest and Path(file).name==name]
    return next(file for file in files if (file.parent/('features.parquet' if name=='feature_report.json' else 'full_labels.parquet')).exists())


def targets(path,spec):
    c=numeric.conn()
    t=c.execute('''SELECT date,code,opportunity15 AS raw,
        opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target
        FROM read_parquet(?) WHERE known15 AND date>=? AND next_date<? ORDER BY date,code''',
        [str(path),spec['training_start'],spec['training_end']]).df();c.close()
    assert t.raw.isin([0,1]).all()
    return t


def predict(q,m):
    scores=numeric.predict(numeric.encode(q),m)
    c=numeric.conn();c.register('visible',q[['date','code',*original.EXPRESSIONS]])
    encoded=','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS X{i:02d}' for i,n in enumerate(original.EXPRESSIONS,1))
    c.sql('SELECT date,code,'+encoded+' FROM visible').create_view('encoded')
    equation=format(m['bias'],'.17e')+'+'+'+'.join(tree_sql(tree) for tree in m['trees'])
    expected=c.sql('SELECT date,code,'+equation+' AS score FROM encoded ORDER BY date,code').df();c.close()
    pd.testing.assert_frame_equal(q[['date','code']].reset_index(drop=True),expected[['date','code']],check_exact=True)
    np.testing.assert_allclose(scores,expected.score,rtol=0,atol=2e-11)
    return scores


def leaf_support(x,d,m,fold,arm):
    date_code=np.unique(d.date.to_numpy(),return_inverse=True)[1]
    weight=1/d.groupby('date').code.transform('size').to_numpy()
    c=numeric.conn();records=[]
    for i,tree in enumerate(m['trees']):
        leaf=numeric.leaf_indices(x,tree)
        frame=pd.DataFrame(dict(day=date_code,leaf=leaf,weight=weight));c.register('leaf_rows',frame)
        expected=c.sql('''WITH day_weights AS(SELECT leaf,day,count(*) AS rows,sum(weight) AS w
            FROM leaf_rows GROUP BY leaf,day)
            SELECT leaf,sum(rows) AS rows,count(*) AS distinct_days,sum(w) AS weight,
            sum(w)*sum(w)/sum(w*w) AS effective_dates,max(w)/sum(w) AS maximum_day_fraction
            FROM day_weights GROUP BY leaf ORDER BY leaf''').df().set_index('leaf')
        for node in sorted(np.unique(leaf)):
            mask=leaf==node;w=np.bincount(date_code[mask],weights=weight[mask]);w=w[w>0]
            row=dict(fold=fold,arm=arm,tree=i,node=int(node),rows=int(mask.sum()),distinct_days=len(w),
                weight=float(w.sum()),effective_dates=float(w.sum()**2/(w*w).sum()),maximum_day_fraction=float(w.max()/w.sum()))
            ex=expected.loc[node]
            for field in ['rows','distinct_days']:assert row[field]==int(ex[field])
            for field in ['weight','effective_dates','maximum_day_fraction']:
                np.testing.assert_allclose(row[field],ex[field],rtol=0,atol=2e-10)
            assert row['rows']==tree['n_node_samples'][node]
            np.testing.assert_allclose(row['weight'],tree['weighted_n_node_samples'][node],rtol=0,atol=1e-8)
            records.append(row)
        c.unregister('leaf_rows')
    c.close();return pd.DataFrame(records)


def main():
    check_runtime();p=json.loads(PROTOCOL.read_text());check_sources(p['source_hashes'])
    assert subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}'])==PROTOCOL.read_bytes()
    assert sha(PROTOCOL) in subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    gate=json.loads(Path(p['completed_receipt']).read_text());assert gate['passed'];check_sources(gate['source_hashes'])
    root=Path(p['output_root']);root.mkdir(exist_ok=False)
    prep=json.loads(Path(p['preparation']).read_text());check_sources(prep['source_hashes'])
    f=pd.read_parquet(p['features']);t=pd.read_parquet(p['targets']);assert len(f)==1815129 and f.date.lt('2026-01-01').all()
    pd.testing.assert_frame_equal(f[['date','code']],t[['date','code']],check_exact=True)
    for name,field in [('positive_before','first_positive_end15'),('space_before','first_space_end15')]:
        t[name]=(t[field].ge(0)&(t.first_bad315.lt(0)|t[field].lt(t.first_bad315))).astype('int64')
    valid=f.loc[f.formula_input_valid].merge(t[['date','code','next_date','known15','known_no_trade','positive_before','space_before']],on=['date','code'],validate='one_to_one')
    frames={arm:{y:checked_selection(Path(folder)) for y,folder in item.items()} for arm,item in p['selections'].items()}
    ranks=[];rank_summary=[];leaves=[];overlap=[];training=[];sources=dict(p['source_hashes'])
    for spec in p['folds']:
        models={arm:json.loads((Path(spec[arm+'_root'])/'model_report.json').read_text()) for arm in ['original','primary']}
        a,b=models.values();assert a['parameters']==b['parameters'] and a['feature_names']==b['feature_names']==list(original.EXPRESSIONS)
        assert all(m['variant']=='relative' and m['last_observation']<spec['evaluation_start'] for m in models.values())
        report=resolve(prep['source_hashes'],a['label_report_sha256'],'full_label_report.json');old_labels=report.parent/'full_labels.parquet'
        assert sha(old_labels)==json.loads(report.read_text())['labels_sha256'];sources[str(old_labels)]=sha(old_labels);sources[str(report)]=sha(report)
        old=targets(old_labels,spec);new=targets(Path(p['primary_labels']),spec)
        pd.testing.assert_frame_equal(old[['date','code']],new[['date','code']],check_exact=True)
        feature_report=resolve(prep['source_hashes'],a['feature_report_sha256'],'feature_report.json');old_features=feature_report.parent/'features.parquet'
        assert sha(old_features)==json.loads(feature_report.read_text())['features_sha256'];sources[str(old_features)]=sha(old_features);sources[str(feature_report)]=sha(feature_report)
        old_f=pd.read_parquet(old_features,columns=['date','code','formula_input_valid',*original.EXPRESSIONS])
        d_old=old_f.loc[old_f.formula_input_valid].merge(old,on=['date','code'],validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
        d_new=f.loc[f.formula_input_valid].merge(new,on=['date','code'],validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
        pd.testing.assert_frame_equal(d_old[['date','code']],d_new[['date','code']],check_exact=True)
        x=numeric.encode(d_new);np.testing.assert_array_equal(numeric.encode(d_old),x)
        assert len(d_old)==a['rows']==b['rows']
        np.testing.assert_array_equal(d_old.groupby('date').code.transform('size'),d_new.groupby('date').code.transform('size'))
        training.append(dict(fold=spec['id'],source_known_rows=len(old),source_raw_labels_changed=int(old.raw.ne(new.raw).sum()),source_dates_changed=int(old.loc[old.raw.ne(new.raw),'date'].nunique()),fitted_rows=len(d_new),fitted_raw_labels_changed=int(d_old.raw.ne(d_new.raw).sum()),fitted_centered_targets_changed=int((d_old.target-d_new.target).abs().gt(2e-12).sum()),encoded_inputs_weights_and_parameters_exactly_equal=True))
        for arm,m in models.items():leaves.append(leaf_support(x,d_new,m,spec['id'],arm))
        for scope in ['training','evaluation']:
            q=valid.loc[valid.date.ge(spec[scope+'_start'])&valid.date.lt(spec[scope+'_end'])].copy().reset_index(drop=True)
            if scope=='training':q=q.loc[q.next_date.lt(spec['training_end'])].reset_index(drop=True)
            for arm,m in models.items():
                q[arm+'_score']=predict(q,m);q[arm+'_chosen']=q[arm+'_score'].gt(cut(m))
                if scope=='evaluation':
                    flags=frames[arm][spec['evaluation_start'][:4]][['date','code','selected']]
                    merged=q[['date','code',arm+'_chosen']].merge(flags,on=['date','code'],validate='one_to_one')
                    assert merged[arm+'_chosen'].equals(merged.selected)
                else:np.testing.assert_allclose(np.quantile(q.loc[q.known15,arm+'_score'],.995),cut(m),rtol=0,atol=2e-10)
                part=q.loc[q.known15,['date','code','positive_before','space_before',arm+'_score']].rename(columns={arm+'_score':'score'});part['utility']=part.positive_before
                d=daily_metrics(part);verify_metrics(part,d)
                # Preserve dates with no known labels as undefined, rather than silently dropping them.
                d=pd.DataFrame(dict(date=sorted(q.date.unique()))).merge(d,on='date',how='left',validate='one_to_one')
                for field in d:
                    if field!='date' and not field.endswith('_auc'):d[field]=d[field].fillna(0).astype('int64')
                d['fold']=spec['id'];d['scope']=scope;d['arm']=arm;ranks.append(d)
                for target in ['positive_before','space_before']:
                    field=target+'_auc';rank_summary.append(dict(fold=spec['id'],scope=scope,arm=arm,target=target,days=len(d),defined_days=int(d[field].notna().sum()),date_equal_auc_defined_only=number(d[field].mean()),weekly_ci=interval(d,field)))
            c=numeric.conn();c.register('scored',q)
            d=c.sql('''SELECT date,count(*) AS rows,corr(original_score,primary_score) AS correlation,
                sum(original_chosen::INT) AS original_selected,sum(primary_chosen::INT) AS primary_selected,
                sum((original_chosen AND primary_chosen)::INT) AS shared,
                sum((original_chosen OR primary_chosen)::INT) AS union_count FROM scored GROUP BY date ORDER BY date''').df();c.close()
            d['fold']=spec['id'];d['scope']=scope;overlap.append(d)
        print(json.dumps(dict(stability_verified=spec['id'])),flush=True)
    for name,parts in [('rank_daily',ranks),('leaf_support',leaves),('overlap_daily',overlap)]:
        path=root/(name+'.parquet');pd.concat(parts,ignore_index=True).to_parquet(path,index=False,compression='zstd');sources[str(path)]=sha(path)
    save_json(root/'report.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),source_hashes=sources,training_comparisons=training,rank_summaries=rank_summary,all_training_keys_encoded_inputs_weights_and_parameters_equal=True,all_scores_and_existing_eval_flags_SQL_verified=True,all_leaf_date_weight_concentration_independently_SQL_verified=True,new_fits=0,new_selection_lists=0,new_economic_groups=0,new_2026_prices_read=False,no_exit_rules=True,effective_dates_are_weight_concentration_not_independent_sample_size=True,not_profit_or_formula_quality_proof=True))
    print(json.dumps(dict(report_sha256=sha(root/'report.json'),rank_groups=len(rank_summary))),flush=True)


if __name__=='__main__':main()
