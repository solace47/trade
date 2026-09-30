"""Reuse the four canonical 48-field controls before any new fitting."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_daily_regression as study
from trade_research import tail_formula_daily_regression_model as model
from trade_research import tail_formula_relative as relative
from trade_research.corporate_cash import save_json, sha

PROTOCOL=Path('config/tail_formula_daily_regression_reuse_protocol.json')
OLD={
    '2024h1':Path('data/research/tail_formula_quarter_history2024/h1'),
    '2024h2':Path('data/research/tail_formula_quarter_history2024/h2'),
    '2025h1':Path('data/research/tail_formula_before1000_model_2024'),
    '2025h2':Path('data/research/tail_formula_before1000_model_recent')}
FEATURES={'2024':Path('data/research/tail_formula_quarter_history2024/inputs'),
          '2025':Path('data/research/tail_formula_float')}
LABELS={'2024':FEATURES['2024'],'2025':Path('data/research/tail_formula_before1000')}


def checked():
    p=json.loads(PROTOCOL.read_text());study.checked()
    assert p['master_protocol_sha256']==sha(study.PROTOCOL)
    assert p['old_models']=={k:str(v) for k,v in OLD.items()}
    for file,digest in p['source_hashes'].items():assert sha(Path(file))==digest
    return p


def prepare():
    checked()
    try:
        model.protocols()
    except AssertionError as error:
        assert str(error)=='Audit and reuse existing fits before proceeding'
    pre=json.loads((study.ROOT/'prefit_lookup_verification.json').read_text())
    assert pre['passed']
    assert not any(x['lookup']['matches'] for x in pre['records'] if x['arm']=='regression')
    return dict(prefit_sha256=sha(study.ROOT/'prefit_lookup_verification.json'),
        counts=pre['counts'],new_regression_fits=4,canonical_controls_to_reuse=4,
        composite_2024_components_explicitly_checked_beyond_model_report_lookup=True)


def reuse(fold):
    checked();p=model.setup('control',fold);root=base.ROOT;old=OLD[fold];year=fold[:4]
    assert not (root/'model_report.json').exists()
    original=json.loads((old/'model_report.json').read_text())
    ov=json.loads((old/'model_verification.json').read_text())
    assert ov['passed'] and ov['model_report_sha256']==sha(old/'model_report.json')
    fr=json.loads((FEATURES[year]/'feature_report.json').read_text())
    lr=json.loads((LABELS[year]/'full_label_report.json').read_text())
    assert original['feature_report_sha256']==sha(FEATURES[year]/'feature_report.json')
    assert original['label_report_sha256']==sha(LABELS[year]/'full_label_report.json')
    assert fr['features_sha256']==sha(FEATURES[year]/'features.parquet')
    assert lr['labels_sha256']==sha(LABELS[year]/'full_labels.parquet')
    fresh=relative.training('relative')
    # Independent SQL uses the original source labels and original source
    # features, not the new projected training frame or the new target builder.
    old_f=pd.read_parquet(FEATURES[year]/'features.parquet',
        columns=['date','code','formula_input_valid',*study.ARMS['control']])
    c=base.conn();c.register('old_features',old_f)
    fields=','.join('f.'+n for n in study.ARMS['control'])
    a=c.sql(f'''WITH labels AS(SELECT date,code,next_date,opportunity15,
        opportunity15-avg(opportunity15) OVER(PARTITION BY date) AS target
        FROM read_parquet('{LABELS[year]}/full_labels.parquet')
        WHERE known15 AND date>='{p['training_start']}' AND next_date<'{p['training_end']}')
        SELECT date,code,next_date,opportunity15,target,
        1./count(*) OVER(PARTITION BY date) AS w,{fields}
        FROM old_features f JOIN labels USING(date,code) WHERE formula_input_valid ORDER BY date,code''').df()
    c.close()
    names=['date','code','next_date','opportunity15',*study.ARMS['control']]
    pd.testing.assert_frame_equal(fresh[names],a[names],check_exact=True,check_dtype=False)
    np.testing.assert_allclose(fresh.target,a.target,rtol=0,atol=2e-12)
    np.testing.assert_array_equal(1/fresh.groupby('date').code.transform('size'),a.w)
    assert len(a)==p['expected_training_rows']==original['rows']
    assert a.date.nunique()==p['expected_training_days']==original['days']
    assert a.next_date.max()==p['expected_last_observation']==original['last_observation']
    if year=='2024':
        component=json.loads((old/'components/full.json').read_text())
        assert component==original['full']
        params=component['fitted_parameters']
        r=dict(parameters=params,feature_names=list(study.ARMS['control']),variant='relative',
            learning_rate=component['learning_rate'],bias=component['bias'],trees=component['trees'],
            rows=original['rows'],days=original['days'],last_observation=original['last_observation'],
            training_start=original['training_start'],training_end=original['training_end'],
            new_2025_score_groups_read=False,new_2025H2_score_groups_read=False,new_2026_prices_read=False,no_exit_rules=True)
        score=base.predict(base.encode(fresh),r)
        r['thresholds']=[dict(id=i,training_quantile=q,threshold=float(np.quantile(score,q)))
            for i,q in enumerate(base.QUANTILES)]
        old_cut=original['thresholds']['full']['threshold']
        np.testing.assert_allclose(r['thresholds'][3]['threshold'],old_cut,rtol=0,atol=2e-11)
        r['thresholds'][3]['threshold']=old_cut
    else:
        component=original;r=original.copy();params=r['parameters'];old_cut=r['thresholds'][3]['threshold']
    assert r['feature_names']==p['feature_names']
    assert all(params[k]==v for k,v in p['parameters'].items())
    r.update(protocol_sha256=sha(base.PROTOCOL),feature_report_sha256=sha(study.INPUTS/'feature_report.json'),
        label_report_sha256=sha(study.INPUTS/'full_label_report.json'),
        reused_model_report_sha256=sha(old/'model_report.json'),no_model_fit_performed=True,
        reused_source_root=str(old),reuse_protocol_sha256=sha(PROTOCOL))
    root.mkdir(parents=True,exist_ok=True);save_json(root/'model_report.json',r)
    assert r['trees']==component['trees'] and r['bias']==component['bias']
    receipt=dict(passed=True,fold=fold,source_root=str(old),
        original_model_report_sha256=sha(old/'model_report.json'),model_report_sha256=sha(root/'model_report.json'),
        reuse_protocol_sha256=sha(PROTOCOL),rows=len(a),days=a.date.nunique(),
        all_training_keys_labels_targets_weights_and_48_raw_inputs_equal=True,
        all_fitted_parameters_trees_bias_and_q995_unchanged=True,
        no_model_fit_performed=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'model_reuse_verification.json',receipt)
    return receipt


def scores(fold):
    checked();model.setup('control',fold);root=base.ROOT;old=OLD[fold]
    original=json.loads((old/'model_report.json').read_text())
    sr=json.loads((old/'score_report.json').read_text())
    sv=json.loads((old/'score_verification.json').read_text())
    assert sv['passed'] and sv['score_report_sha256']==sha(old/'score_report.json')
    assert sr['scores_sha256']==sha(old/'scores.parquet')
    old_s=pd.read_parquet(old/'scores.parquet')
    score_name='full_score' if fold.startswith('2024') else 'score'
    cols=[*study.META,score_name]
    old_s=old_s[cols].rename(columns={score_name:'score'})
    new=pd.read_parquet(root/'scores.parquet')
    overlap=new.loc[new.date.between(old_s.date.min(),old_s.date.max())].reset_index(drop=True)
    pd.testing.assert_frame_equal(overlap.drop(columns='score'),old_s.drop(columns='score'),check_exact=True)
    np.testing.assert_allclose(overlap.score,old_s.score,rtol=0,atol=2e-11,equal_nan=True)
    cut=original['thresholds']['full']['threshold'] if fold.startswith('2024') else original['thresholds'][3]['threshold']
    np.testing.assert_array_equal(overlap.score.gt(cut),old_s.score.gt(cut))
    receipt=dict(passed=True,fold=fold,source_root=str(old),old_scores_sha256=sha(old/'scores.parquet'),
        score_report_sha256=sha(root/'score_report.json'),overlap_rows=len(overlap),
        all_old_score_metadata_values_and_q995_flags_equal=True,
        additional_2023_or_2025_rows_only_equation_replay_not_new_model_fit=True,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'score_reuse_verification.json',receipt)
    return receipt


def complete():
    checked();receipts={};records=[]
    for fold in OLD:
        root=study.ROOT/'control'/fold
        for file,key in [('model_reuse_verification.json','model_report_sha256'),
                         ('score_reuse_verification.json','score_report_sha256')]:
            v=json.loads((root/file).read_text());assert v['passed']
            assert v[key]==sha(root/('model_report.json' if key.startswith('model') else 'score_report.json'))
            receipts[str(root/file)]=sha(root/file)
        m=json.loads((root/'model_report.json').read_text());assert m['no_model_fit_performed']
        for kind in ['model','score']:
            v=json.loads((root/(kind+'_verification.json')).read_text())
            assert v['passed'] and v[kind+'_report_sha256']==sha(root/(kind+'_report.json'))
        records.append(dict(fold=fold,source=str(OLD[fold]),no_model_fit_performed=True))
    out=dict(passed=True,reuse_protocol_sha256=sha(PROTOCOL),source_hashes=receipts,records=records,
        all_four_canonical_controls_reused_before_new_group_results=True,
        four_regression_models_are_the_only_new_fits=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(study.ROOT/'control_reuse_verification.json',out)
    return dict(control_reuse_sha256=sha(study.ROOT/'control_reuse_verification.json'))


def checked_complete():
    checked();p=study.ROOT/'control_reuse_verification.json';r=json.loads(p.read_text())
    assert r['passed'] and r['reuse_protocol_sha256']==sha(PROTOCOL)
    for file,digest in r['source_hashes'].items():assert sha(Path(file))==digest
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['prepare','reuse','scores','complete']);p.add_argument('--fold',choices=list(OLD))
    a=p.parse_args();result=globals()[a.stage](a.fold) if a.stage in ['reuse','scores'] else globals()[a.stage]()
    print(json.dumps(result,ensure_ascii=False,indent=2))
