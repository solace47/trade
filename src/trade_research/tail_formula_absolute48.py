"""Evaluate the already-fitted absolute opportunity score without the rank gate."""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as inputs
from .corporate_cash import save_json,sha
from .tail_formula_1000_daily import analyze as common_analysis

COMBINED=Path('data/research/tail_formula_absolute48_2025')
COMBINED_PROTOCOL=Path('config/tail_formula_absolute48_combined_protocol.json')


def root(fold):
    return Path('data/research/tail_formula_absolute48_'+fold)


def protocol(fold):
    return Path('config/tail_formula_absolute48_'+fold+'_protocol.json')


def source(fold):
    config=json.loads(protocol(fold).read_text());path=Path(config['source_root'])
    assert config['absolute_score_cut']==.60 and not config['model_refitted']
    assert config['feature_report_sha256']==sha(inputs.ROOT/'feature_report.json')
    for name,digest in config['source_sha256'].items():
        assert sha(path/name)==digest
    m=json.loads((path/'model_report.json').read_text());s=json.loads((path/'score_report.json').read_text())
    for kind in ['model','score']:
        p=json.loads((path/f'{kind}_verification.json').read_text())
        assert p['passed'] and p[f'{kind}_report_sha256']==sha(path/f'{kind}_report.json')
    assert s['model_report_sha256']==sha(path/'model_report.json') and s['scores_sha256']==sha(path/'scores.parquet')
    assert m['variant']=='absolute' and m['feature_names']==list(inputs.EXPRESSIONS)
    assert m['feature_report_sha256']==config['feature_report_sha256']
    for key in ['training_start','training_end']:
        assert m[key]==config[key]
    assert m['last_observation']<config['evaluation_start']
    gate=json.loads((path/'selection_report.json').read_text())
    assert gate['absolute_score_cut']==.60 and gate['absolute_core_sha256']==sha(path/'absolute_numeric_core.tdx')
    assert (path/'absolute_numeric_core.tdx').read_text()==base.native_core(m,.60,inputs.EXPRESSIONS,inputs.HEADER)
    return config,path,m


def freeze(fold):
    path=root(fold)
    if (path/'selection_report.json').exists():
        raise ValueError('Do not replace the frozen standalone absolute-score selection')
    config,prior,m=source(fold);f=pd.read_parquet(prior/'scores.parquet')
    out=f[['date','code','half','board','decision_shares']].copy()
    out['selected']=(f.formula_input_valid&f.score.gt(.60)
        &f.date.ge(config['evaluation_start'])&f.date.lt(config['evaluation_end']))
    path.mkdir(parents=True,exist_ok=True);out.to_parquet(path/'selection.parquet',index=False,compression='zstd')
    (path/'model_report.json').write_bytes((prior/'model_report.json').read_bytes())
    (path/'frozen_numeric_core.tdx').write_bytes((prior/'absolute_numeric_core.tdx').read_bytes())
    r=dict(protocol_sha256=sha(protocol(fold)),model_report_sha256=sha(path/'model_report.json'),
        source_score_report_sha256=sha(prior/'score_report.json'),source_selection_report_sha256=sha(prior/'selection_report.json'),
        selection_sha256=sha(path/'selection.parquet'),core_sha256=sha(path/'frozen_numeric_core.tdx'),
        absolute_score_cut=.60,chosen_threshold=dict(kind='fixed_absolute_score',threshold=.60,training_quantile=None),
        selected=int(out.selected.sum()),by_half=out.groupby('half').selected.agg(['size','sum']).reset_index().to_dict('records'),
        evaluation_start=config['evaluation_start'],evaluation_end=config['evaluation_end'],
        model_refitted=False,rank_intersection_used=False,score_is_uncalibrated=True,
        new_group_outcomes_read=False,original_rank_gate_outcomes_previously_seen=True,
        year_2025_is_exploratory=True,new_2026_prices_read=False,no_exit_rules=True,software_compilation_verified=False)
    save_json(path/'selection_report.json',r);return r


def verify(fold):
    config,prior,m=source(fold);path=root(fold);r=json.loads((path/'selection_report.json').read_text())
    for key,file in [('protocol_sha256',protocol(fold)),('model_report_sha256',path/'model_report.json'),
        ('source_score_report_sha256',prior/'score_report.json'),('source_selection_report_sha256',prior/'selection_report.json'),
        ('selection_sha256',path/'selection.parquet'),('core_sha256',path/'frozen_numeric_core.tdx')]:
        assert r[key]==sha(file)
    assert sha(path/'model_report.json')==sha(prior/'model_report.json')
    assert sha(path/'frozen_numeric_core.tdx')==sha(prior/'absolute_numeric_core.tdx')
    assert r['absolute_score_cut']==.60 and r['chosen_threshold']==dict(kind='fixed_absolute_score',threshold=.60,training_quantile=None)
    c=base.conn();start,end=config['evaluation_start'],config['evaluation_end']
    expected=c.sql(f'''SELECT date,code,half,board,decision_shares,
        formula_input_valid AND score>0.60 AND date>='{start}' AND date<'{end}' AS selected
        FROM read_parquet('{prior}/scores.parquet') ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(path/'selection.parquet'),expected,check_exact=True)
    assert int(expected.selected.sum())==r['selected']
    assert expected.loc[expected.selected,'date'].ge(m['training_end']).all()
    p=dict(passed=True,selection_report_sha256=sha(path/'selection_report.json'),rows=len(expected),
        all_standalone_absolute_flags_rebuilt=True,original_model_score_and_numeric_core_preserved=True,
        no_training_period_selection=True,no_new_group_outcomes_read=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(path/'selection_verification.json',p);return p


def checked(fold):
    config,_,m=source(fold);path=root(fold)
    s=json.loads((path/'selection_report.json').read_text());p=json.loads((path/'selection_verification.json').read_text())
    assert p['passed'] and p['selection_report_sha256']==sha(path/'selection_report.json')
    assert s['protocol_sha256']==sha(protocol(fold)) and s['selection_sha256']==sha(path/'selection.parquet')
    assert s['model_report_sha256']==sha(path/'model_report.json') and s['core_sha256']==sha(path/'frozen_numeric_core.tdx')
    assert s['evaluation_start']==config['evaluation_start'] and s['evaluation_end']==config['evaluation_end']
    return config,s,m


def combine():
    if (COMBINED/'selection_report.json').exists():
        raise ValueError('Do not replace the fixed absolute-score linkage')
    assert all(not (root(fold)/'analysis_report.json').exists() for fold in ['2024','recent'])
    COMBINED.mkdir(parents=True,exist_ok=True);folds=[];frames=[]
    for fold,half in [('2024','2025H1'),('recent','2025H2')]:
        cfg,s,m=checked(fold);path=root(fold);f=pd.read_parquet(path/'selection.parquet')
        assert f.loc[f.selected,'date'].ge(cfg['evaluation_start']).all() and f.loc[f.selected,'date'].lt(cfg['evaluation_end']).all()
        core=COMBINED/f'frozen_numeric_core_{half}.tdx';core.write_bytes((path/'frozen_numeric_core.tdx').read_bytes())
        folds.append(dict(fold=fold,half=half,start=cfg['evaluation_start'],end=cfg['evaluation_end'],root=str(path),
            protocol_sha256=sha(protocol(fold)),selection_report_sha256=sha(path/'selection_report.json'),
            model_report_sha256=sha(path/'model_report.json'),core_sha256=sha(core),selected=int(f.selected.sum())))
        frames.append(f)
    pd.testing.assert_frame_equal(frames[0].drop(columns='selected'),frames[1].drop(columns='selected'),check_exact=True)
    assert not (frames[0].selected&frames[1].selected).any()
    out=frames[0].copy();out['selected'] |= frames[1].selected
    out.to_parquet(COMBINED/'selection.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(COMBINED_PROTOCOL),folds=folds,selection_sha256=sha(COMBINED/'selection.parquet'),
        absolute_score_cut=.60,selected=int(out.selected.sum()),
        by_half=out.groupby('half').selected.agg(['size','sum']).reset_index().to_dict('records'),
        identical_coefficients_all_year=False,model_refitted=False,rank_intersection_used=False,
        no_new_group_outcomes_read=True,original_rank_gate_outcomes_previously_seen=True,
        year_2025_is_exploratory=True,new_2026_prices_read=False,no_exit_rules=True,software_compilation_verified=False)
    save_json(COMBINED/'selection_report.json',r);return r


def verify_combined():
    r=json.loads((COMBINED/'selection_report.json').read_text())
    assert r['protocol_sha256']==sha(COMBINED_PROTOCOL) and r['selection_sha256']==sha(COMBINED/'selection.parquet')
    assert [x['fold'] for x in r['folds']]==['2024','recent'] and r['absolute_score_cut']==.60
    queries=[]
    for fold in r['folds']:
        cfg,s,m=checked(fold['fold']);path=Path(fold['root'])
        assert fold['selection_report_sha256']==sha(path/'selection_report.json')
        assert fold['protocol_sha256']==sha(protocol(fold['fold'])) and fold['model_report_sha256']==sha(path/'model_report.json')
        assert fold['core_sha256']==sha(COMBINED/f"frozen_numeric_core_{fold['half']}.tdx")
        assert fold['start']==cfg['evaluation_start'] and fold['end']==cfg['evaluation_end']
        queries.append(f"SELECT date,code,half,board,decision_shares,selected AND date>='{fold['start']}' AND date<'{fold['end']}' AS selected FROM read_parquet('{path}/selection.parquet')")
    c=base.conn()
    expected=c.sql('SELECT date,code,half,board,decision_shares,bool_or(selected) AS selected FROM ('+
        ' UNION ALL '.join(queries)+') GROUP BY date,code,half,board,decision_shares ORDER BY date,code').df();c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(COMBINED/'selection.parquet'),expected,check_exact=True)
    assert int(expected.selected.sum())==r['selected']==sum(f['selected'] for f in r['folds'])
    p=dict(passed=True,selection_report_sha256=sha(COMBINED/'selection_report.json'),rows=len(expected),
        all_time_windows_and_flags_rebuilt=True,no_new_group_outcomes_read=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(COMBINED/'selection_verification.json',p);return p


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold',choices=['2024','recent','combined'])
    p.add_argument('stage',choices=['freeze','verify','analyze'])
    a=p.parse_args()
    if a.fold=='combined':
        r=combine() if a.stage=='freeze' else verify_combined() if a.stage=='verify' else common_analysis(COMBINED,COMBINED_PROTOCOL)
    elif a.stage=='analyze':
        assert (COMBINED/'selection_verification.json').exists()
        r=common_analysis(root(a.fold),protocol(a.fold))
    else:
        r=globals()[a.stage](a.fold)
    print(json.dumps(r,ensure_ascii=False,indent=2))
