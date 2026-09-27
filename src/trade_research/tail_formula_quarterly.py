"""Quarterly rolling-12-month updates of the unchanged 48-input formula."""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as inputs
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json,sha
from .tail_formula_1000_daily import analyze as common_analysis

COMBINED=Path('data/research/tail_formula_quarterly_2025')
COMBINED_PROTOCOL=Path('config/tail_formula_quarterly_combined_protocol.json')


def root(fold):
    return Path('data/research/tail_formula_quarterly_'+fold)


def protocol(fold):
    return Path('config/tail_formula_quarterly_'+fold+'_protocol.json')


def setup(fold):
    inputs.setup('2024' if fold in ['q1','q2'] else 'recent')
    base.ROOT=root(fold);base.PROTOCOL=protocol(fold);relative.PROTOCOL=protocol(fold)
    study.ROOT=base.ROOT;study.PROTOCOL=base.PROTOCOL
    config=json.loads(base.PROTOCOL.read_text())
    assert config['feature_report_sha256']==sha(inputs.ROOT/'feature_report.json')


def source(config):
    path=Path(config['source_root'])
    for name,key in [('model_report.json','source_model_report_sha256'),
        ('score_report.json','source_score_report_sha256'),('selection_report.json','source_selection_report_sha256'),
        ('frozen_numeric_core.tdx','source_core_sha256')]:
        assert sha(path/name)==config[key]
    for name,key,target in [('model_verification.json','model_report_sha256','model_report.json'),
        ('score_verification.json','score_report_sha256','score_report.json'),
        ('selection_verification.json','selection_report_sha256','selection_report.json')]:
        p=json.loads((path/name).read_text());assert p['passed'] and p[key]==sha(path/target)
    m=json.loads((path/'model_report.json').read_text())
    score=json.loads((path/'score_report.json').read_text())
    assert score['scores_sha256']==sha(path/'scores.parquet')
    assert score['model_report_sha256']==sha(path/'model_report.json')
    assert m['training_start']==config['training_start'] and m['training_end']==config['training_end']
    assert m['last_observation']<config['evaluation_start'] and m['feature_names']==list(inputs.EXPRESSIONS)
    return path,m


def reuse(fold,stage):
    path=root(fold);config=json.loads(protocol(fold).read_text());prior,m=source(config)
    assert config['reuse_existing_model'] and fold in ['q1','q3']
    start,end=config['evaluation_start'],config['evaluation_end'];cut=m['thresholds'][3]
    assert cut['training_quantile']==.995
    if stage=='freeze':
        if (path/'selection_report.json').exists():
            raise ValueError('Do not replace a frozen reused quarter')
        f=pd.read_parquet(prior/'scores.parquet')
        out=f[['date','code','half','board','decision_shares']].copy()
        out['selected']=f.formula_input_valid&f.score.gt(cut['threshold'])&f.date.ge(start)&f.date.lt(end)
        path.mkdir(parents=True,exist_ok=True);out.to_parquet(path/'selection.parquet',index=False,compression='zstd')
        for file in ['model_report.json','frozen_numeric_core.tdx']:
            (path/file).write_bytes((prior/file).read_bytes())
        r=dict(protocol_sha256=sha(protocol(fold)),model_report_sha256=sha(path/'model_report.json'),
            core_sha256=sha(path/'frozen_numeric_core.tdx'),source_root=str(prior),
            source_score_report_sha256=sha(prior/'score_report.json'),source_selection_report_sha256=sha(prior/'selection_report.json'),
            selection_sha256=sha(path/'selection.parquet'),chosen_threshold=cut,selected=int(out.selected.sum()),
            evaluation_start=start,evaluation_end=end,evaluation_quarter=fold,model_reused_without_refit=True,
            original_model_outcomes_previously_seen=True,new_schedule_outcomes_read=False,
            year_2025_is_exploratory=True,new_2026_prices_read=False,no_exit_rules=True,software_compilation_verified=False)
        save_json(path/'selection_report.json',r);return r
    assert stage=='verify'
    r=json.loads((path/'selection_report.json').read_text())
    for key,file in [('protocol_sha256',protocol(fold)),('selection_sha256',path/'selection.parquet'),
        ('model_report_sha256',path/'model_report.json'),('core_sha256',path/'frozen_numeric_core.tdx'),
        ('source_score_report_sha256',prior/'score_report.json'),('source_selection_report_sha256',prior/'selection_report.json')]:
        assert r[key]==sha(file)
    assert sha(path/'model_report.json')==sha(prior/'model_report.json')
    assert sha(path/'frozen_numeric_core.tdx')==sha(prior/'frozen_numeric_core.tdx')
    assert r['evaluation_start']==start and r['evaluation_end']==end and r['chosen_threshold']==cut
    c=base.conn()
    expected=c.sql(f'''SELECT date,code,half,board,decision_shares,
        date>='{start}' AND date<'{end}' AND formula_input_valid AND score>{cut['threshold']:.17e} AS selected
        FROM read_parquet('{prior}/scores.parquet') ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(pd.read_parquet(path/'selection.parquet'),expected,check_exact=True)
    assert int(expected.selected.sum())==r['selected']
    p=dict(passed=True,selection_report_sha256=sha(path/'selection_report.json'),rows=len(expected),
        all_reused_model_flags_independently_rebuilt=True,original_model_and_core_bytes_unchanged=True,
        all_selected_dates_after_training=True,new_schedule_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(path/'selection_verification.json',p);return p


def checked_fold(fold):
    path=root(fold);s=json.loads((path/'selection_report.json').read_text())
    p=json.loads((path/'selection_verification.json').read_text());m=json.loads((path/'model_report.json').read_text())
    cfg=json.loads(protocol(fold).read_text())
    assert p['passed'] and p['selection_report_sha256']==sha(path/'selection_report.json')
    assert s['protocol_sha256']==sha(protocol(fold)) and s['selection_sha256']==sha(path/'selection.parquet')
    assert s['model_report_sha256']==sha(path/'model_report.json') and s['core_sha256']==sha(path/'frozen_numeric_core.tdx')
    assert s['evaluation_start']==cfg['evaluation_start'] and s['evaluation_end']==cfg['evaluation_end']
    assert m['training_start']==cfg['training_start'] and m['training_end']==cfg['training_end']
    assert m['last_observation']<cfg['evaluation_start'] and s['chosen_threshold']['training_quantile']==.995
    return s,m,cfg


def combine():
    if (COMBINED/'selection_report.json').exists():
        raise ValueError('Do not replace the frozen quarterly schedule')
    assert all(not (root(f'q{i}')/'analysis_report.json').exists() for i in range(1,5))
    COMBINED.mkdir(parents=True,exist_ok=True);frames=[];folds=[]
    for i in range(1,5):
        fold=f'q{i}';path=root(fold);s,m,cfg=checked_fold(fold)
        f=pd.read_parquet(path/'selection.parquet');selected=f.loc[f.selected,'date']
        assert selected.ge(cfg['evaluation_start']).all() and selected.lt(cfg['evaluation_end']).all()
        frames.append(f);core=COMBINED/f'frozen_numeric_core_{fold}.tdx'
        core.write_bytes((path/'frozen_numeric_core.tdx').read_bytes())
        folds.append(dict(quarter=fold,root=str(path),start=cfg['evaluation_start'],end=cfg['evaluation_end'],
            protocol_sha256=sha(protocol(fold)),selection_report_sha256=sha(path/'selection_report.json'),
            model_report_sha256=sha(path/'model_report.json'),core_sha256=sha(core),selected=int(f.selected.sum()),
            model_reused=cfg['reuse_existing_model'],training_includes_2025H2=cfg['training_includes_2025H2']))
    out=frames[0].copy();out['selected']=False
    for f in frames:
        pd.testing.assert_frame_equal(out.drop(columns='selected'),f.drop(columns='selected'),check_exact=True)
        assert not (out.selected&f.selected).any();out['selected'] |= f.selected
    out.to_parquet(COMBINED/'selection.parquet',index=False,compression='zstd')
    old=Path('data/research/tail_formula_float_2025/selection_report.json')
    config=json.loads(COMBINED_PROTOCOL.read_text())
    assert sha(old)==config['original_48_half_year_selection_report_sha256']
    r=dict(protocol_sha256=sha(COMBINED_PROTOCOL),folds=folds,selection_sha256=sha(COMBINED/'selection.parquet'),
        original_half_year_selection_report_sha256=sha(old),selected=int(out.selected.sum()),
        by_half=out.groupby('half').selected.agg(['size','sum']).reset_index().to_dict('records'),
        updates_per_year=4,identical_coefficients_all_year=False,all_evaluation_dates_after_model_training=True,
        new_quarter_schedule_outcomes_read=False,previous_2025_outcomes_seen=True,year_2025_is_exploratory=True,
        new_2026_prices_read=False,no_exit_rules=True,software_compilation_verified=False)
    save_json(COMBINED/'selection_report.json',r);return r


def verify_combined():
    r=json.loads((COMBINED/'selection_report.json').read_text())
    assert r['protocol_sha256']==sha(COMBINED_PROTOCOL) and r['selection_sha256']==sha(COMBINED/'selection.parquet')
    assert [f['quarter'] for f in r['folds']]==['q1','q2','q3','q4'] and r['updates_per_year']==4
    queries=[]
    for fold in r['folds']:
        s,m,cfg=checked_fold(fold['quarter']);path=Path(fold['root'])
        assert fold['selection_report_sha256']==sha(path/'selection_report.json')
        assert fold['model_report_sha256']==sha(path/'model_report.json') and fold['protocol_sha256']==sha(protocol(fold['quarter']))
        assert fold['core_sha256']==sha(COMBINED/f"frozen_numeric_core_{fold['quarter']}.tdx")
        assert fold['start']==cfg['evaluation_start'] and fold['end']==cfg['evaluation_end']
        queries.append(f"SELECT date,code,half,board,decision_shares,selected AND date>='{fold['start']}' AND date<'{fold['end']}' AS selected FROM read_parquet('{path}/selection.parquet')")
    c=base.conn()
    expected=c.sql('''SELECT date,code,half,board,decision_shares,bool_or(selected) AS selected
        FROM ('''+ ' UNION ALL '.join(queries)+''') GROUP BY date,code,half,board,decision_shares ORDER BY date,code''').df()
    c.close();pd.testing.assert_frame_equal(pd.read_parquet(COMBINED/'selection.parquet'),expected,check_exact=True)
    assert int(expected.selected.sum())==r['selected']==sum(f['selected'] for f in r['folds'])
    p=dict(passed=True,selection_report_sha256=sha(COMBINED/'selection_report.json'),rows=len(expected),
        all_four_time_windows_and_flags_rebuilt=True,all_evaluation_dates_after_training=True,
        new_schedule_outcomes_read=False,year_2025_is_exploratory=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(COMBINED/'selection_verification.json',p);return p


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold',choices=['q1','q2','q3','q4','combined'])
    p.add_argument('stage',choices=['model','verify_model','scores','freeze','verify','analyze'])
    a=p.parse_args()
    if a.fold=='combined':
        assert a.stage in ['freeze','verify','analyze']
        r=combine() if a.stage=='freeze' else verify_combined() if a.stage=='verify' else common_analysis(COMBINED,COMBINED_PROTOCOL)
    else:
        setup(a.fold);cfg=json.loads(base.PROTOCOL.read_text())
        if a.stage=='analyze':
            assert (COMBINED/'selection_verification.json').exists()
            r=common_analysis(base.ROOT,base.PROTOCOL)
        elif cfg['reuse_existing_model']:
            assert a.stage in ['freeze','verify'],'Existing models must not be refitted or rescored'
            r=reuse(a.fold,a.stage)
        elif a.stage in ['model','verify_model']:
            r=getattr(relative,a.stage)('relative')
        elif a.stage in ['freeze','verify']:
            r=getattr(study,a.stage)()
        else:
            r=base.scores()
    print(json.dumps(r,ensure_ascii=False,indent=2))
