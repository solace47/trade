"""A separate zero-cut selection from four already-frozen absolute-price models."""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_boundary_evaluation as evaluation
from . import tail_formula_float as inputs
from .tail_formula_context_2024 import normalized_selection_flags
from .corporate_cash import save_json, sha

STEM='tail_formula_absolute_zero'
PROTOCOL=Path('config')/(STEM+'_protocol.json')


def root_for(arm,fold):
    return Path('data/research')/(STEM+'_'+arm+'_'+fold)


def sources():
    p=json.loads(PROTOCOL.read_text())
    assert p['threshold']==0. and p['arms']==['squared','huber'] and not p['new_2026_prices_allowed']
    assert p['source_master_protocol_sha256']==sha(Path('config/tail_formula_endpoint_absolute_protocol.json'))
    for file,digest in p['source_hashes'].items():assert sha(Path(file))==digest
    for arm in p['arms']:
        for fold in p['folds']:
            root=Path('data/research/tail_formula_endpoint_absolute_'+arm+'_'+fold['name'])
            model=json.loads((root/'model_report.json').read_text())
            assert model['target_is_date_centered'] is False and model['variant']=='endpoint_absolute_'+arm
            assert model['feature_names']==list(inputs.EXPRESSIONS)
            assert model['last_observation']<model['training_end']<=fold['start']<fold['end']
            for stage in ['model','score']:
                v=json.loads((root/(stage+'_verification.json')).read_text())
                assert v['passed'] and v[stage+'_report_sha256']==sha(root/(stage+'_report.json'))
            score=json.loads((root/'score_report.json').read_text())
            assert score['scores_sha256']==sha(root/'scores.parquet')
            assert score['model_report_sha256']==sha(root/'model_report.json')
            assert score['feature_report_sha256']==sha(inputs.ROOT/'feature_report.json')
            original=json.loads((root/'selection_report.json').read_text())
            assert original['core_sha256']==sha(root/'frozen_numeric_core.tdx')
            assert (root/'frozen_numeric_core.tdx').read_text()==base.native_core(
                model,model['thresholds'][3]['threshold'],inputs.EXPRESSIONS,inputs.HEADER)
    return p


def freeze():
    p=sources()
    for arm in p['arms']:
        for fold in ['2024','recent','2025']:
            assert not (root_for(arm,fold)/'selection_report.json').exists()
            assert not (root_for(arm,fold)/'analysis_report.json').exists()
    result=[]
    for arm in p['arms']:
        frames=[];folds=[]
        for fold in p['folds']:
            root=root_for(arm,fold['name']);source=Path('data/research/tail_formula_endpoint_absolute_'+arm+'_'+fold['name'])
            root.mkdir(parents=True,exist_ok=True)
            m=json.loads((source/'model_report.json').read_text());f=pd.read_parquet(source/'scores.parquet')
            assert f.formula_input_valid.notna().all() and f.score.notna().equals(f.formula_input_valid)
            out=f[['date','code','half','board','decision_shares']].copy()
            out['selected']=f.formula_input_valid & f.score.gt(0.) & f.date.ge(fold['start']) & f.date.lt(fold['end'])
            out=normalized_selection_flags(out);out.to_parquet(root/'selection.parquet',index=False,compression='zstd')
            (root/'frozen_numeric_core.tdx').write_text(base.native_core(m,0.,inputs.EXPRESSIONS,inputs.HEADER))
            r=dict(protocol_sha256=sha(PROTOCOL),source_root=str(source),model_report_sha256=sha(source/'model_report.json'),
                score_report_sha256=sha(source/'score_report.json'),selection_sha256=sha(root/'selection.parquet'),
                core_sha256=sha(root/'frozen_numeric_core.tdx'),score_threshold=0.,selected=int(out.selected.sum()),
                evaluation_start=fold['start'],evaluation_end=fold['end'],model_refitted=False,
                old_q995_selection_unchanged=True,new_group_outcomes_read=False,year_2025_is_exploratory=True,
                new_2026_prices_read=False,no_exit_rules=True,software_compilation_verified=False)
            save_json(root/'selection_report.json',r);frames.append(out)
            folds.append(dict(root=str(root),selection_report_sha256=sha(root/'selection_report.json'),**fold))
            result.append(dict(arm=arm,fold=fold['name'],selected=r['selected']))
        pd.testing.assert_frame_equal(frames[0].drop(columns='selected'),frames[1].drop(columns='selected'),check_exact=True)
        combined=root_for(arm,'2025');combined.mkdir(parents=True,exist_ok=True)
        out=frames[0].copy();out['selected']=frames[0].selected | frames[1].selected
        out.to_parquet(combined/'selection.parquet',index=False,compression='zstd')
        for fold in folds:
            (combined/('frozen_numeric_core_'+fold['name']+'.tdx')).write_bytes((Path(fold['root'])/'frozen_numeric_core.tdx').read_bytes())
        save_json(combined/'selection_report.json',dict(protocol_sha256=sha(PROTOCOL),folds=folds,
            selection_sha256=sha(combined/'selection.parquet'),selected=int(out.selected.sum()),
            score_threshold=0.,model_refitted=False,method_updates_twice_per_year=True,year_2025_is_exploratory=True,
            new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True,software_compilation_verified=False))
        result.append(dict(arm=arm,fold='2025',selected=int(out.selected.sum())))
    return result


def verify():
    p=sources();proofs=[]
    for arm in p['arms']:
        for fold in p['folds']:
            root=root_for(arm,fold['name']);r=json.loads((root/'selection_report.json').read_text())
            source=Path('data/research/tail_formula_endpoint_absolute_'+arm+'_'+fold['name'])
            for field,file in [('protocol_sha256',PROTOCOL),('selection_sha256',root/'selection.parquet'),
                ('core_sha256',root/'frozen_numeric_core.tdx'),('model_report_sha256',source/'model_report.json'),
                ('score_report_sha256',source/'score_report.json')]:assert r[field]==sha(file)
            assert r['source_root']==str(source) and r['score_threshold']==0.
            assert r['evaluation_start']==fold['start'] and r['evaluation_end']==fold['end'] and not r['model_refitted']
            m=json.loads((source/'model_report.json').read_text())
            assert (root/'frozen_numeric_core.tdx').read_text()==base.native_core(m,0.,inputs.EXPRESSIONS,inputs.HEADER)
            c=base.conn();c.read_parquet(str(source/'scores.parquet')).create_view('scores')
            expected=c.sql(f'''SELECT date,code,half,board,decision_shares,
                date>='{fold['start']}' AND date<'{fold['end']}' AND formula_input_valid AND score>0. AS selected
                FROM scores ORDER BY date,code''').df();c.close()
            pd.testing.assert_frame_equal(normalized_selection_flags(pd.read_parquet(root/'selection.parquet')),
                normalized_selection_flags(expected),check_exact=True)
            assert int(expected.selected.sum())==r['selected']
            proof=dict(passed=True,selection_report_sha256=sha(root/'selection_report.json'),rows=len(expected),
                all_score_flags_dates_and_unchanged_model_source_rebuilt=True,zero_cut_native_core_verified=True,
                original_q995_selection_unchanged=True,no_training_period_selection=True,new_2026_prices_read=False)
            save_json(root/'selection_verification.json',proof);proofs.append(proof)
        root=root_for(arm,'2025');r=json.loads((root/'selection_report.json').read_text())
        assert r['protocol_sha256']==sha(PROTOCOL) and r['selection_sha256']==sha(root/'selection.parquet')
        assert r['score_threshold']==0. and len(r['folds'])==2 and not r['model_refitted']
        for fold in r['folds']:
            source=Path(fold['root']);assert fold['selection_report_sha256']==sha(source/'selection_report.json')
            assert (root/('frozen_numeric_core_'+fold['name']+'.tdx')).read_bytes()==(source/'frozen_numeric_core.tdx').read_bytes()
        c=base.conn()
        c.read_parquet(str(root_for(arm,'2024')/'selection.parquet')).create_view('a')
        c.read_parquet(str(root_for(arm,'recent')/'selection.parquet')).create_view('b')
        expected=c.sql('''SELECT a.date,a.code,a.half,a.board,a.decision_shares,
            (a.date>='2025-01-01' AND a.date<'2025-07-01' AND a.selected) OR
            (a.date>='2025-07-01' AND a.date<'2026-01-01' AND b.selected) AS selected
            FROM a JOIN b USING(date,code) ORDER BY date,code''').df();c.close()
        pd.testing.assert_frame_equal(normalized_selection_flags(pd.read_parquet(root/'selection.parquet')),
            normalized_selection_flags(expected),check_exact=True)
        assert int(expected.selected.sum())==r['selected']
        proof=dict(passed=True,selection_report_sha256=sha(root/'selection_report.json'),rows=len(expected),
            all_period_links_and_selection_flags_rebuilt=True,new_2026_prices_read=False)
        save_json(root/'selection_verification.json',proof);proofs.append(proof)
    return proofs


def analyze(arm,fold):
    p=sources()
    for a in p['arms']:
        for f in ['2024','recent','2025']:
            root=root_for(a,f);v=json.loads((root/'selection_verification.json').read_text())
            assert v['passed'] and v['selection_report_sha256']==sha(root/'selection_report.json')
    return evaluation.analyze(root_for(arm,'2025' if fold=='combined' else fold),PROTOCOL)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['freeze','verify','analyze'])
    parser.add_argument('--arm',choices=['squared','huber'],default='squared')
    parser.add_argument('--fold',choices=['2024','recent','combined'],default='combined')
    args=parser.parse_args()
    result=analyze(args.arm,args.fold) if args.stage=='analyze' else globals()[args.stage]()
    print(json.dumps(result,ensure_ascii=False,indent=2))
