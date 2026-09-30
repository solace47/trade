"""Freeze two new morning-range models and two reused controls before statistics."""
import json
from pathlib import Path

import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_morning_range as study
from trade_research import tail_formula_morning_range_model as model
from trade_research.corporate_cash import save_json, sha
from freeze_tail_formula_bipower_gap import checked_selection


def freeze():
    p = model.checked(); assert not (study.ROOT/'joint_selection_freeze.json').exists()
    configs = {}; receipts = {str(model.PROTOCOL):sha(model.PROTOCOL)}; models = []
    for arm in study.ARMS:
        for fold in p['folds']:
            q = model.setup(arm,fold); root = base.ROOT
            m = json.loads((root/'model_report.json').read_text()); sr = json.loads((root/'score_report.json').read_text())
            assert m['protocol_sha256']==sr['protocol_sha256']==sha(base.PROTOCOL)
            assert sr['model_report_sha256']==sha(root/'model_report.json') and sr['scores_sha256']==sha(root/'scores.parquet')
            assert m['feature_names']==list(study.ARMS[arm]) and m['variant']=='relative'
            assert m['feature_report_sha256']==sr['feature_report_sha256']==sha(study.INPUTS/'feature_report.json')
            assert m['label_report_sha256']==sha(study.INPUTS/'full_label_report.json')
            assert m['rows']==q['expected_training_rows'] and m['days']==q['expected_training_days']
            assert m['last_observation']==q['expected_last_observation']<q['evaluation_start']
            assert all(m['parameters'][k]==v for k,v in p['parameters'].items())
            assert m['thresholds'][3]['training_quantile']==.995
            for kind in ['model','score']:
                v = json.loads((root/(kind+'_verification.json')).read_text())
                assert v['passed'] and v[kind+'_report_sha256']==sha(root/(kind+'_report.json'))
            if arm=='control':
                v = json.loads((root/'control_reuse_verification.json').read_text())
                assert v['passed'] and v['master_protocol_sha256']==sha(model.PROTOCOL) and m['no_model_fit_performed']
                for f,d in v['source_hashes'].items():assert sha(Path(f))==d
                receipts[str(root/'control_reuse_verification.json')]=sha(root/'control_reuse_verification.json')
            else:
                assert not m.get('no_model_fit_performed',False)
            configs[(arm,fold)]=(q,m)
            for file in ['model_report.json','model_verification.json','score_report.json','score_verification.json','scores.parquet']:
                receipts[str(root/file)]=sha(root/file)
            receipts[str(base.PROTOCOL)]=sha(base.PROTOCOL)
    f = pd.read_parquet(study.INPUTS/'features.parquet',columns=study.META)
    meta = study.META[:-1]; keys = f.loc[f.date.ge('2024-01-01'),meta].reset_index(drop=True)
    assert len(keys)==1258085 and keys.date.lt('2026-01-01').all()
    selections = []; equality = []
    for arm in study.ARMS:
        flags = []; queries = []
        for fold in p['folds']:
            q,m = configs[(arm,fold)]; root = study.ROOT/arm/fold; cut = m['thresholds'][3]['threshold']
            d = pd.read_parquet(root/'scores.parquet',filters=[('date','>=',q['evaluation_start']),('date','<',q['evaluation_end'])])
            d['selected']=d.formula_input_valid & d.score.gt(cut); flags.append(d[['date','code','selected']])
            pd.testing.assert_frame_equal(d[meta].reset_index(drop=True),keys.loc[keys.date.ge(q['evaluation_start']) & keys.date.lt(q['evaluation_end'])].reset_index(drop=True),check_exact=True)
            queries.append(f'''SELECT date,code,coalesce(formula_input_valid AND score>{cut:.17e},false) AS selected
                FROM read_parquet('{root}/scores.parquet') WHERE date>='{q['evaluation_start']}' AND date<'{q['evaluation_end']}' ''')
            core = root/'frozen_numeric_core.tdx'; core.write_text(base.native_core(m,cut,study.ARMS[arm],study.HEADER))
            receipts[str(core)]=sha(core)
            models.append(dict(arm=arm,fold=fold,rows=m['rows'],days=m['days'],last_observation=m['last_observation'],
                selected=int(d.selected.sum()),signal_days=d.loc[d.selected,'date'].nunique(),
                new_morning_split_nodes=sum(int(i>=48) for t in m['trees'] for i in t['feature'] if i>=0),
                new_fit=arm=='morning',software_compilation_verified=False,native_source_parity_verified=False))
        out = keys.merge(pd.concat(flags,ignore_index=True),on=['date','code'],how='left',validate='one_to_one')
        out['selected']=out.selected.eq(True)
        c = base.conn(); c.register('keys',keys); c.sql(' UNION ALL '.join(queries)).create_view('flags')
        expected = c.sql('SELECT k.*,coalesce(selected,false) AS selected FROM keys k LEFT JOIN flags USING(date,code) ORDER BY date,code').df(); c.close()
        pd.testing.assert_frame_equal(out,expected,check_exact=True)
        assert out.loc[out.selected,'date'].ge('2025-01-01').all()
        group = arm+'2025'; root = study.ROOT/group; root.mkdir(parents=True,exist_ok=True)
        assert not (root/'selection_report.json').exists(); out.to_parquet(root/'selection.parquet',index=False,compression='zstd')
        r = dict(protocol_sha256=sha(model.PROTOCOL),group=group,selection_sha256=sha(root/'selection.parquet'),
            rows=len(out),selected=int(out.selected.sum()),days=out.loc[out.selected,'date'].nunique(),source_hashes=receipts.copy(),
            no_new_group_evaluation=True,new_2026_prices_read=False,no_exit_rules=True)
        save_json(root/'selection_report.json',r)
        save_json(root/'selection_verification.json',dict(passed=True,selection_report_sha256=sha(root/'selection_report.json'),
            rows=len(out),all_original_pool_metadata_and_half_flags_sql_verified=True,no_training_period_selection=True,
            new_2026_prices_read=False,no_exit_rules=True))
        selections.append(dict(group=group,root=str(root),selected=r['selected'],days=r['days'],rows=len(out)))
        for name,path in p['controls'].items():
            equality.append(dict(left=group,right=name,full_frame_equal=out.equals(checked_selection(Path(path)))))
        for file in ['selection.parquet','selection_report.json','selection_verification.json']:
            receipts[str(root/file)]=sha(root/file)
    assert checked_selection(study.ROOT/'control2025').equals(checked_selection(Path(p['controls']['control2025'])))
    out = dict(passed=True,model_protocol_sha256=sha(model.PROTOCOL),source_hashes=receipts,models=models,
        selections=selections,equality=equality,all_two_new_models_two_reused_controls_and_two_full_lists_jointly_frozen=True,
        no_parent_refitting_or_prediction=True,no_new_raw_extraction=True,year_2025_is_exploratory=True,
        no_new_group_evaluation=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(study.ROOT/'joint_selection_freeze.json',out)
    return dict(joint_sha256=sha(study.ROOT/'joint_selection_freeze.json'),models=models,selections=selections,equality=equality)


if __name__=='__main__':
    print(json.dumps(freeze(),ensure_ascii=False,indent=2))
