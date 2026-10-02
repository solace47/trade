"""Matched four-fold test of five-day cost-adjusted quote history."""
import argparse
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as numeric
from trade_research import tail_formula_baseline as baseline
from trade_research import tail_formula_relative as relative
from trade_research.research_io import check_runtime, check_sources, save_json, sha
from find_existing_tail_formula_models import find
from verify_tail_formula_additive import tree_sql
from tail_formula_reports import checked_selection
import finish_tail_formula_rule_search as shared
import finish_tail_formula_profit_rule_search as comparisons

ROOT = Path('data/research/tail_formula_cost_history')
PROTOCOL = Path('config/tail_formula_cost_history_model_protocol.json')
EVALUATION = Path('config/tail_formula_cost_history_evaluation.json')
KEYS = shared.KEYS


def committed(path):
    assert subprocess.check_output(['git','show',f'HEAD:{path}']) == path.read_bytes()
    assert sha(path) in subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()


def checked():
    check_runtime(); committed(PROTOCOL)
    p = json.loads(PROTOCOL.read_text()); check_sources(p['source_hashes'])
    assert p['maximum_new_fits'] == 8 and p['training_quantile'] == .995
    assert p['target'] == 'relative' and not p['new_2026_prices_allowed']
    return p


def audit_atoms(p):
    """Rebuild every atom with independent SQL cash arithmetic and stock lags."""
    c = numeric.conn()
    c.read_parquet(str(ROOT/'stock_days.parquet')).create_view('calendar')
    c.read_parquet(str(ROOT/'windows.parquet')).create_view('windows')
    rebuilt = c.sql('''WITH b AS(SELECT *,floor(20000/q/100)*100 AS shares,
        coalesce(q_count=1 AND q_rows=1 AND shares>=100 AND entry_rows=4
          AND entry_clocks=4 AND entry_good=4 AND entry_capacity>=10*shares,false) AS entry_valid,
        shares*(entry_high+greatest(.0015*entry_high,.005)) AS purchase,
        purchase+greatest(.0003*purchase,5)+.00001*purchase AS cash
        FROM calendar LEFT JOIN windows USING(date,code)),
        l AS(SELECT *,lag(date) OVER w AS reference_date,lag(shares) OVER w AS previous_shares,
        lag(cash) OVER w AS previous_cash,lag(entry_valid) OVER w AS previous_valid
        FROM b WINDOW w AS(PARTITION BY code ORDER BY date)),
        t AS(SELECT *,coalesce(previous_valid AND reference_date<date AND morning_rows=29
          AND morning_clocks=29 AND morning_good=29 AND morning_close IS NOT NULL,false) AS atom_valid,
        previous_shares*(morning_close-greatest(.0015*morning_close,.005)) AS sale_mark,
        previous_shares*(sustained-greatest(.0015*sustained,.005)) AS sale_positive,
        CASE WHEN date>='2023-08-28' THEN .00051 ELSE .00101 END AS levy FROM l)
        SELECT date,code,reference_date,atom_valid,
        CASE WHEN atom_valid THEN 100*((sale_mark-greatest(.0003*sale_mark,5)-levy*sale_mark)/previous_cash-1) END AS proxy_mark,
        CASE WHEN atom_valid THEN coalesce((sale_positive-greatest(.0003*sale_positive,5)-levy*sale_positive)/previous_cash>1,false)::DOUBLE END AS proxy_positive
        FROM t ORDER BY code,date''').df(); c.close()
    actual = pd.read_parquet(ROOT/'history_atoms.parquet')
    pd.testing.assert_frame_equal(actual[['date','code','reference_date','atom_valid']],
                                  rebuilt[['date','code','reference_date','atom_valid']],check_dtype=False,check_exact=True)
    np.testing.assert_allclose(actual[['proxy_mark','proxy_positive']],rebuilt[['proxy_mark','proxy_positive']],
                              rtol=0,atol=2e-10,equal_nan=True)
    return len(actual)


def prefit():
    p = checked(); destination = ROOT/'prefit_verified.json'; assert not destination.exists()
    fr = json.loads((ROOT/'feature_report.json').read_text())
    fv = json.loads((ROOT/'feature_verification.json').read_text())
    assert fv['passed'] and fv['feature_report_sha256'] == sha(ROOT/'feature_report.json')
    assert fr['valid'] == p['expected_input_valid'] and fr['features_sha256'] == sha(ROOT/'features.parquet')
    atom_rows = audit_atoms(p)
    f = pd.read_parquet(ROOT/'features.parquet')
    old = baseline.original()
    pd.testing.assert_frame_equal(f[KEYS+list(baseline.EXPRESSIONS)],old[KEYS+list(baseline.EXPRESSIONS)],check_exact=True)
    assert (f.formula_input_valid <= old.formula_input_valid).all()
    labels = Path(p['label_root'])
    lv = json.loads((labels/'full_label_verification.json').read_text())
    lr = json.loads((labels/'full_label_report.json').read_text())
    assert lv['passed'] and lv['label_report_sha256'] == sha(labels/'full_label_report.json')
    assert lr['labels_sha256'] == sha(labels/'full_labels.parquet')
    l = pd.read_parquet(labels/'full_labels.parquet')
    bridge = pd.read_parquet('data/research/tail_formula_phase_split/targets.parquet',columns=l.columns)
    pd.testing.assert_frame_equal(l.sort_values(['date','code']).reset_index(drop=True),bridge,check_exact=True)
    assert l.date.lt('2026-01-01').all() and not l.duplicated(['date','code']).any()
    pd.testing.assert_frame_equal(l[['date','code']],f[['date','code']],check_exact=True)
    records = []
    for fold in p['folds']:
        t = f.loc[f.formula_input_valid].merge(l.loc[l.known15 & l.date.ge(fold['training_start'])
                 & l.next_date.lt(fold['training_end'])],on=['date','code'],validate='one_to_one')
        assert t.next_date.max() < fold['evaluation_start']
        counts = dict(expected_training_rows=len(t),expected_training_days=t.date.nunique(),
                      expected_last_observation=t.next_date.max())
        for arm, names in p['arms'].items():
            q = dict(master_protocol_sha256=sha(PROTOCOL),arm=arm,fold=fold['id'],**fold,**counts,
                     feature_names=names,expected_features=len(names),parameters=p['parameters'],
                     target='relative',model_max_depth=3,encoding_multiplier=1)
            path = ROOT/'protocols'/(arm+'_'+fold['id']+'.json')
            path.parent.mkdir(exist_ok=True); assert not path.exists(); save_json(path,q)
            lookup = find(path)
            assert not lookup['matches'], 'Resolve complete candidate equivalence before any fit'
            records.append(dict(arm=arm,fold=fold['id'],protocol=str(path),protocol_sha256=sha(path),
                                lookup=lookup,**counts))
    save_json(destination,dict(passed=True,model_protocol_sha256=sha(PROTOCOL),records=records,
        atom_rows=atom_rows,all_atom_cash_stock_lags_and_validity_SQL_rebuilt=True,
        same_qualification_keys_target_dates_weights_for_both_arms=True,
        historical_2023_and_current_labels_exact_existing_strict_projection=True,
        no_cached_outcome_flags_used_as_predictors=True,new_fits=0,new_economic_groups_read=False,new_2026_prices_read=False))
    return dict(prefit_sha256=sha(destination),atom_rows=atom_rows,counts=[{k:r[k] for k in
                ['fold','expected_training_rows','expected_training_days']} for r in records if r['arm']=='control'])


def setup(arm,fold):
    p = checked()
    proof = json.loads((ROOT/'prefit_verified.json').read_text())
    assert proof['passed'] and proof['model_protocol_sha256'] == sha(PROTOCOL)
    record = next(r for r in proof['records'] if r['arm']==arm and r['fold']==fold)
    path = Path(record['protocol']); assert sha(path) == record['protocol_sha256']
    q = json.loads(path.read_text()); assert q['master_protocol_sha256'] == sha(PROTOCOL)
    numeric.ROOT = ROOT/arm/fold; numeric.FEATURES = ROOT; numeric.SOURCE = Path(p['label_root'])
    numeric.PROTOCOL = relative.PROTOCOL = path
    numeric.EXPRESSIONS = {n:baseline.EXPRESSIONS.get(n,p['new_feature_definitions'].get(n)) for n in p['arms'][arm]}
    numeric.QUANTILES = [.995]
    return p,q


def fit(arm,fold):
    p,q = setup(arm,fold); root = numeric.ROOT
    assert not (root/'model_report.json').exists()
    print(json.dumps(dict(fitting=arm+'_'+fold,rows=q['expected_training_rows'],features=q['expected_features'])),flush=True)
    relative.model('relative'); relative.verify_model('relative')
    m = json.loads((root/'model_report.json').read_text())
    assert m['rows']==q['expected_training_rows'] and m['days']==q['expected_training_days']
    assert m['last_observation']==q['expected_last_observation']<q['evaluation_start']
    assert all(m['parameters'][k]==v for k,v in p['parameters'].items())
    numeric.scores()
    f = pd.read_parquet(ROOT/'features.parquet',columns=[*KEYS,'formula_input_valid',*q['feature_names']])
    c = numeric.conn(); c.register('features',f)
    encoding = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS X{i:02d}'
                        for i,n in enumerate(q['feature_names'],1))
    c.sql('SELECT date,code,'+encoding+' FROM features WHERE formula_input_valid').create_view('encoded')
    equation = format(m['bias'],'.17e')+'+'+'+'.join(tree_sql(t) for t in m['trees'])
    expected = c.sql('SELECT f.date,f.code,f.half,f.board,f.decision_shares,f.formula_input_valid,e.score FROM features f '
        'LEFT JOIN (SELECT date,code,'+equation+' AS score FROM encoded) e USING(date,code) ORDER BY date,code').df(); c.close()
    actual = pd.read_parquet(root/'scores.parquet')
    pd.testing.assert_frame_equal(actual.drop(columns='score'),expected.drop(columns='score'),check_dtype=False,check_exact=True)
    np.testing.assert_allclose(actual.score,expected.score,rtol=0,atol=2e-11,equal_nan=True)
    cut = m['thresholds'][0]; assert cut['training_quantile']==.995
    np.testing.assert_array_equal(actual.score.gt(cut['threshold']),expected.score.gt(cut['threshold']))
    t = numeric.training(start=q['training_start'],end=q['training_end'])
    train_scores = expected.merge(t[['date','code']],on=['date','code'],validate='one_to_one')
    np.testing.assert_allclose(np.quantile(train_scores.score,.995),cut['threshold'],rtol=0,atol=2e-11)
    save_json(root/'score_verification.json',dict(passed=True,score_report_sha256=sha(root/'score_report.json'),
        all_integer_encodings_tree_scores_training_quantile_and_flags_SQL_rebuilt=True,
        model_report_sha256=sha(root/'model_report.json'),rows=len(actual),valid=int(actual.formula_input_valid.sum()),
        new_2026_prices_read=False,native_client_parity_verified=False))
    added = {i for i,n in enumerate(m['feature_names']) if n not in baseline.EXPRESSIONS}
    return dict(arm=arm,fold=fold,rows=m['rows'],days=m['days'],
                new_input_nodes=sum(i in added for t in m['trees'] for i in t['feature']))


def checked_evaluation():
    p = checked(); committed(EVALUATION)
    e = json.loads(EVALUATION.read_text()); check_sources(e['source_hashes'])
    assert e['input_protocol_sha256'] == sha(PROTOCOL)
    return p,e


def freeze():
    p,e = checked_evaluation(); destination = ROOT/'joint_selection_freeze.json'; assert not destination.exists()
    f = pd.read_parquet(ROOT/'features.parquet',columns=[*KEYS,'formula_input_valid'])
    keys = f.loc[f.date.ge('2024-01-01'),KEYS].reset_index(drop=True)
    receipts = dict(e['source_hashes'],**{str(PROTOCOL):sha(PROTOCOL),str(EVALUATION):sha(EVALUATION)})
    lists = []; models = []
    for arm in p['arms']:
        flags = np.zeros(len(keys),bool)
        for fold in p['folds']:
            _,q = setup(arm,fold['id']); root = numeric.ROOT
            m = json.loads((root/'model_report.json').read_text())
            for kind in ['model','score']:
                v = json.loads((root/(kind+'_verification.json')).read_text())
                assert v['passed'] and v[kind+'_report_sha256']==sha(root/(kind+'_report.json'))
            sr = json.loads((root/'score_report.json').read_text())
            assert sr['scores_sha256']==sha(root/'scores.parquet')
            d = pd.read_parquet(root/'scores.parquet',filters=[('date','>=',q['evaluation_start']),('date','<',q['evaluation_end'])])
            scope = keys.date.ge(q['evaluation_start']) & keys.date.lt(q['evaluation_end'])
            pd.testing.assert_frame_equal(d[KEYS].reset_index(drop=True),keys.loc[scope].reset_index(drop=True),check_exact=True)
            cut = m['thresholds'][0]['threshold']
            values = d.formula_input_valid & d.score.gt(cut)
            c = numeric.conn(); c.register('d',d)
            rebuilt = c.execute('SELECT coalesce(formula_input_valid AND score>?,false) AS selected FROM d',[cut]).df(); c.close()
            np.testing.assert_array_equal(values,rebuilt.selected); flags[scope] = values
            models.append(dict(arm=arm,fold=fold['id'],last_observation=m['last_observation'],cut=cut))
            for name in ['model_report.json','model_verification.json','score_report.json','score_verification.json','scores.parquet']:
                receipts[str(root/name)] = sha(root/name)
            receipts[str(numeric.PROTOCOL)] = sha(numeric.PROTOCOL)
        for year in ['2024','2025']:
            group = ('rule' if arm=='memory' else 'matched')+year; root = ROOT/group; assert not root.exists(); root.mkdir()
            out = keys.copy(); out['selected'] = flags & keys.date.str.startswith(year)
            out.to_parquet(root/'selection.parquet',index=False,compression='zstd')
            chosen = out.loc[out.selected]; sizes = chosen.groupby('date').size()
            save_json(root/'selection_report.json',dict(protocol_sha256=sha(EVALUATION),selection_sha256=sha(root/'selection.parquet'),
                rows=len(out),selected=len(chosen),days=len(sizes),half_counts=chosen.groupby('half').agg(rows=('code','size'),days=('date','nunique')).reset_index().to_dict('records'),
                median_daily=float(sizes.median()) if len(sizes) else None,max_daily=int(sizes.max()) if len(sizes) else None,
                no_outcome_or_fill_filter=True,new_2026_prices_read=False,no_exit_rules=True))
            save_json(root/'selection_verification.json',dict(passed=True,selection_report_sha256=sha(root/'selection_report.json'),
                all_scores_quantiles_qualification_scopes_and_full_metadata_SQL_rebuilt=True))
            lists.append(dict(group=group,root=str(root),selected=len(chosen),days=len(sizes)))
            for name in ['selection.parquet','selection_report.json','selection_verification.json']: receipts[str(root/name)] = sha(root/name)
    for year,control in e['controls'].items():
        frame = checked_selection(Path(control)); pd.testing.assert_frame_equal(keys,frame[KEYS],check_exact=True)
        lists.append(dict(group='control'+year,root=control,original_unchanged=True))
        for name in ['selection.parquet','selection_report.json','selection_verification.json']: receipts[str(Path(control)/name)] = sha(Path(control)/name)
    save_json(destination,dict(passed=True,input_protocol_sha256=sha(PROTOCOL),execution_protocol_sha256=sha(EVALUATION),
        source_hashes=receipts,selections=lists,models=models,fits_completed=8,
        all_eight_models_and_full_annual_lists_fixed_before_economics=True,new_economic_outcomes_read=False,new_2026_prices_read=False))
    return dict(joint_sha256=sha(destination),selections=lists)


def finish():
    comparisons.finish()
    path = ROOT/'complete_results_manifest.json'; m = json.loads(path.read_text())
    m.pop('complete_two_year_four_half_comparisons_with_all_three_controls',None)
    m.update(new_tree_fits=8,new_input_features=2,control_families=2,
             all_two_year_four_half_comparisons_with_matched_qualification_and_original_controls=True)
    save_json(path,m)
    return dict(complete_sha256=sha(path),criteria=m['criteria'],comparisons=len(m['comparisons']),fingerprints=len(m['source_hashes']))


shared.ROOT = comparisons.ROOT = ROOT
shared.EXECUTION = comparisons.EXECUTION = EVALUATION
shared.fit = SimpleNamespace(PROTOCOL=PROTOCOL,EXECUTION=PROTOCOL)
shared.checked = comparisons.checked = checked_evaluation

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prefit','fit','freeze','analyze','finish'])
    parser.add_argument('--arm',choices=['control','memory']); parser.add_argument('--fold')
    a = parser.parse_args()
    if a.stage == 'fit': result = fit(a.arm,a.fold)
    elif a.stage == 'analyze': result = shared.analyze()
    else: result = globals()[a.stage]()
    print(json.dumps(result,ensure_ascii=False),flush=True)
