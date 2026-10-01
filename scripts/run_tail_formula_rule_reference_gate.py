"""Fit a reference-profit guard within the unchanged frozen condition bank."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research import tail_formula_baseline as original
from trade_research import tail_formula_additive as numeric
from trade_research import tail_formula_rule_search as rules
from trade_research.research_io import check_sources, save_json, sha
from run_tail_formula_rule_search import condition_key


ROOT = Path('data/research/tail_formula_rule_reference_gate')
PROTOCOL = Path('config/tail_formula_rule_reference_gate_protocol.json')
EXECUTION = Path('config/tail_formula_rule_reference_gate_execution.json')
TARGETS = Path('data/research/tail_formula_phase_split/targets.parquet')


def checked():
    p = json.loads(PROTOCOL.read_text())
    e = json.loads(EXECUTION.read_text())
    assert e['input_protocol_sha256'] == sha(PROTOCOL)
    check_sources(p['source_hashes'])
    check_sources(e['source_hashes'])
    document = subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    assert sha(PROTOCOL) in document and sha(EXECUTION) in document
    return p, e


def fit_all():
    p, e = checked()
    f = original.original()
    targets = pd.read_parquet(TARGETS)
    pd.testing.assert_frame_equal(f[['date','code']],targets[['date','code']],check_exact=True)
    reports = []
    for fold in p['folds']:
        root = ROOT / fold['id']
        assert not root.exists(), 'Never refit a completed selector'
        root.mkdir()
        prior = Path(p['original_bank']) / fold['id']
        model = json.loads((prior / 'model_report.json').read_text())
        bank = json.loads((prior / 'search_trace.json').read_text())
        scope = f.formula_input_valid & f.date.ge(fold['training_start']) & f.date.lt(fold['training_end'])
        visible = f.loc[scope].reset_index(drop=True)
        labels = targets.loc[scope].reset_index(drop=True)
        x = numeric.encode(visible)
        dates, day_ids = np.unique(visible.date.to_numpy(), return_inverse=True)
        half_ids = (dates >= fold['split']).astype(int)
        known = (labels.next_date.lt(fold['training_end']) & labels.known15 & labels.target_valid & np.isfinite(labels.net)).to_numpy()
        reference = np.expm1(labels.net.to_numpy()/100)
        con = numeric.conn()
        con.register('visible',visible[['date','code',*p['feature_names']]])
        enc = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in p['feature_names'])
        con.execute(f'''CREATE TABLE observations AS SELECT v.date,v.code,{enc},
            exp(t.net/100)-1 AS reference FROM visible v JOIN read_parquet('{TARGETS}') t USING(date,code)
            WHERE t.next_date<'{fold['training_end']}' AND t.known15 AND t.target_valid AND isfinite(t.net)''')
        source = con.sql('SELECT * FROM observations ORDER BY date,code').df()
        pd.testing.assert_frame_equal(visible.loc[known,['date','code']].reset_index(drop=True),source[['date','code']],check_exact=True)
        np.testing.assert_array_equal(x[known],source[p['feature_names']].to_numpy('int32'))
        np.testing.assert_allclose(reference[known],source.reference,rtol=0,atol=2e-12)
        records, feasible = [], []
        for i, record in enumerate(bank):
            if not record['eligible']:
                continue
            key = condition_key(record['conditions'])
            selected = rules.masks(x,key) & known
            count = np.bincount(day_ids[selected],minlength=len(dates))
            sums = np.bincount(day_ids[selected],weights=reference[selected],minlength=len(dates))
            stats = []
            for half in [0,1]:
                active = (half_ids==half) & (count>0)
                stats.append(dict(days=int(active.sum()),rows=int(count[active].sum()),
                    reference=float(np.mean(sums[active]/count[active])) if active.any() else None))
            terms = [f"{p['feature_names'][j]} {'<=' if op==0 else '>'} {cut}" for j,op,cut in key]
            independent = con.sql(f'''WITH daily AS(SELECT date,count(*) AS rows,avg(reference) AS reference
                FROM observations WHERE {' AND '.join(terms)} GROUP BY date)
                SELECT (date>='{fold['split']}')::INT AS half,count(*) AS days,sum(rows) AS rows,
                avg(reference) AS reference FROM daily GROUP BY half ORDER BY half''').df()
            for half,s in enumerate(stats):
                a = independent.loc[independent.half.eq(half)]
                if a.empty:
                    assert s['days']==s['rows']==0 and s['reference'] is None
                else:
                    a = a.iloc[0]
                    assert (s['days'],s['rows'])==(int(a.days),int(a.rows))
                    assert abs(s['reference']-a.reference)<=2e-12
            eligible = all(s['days']>=20 and s['rows']>=100 and s['reference'] is not None and s['reference']>0 for s in stats)
            sql_eligible = len(independent)==2 and all(int(a.days)>=20 and int(a.rows)>=100 and a.reference>0 for a in independent.itertuples())
            assert eligible==sql_eligible
            item = dict(record,conditions=key,reference_stats=stats,reference_feasible=eligible)
            records.append(item)
            if eligible:
                feasible.append(item)
            if len(records)%1000==0:
                print(json.dumps(dict(fold=fold['id'],verified_candidates=len(records),feasible=len(feasible))),flush=True)
        con.close()
        feasible.sort(key=rules.ranking)
        best = feasible[0] if feasible else None
        chosen = best if best and min(h['lower'] for h in best['halves'])>.5 else None
        save_json(root/'search_trace.json',records)
        report = dict(model,protocol_sha256=sha(PROTOCOL),execution_protocol_sha256=sha(EXECUTION),
            variant='profit_feasible_frozen_condition_bank',original_model_report_sha256=sha(prior/'model_report.json'),
            original_candidate_bank_sha256=sha(prior/'search_trace.json'),best_training_rule=best,chosen_rule=chosen,
            checked_original_eligible_candidates=len(records),profit_feasible_candidates=len(feasible),
            search_trace_sha256=sha(root/'search_trace.json'),new_candidate_conditions=0,new_tree_fits=0)
        save_json(root/'model_report.json',report)
        save_json(root/'model_verification.json',dict(passed=True,model_report_sha256=sha(root/'model_report.json'),
            every_candidate_price_maturity_mask_support_and_two_half_reference_SQL_rebuilt=True,
            all_original_rank_keys_and_conditions_preserved=True,candidates=len(records),new_2026_prices_read=False))
        reports.append(dict(fold=fold['id'],feasible=len(feasible),chosen=chosen,model_report_sha256=sha(root/'model_report.json')))
        print(json.dumps(dict(fold_finished=fold['id'],feasible=len(feasible),chosen=chosen['conditions'] if chosen else None)),flush=True)
    save_json(ROOT/'all_models_verified.json',dict(passed=True,folds=reports,protocol_sha256=sha(PROTOCOL),
        execution_protocol_sha256=sha(EXECUTION),no_new_conditions=True,new_tree_fits=0,selector_fits=4,new_2026_prices_read=False))
    return reports


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['fit'])
    parser.parse_args()
    print(json.dumps(fit_all(),ensure_ascii=False))
