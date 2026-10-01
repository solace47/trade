"""Fit all profit-directed rule paths before reading new annual economics."""
import argparse
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research import tail_formula_baseline as original
from trade_research import tail_formula_additive as numeric
from trade_research import tail_formula_profit_rule_search as rules
from trade_research.research_io import check_sources, save_json, sha
from run_tail_formula_rule_search import condition_key


ROOT = Path('data/research/tail_formula_profit_rule_search')
PROTOCOL = Path('config/tail_formula_profit_rule_search_protocol.json')
EXECUTION = Path('config/tail_formula_profit_rule_search_execution.json')
TARGETS = Path('data/research/tail_formula_phase_split/targets.parquet')


def checked():
    p, e = json.loads(PROTOCOL.read_text()), json.loads(EXECUTION.read_text())
    assert e['input_protocol_sha256'] == sha(PROTOCOL)
    check_sources(p['source_hashes'])
    check_sources(e['source_hashes'])
    document = subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    assert sha(PROTOCOL) in document and sha(EXECUTION) in document
    return p, e


def independent_audit(con, atoms, trace, best, chosen, p, fold):
    seen, beam, rebuilt_best, rebuilt_chosen, examined = set(), [], None, None, 0
    def rank(r):
        if 'robust_ranks' in r:
            return (*(-n for n in r['robust_ranks']),sum(h['rows'] for h in r['halves']),r['conditions'])
        return (-r['minimum_integer'], -r['mean_integer'], sum(h['rows'] for h in r['halves']), r['conditions'])
    for depth in range(1, p['max_conditions'] + 1):
        expected = set()
        for parent in [()] if depth == 1 else [r['conditions'] for r in beam]:
            for atom in atoms:
                if atom[0] in {c[0] for c in parent}:
                    continue
                key = tuple(sorted((*parent, atom)))
                if key not in seen:
                    expected.add(key)
                    seen.add(key)
        layer_trace = [r for r in trace if r['depth'] == depth]
        assert {condition_key(r['conditions']) for r in layer_trace} == expected
        layer = []
        for record in layer_trace:
            key = condition_key(record['conditions'])
            terms = [f"{p['feature_names'][j]} {'<=' if op == 0 else '>'} {cut}" for j, op, cut in key]
            table = con.sql(f'''WITH daily AS(SELECT date,count(*) AS rows,sum(known::INT) AS known,
                sum(success::INT) AS success,count(reference) AS reference_rows,avg(reference) AS reference
                FROM observations WHERE {' AND '.join(terms)} GROUP BY date)
                SELECT (date>='{fold['split']}')::INT AS half,count(*) AS days,sum(rows) AS rows,
                sum(known) AS known,sum(success) AS success,avg(success*1.0/rows) AS lower,
                count(reference) AS reference_days,sum(reference_rows) AS reference_rows,
                avg(reference) AS reference,median(reference) AS reference_median,
                avg(least(greatest(reference,-.03),.03)) FILTER(WHERE reference IS NOT NULL) AS reference_clipped,
                avg((reference>0)::INT) AS positive_reference_fraction FROM daily GROUP BY half ORDER BY half''').df()
            halves = []
            for half in [0, 1]:
                row = table.loc[table.half.eq(half)]
                h = dict(days=0, rows=0, known=0, success=0, lower=None,
                         reference_days=0, reference_rows=0, reference=None)
                if p.get('robust_objective',False):
                    h.update(reference_median=None,reference_clipped=None,positive_reference_fraction=None)
                if not row.empty:
                    row = row.iloc[0]
                    for name in ['days', 'rows', 'known', 'success', 'reference_days', 'reference_rows']:
                        h[name] = int(row[name])
                    h['lower'] = float(row['lower'])
                    h['reference'] = float(row.reference) if pd.notna(row.reference) else None
                    if p.get('robust_objective',False):
                        for name in ['reference_median','reference_clipped','positive_reference_fraction']:
                            h[name] = float(row[name]) if pd.notna(row[name]) else None
                target = record['halves'][half]
                for name in h:
                    if name in ['lower','reference','reference_median','reference_clipped','positive_reference_fraction']:
                        assert (h[name] is None) == (target[name] is None)
                        if h[name] is not None:
                            assert abs(h[name] - target[name]) <= 2e-12
                    else:
                        assert h[name] == target[name]
                halves.append(h)
            eligible = all(h['days'] >= p['minimum_signal_days_each_training_half']
                and h['known'] >= p['minimum_known_opportunity_rows_each_training_half']
                and h['reference_days'] >= p['minimum_reference_days_each_training_half']
                and h['reference_rows'] >= p['minimum_known_reference_rows_each_training_half'] for h in halves)
            assert eligible == record['eligible']
            if eligible:
                values = [h['reference'] for h in halves]
                scale = p['objective_integer_scale']
                item = dict(record, conditions=key, halves=halves,
                    minimum_integer=int(math.floor(min(values)*scale+.5)),
                    mean_integer=int(math.floor(math.fsum(values)/2*scale+.5)))
                assert (item['minimum_integer'], item['mean_integer']) == (record['minimum_integer'], record['mean_integer'])
                if p.get('robust_objective',False):
                    medians=[h['reference_median'] for h in halves]
                    clipped=[h['reference_clipped'] for h in halves]
                    item['robust_ranks']=[int(math.floor(v*scale+.5)) for v in
                        [min(medians),math.fsum(medians)/2,min(clipped),math.fsum(clipped)/2]]
                    assert item['robust_ranks']==record['robust_ranks']
                layer.append(item)
                if all(h['reference']>0 and h['lower']>.5 and (not p.get('robust_objective',False)
                    or (h['reference_median']>0 and h['reference_clipped']>0 and h['positive_reference_fraction']>.5)) for h in halves):
                    if rebuilt_chosen is None or rank(item) < rank(rebuilt_chosen):
                        rebuilt_chosen = item
            examined += 1
            if examined % 1000 == 0:
                print(json.dumps(dict(fold=fold['id'],independent_candidates=examined)),flush=True)
        layer.sort(key=rank)
        beam = layer[:p['beam_width']]
        if beam and (rebuilt_best is None or rank(beam[0]) < rank(rebuilt_best)):
            rebuilt_best = beam[0]
        if not beam:
            break
    assert examined == len(trace)
    for independent, result in [(rebuilt_best, best), (rebuilt_chosen, chosen)]:
        assert (independent is None) == (result is None)
        if result:
            assert independent['conditions'] == condition_key(result['conditions'])
    return dict(all_candidates=len(trace),every_candidate_support_counts_opportunity_and_profit_SQL_rebuilt=True,
        search_layers_beams_all_ranks_and_final_quality_gate_independently_rebuilt=True)


def fit_all():
    p, _ = checked()
    f, targets = original.original(), pd.read_parquet(TARGETS)
    assert len(f) == p['original_feature_keys'] and int(f.formula_input_valid.sum()) == p['original_valid_keys']
    pd.testing.assert_frame_equal(f[['date','code']], targets[['date','code']],check_exact=True)
    reports = []
    for fold in p['folds']:
        root = ROOT / fold['id']
        assert not root.exists(), 'Never repeat a fitted fold'
        root.mkdir()
        prior = Path(p['original_bank']) / fold['id']
        original_model = json.loads((prior / 'model_report.json').read_text())
        atoms = list(condition_key(original_model['atoms']))
        scope = f.formula_input_valid & f.date.ge(fold['training_start']) & f.date.lt(fold['training_end'])
        train, label = f.loc[scope].reset_index(drop=True), targets.loc[scope].reset_index(drop=True)
        x = numeric.encode(train)
        dates, day_ids = np.unique(train.date.to_numpy(),return_inverse=True)
        half_ids = (dates >= fold['split']).astype(int)
        known = (label.next_date.lt(fold['training_end']) & label.known15).to_numpy()
        success = known & label.opportunity15.eq(1).to_numpy()
        reference_known = known & label.target_valid.to_numpy() & np.isfinite(label.net.to_numpy())
        reference = np.expm1(label.net.to_numpy()/100)
        assert len(train) == original_model['rows'] and int(known.sum()) == original_model['mature_known']
        assert np.isfinite(reference[reference_known]).all()
        con = numeric.conn()
        con.register('visible',train[['date','code',*p['feature_names']]])
        enc = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in p['feature_names'])
        con.execute(f'''CREATE TABLE observations AS SELECT v.date,v.code,{enc},
            coalesce(t.next_date<'{fold['training_end']}' AND t.known15,false) AS known,
            coalesce(t.next_date<'{fold['training_end']}' AND t.known15 AND t.opportunity15=1,false) AS success,
            CASE WHEN t.next_date<'{fold['training_end']}' AND t.known15 AND t.target_valid AND isfinite(t.net)
            THEN exp(t.net/100)-1 ELSE NULL END AS reference
            FROM visible v LEFT JOIN read_parquet('{TARGETS}') t USING(date,code)''')
        independent = con.sql('SELECT * FROM observations ORDER BY date,code').df()
        pd.testing.assert_frame_equal(train[['date','code']],independent[['date','code']],check_exact=True)
        np.testing.assert_array_equal(x,independent[p['feature_names']].to_numpy('int32'))
        np.testing.assert_array_equal(known,independent.known)
        np.testing.assert_array_equal(success,independent.success)
        np.testing.assert_array_equal(reference_known,independent.reference.notna())
        np.testing.assert_allclose(reference[reference_known],independent.loc[reference_known,'reference'],rtol=0,atol=2e-12)
        progress = lambda r: print(json.dumps(dict(fold=fold['id'],**r)),flush=True)
        trace,best,chosen = rules.learn(x,atoms,day_ids,half_ids,known,success,reference_known,reference,p,progress)
        audit = independent_audit(con,atoms,trace,best,chosen,p,fold)
        con.close()
        save_json(root/'search_trace.json',trace)
        report = dict(protocol_sha256=sha(PROTOCOL),execution_protocol_sha256=sha(EXECUTION),
            variant='robust_profit_same_atomic_condition_search' if p.get('robust_objective',False)
                else 'profit_directed_same_atomic_condition_search',fold=fold,feature_names=p['feature_names'],
            rows=len(train),days=len(dates),mature_known=int(known.sum()),mature_reference=int(reference_known.sum()),
            last_observation=str(label.loc[known,'next_date'].max()),atoms=atoms,
            original_model_report_sha256=sha(prior/'model_report.json'),best_training_rule=best,chosen_rule=chosen,
            search_trace_sha256=sha(root/'search_trace.json'),new_input_features=0,new_atomic_conditions=0,
            new_2026_prices_read=False,no_exit_rules=True,training_metric_is_not_held_out_economics=True)
        assert report['last_observation'] < fold['evaluation_start']
        save_json(root/'model_report.json',report)
        save_json(root/'model_verification.json',dict(passed=True,model_report_sha256=sha(root/'model_report.json'),**audit))
        reports.append(dict(fold=fold['id'],candidates=len(trace),chosen=chosen,model_report_sha256=sha(root/'model_report.json')))
        print(json.dumps(dict(fold_finished=fold['id'],chosen=chosen['conditions'] if chosen else None)),flush=True)
    save_json(ROOT/'all_models_verified.json',dict(passed=True,folds=reports,protocol_sha256=sha(PROTOCOL),
        execution_protocol_sha256=sha(EXECUTION),all_four_models_before_new_economics=True,
        selector_fits=4,new_tree_fits=0,new_2026_prices_read=False))
    return dict(folds=[dict(fold=r['fold'],candidates=r['candidates'],chosen=r['chosen']['conditions'] if r['chosen'] else None) for r in reports])


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['fit'])
    parser.parse_args()
    print(json.dumps(fit_all(),ensure_ascii=False))
