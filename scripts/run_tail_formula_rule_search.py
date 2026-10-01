"""Fit and freeze a bounded conjunction learner; economics are a later stage."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research import tail_formula_baseline as original
from trade_research import tail_formula_additive as numeric
from trade_research import tail_formula_rule_search as rules
from trade_research.research_io import check_sources, save_json, sha


ROOT = Path('data/research/tail_formula_rule_search')
PROTOCOL = Path('config/tail_formula_rule_search_protocol.json')
EXECUTION = Path('config/tail_formula_rule_search_execution.json')
TARGETS = Path('data/research/tail_formula_phase_split/targets.parquet')


def checked():
    p = json.loads(PROTOCOL.read_text())
    e = json.loads(EXECUTION.read_text())
    assert e['input_protocol_sha256'] == sha(PROTOCOL)
    check_sources(p['source_hashes'])
    check_sources(e['source_hashes'])
    text = subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    assert sha(PROTOCOL) in text and sha(EXECUTION) in text
    return p, e


def condition_key(conditions):
    return tuple(tuple(map(int, a)) for a in conditions)


def bitset(flags):
    return int.from_bytes(np.packbits(flags, bitorder='little').tobytes(), 'little')


def independent_search_audit(train, sql_x, sql_known, sql_success, atoms, trace, best, chosen, p):
    """Rebuild every candidate from SQL inputs using integer bitsets by date."""
    dates, day_id = np.unique(train.date.to_numpy(), return_inverse=True)
    rows_by_day = [np.flatnonzero(day_id == i) for i in range(len(dates))]
    blocks = []
    for indexes in rows_by_day:
        block = []
        for feature, operator, threshold in atoms:
            value = sql_x[indexes, feature]
            block.append(bitset(value <= threshold if operator == 0 else value > threshold))
        blocks.append((block, bitset(sql_known[indexes]), bitset(sql_success[indexes])))
    positions = {atom: i for i, atom in enumerate(atoms)}
    half_ids = (dates >= p['active_fold']['split']).astype(int)
    audited = []
    seen, beam, independent_best = set(), [], None
    for depth in range(1, p['max_conditions'] + 1):
        parents = [()] if depth == 1 else [r['conditions'] for r in beam]
        expected_keys = set()
        for parent in parents:
            for atom in atoms:
                if atom[0] in {a[0] for a in parent}:
                    continue
                key = tuple(sorted((*parent, atom)))
                if key not in seen:
                    expected_keys.add(key)
                    seen.add(key)
        layer_trace = [r for r in trace if r['depth'] == depth]
        assert {condition_key(r['conditions']) for r in layer_trace} == expected_keys
        layer = []
        for record in layer_trace:
            key = condition_key(record['conditions'])
            daily = []
            for i, (atom_bits, known_bits, success_bits) in enumerate(blocks):
                selected = atom_bits[positions[key[0]]]
                for atom in key[1:]:
                    selected &= atom_bits[positions[atom]]
                count = selected.bit_count()
                if count:
                    daily.append((int(half_ids[i]), count,
                                  (selected & known_bits).bit_count(),
                                  (selected & success_bits).bit_count()))
            halves = []
            for half in [0, 1]:
                a = [d for d in daily if d[0] == half]
                halves.append(dict(days=len(a), rows=sum(d[1] for d in a),
                    known=sum(d[2] for d in a), success=sum(d[3] for d in a),
                    lower=math.fsum(d[3]/d[1] for d in a)/len(a) if a else None))
            eligible = all(h['days'] >= p['minimum_signal_days_each_training_half']
                and h['known'] >= p['minimum_known_opportunity_rows_each_training_half'] for h in halves)
            assert record['eligible'] == eligible
            if not eligible:
                continue
            for left, right in zip(halves, record['halves']):
                assert all(left[k] == right[k] for k in ['days', 'rows', 'known', 'success'])
                assert abs(left['lower'] - right['lower']) <= 2e-12
            values = [h['lower'] for h in halves]
            scale = p['objective_integer_scale']
            minimum = int(math.floor(min(values)*scale + .5))
            mean = int(math.floor(math.fsum(values)/2*scale + .5))
            assert (minimum, mean) == (record['minimum_integer'], record['mean_integer'])
            item = dict(record, conditions=key)
            layer.append(item)
        layer.sort(key=rules.ranking)
        beam = layer[:p['beam_width']]
        if beam and (independent_best is None or rules.ranking(beam[0]) < rules.ranking(independent_best)):
            independent_best = beam[0]
        audited.extend(layer_trace)
        if not beam:
            break
    assert len(audited) == len(trace)
    assert (independent_best is None) == (best is None)
    if best:
        assert independent_best['conditions'] == condition_key(best['conditions'])
    expected_chosen = independent_best if independent_best and min(h['lower'] for h in independent_best['halves']) > .5 else None
    assert (expected_chosen is None) == (chosen is None)
    if chosen:
        assert expected_chosen['conditions'] == condition_key(chosen['conditions'])
    return dict(all_candidates=len(trace), SQL_encoded_inputs_and_mature_labels=True,
        all_candidate_masks_and_daily_counts_rebuilt_by_integer_bitsets=True,
        all_eligibility_objectives_search_layers_beams_and_final_rule_rebuilt=True)


def fit_all():
    p, e = checked()
    f = original.original()
    assert len(f) == p['original_feature_keys'] and f.formula_input_valid.sum() == p['original_valid_keys']
    targets = pd.read_parquet(TARGETS)
    pd.testing.assert_frame_equal(f[['date', 'code']], targets[['date', 'code']], check_exact=True)
    reports = []
    for fold in p['folds']:
        root = ROOT / fold['id']
        assert not root.exists(), 'Never repeat a fitted fold'
        root.mkdir()
        scope = f.formula_input_valid & f.date.ge(fold['training_start']) & f.date.lt(fold['training_end'])
        train = f.loc[scope].reset_index(drop=True)
        label = targets.loc[scope].reset_index(drop=True)
        known = (label.next_date.lt(fold['training_end']) & label.known15).to_numpy()
        success = known & label.opportunity15.eq(1).to_numpy()
        x = numeric.encode(train)
        dates, day_ids = np.unique(train.date.to_numpy(), return_inverse=True)
        half_ids = (dates >= fold['split']).astype(int)
        con = numeric.conn()
        con.register('visible', train[['date', 'code', *p['feature_names']]])
        encoding = ','.join(f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in p['feature_names'])
        sql = con.sql(f'''SELECT v.date,v.code,{encoding},
            coalesce(t.next_date<'{fold['training_end']}' AND t.known15,false) AS known,
            coalesce(t.next_date<'{fold['training_end']}' AND t.known15 AND t.opportunity15=1,false) AS success
            FROM visible v LEFT JOIN read_parquet('{TARGETS}') t USING(date,code) ORDER BY v.date,v.code''').df()
        pd.testing.assert_frame_equal(train[['date', 'code']], sql[['date', 'code']], check_exact=True)
        sql_x = sql[p['feature_names']].to_numpy('int32')
        np.testing.assert_array_equal(x, sql_x)
        np.testing.assert_array_equal(known, sql.known)
        np.testing.assert_array_equal(success, sql.success)
        con.register('encoded', sql)
        # Rebuild lower-quantile atoms with explicit SQL order statistics.
        queries = []
        for n in p['feature_names']:
            for q in p['atomic_training_quantiles']:
                queries.append(f'list_sort(list({n}))[floor({q}*(count(*)-1))::BIGINT+1]')
        cuts = con.sql('SELECT ' + ','.join(queries) + ' FROM encoded').fetchone()
        con.close()
        expected_atoms = []
        for j in range(len(p['feature_names'])):
            for cut in sorted(set(map(int, cuts[3*j:3*j+3]))):
                for operator in [0, 1]:
                    a = (j, operator, cut)
                    flag = sql_x[:, j] <= cut if operator == 0 else sql_x[:, j] > cut
                    if flag.any() and not flag.all():
                        expected_atoms.append(a)
        active = dict(p, active_fold=fold)
        progress = lambda r: print(json.dumps(dict(fold=fold['id'], **r)), flush=True)
        atoms, trace, best, chosen = rules.learn(x, day_ids, half_ids, known, success, active, progress)
        assert atoms == expected_atoms
        audit = independent_search_audit(train, sql_x, sql.known.to_numpy(), sql.success.to_numpy(),
                                         atoms, trace, best, chosen, active)
        save_json(root / 'search_trace.json', trace)
        report = dict(protocol_sha256=sha(PROTOCOL), execution_protocol_sha256=sha(EXECUTION),
            variant='bounded_three_condition_maximin_lower_opportunity', fold=fold,
            feature_names=p['feature_names'], rows=len(train), days=len(dates),
            mature_known=int(known.sum()), last_observation=str(label.loc[known, 'next_date'].max()),
            atoms=atoms, best_training_rule=best, chosen_rule=chosen,
            search_trace_sha256=sha(root / 'search_trace.json'),
            new_2026_prices_read=False, no_exit_rules=True, training_metric_is_not_held_out_economics=True)
        assert report['last_observation'] < fold['evaluation_start']
        save_json(root / 'model_report.json', report)
        save_json(root / 'model_verification.json', dict(passed=True,
            model_report_sha256=sha(root / 'model_report.json'), **audit))
        reports.append(dict(fold=fold['id'], candidates=len(trace), chosen=chosen,
            model_report_sha256=sha(root / 'model_report.json')))
        print(json.dumps(dict(fold_finished=fold['id'], selected_conditions=chosen['conditions'] if chosen else None)), flush=True)
    save_json(ROOT / 'all_models_verified.json', dict(passed=True, folds=reports,
        protocol_sha256=sha(PROTOCOL), execution_protocol_sha256=sha(EXECUTION),
        all_four_models_before_new_economics=True, new_2026_prices_read=False))
    return reports


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['fit'])
    args = parser.parse_args()
    print(json.dumps(fit_all(), ensure_ascii=False))
