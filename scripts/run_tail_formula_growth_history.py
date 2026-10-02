"""One extra morning input with unchanged primary target and exact old controls."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_tail_formula_target as workflow
from find_existing_tail_formula_models import find
from tail_formula_order_statistics import COUNTS, METRICS
from tail_formula_statistics import eq, interval
import tail_formula_statistics as statistics
from trade_research import tail_formula_additive as numeric
from trade_research import tail_formula_relative as relative
from trade_research import tail_formula_baseline as baseline
from trade_research.research_io import check_sources, save_json, sha
from trade_research.tail_formula_boundary_evaluation import number, period

ROOT = Path('data/research/tail_formula_growth_history')
PROTOCOL = Path('config/tail_formula_growth_history_execution.json')


def prepare(p):
    gate = workflow.committed_file(Path(p['source_gate']))
    assert gate['effective_input_intersection_unchanged']
    assert not (ROOT / 'preparation.json').exists()
    old_features = pd.read_parquet(p['original_features'])
    f = pd.read_parquet(Path(p['features_root']) / 'features.parquet')
    pd.testing.assert_frame_equal(f[old_features.columns], old_features, check_exact=True)
    labels = Path(p['training_label_root'])
    label_report = json.loads((labels / 'full_label_report.json').read_text())
    label_proof = json.loads((labels / 'full_label_verification.json').read_text())
    assert label_proof['passed'] and label_proof['label_report_sha256'] == sha(labels / 'full_label_report.json')
    assert label_report['labels_sha256'] == sha(labels / 'full_labels.parquet')
    configs = ROOT / 'model_configs'
    configs.mkdir(exist_ok=False)
    records, lookup, controls = [], [], []
    receipts = {**gate['source_hashes'], **p['source_hashes']}
    for spec in p['folds']:
        old_root = Path(p['matched_models']) / spec['id']
        old = json.loads((old_root / 'model_report.json').read_text())
        proof = json.loads((old_root / 'model_verification.json').read_text())
        assert proof['passed'] and proof['model_report_sha256'] == sha(old_root / 'model_report.json')
        assert proof['all_targets_integer_inputs_day_weights_residual_means_and_variances_rebuilt']
        assert old['label_report_sha256'] == sha(labels / 'full_label_report.json')
        assert old['feature_report_sha256'] == sha(Path(p['original_features']).parent / 'feature_report.json')
        assert old['feature_names'] == list(baseline.EXPRESSIONS)
        assert old['variant'] == 'relative' and old['training_event'] == 'costed_positive_before_bad3'
        assert all(old['parameters'][k] == value for k, value in p['fit_parameters'].items())
        config = dict(master_input_protocol_sha256=sha(PROTOCOL), target='relative',
                      utility='binary_primary_costed_positive_before_bad3', model_max_depth=3,
                      training_start=spec['training_start'], training_end=spec['training_end'],
                      feature_names=list(p['expressions']), minimum_leaf_training_days=0,
                      training_event='costed_positive_before_bad3', training_allowed_utility_values=[0, 1],
                      parameters=p['fit_parameters'])
        path = configs / (spec['id']+'.json')
        save_json(path, config)
        item = dict(fold=spec['id'], config=str(path))
        workflow.setup(item, p)
        train = relative.training('relative')
        target = workflow.target_values(labels / 'full_labels.parquet', spec['training_start'], spec['training_end'], 'relative')
        expected = old_features.loc[old_features.formula_input_valid].merge(target, on=['date', 'code'], validate='one_to_one').sort_values(['date', 'code']).reset_index(drop=True)
        pd.testing.assert_frame_equal(train[['date', 'code']], expected[['date', 'code']], check_exact=True)
        np.testing.assert_allclose(train.target, expected.target, rtol=0, atol=2e-12)
        old_x = np.floor(np.clip(100*expected[list(baseline.EXPRESSIONS)].to_numpy()+10000+.000001, 0, 999999)).astype('int32')
        x = numeric.encode(train)
        np.testing.assert_array_equal(x[:, :50], old_x)
        assert not any(np.array_equal(x[:, 50], x[:, i]) for i in range(50))
        assert old['rows'] == len(train) and old['days'] == train.date.nunique()
        assert old['last_observation'] == train.next_date.max() < spec['evaluation_start']
        assert old['training_start'] == spec['training_start'] and old['training_end'] == spec['training_end']
        counts = train.groupby('date').code.transform('size').to_numpy()
        np.testing.assert_array_equal(counts, expected.groupby('date').code.transform('size').to_numpy())
        result = find(path, variant='relative', training_event='costed_positive_before_bad3')
        assert not result['matches'], 'An old candidate requires full X/Y/weight equality before fitting'
        lookup.append(dict(fold=spec['id'], metadata_lookup=result,
                           source_definition_review=p['source_definition_review'],
                           new_column_not_equal_any_original_integer_column=True))
        controls.append(dict(fold=spec['id'], rows=len(train), days=int(train.date.nunique()),
                             last_observation=train.next_date.max(), matched_model=str(old_root),
                             all_original_inputs_mature_keys_targets_and_date_weights_exact=True,
                             existing_full_arithmetic_proof_reused_after_exact_input_and_label_check=True))
        records.append(item)
        for file in [path, old_root/'model_report.json', old_root/'model_verification.json']:
            receipts[str(file)] = sha(file)
        print(json.dumps(dict(prepared=spec['id'], rows=len(train), old_metadata_examined=result['existing_model_metadata_examined'])), flush=True)
    save_json(ROOT / 'model_lookup.json', dict(passed=True, records=lookup, exact_controls=controls, no_new_fits=True))
    receipts[str(ROOT / 'model_lookup.json')] = sha(ROOT / 'model_lookup.json')
    save_json(ROOT / 'preparation.json', dict(passed=True, protocol_sha256=sha(PROTOCOL),
              source_hashes=receipts, models=records, all_original_X_Y_keys_and_weights_preserved=True,
              maximum_new_fits=4, new_fits=0, new_2026_prices_read=False))
    print(json.dumps(dict(preparation_sha256=sha(ROOT/'preparation.json'))), flush=True)


def finish(p):
    joint = workflow.committed_receipt('joint_selection_freeze.json')
    destination = ROOT / 'complete_receipt.json'
    assert not destination.exists()
    report_path = ROOT / 'economic_report.json'
    report = json.loads(report_path.read_text())
    assert report['passed'] and report['protocol_sha256'] == sha(PROTOCOL)
    check_sources(report['source_hashes'])
    d = pd.read_parquet(ROOT / 'economic_daily.parquet')
    c = numeric.conn()
    checks_before = statistics.checked
    for s in report['summaries']:
        q = period(d.loc[d.group.eq(s['group']) & d.arm.eq(s['arm']) & d.bps.eq(s['bps']) & d.sensitive.eq(s['sensitive'])], s['period'])
        c.register('daily', q)
        sql = 'SELECT count(*) AS days,'+','.join(f'sum("{n}") AS "{n}"' for n in COUNTS)+','+','.join(f'avg("{n}") AS "{n}"' for n in METRICS)+' FROM daily'
        rebuilt = c.sql(sql).df().iloc[0]
        eq(s['days'], len(q), s['group'])
        for name in COUNTS:
            eq(s[name], int(rebuilt[name]) if len(q) else 0, (s['group'], name))
        for name in METRICS:
            eq(s[name], number(rebuilt[name]), (s['group'], name))
        for name in ['positive_before_rate', 'space_before_rate', 'lower', 'upper']:
            eq(s[name+'_ci'], interval(q, name), (s['group'], name+'_ci'))
    c.close()
    pair_path = ROOT / 'paired_daily.parquet'
    pairs = pd.read_parquet(pair_path) if pair_path.exists() else pd.DataFrame()
    for comparison in report['comparisons']:
        reused = next((r for r in report['exact_reuse'] if r.get('left') == comparison['left'] and r.get('right') == comparison['right']), None)
        if reused:
            continue  # Exact prior result and proof are retained in the frozen graph.
        for s in comparison['summaries']:
            q = period(pairs.loc[pairs.left.eq(s['left']) & pairs.right.eq(s['right'])
                & pairs.bps.eq(s['bps']) & pairs.sensitive.eq(s['sensitive']) & pairs.target.eq(s['target'])], s['period'])
            eq(s['days'], len(q), (s['left'], s['period']))
            for name in ['lower', 'upper', 'unknown_weight']:
                eq(s[name], number(q[name].mean()), (s['left'], name))
                eq(s[name+'_ci'], interval(q, name), (s['left'], name+'_ci'))
    receipts = {**joint['source_hashes'], **report['source_hashes']}
    for file in [report_path, ROOT/'economic_daily.parquet', pair_path]:
        if file.exists(): receipts[str(file)] = sha(file)
    save_json(destination, dict(passed=True, source_hashes=receipts, economic_report_sha256=sha(report_path),
        summary_groups=len(report['summaries']), paired_comparisons=len(report['comparisons']),
        checks=statistics.checked-checks_before, all_daily_counts_SQL_verified_during_analysis=True,
        all_summary_counts_means_and_intervals_independently_rebuilt=True,
        all_new_paired_bounds_and_intervals_independently_rebuilt=True,
        exact_prior_result_reuse=report['exact_reuse'], criteria=report['criteria'], new_fits=4,
        new_2026_prices_read=False, no_exit_rules=True, not_realized_profit=True))
    print(json.dumps(dict(complete_sha256=sha(destination), criteria=report['criteria'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'fit', 'freeze', 'analyze', 'finish'])
    args = parser.parse_args()
    workflow.ROOT, workflow.PROTOCOL = ROOT, PROTOCOL
    action = globals().get(args.stage, getattr(workflow, args.stage, None))
    action(workflow.checked())
