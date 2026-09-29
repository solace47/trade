"""Explain fixed 2025 scores by exact tree paths; do not read outcome labels."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research.corporate_cash import save_json, sha
from freeze_tail_formula_bipower_gap import checked_selection

OUTPUT = Path('data/research/tail_formula_dual_units/score_path_description.json')
SPECS = [
    ('normalized', 'tail_formula_before1000_model', 'tail_formula_float'),
    ('raw', 'tail_formula_unscaled_prices', 'tail_formula_unscaled_prices'),
    ('dual', 'tail_formula_dual_units_dual', 'tail_formula_dual_units'),
]


def decompose(x, model):
    # Credit each parent-to-child value difference to the splitting feature.
    # This telescopes exactly; it is order-dependent accounting, not causal SHAP.
    out = np.zeros((len(x), x.shape[1])); baseline = model['bias']
    for tree in model['trees']:
        value = np.asarray(tree['value']); left = np.asarray(tree['children_left'])
        right = np.asarray(tree['children_right']); feature = np.asarray(tree['feature'])
        threshold = np.asarray(tree['threshold']); node = np.zeros(len(x), dtype='int32')
        baseline += model['learning_rate'] * value[0]
        for _ in range(len(value)):
            indices = np.flatnonzero(left[node] >= 0)
            if not len(indices):
                break
            parent = node[indices]; column = feature[parent]
            child = np.where(x[indices, column] <= threshold[parent], left[parent], right[parent])
            out[indices, column] += model['learning_rate'] * (value[child] - value[parent])
            node[indices] = child
        assert (left[node] < 0).all()
    return baseline, out


def main():
    assert not OUTPUT.exists()
    records = []; sources = {}
    for label, stem, inputs in SPECS:
        for fold in ['2024', 'recent']:
            root = Path('data/research') / (stem + '_' + fold)
            feature_root = Path('data/research') / inputs
            m = json.loads((root / 'model_report.json').read_text())
            for kind in ['model', 'score']:
                proof = json.loads((root / (kind + '_verification.json')).read_text())
                assert proof['passed'] and proof[kind + '_report_sha256'] == sha(root / (kind + '_report.json'))
            sr = json.loads((root / 'score_report.json').read_text())
            fr = json.loads((feature_root / 'feature_report.json').read_text())
            assert sr['feature_report_sha256'] == sha(feature_root / 'feature_report.json')
            assert sr['scores_sha256'] == sha(root / 'scores.parquet')
            assert fr['features_sha256'] == sha(feature_root / 'features.parquet')
            selected = checked_selection(root)
            dates = sorted(selected.loc[selected.selected, 'date'].unique())
            f = pd.read_parquet(feature_root / 'features.parquet',
                columns=['date', 'code', 'formula_input_valid', *m['feature_names']], filters=[('date', 'in', dates)])
            f = f.loc[f.formula_input_valid].sort_values(['date', 'code']).reset_index(drop=True)
            flags = f[['date', 'code']].merge(selected, on=['date', 'code'], validate='one_to_one').selected.to_numpy()
            x = np.floor(np.clip(100 * f[m['feature_names']].to_numpy(float) + 10000 + .000001,
                                 0, 999999)).astype('int32')
            baseline, contributions = decompose(x, m)
            scores = pd.read_parquet(root / 'scores.parquet', filters=[('date', 'in', dates)])
            expected = f[['date', 'code']].merge(scores, on=['date', 'code'], validate='one_to_one').score.to_numpy()
            np.testing.assert_allclose(baseline + contributions.sum(axis=1), expected, rtol=0, atol=2e-12)
            frame = pd.DataFrame(contributions, columns=m['feature_names']); frame['date'] = f.date.to_numpy()
            chosen_mean = frame.loc[flags].groupby('date').mean().mean()
            universe_mean = frame.groupby('date').mean().mean()
            mean_score = pd.DataFrame(dict(date=f.date, score=expected)).loc[flags].groupby('date').score.mean().mean()
            np.testing.assert_allclose(baseline + chosen_mean.sum(), mean_score, rtol=0, atol=2e-12)
            ordered = (chosen_mean - universe_mean).sort_values(ascending=False)
            records.append(dict(representation=label, fold=fold, selected=int(flags.sum()), days=len(dates),
                valid_same_date_universe=len(f), bias_plus_root_values=float(baseline), selected_date_equal_score=float(mean_score),
                training_threshold=m['thresholds'][3]['threshold'],
                contributions=[dict(feature=n, selected=float(chosen_mean[n]), same_dates_universe=float(universe_mean[n]),
                                    selected_minus_universe=float(ordered[n])) for n in ordered.index]))
            for path in [feature_root / 'feature_report.json', root / 'model_report.json', root / 'model_verification.json',
                         root / 'score_report.json', root / 'score_verification.json', root / 'selection_report.json']:
                sources[str(path)] = sha(path)
            print(json.dumps(dict(completed=label, fold=fold, days=len(dates))), flush=True)
    save_json(OUTPUT, dict(passed=True, implementation_sha256=sha(Path(__file__)), source_hashes=sources, records=records,
        all_path_sums_match_previously_verified_full_scores=True, date_equal_contribution_means=True,
        interpretation='Exact parent-child telescoping score accounting; depends on fitted tree order, not causal attribution or new information.',
        descriptive_after_existing_results=True, no_new_selection_or_threshold=True, outcome_labels_read=False,
        new_2024_test_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True))
    return dict(path=str(OUTPUT), sha256=sha(OUTPUT))


if __name__ == '__main__':
    print(json.dumps(main(), ensure_ascii=False, indent=2))
