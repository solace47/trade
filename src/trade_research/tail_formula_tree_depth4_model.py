"""Change only the fixed 48-input boosting tree depth from three to four."""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from . import tail_formula_additive as base
from . import tail_formula_float as original
from . import tail_formula_prior_day as adapter
from . import tail_formula_relative as relative
from . import tail_formula_recent as study
from . import tail_formula_context_2024 as linkage
from . import tail_formula_boundary_evaluation as evaluation
from .tail_formula_offset_logit48 import verify_scores
from .corporate_cash import save_json, sha

STEM = 'tail_formula_tree_depth4'
ROOT = Path('data/research') / STEM
MASTER = Path('config') / (STEM+'_protocol.json')


def checked_sources():
    p = json.loads(MASTER.read_text())
    for path, digest in p['references'].items():
        assert sha(Path(path)) == digest
    r = json.loads((original.ROOT / 'feature_report.json').read_text())
    v = json.loads((original.ROOT / 'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(original.ROOT / 'feature_report.json')
    assert r['features_sha256'] == sha(original.ROOT / 'features.parquet')
    assert r['valid'] == 1117397 and r['rows'] == 1258085
    records = []
    for fold in ['2024', 'recent']:
        path = Path('data/research') / ('tail_formula_before1000_model_'+fold)
        m = json.loads((path / 'model_report.json').read_text())
        proof = json.loads((path / 'model_verification.json').read_text())
        cfg = json.loads((Path('config') / (STEM+'_'+fold+'_protocol.json')).read_text())
        assert proof['passed'] and proof['model_report_sha256'] == sha(path / 'model_report.json')
        assert cfg['inputs_protocol_sha256'] == sha(MASTER)
        assert cfg['feature_report_sha256'] == m['feature_report_sha256'] == sha(original.ROOT / 'feature_report.json')
        assert cfg['label_report_sha256'] == m['label_report_sha256']
        assert m['feature_names'] == list(original.EXPRESSIONS)
        assert m['rows'] == cfg['expected_rows'] and m['days'] == cfg['expected_training_days'] == 241
        assert m['training_start'] == cfg['training_start'] and m['training_end'] == cfg['training_end']
        assert m['variant'] == 'relative' and m['thresholds'][3]['training_quantile'] == cfg['training_quantile'] == .995
        assert cfg['parameters'] == p['parameters'] and cfg['model_max_depth'] == 4
        for key, value in p['parameters'].items():
            assert m['parameters'][key] == (3 if key == 'max_depth' else value)
        records.append(dict(fold=fold, model_report_sha256=sha(path / 'model_report.json'),
            rows=m['rows'], days=m['days'], all_other_frozen_parameters_inputs_and_targets_match=True))
    return records


def inputs():
    records = checked_sources()
    ROOT.mkdir(parents=True, exist_ok=True)
    output = ROOT / 'input_reuse_verification.json'
    assert not output.exists()
    save_json(output, dict(passed=True, protocol_sha256=sha(MASTER), matched_controls=records,
        original_48_inputs_and_labels_reused=True, no_new_input_quality_gate=True,
        new_group_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True))
    return dict(passed=True, input_reuse_sha256=sha(output), matched_controls=records)


def setup(fold):
    checked_sources()
    proof = json.loads((ROOT / 'input_reuse_verification.json').read_text())
    assert proof['passed'] and proof['protocol_sha256'] == sha(MASTER)
    adapter.STEM = STEM
    adapter.ROOT = original.ROOT
    adapter.EXPRESSIONS = original.EXPRESSIONS
    adapter.HEADER = original.HEADER
    adapter.COMBINED_PROTOCOL = Path('config') / (STEM+'_combined_protocol.json')
    adapter.setup(fold)
    base.SOURCE = Path('data/research/tail_formula_before1000')


def model():
    root = base.ROOT
    assert not (root / 'model_report.json').exists()
    cfg = json.loads(base.PROTOCOL.read_text())
    t = relative.training('relative')
    assert len(t) == cfg['expected_rows'] and t.date.nunique() == 241
    x, y = base.encode(t), t.target.to_numpy()
    w = 1/t.groupby('date').code.transform('size').to_numpy()
    estimator = GradientBoostingRegressor(**cfg['parameters']).fit(x, y, sample_weight=w)
    trees = []
    for fitted in estimator.estimators_.ravel():
        q = fitted.tree_
        trees.append(dict(feature=q.feature.tolist(), threshold=q.threshold.tolist(), children_left=q.children_left.tolist(),
            children_right=q.children_right.tolist(), n_node_samples=q.n_node_samples.tolist(),
            weighted_n_node_samples=q.weighted_n_node_samples.tolist(), value=q.value.reshape(-1).tolist(), impurity=q.impurity.tolist()))
    m = dict(protocol_sha256=sha(base.PROTOCOL), inputs_protocol_sha256=sha(MASTER),
        feature_report_sha256=sha(original.ROOT / 'feature_report.json'), label_report_sha256=sha(base.SOURCE / 'full_label_report.json'),
        input_reuse_verification_sha256=sha(ROOT / 'input_reuse_verification.json'),
        rows=len(t), days=t.date.nunique(), last_observation=t.next_date.max(), parameters=estimator.get_params(),
        feature_names=list(base.EXPRESSIONS), variant='relative', learning_rate=.05,
        bias=float(np.ravel(estimator.init_.constant_)[0]), trees=trees,
        training_start=cfg['training_start'], training_end=cfg['training_end'],
        new_2025_score_groups_read=bool(t.date.ge('2025-01-01').any()),
        new_2025H2_score_groups_read=bool(t.date.ge('2025-07-01').any()), new_2026_prices_read=False, no_exit_rules=True)
    score = base.predict(x, m)
    np.testing.assert_allclose(score, estimator.predict(x), rtol=0, atol=2e-12)
    m['thresholds'] = [dict(id=i, training_quantile=q, threshold=float(np.quantile(score, q))) for i, q in enumerate(base.QUANTILES)]
    root.mkdir(parents=True, exist_ok=True)
    save_json(root / 'model_report.json', m)
    return {k: v for k, v in m.items() if k != 'trees'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['inputs', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'verify', 'analyze'])
    parser.add_argument('--fold', choices=['2024', 'recent', 'combined'], default='2024')
    args = parser.parse_args()
    if args.stage == 'inputs':
        result = inputs()
    else:
        setup(args.fold)
        if args.stage == 'analyze':
            assert args.fold == 'combined'
            result = evaluation.analyze(linkage.COMBINED, linkage.PROTOCOL)
        elif args.fold == 'combined':
            assert args.stage in ['freeze', 'verify']
            result = linkage.combine() if args.stage == 'freeze' else linkage.verify_combined()
        elif args.stage == 'model':
            result = model()
        elif args.stage == 'verify_model':
            r = json.loads((base.ROOT / 'model_report.json').read_text())
            p = json.loads(base.PROTOCOL.read_text())
            assert r['feature_names'] == list(original.EXPRESSIONS)
            assert all(r['parameters'][k] == v for k, v in p['parameters'].items())
            result = relative.verify_model('relative')
        elif args.stage == 'verify_scores':
            result = verify_scores()
        elif args.stage == 'scores':
            result = base.scores()
        elif args.stage == 'freeze':
            result = study.freeze()
        else:
            m = json.loads((base.ROOT / 'model_report.json').read_text())
            assert (base.ROOT / 'frozen_numeric_core.tdx').read_text() == base.native_core(m, m['thresholds'][3]['threshold'], base.EXPRESSIONS, base.HEADER)
            result = study.verify()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
