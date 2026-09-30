"""Freeze an exact parent-list intersection; reuse generic full-year evaluation."""
import argparse
import json
from pathlib import Path

import pandas as pd

from trade_research import tail_formula_stock_agreement as study
from trade_research.corporate_cash import save_json, sha
import run_tail_formula_paired_study as shared


def freeze():
    p = shared.model.checked()
    joint_path = study.ROOT / 'joint_selection_freeze.json'
    assert not joint_path.exists()
    parent = study.parent.ROOT
    receipts = {str(shared.model.PROTOCOL): sha(shared.model.PROTOCOL),
                str(parent / 'joint_selection_freeze.json'): sha(parent / 'joint_selection_freeze.json')}
    cross = shared.checked_selection(parent / 'cross2025')
    own = shared.checked_selection(parent / 'own2025')
    meta = study.META[:-1]
    pd.testing.assert_frame_equal(cross[meta], own[meta], check_exact=True)
    f = pd.read_parquet(study.INPUTS / 'features.parquet', columns=study.META)
    assert len(f) == 1258085 and f.formula_input_valid.sum() == 1117397
    pd.testing.assert_frame_equal(cross[meta], f[meta], check_exact=True)
    assert not f.duplicated(['date', 'code']).any() and f.date.lt('2026-01-01').all()
    candidate = cross.copy()
    candidate['selected'] = cross.selected & own.selected
    queries, models = [], []
    for fold, dates in p['folds'].items():
        cuts = []
        for group in [0, 1]:
            root = study.parent.model_root(fold, group)
            calibration = json.loads((root / 'calibration_report.json').read_text())
            m = json.loads((root / 'model_report.json').read_text())
            s = json.loads((root / 'score_report.json').read_text())
            assert m['variant'] == 'relative' and m['training_group'] == group
            assert m['feature_names'] == list(study.ARMS['agreement'])
            assert len(m['trees']) == 64 and m['last_observation'] < dates['evaluation_start']
            assert calibration['quantile'] == .995 and calibration['training_group'] == group
            assert calibration['calibration_group'] == 1-group
            assert calibration['model_report_sha256'] == s['model_report_sha256'] == sha(root / 'model_report.json')
            assert calibration['scores_sha256'] == s['scores_sha256'] == sha(root / 'scores.parquet')
            for kind in ['model', 'score', 'calibration']:
                report = root / (kind + '_report.json')
                proof = root / (kind + '_verification.json')
                v = json.loads(proof.read_text())
                assert v['passed'] and v[kind + '_report_sha256'] == sha(report)
                receipts[str(report)] = sha(report); receipts[str(proof)] = sha(proof)
            receipts[str(root / 'scores.parquet')] = sha(root / 'scores.parquet')
            cuts.append(calibration['threshold'])
            models.append(dict(fold=fold, training_group=group, root=str(root), threshold=cuts[-1],
                               new_fit=False, new_prediction=False, software_compilation_verified=False,
                               native_source_parity_verified=False))
        queries.append(f"""SELECT f.date,f.code,coalesce(f.formula_input_valid
            AND a.score>{cuts[0]:.17e} AND b.score>{cuts[1]:.17e},false) AS selected
            FROM read_parquet('{study.INPUTS}/features.parquet') f
            JOIN read_parquet('{study.parent.model_root(fold,0)}/scores.parquet') a USING(date,code)
            JOIN read_parquet('{study.parent.model_root(fold,1)}/scores.parquet') b USING(date,code)
            WHERE f.date>='{dates['evaluation_start']}' AND f.date<'{dates['evaluation_end']}'""")
        old_cores = [parent / (fold + '_' + arm + '_frozen_numeric_core.tdx') for arm in ['cross', 'control']]
        texts = [core.read_text() for core in old_cores]
        assert all(t.count('HG:=') == t.count('CORE:') == 1 for t in texts)
        body = texts[0].split('HG:=', 1)[0]
        assert body == texts[1].split('HG:=', 1)[0]
        assert body.count('H0SC:=') == body.count('H1SC:=') == 1
        core = study.ROOT / (fold + '_agreement_frozen_numeric_core.tdx')
        core.parent.mkdir(parents=True, exist_ok=True)
        core.write_text(body + f'CORE:{study.CORE_GATE} AND H0SC>{cuts[0]:.17g} AND H1SC>{cuts[1]:.17g};\n')
        for file in [*old_cores, core]: receipts[str(file)] = sha(file)
    c = shared.base.conn(); c.register('keys', f[meta])
    c.sql(' UNION ALL '.join(queries)).create_view('flags')
    expected = c.sql('SELECT k.*,coalesce(selected,false) AS selected FROM keys k LEFT JOIN flags USING(date,code) ORDER BY date,code').df()
    c.close()
    pd.testing.assert_frame_equal(candidate, expected, check_exact=True)
    assert candidate.loc[candidate.selected, 'date'].ge('2025-01-01').all()
    selections, equality = [], []
    for group, frame in [(p['candidate_group'], candidate), (p['screen_comparison_group'], cross)]:
        root = study.ROOT / group; root.mkdir(parents=True, exist_ok=True)
        assert not (root / 'selection_report.json').exists()
        frame.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
        r = dict(protocol_sha256=sha(shared.model.PROTOCOL), group=group, rows=len(frame),
            selected=int(frame.selected.sum()), days=frame.loc[frame.selected, 'date'].nunique(),
            selection_sha256=sha(root / 'selection.parquet'), source_hashes=receipts.copy(),
            no_new_group_evaluation=True, new_2026_prices_read=False, no_exit_rules=True)
        save_json(root / 'selection_report.json', r)
        save_json(root / 'selection_verification.json', dict(passed=True,
            selection_report_sha256=sha(root / 'selection_report.json'), rows=len(frame),
            full_parent_metadata_and_intersection_independently_verified_from_fixed_scores=True,
            unchanged_parent_cross_control=group == p['screen_comparison_group'],
            zero_new_fits_predictions_or_calibration=True, new_2026_prices_read=False, no_exit_rules=True))
        selections.append(dict(group=group, root=str(root), rows=len(frame), selected=r['selected'], days=r['days']))
        for name, path in p['controls'].items():
            equality.append(dict(left=group, right=name, full_frame_equal=frame.equals(shared.checked_selection(Path(path)))))
        for file in ['selection.parquet', 'selection_report.json', 'selection_verification.json']:
            receipts[str(root / file)] = sha(root / file)
    assert shared.checked_selection(study.ROOT / 'cross2025').equals(cross)
    joint = dict(passed=True, model_protocol_sha256=sha(shared.model.PROTOCOL), source_hashes=receipts,
        models=models, selections=selections, equality=equality,
        zero_new_model_fits_predictions_calibrations_or_threshold_searches=True,
        both_full_annual_lists_jointly_frozen_before_economics=True,
        no_new_raw_extraction=True, year_2025_is_exploratory=True, no_new_group_evaluation=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(joint_path, joint)
    return dict(joint_sha256=sha(joint_path), selections=selections, equality=equality)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['freeze', 'analyze', 'finish'])
    args = parser.parse_args(); shared.configure(study.STEM)
    result = freeze() if args.stage == 'freeze' else getattr(shared, args.stage)()
    print(json.dumps(result, ensure_ascii=False, indent=2))
