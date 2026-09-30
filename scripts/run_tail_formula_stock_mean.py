"""Compose and independently verify two score means, then freeze full lists."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_stock_mean as study
from trade_research.corporate_cash import save_json, sha
import run_tail_formula_paired_study as shared


def compose():
    p = shared.model.checked(); records = []
    for fold, dates in p['folds'].items():
        root = study.ROOT / fold; root.mkdir(parents=True, exist_ok=True)
        assert not (root / 'score_report.json').exists()
        parents = [study.parent.model_root(fold, group) for group in [0, 1]]
        frames = [pd.read_parquet(parent / 'scores.parquet') for parent in parents]
        pd.testing.assert_frame_equal(frames[0][study.META], frames[1][study.META], check_exact=True)
        f = frames[0][study.META].copy()
        assert len(f) == 1258085 and f.formula_input_valid.sum() == 1117397
        assert f.date.lt('2026-01-01').all() and not f.duplicated(['date', 'code']).any()
        f['score'] = (frames[0].score.to_numpy(dtype=float) + frames[1].score.to_numpy(dtype=float)) / 2
        mask = f.date.ge(dates['training_start']) & f.date.lt(dates['training_end'])
        valid = mask & f.formula_input_valid
        assert np.isfinite(f.loc[f.formula_input_valid, 'score']).all()
        cut = float(np.quantile(f.loc[valid, 'score'], .995, method='linear'))
        c = shared.base.conn()
        names = ','.join('f.' + name for name in study.META)
        expected = c.sql(f"""SELECT {names},(a.score+b.score)/2 AS score
            FROM read_parquet('{study.INPUTS}/features.parquet') f
            JOIN read_parquet('{parents[0]}/scores.parquet') a USING(date,code)
            JOIN read_parquet('{parents[1]}/scores.parquet') b USING(date,code) ORDER BY f.date,f.code""").df()
        pd.testing.assert_frame_equal(f[study.META], expected[study.META], check_exact=True)
        np.testing.assert_allclose(f.score, expected.score, rtol=0, atol=2e-12, equal_nan=True)
        c.register('independent', expected)
        v = c.execute('''SELECT count(*) AS rows,count(*) FILTER(WHERE formula_input_valid) AS valid,
            count(DISTINCT date) AS days,quantile_cont(score,.995) FILTER(WHERE formula_input_valid) AS cutoff
            FROM independent WHERE date>=? AND date<?''', [dates['training_start'], dates['training_end']]).df().iloc[0]
        c.close()
        assert v.rows == mask.sum() and v.valid == valid.sum() and v.days == f.loc[mask, 'date'].nunique()
        np.testing.assert_allclose(v.cutoff, cut, rtol=0, atol=2e-12)
        f.to_parquet(root / 'scores.parquet', index=False, compression='zstd')
        receipts = {str(parent / file): sha(parent / file) for parent in parents for file in
                    ['model_report.json', 'model_verification.json', 'score_report.json', 'score_verification.json', 'scores.parquet']}
        r = dict(protocol_sha256=sha(shared.model.PROTOCOL), fold=fold, source_hashes=receipts,
            feature_report_sha256=sha(study.INPUTS / 'feature_report.json'), scores_sha256=sha(root / 'scores.parquet'),
            rows=len(f), valid=int(f.formula_input_valid.sum()), weights=[.5, .5], threshold=cut,
            calibration_start=dates['training_start'], calibration_end=dates['training_end'],
            calibration_rows=int(v.rows), calibration_valid=int(v.valid), calibration_days=int(v.days),
            quantile=.995, quantile_method='linear_unweighted_all_visible_valid_input_rows',
            calibration_is_partly_in_sample=True, no_calibration_labels_or_execution_values_read=True,
            no_model_fit_or_prediction=True, year_2025_is_exploratory=True, new_2026_prices_read=False, no_exit_rules=True)
        save_json(root / 'score_report.json', r)
        save_json(root / 'score_verification.json', dict(passed=True, score_report_sha256=sha(root / 'score_report.json'),
            all_keys_metadata_validity_score_means_and_fixed_quantiles_independently_sql_rebuilt=True,
            no_new_model_fit_or_prediction=True, no_calibration_labels_read=True,
            software_compilation_verified=False, native_source_parity_verified=False,
            new_2026_prices_read=False, no_exit_rules=True))
        records.append({k: r[k] for k in ['fold', 'threshold', 'calibration_rows', 'calibration_valid', 'calibration_days']})
    return records


def checked_scores(fold):
    root = study.ROOT / fold
    r = json.loads((root / 'score_report.json').read_text())
    v = json.loads((root / 'score_verification.json').read_text())
    assert r['protocol_sha256'] == sha(shared.model.PROTOCOL)
    assert v['passed'] and v['score_report_sha256'] == sha(root / 'score_report.json')
    assert r['scores_sha256'] == sha(root / 'scores.parquet') and r['weights'] == [.5, .5]
    assert r['quantile'] == .995 and r['no_model_fit_or_prediction']
    for file, digest in r['source_hashes'].items(): assert sha(Path(file)) == digest, file
    return root, r


def freeze():
    p = shared.model.checked(); joint_path = study.ROOT / 'joint_selection_freeze.json'
    assert not joint_path.exists()
    f = pd.read_parquet(study.INPUTS / 'features.parquet', columns=study.META)
    keys = f[study.META[:-1]]; flags, queries, compositions = [], [], []
    receipts = {str(shared.model.PROTOCOL): sha(shared.model.PROTOCOL)}
    for fold, dates in p['folds'].items():
        root, r = checked_scores(fold); cut = r['threshold']
        d = pd.read_parquet(root / 'scores.parquet', filters=[('date', '>=', dates['evaluation_start']), ('date', '<', dates['evaluation_end'])])
        d['selected'] = d.formula_input_valid & d.score.gt(cut)
        flags.append(d[['date', 'code', 'selected']])
        queries.append(f"SELECT date,code,coalesce(formula_input_valid AND score>{cut:.17e},false) AS selected FROM read_parquet('{root}/scores.parquet') WHERE date>='{dates['evaluation_start']}' AND date<'{dates['evaluation_end']}'")
        old_cores = [study.parent.ROOT / (fold + '_' + arm + '_frozen_numeric_core.tdx') for arm in ['cross', 'control']]
        texts = [core.read_text() for core in old_cores]
        assert all(t.count('HG:=') == t.count('CORE:') == 1 for t in texts)
        body = texts[0].split('HG:=', 1)[0]; assert body == texts[1].split('HG:=', 1)[0]
        core = root / 'frozen_numeric_core.tdx'
        core.write_text(body + f'MSC:=(H0SC+H1SC)/2;\nCORE:{study.CORE_GATE} AND MSC>{cut:.17g};\n')
        for file in [*old_cores, core, root/'score_report.json', root/'score_verification.json', root/'scores.parquet']:
            receipts[str(file)] = sha(file)
        compositions.append(dict(fold=fold, root=str(root), threshold=cut,
            new_fit=False, new_model_prediction=False, deterministic_mean_only=True,
            software_compilation_verified=False, native_source_parity_verified=False))
    candidate = keys.merge(pd.concat(flags, ignore_index=True), on=['date', 'code'], how='left', validate='one_to_one')
    candidate['selected'] = candidate.selected.eq(True)
    c = shared.base.conn(); c.register('keys', keys); c.sql(' UNION ALL '.join(queries)).create_view('flags')
    expected = c.sql('SELECT k.*,coalesce(selected,false) AS selected FROM keys k LEFT JOIN flags USING(date,code) ORDER BY date,code').df()
    c.close(); pd.testing.assert_frame_equal(candidate, expected, check_exact=True)
    assert len(candidate) == 1258085 and candidate.loc[candidate.selected, 'date'].ge('2025-01-01').all()
    control = shared.checked_selection(study.previous.ROOT / 'agreement2025')
    pd.testing.assert_frame_equal(candidate[study.META[:-1]], control[study.META[:-1]], check_exact=True)
    selections, equality = [], []
    for group, frame in [(p['candidate_group'], candidate), (p['screen_comparison_group'], control)]:
        root = study.ROOT / group; root.mkdir(parents=True, exist_ok=True)
        assert not (root / 'selection_report.json').exists()
        frame.to_parquet(root / 'selection.parquet', index=False, compression='zstd')
        r = dict(protocol_sha256=sha(shared.model.PROTOCOL), group=group, rows=len(frame),
            selected=int(frame.selected.sum()), days=frame.loc[frame.selected, 'date'].nunique(),
            selection_sha256=sha(root / 'selection.parquet'), source_hashes=receipts.copy(),
            no_new_group_evaluation=True, new_2026_prices_read=False, no_exit_rules=True)
        save_json(root / 'selection_report.json', r)
        save_json(root / 'selection_verification.json', dict(passed=True,
            selection_report_sha256=sha(root / 'selection_report.json'),
            all_metadata_and_fixed_mean_score_flags_independently_sql_verified=True,
            unchanged_parent_agreement_control=group == p['screen_comparison_group'],
            zero_new_model_fit_or_prediction=True, new_2026_prices_read=False, no_exit_rules=True))
        selections.append(dict(group=group, root=str(root), rows=len(frame), selected=r['selected'], days=r['days']))
        for name, parent in p['controls'].items():
            equality.append(dict(left=group, right=name, full_frame_equal=frame.equals(shared.checked_selection(Path(parent)))))
        for file in ['selection.parquet', 'selection_report.json', 'selection_verification.json']:
            receipts[str(root / file)] = sha(root / file)
    joint = dict(passed=True, model_protocol_sha256=sha(shared.model.PROTOCOL), source_hashes=receipts,
        compositions=compositions, selections=selections, equality=equality,
        all_four_parent_models_reused_zero_new_fit_or_prediction=True,
        both_full_annual_selection_frames_frozen_before_economics=True,
        no_new_raw_extraction=True, year_2025_is_exploratory=True, no_new_group_evaluation=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(joint_path, joint)
    return dict(joint_sha256=sha(joint_path), compositions=compositions, selections=selections, equality=equality)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['compose', 'freeze', 'analyze', 'finish'])
    args = parser.parse_args(); shared.configure(study.STEM)
    result = globals()[args.stage]() if args.stage in ['compose', 'freeze'] else getattr(shared, args.stage)()
    print(json.dumps(result, ensure_ascii=False, indent=2))
