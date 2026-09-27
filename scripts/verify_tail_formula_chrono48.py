"""Independent SQL calibration, interval, and chronological selection checks."""
import argparse
from datetime import date
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha
from trade_research.tail_formula_chrono48 import STEM, LABELS, KEYS


def numeric(value):
    return float(value) if pd.notna(value) and np.isfinite(value) else None


def percentile(d, spec):
    weeks = {}
    for day, value in d[['date', 'lower_delta']].itertuples(index=False, name=None):
        if not np.isfinite(value):
            return None
        iso = date.fromisoformat(day).isocalendar()
        weeks.setdefault((iso.year, iso.week), []).append(value)
    if len(weeks) < 2:
        return None
    values = [weeks[k] for k in sorted(weeks)]
    totals = np.array([sum(x) for x in values]); counts = np.array([len(x) for x in values])
    draws = np.random.default_rng(spec['seed']).choice(len(values), size=(spec['replicates'], len(values)), replace=True)
    samples = np.take(totals, draws).sum(axis=1) / np.take(counts, draws).sum(axis=1)
    return float(np.percentile(samples, spec['lower_quantile'] * 100))


def equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        return sum(equal(actual[k], v) for k, v in expected.items())
    if expected is None:
        assert actual is None
    elif isinstance(expected, (float, np.floating)):
        np.testing.assert_allclose(actual, expected, atol=2e-12, rtol=0)
    else:
        assert actual == expected, (actual, expected)
    return 1


def calibration(root, protocol):
    r = json.loads((root / 'calibration_report.json').read_text()); p = json.loads(protocol.read_text())
    for key, path in [('protocol_sha256', protocol), ('model_report_sha256', root / 'model_report.json'),
        ('score_report_sha256', root / 'score_report.json'), ('calibration_days_sha256', root / 'calibration_days.parquet'),
        ('calibration_label_report_sha256', LABELS / 'full_label_report.json')]:
        assert r[key] == sha(path)
    model = json.loads((root / 'model_report.json').read_text())
    proof = json.loads((root / 'score_verification.json').read_text())
    assert proof['passed'] and proof['score_report_sha256'] == sha(root / 'score_report.json')
    assert model['last_observation'] < p['calibration_start'] == p['training_end']
    lr = json.loads((LABELS / 'full_label_report.json').read_text())
    assert lr['labels_sha256'] == sha(LABELS / 'full_labels.parquet')
    c = duckdb.connect(); c.execute('SET threads=4')
    c.execute(f"CREATE VIEW labels AS SELECT * FROM read_parquet('{LABELS}/full_labels.parquet') WHERE date>='{p['calibration_start']}' AND next_date<'{p['calibration_end']}'")
    first, last, observation = c.sql('SELECT min(date),max(date),max(next_date) FROM labels').fetchone()
    assert (first, last, observation) == (r['first_signal'], r['last_signal'], r['last_observation'])
    assert first >= p['training_end'] and observation < p['evaluation_start'] == p['calibration_end']
    c.execute(f"CREATE VIEW scores AS SELECT * FROM read_parquet('{root}/scores.parquet')")
    c.execute('''CREATE VIEW baseline AS SELECT date,
        count(*) FILTER(WHERE known15 AND opportunity15=1)*1./nullif(count(*) FILTER(WHERE known15),0) AS base_rate,
        (count(*) FILTER(WHERE known15 AND opportunity15=1)+count(*) FILTER(WHERE NOT known15 AND NOT known_no_trade))*1./count(*) AS base_upper
        FROM labels GROUP BY date''')
    parts = []; summaries = []; checks = 0
    assert len(model['thresholds']) == len(r['summaries']) == 7
    for cut, saved in zip(model['thresholds'], r['summaries']):
        d = c.sql(f'''WITH q AS (SELECT l.* FROM scores s JOIN labels l USING(date,code)
            WHERE s.formula_input_valid AND s.score>{cut['threshold']:.17e}),
            counts AS (SELECT date,count(*) AS rows,count(*) FILTER(WHERE known15) AS known,
                count(*) FILTER(WHERE known15 AND opportunity15=1) AS success,
                count(*) FILTER(WHERE NOT known15 AND NOT known_no_trade) AS unknown,
                count(*) FILTER(WHERE known_no_trade) AS no_trade,
                count(*) FILTER(WHERE known15 AND isfinite(sustained_return15) AND sustained_return15>=.01) AS one_success,
                count(*) FILTER(WHERE NOT coalesce(known15 AND isfinite(sustained_return15),FALSE) AND NOT known_no_trade) AS one_unknown,
                count(*) FILTER(WHERE known15 AND isfinite(adverse_return15) AND adverse_return15<=-.03) AS bad_success,
                count(*) FILTER(WHERE NOT coalesce(known15 AND isfinite(adverse_return15),FALSE) AND NOT known_no_trade) AS bad_unknown,
                avg(CASE WHEN known15 AND isfinite(mark_1000_return15) THEN mark_1000_return15 END) AS mean1000
                FROM q GROUP BY date),
            daily AS (SELECT *,success*1./nullif(known,0) AS rate,success*1./rows AS lower,
                (success+unknown)*1./rows AS upper,one_success*1./rows AS one_lower,
                (bad_success+bad_unknown)*1./rows AS bad_upper FROM counts)
            SELECT daily.*,base_rate,base_upper,rate-base_rate AS delta,lower-base_upper AS lower_delta,
                {cut['id']} AS cut_id FROM daily JOIN baseline USING(date) ORDER BY date''').df()
        parts.append(d)
        s = dict(**cut, rows=int(d.rows.sum()), known=int(d.known.sum()), unknown=int(d.unknown.sum()),
            no_trade=int(d.no_trade.sum()), days=len(d))
        for field, column in [('rate', 'rate'), ('lower', 'lower'), ('delta', 'delta'),
            ('mean_selected', 'rows'), ('one_lower', 'one_lower'), ('bad_upper', 'bad_upper'),
            ('mean1000', 'mean1000'), ('lower_delta', 'lower_delta')]:
            values = d[column].dropna().to_list()
            s[field] = numeric(sum(values) / len(values)) if values else None
        s['p95_selected'] = float(np.percentile(d.rows, 95)) if len(d) else None
        s['lower_delta_bootstrap'] = percentile(d, p['bootstrap'])
        s['conditional_gap_days'] = int((d.rate.isna() | d.delta.isna() | d.mean1000.isna()).sum())
        g = p['quality_gates']
        ge = lambda field, limit: s[field] is not None and s[field] >= g[limit]
        le = lambda field, limit: s[field] is not None and s[field] <= g[limit]
        s['gates'] = dict(days=s['days'] >= g['minimum_days'], known=s['known'] >= g['minimum_known'],
            conditional_complete=s['conditional_gap_days'] == 0,
            rate=ge('rate', 'minimum_rate'), lower=ge('lower', 'minimum_complete_lower'),
            delta=ge('delta', 'minimum_same_day_delta'), mean_selected=le('mean_selected', 'maximum_mean_selected'),
            p95_selected=le('p95_selected', 'maximum_p95_selected'), one_lower=ge('one_lower', 'minimum_one_percent_complete_lower'),
            bad_upper=le('bad_upper', 'maximum_bad3_complete_upper'),
            mean1000=s['mean1000'] is not None and s['mean1000'] > g['minimum_known_mean1000_exclusive'],
            conservative_weekly_increment=s['lower_delta_bootstrap'] is not None and s['lower_delta_bootstrap'] > 0)
        s['eligible'] = all(value for key, value in s['gates'].items() if key != 'conditional_complete')
        summaries.append(s); checks += equal(saved, s)
    expected = pd.concat(parts, ignore_index=True)
    saved = pd.read_parquet(root / 'calibration_days.parquet')
    pd.testing.assert_frame_equal(saved, expected[saved.columns], check_dtype=False, rtol=0, atol=2e-12)
    candidates = [s for s in summaries if s['eligible']]
    chosen = max(candidates, key=lambda s: (s['lower'], s['delta'], s['known'], -s['id'])) if candidates else None
    equal(r['chosen_threshold'], chosen)
    c.close()
    result = dict(passed=True, calibration_report_sha256=sha(root / 'calibration_report.json'),
        daily_rows=len(expected), scalar_checks=checks, all_seven_cut_statistics_and_gates_rebuilt=True,
        chosen_id=chosen['id'] if chosen else None, strict_fit_calibration_evaluation_order_checked=True,
        unknown_and_nonfinite_marks_preserved=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'calibration_verification.json', result); return result


def selection(root, protocol):
    r = json.loads((root / 'selection_report.json').read_text()); p = json.loads(protocol.read_text())
    for key, path in [('protocol_sha256', protocol), ('selection_sha256', root / 'selection.parquet'),
                      ('core_sha256', root / 'frozen_numeric_core.tdx')]:
        assert r[key] == sha(path)
    model_root = Path(r['model_root'])
    for key, name in [('model_report_sha256', 'model_report.json'), ('score_report_sha256', 'score_report.json'),
                      ('calibration_report_sha256', 'calibration_report.json')]:
        assert r[key] == sha(model_root / name)
    cal = json.loads((model_root / 'calibration_report.json').read_text())
    model = json.loads((model_root / 'model_report.json').read_text())
    proof = json.loads((model_root / 'calibration_verification.json').read_text())
    assert proof['passed'] and proof['calibration_report_sha256'] == sha(model_root / 'calibration_report.json')
    cut = model['thresholds'][3] if r['arm'] == 'control' else cal['chosen_threshold']
    assert cut == r['chosen_threshold']
    assert model['last_observation'] < p['calibration_start'] <= cal['first_signal']
    assert cal['last_observation'] < r['evaluation_start'] == p['evaluation_start']
    assert r['evaluation_end'] == p['evaluation_end']
    clause = (f"date>='{p['evaluation_start']}' AND date<'{p['evaluation_end']}' AND formula_input_valid AND score>{cut['threshold']:.17e}") if cut else 'FALSE'
    c = duckdb.connect()
    expected = c.sql(f"SELECT {','.join(KEYS)},{clause} AS selected FROM read_parquet('{model_root}/scores.parquet') ORDER BY date,code").df()
    actual = pd.read_parquet(root / 'selection.parquet')
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False, check_exact=True)
    core = (root / 'frozen_numeric_core.tdx').read_text()
    if cut:
        from trade_research import tail_formula_additive as base
        from trade_research import tail_formula_float as original
        assert core == base.native_core(model, cut['threshold'], original.EXPRESSIONS, original.HEADER)
    else:
        assert core == 'CORE:0;\n'
    assert int(expected.selected.sum()) == r['selected']
    c.close()
    result = dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'), rows=len(expected),
        selected=r['selected'], all_selection_flags_and_core_checked=True,
        no_fit_or_calibration_period_selection=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'selection_verification.json', result); return result


def combined(root, protocol):
    r = json.loads((root / 'selection_report.json').read_text())
    assert r['protocol_sha256'] == sha(protocol) and r['selection_sha256'] == sha(root / 'selection.parquet')
    p = json.loads(protocol.read_text()); frames = []
    for i, f in enumerate(r['folds']):
        source = Path(f['root']); s = json.loads((source / 'selection_report.json').read_text())
        v = json.loads((source / 'selection_verification.json').read_text())
        assert v['passed'] and f['selection_report_sha256'] == v['selection_report_sha256'] == sha(source / 'selection_report.json')
        assert s['protocol_sha256'] == sha(Path(p['fold_protocols'][i]))
        assert s['selection_sha256'] == sha(source / 'selection.parquet')
        assert f['core_sha256'] == sha(root / ('frozen_numeric_core_' + ['2024', 'recent'][i] + '.tdx'))
        frames.append(source / 'selection.parquet')
    c = duckdb.connect()
    expected = c.sql(f"SELECT a.date,a.code,a.half,a.board,a.decision_shares,((a.date<'2025-07-01' AND a.selected) OR (b.date>='2025-07-01' AND b.selected)) AS selected FROM read_parquet('{frames[0]}') a JOIN read_parquet('{frames[1]}') b USING(date,code) ORDER BY a.date,a.code").df()
    pd.testing.assert_frame_equal(pd.read_parquet(root / 'selection.parquet'), expected, check_exact=True, check_dtype=False)
    assert r['selected'] == int(expected.selected.sum()) == sum(f['selected'] for f in r['folds'])
    c.close()
    result = dict(passed=True, selection_report_sha256=sha(root / 'selection_report.json'), rows=len(expected),
        selected=r['selected'], all_fold_provenance_and_flags_rebuilt=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(root / 'selection_verification.json', result); return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('fold', choices=['2024', 'recent', 'combined'])
    p.add_argument('stage', choices=['calibration', 'selection'])
    p.add_argument('--control', action='store_true'); a = p.parse_args()
    part = '2025' if a.fold == 'combined' else a.fold
    root = Path('data/research') / (STEM + '_' + part + ('_control' if a.control else ''))
    protocol = Path('config') / (STEM + '_' + a.fold + '_protocol.json')
    function = combined if a.fold == 'combined' else globals()[a.stage]
    print(json.dumps(function(root, protocol), ensure_ascii=False, indent=2))
