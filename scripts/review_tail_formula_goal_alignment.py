"""Check existing strict reports against the user's morning opportunity goal."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research.research_io import check_runtime, check_sources, save_json, sha
from trade_research.tail_formula_additive import conn

PROTOCOL = Path('config/tail_formula_goal_alignment_review.json')
ROOT = Path('data/research/tail_formula_goal_alignment_review')
PERIODS = ['2024H1', '2024H2', '2025H1', '2025H2', '2024', '2025']
KEYS = ['date', 'code', 'half', 'board', 'decision_shares', 'selected']


def review(protocol=PROTOCOL):
    global PROTOCOL, ROOT
    PROTOCOL = protocol
    check_runtime()
    p = json.loads(PROTOCOL.read_text())
    ROOT = Path(p.get('output_root', str(ROOT)))
    assert subprocess.check_output(['git', 'show', f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    assert sha(PROTOCOL) in subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    check_sources(p['source_hashes'])
    index = json.loads(Path(p['report_index']).read_text())
    check_sources(index['source_hashes'])
    assert not (ROOT/'review.json').exists()
    assert p['minimum_half_days'] == 20 and p['reference_required'] is False
    rows, exclusions, fingerprints, aliases, representatives = [], [], {}, {}, {}
    template = pd.read_parquet(p['canonical_selection'], columns=KEYS[:-1])
    assert len(template) == 1258085
    if p.get('complete_metadata_scope') == '2024':
        template = template.loc[template.date.str.startswith('2024')].reset_index(drop=True)
        assert len(template) == 587276
    for i, item in enumerate(index['reports']):
        root = Path(item['root'])
        ar = json.loads((root/'analysis_report.json').read_text())
        av = json.loads((root/'analysis_verification.json').read_text())
        sr = json.loads((root/'selection_report.json').read_text())
        sv = json.loads((root/'selection_verification.json').read_text())
        lr = json.loads((root/'full_label_report.json').read_text())
        lv = json.loads((root/'full_label_verification.json').read_text())
        assert av['passed'] and av['analysis_report_sha256'] == item['analysis_report_sha256']
        assert sv['passed'] and sv['selection_report_sha256'] == sha(root/'selection_report.json')
        assert ar['selection_report_sha256'] == sha(root/'selection_report.json')
        assert lv['passed'] and lv['label_report_sha256'] == ar['label_report_sha256'] == sha(root/'full_label_report.json')
        assert str((root/'full_labels.parquet').resolve()) == item['actual_label_file']
        assert lr['labels_sha256'] == item['actual_label_sha256']
        assert ar['daily_summary_sha256'] == item['daily_summary_sha256']
        assert not ar['new_2026_prices_read']
        scope = (ar.get('window_start'), ar.get('window_end'), ar.get('reference_label'))
        if scope != ('09:31', '09:59', '09:59') or lr['labels_sha256'] != p['canonical_labels_sha256']:
            exclusions.append(dict(root=str(root), scope=scope, label_sha256=lr['labels_sha256'],
                                   reason='Different window or label bytes; no extrapolation'))
            continue
        # A verified whole-table byte match reuses its logical fingerprint.
        digest = sr['selection_sha256']
        if digest not in fingerprints:
            f = pd.read_parquet(root/'selection.parquet', columns=KEYS)
            if not f[KEYS[:-1]].equals(template):
                exclusions.append(dict(root=str(root), reason='Different complete metadata keys', rows=len(f)))
                continue
            assert len(f) == len(template) and not f.duplicated(['date', 'code']).any()
            assert f.date.lt('2026-01-01').all() and f.loc[f.selected, 'date'].ge('2024-01-01').all()
            import hashlib
            h = hashlib.sha256(pd.util.hash_pandas_object(f, index=False).to_numpy(dtype='uint64').tobytes()).hexdigest()
            if h in representatives:
                previous = pd.read_parquet(representatives[h], columns=KEYS)
                pd.testing.assert_frame_equal(f, previous, check_exact=True)
            else:
                representatives[h] = root/'selection.parquet'
            fingerprints[digest] = h
        h = fingerprints[digest]
        aliases.setdefault(h, []).append(str(root))
        for period in PERIODS:
            def record(arm):
                found = [s for s in ar['summaries'] if s['period'] == period and s['arm'] == arm
                         and s['bps'] == 15 and not s['sensitive']]
                assert len(found) == 1
                return found[0]
            a, b, d = record('formula'), record('base_same_dates'), record('same_day_difference')
            assert a['days'] == b['days'] == d['days']
            ci = d['lower_delta_ci']
            rows.append(dict(root=str(root), complete_frame_fingerprint=h, period=period,
                days=a['days'], selected_rows=a['rows'], known=a['known'], unknown=a['unknown'],
                no_trade=a['no_trade'], opportunity=a['rate'], base_opportunity=b['rate'],
                one_percent=a['one_percent_rate'], base_one_percent=b['one_percent_rate'],
                bad3=a['bad3'], base_bad3=b['bad3'], reference=a['mean_reference'],
                opportunity_delta_lower_ci=None if ci is None else ci[0],
                opportunity_delta_upper_ci=None if ci is None else ci[1]))
        if (i+1) % 50 == 0:
            print(json.dumps(dict(reports_checked=i+1, distinct_whole_table_bytes=len(fingerprints))), flush=True)
    out = pd.DataFrame(rows)
    out['coverage'] = out.days.ge(20)
    out['opportunity_increment'] = out.opportunity_delta_lower_ci.gt(0)
    out['space_increment'] = out.one_percent.gt(out.base_one_percent)
    out['risk_not_above'] = out.bad3.le(out.base_bad3)
    out['passes_necessary_opportunity_checks'] = out[['coverage', 'opportunity_increment',
        'space_increment', 'risk_not_above']].all(axis=1)
    out['reference_negative_only_auxiliary'] = out.reference.lt(0)
    c = conn(); c.register('review_rows', out)
    expected = c.sql('''SELECT root,period,days>=20 AS coverage,
        coalesce(opportunity_delta_lower_ci>0,false) AS opportunity_increment,
        coalesce(one_percent>base_one_percent,false) AS space_increment,
        coalesce(bad3<=base_bad3,false) AS risk_not_above,
        days>=20 AND coalesce(opportunity_delta_lower_ci>0,false)
        AND coalesce(one_percent>base_one_percent,false) AND coalesce(bad3<=base_bad3,false)
        AS passes_necessary_opportunity_checks, coalesce(reference<0,false) AS reference_negative_only_auxiliary
        FROM review_rows ORDER BY root,period''').df(); c.close()
    columns = list(expected.columns)
    pd.testing.assert_frame_equal(out.sort_values(['root', 'period'])[columns].reset_index(drop=True), expected,
                                  check_exact=True, check_dtype=False)
    out.to_parquet(ROOT/'all_report_checks.parquet', index=False, compression='zstd')
    distinct = out.drop_duplicates(['complete_frame_fingerprint', 'period'])
    selected = distinct.loc[distinct.passes_necessary_opportunity_checks]
    save_json(ROOT/'complete_frame_aliases.json', dict(passed=True, aliases=aliases,
        logical_fingerprint='pandas whole KEYS frame hash; no claim of independent trials'))
    summaries = []
    for period in PERIODS:
        q = distinct.loc[distinct.period.eq(period)]
        summaries.append(dict(period=period, complete_frame_views=len(q), supported=int(q.coverage.sum()),
            opportunity_increment=int((q.coverage & q.opportunity_increment).sum()),
            passes_all_necessary_checks=int(q.passes_necessary_opportunity_checks.sum()),
            passes_with_negative_reference=int((q.passes_necessary_opportunity_checks & q.reference_negative_only_auxiliary).sum())))
    save_json(ROOT/'review.json', dict(passed=True, protocol_sha256=sha(PROTOCOL),
        source_hashes=p['source_hashes'], reports_in_scope=len(index['reports']), excluded=exclusions,
        rows=len(out), summaries=summaries,
        possible_followup_views=selected.astype(object).where(pd.notna(selected), None).to_dict('records'),
        output_sha256=sha(ROOT/'all_report_checks.parquet'), aliases_sha256=sha(ROOT/'complete_frame_aliases.json'),
        all_new_boolean_diagnoses_SQL_rebuilt=True, existing_statistics_reused=True,
        no_new_economic_aggregation=True, no_new_fits=True, no_new_selection_lists=True,
        cannot_publish_from_retrospective_screen=True, no_2026_prices_read=True, no_exit_rules=True))
    return dict(review_sha256=sha(ROOT/'review.json'), summaries=summaries, excluded=len(exclusions))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    print(json.dumps(review(parser.parse_args().protocol), ensure_ascii=False), flush=True)
