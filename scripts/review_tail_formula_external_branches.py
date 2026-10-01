"""Complete retrospective branch diagnostics; never choose a winning branch."""
import json
from pathlib import Path
import subprocess

import pandas as pd

from trade_research.research_io import check_sources, save_json, sha
from trade_research.tail_formula_additive import conn
from trade_research.tail_formula_boundary_evaluation import daily_summary, METRICS, number
from tail_formula_statistics import weekly_interval


ROOT = Path('data/research/tail_formula_external_pattern')
PROTOCOL = ROOT / 'branch_diagnosis_protocol.json'


def review():
    output = ROOT / 'branch_diagnosis.json'
    assert not output.exists(), 'Reuse finished diagnostics'
    p = json.loads(PROTOCOL.read_text())
    text = subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    assert sha(PROTOCOL) in text
    check_sources(p['source_hashes'])
    f = pd.read_parquet(ROOT / 'prefix_features.parquet',
        columns=['date', 'code', 'formula_input_valid', 'prefix_quality_valid', 'branch'])
    assert f.date.ge('2024-01-01').all() and f.date.lt('2026-01-01').all()
    f = f.loc[f.formula_input_valid & f.prefix_quality_valid, ['date', 'code', 'branch']]
    assert f.branch.between(0, 9).all()
    columns = ['date', 'code', 'half', 'known_no_trade']
    for bps in [5, 15]:
        columns.extend([f'{name}{bps}' for name in ['known', 'sensitive_known', 'opportunity',
            'one_percent', 'any_opportunity', 'mark_0959_return', 'adverse_return']])
    labels = Path('data/research/tail_formula_before1000/full_labels.parquet')
    r = f.merge(pd.read_parquet(labels, columns=columns), on=['date', 'code'], validate='one_to_one')
    assert len(r) == len(f) == 1003466
    con = conn()
    con.register('rows', r)
    frames, summaries = [], []
    for branch, title in enumerate(p['branches']):
        rows = r.loc[r.branch.eq(branch)]
        for bps in [5, 15]:
            for sensitive in [False, True]:
                d = daily_summary(rows, bps, sensitive)
                known = f'sensitive_known{bps}' if sensitive else f'known{bps}'
                q = con.sql(f'''WITH daily AS(SELECT date,half,count(*) AS rows,
                    count(*) FILTER(WHERE {known}) AS known,
                    count(*) FILTER(WHERE {known} AND opportunity{bps}=1) AS success,
                    count(*) FILTER(WHERE NOT {known} AND NOT known_no_trade) AS unknown,
                    count(*) FILTER(WHERE known_no_trade) AS no_trade,
                    count(*) FILTER(WHERE {known} AND one_percent{bps}=1) AS one_percent,
                    count(*) FILTER(WHERE {known} AND any_opportunity{bps}=1) AS any_success,
                    avg(mark_0959_return{bps}) FILTER(WHERE {known}) AS mean_reference,
                    avg((mark_0959_return{bps}<0)::INT) FILTER(WHERE {known}) AS negative_reference,
                    avg(adverse_return{bps}) FILTER(WHERE {known}) AS adverse_mean,
                    avg((adverse_return{bps}<=-.03)::INT) FILTER(WHERE {known}) AS bad3
                    FROM rows WHERE branch={branch} GROUP BY date,half)
                    SELECT *,success/nullif(known,0) AS rate,success/rows AS lower,
                    (success+unknown)/rows AS upper,one_percent/nullif(known,0) AS one_percent_rate,
                    any_success/nullif(known,0) AS any_rate FROM daily ORDER BY date''').df()
                pd.testing.assert_frame_equal(d[q.columns], q, check_dtype=False, atol=2e-10, rtol=0)
                d['branch'], d['title'], d['bps'], d['sensitive'] = branch, title, bps, sensitive
                frames.append(d)
                for period in p['periods']:
                    scope = d.half.eq(period) if 'H' in period else d.date.str.startswith(period)
                    a = d.loc[scope].set_index('date')
                    record = dict(branch=branch, title=title, bps=bps, sensitive=sensitive, period=period,
                        days=len(a), rows=int(a.rows.sum()), known=int(a.known.sum()),
                        unknown=int(a.unknown.sum()), no_trade=int(a.no_trade.sum()),
                        all_unknown_days=int(a.known.eq(0).sum()))
                    for name in METRICS:
                        record[name] = number(a[name].mean())
                        independent = con.sql(f'''SELECT avg({name}) AS value FROM (
                            SELECT * FROM q WHERE {'half' if 'H' in period else 'substr(date,1,4)'}='{period}')''').df().value.iloc[0]
                        assert number(independent) is None if record[name] is None else abs(independent-record[name]) <= 2e-10
                    for name in ['rate', 'lower', 'upper']:
                        record[name + '_ci'] = weekly_interval(a[name])
                    summaries.append(record)
    con.close()
    pd.concat(frames, ignore_index=True).to_parquet(ROOT / 'branch_daily.parquet', index=False, compression='zstd')
    result = dict(passed=True, protocol_sha256=sha(PROTOCOL), implementation_sha256=sha(Path(__file__)),
        source_hashes=p['source_hashes'], summaries=summaries, all_ten_branches_and_all_periods_reported=True,
        all_daily_and_group_means_SQL_rebuilt=True, daily_sha256=sha(ROOT / 'branch_daily.parquet'),
        retrospective_diagnosis=True, new_candidate_claims=False, new_fits=0, original_selection_changes=False,
        new_raw_minutes_read=False, new_2026_prices_read=False)
    save_json(output, result)
    return dict(sha256=sha(output), summary_count=len(summaries))


if __name__ == '__main__':
    print(json.dumps(review(), ensure_ascii=False))
