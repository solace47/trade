"""Diagnose frozen positive-score/rank coverage without labels or new fits."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research.tail_formula_additive import conn
from trade_research.research_io import check_sources, save_json, sha


ROOT = Path('data/research/tail_formula_phase_split')
PROTOCOL = ROOT / 'score_gate_diagnosis_protocol.json'


def review():
    destination = ROOT / 'score_gate_diagnosis.json'
    assert not destination.exists(), 'Reuse the completed diagnosis; do not overwrite it'
    protocol = json.loads(PROTOCOL.read_text())
    committed = subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    assert sha(PROTOCOL) in committed
    assert protocol['no_outcome_labels_or_prices'] and protocol['no_new_fits']
    assert protocol['no_selection_changes']
    assert sha(ROOT / 'joint_selection_freeze.json') == protocol['joint_sha256']
    check_sources(protocol['source_hashes'])
    records, summaries = [], []
    con = conn()
    columns = ['valid_inputs', 'positive_scores', 'before_guard', 'suppressed', 'selected']
    for path in protocol['source_hashes']:
        arm, fold = Path(path).parts[-3:-1]
        d = pd.read_parquet(path, columns=['date', 'code', 'formula_input_valid', 'score', 'selected'])
        assert d.date.str.startswith(fold[:4]).all() and not d.duplicated(['date', 'code']).any()
        eligible = d.formula_input_valid & d.score.gt(0)
        p = d.loc[eligible, ['date', 'code', 'score']].copy()
        p['integer_score'] = np.floor(p.score * 1000000 + .5).astype('int64')
        p['rank'] = p.groupby('date').integer_score.rank(method='min', ascending=False)
        p['before_guard'] = p['rank'].le(5)
        n = p.loc[p.before_guard].groupby('date').size()
        suppressed_dates = n.loc[n.gt(5)].index
        p['selected'] = p.before_guard & ~p.date.isin(suppressed_dates)
        rebuilt = d[['date', 'code']].merge(p[['date', 'code', 'selected']],
            on=['date', 'code'], how='left', validate='one_to_one')
        np.testing.assert_array_equal(d.selected, rebuilt.selected.fillna(False).astype(bool))
        daily = pd.DataFrame(index=pd.Index(sorted(d.date.unique()), name='date'))
        daily['valid_inputs'] = d.groupby('date').formula_input_valid.sum()
        daily['positive_scores'] = p.groupby('date').size()
        daily['before_guard'] = n
        daily['suppressed'] = daily.index.isin(suppressed_dates)
        daily['selected'] = d.groupby('date').selected.sum()
        daily = daily.fillna(0).reset_index()
        con.register('scores', d)
        sql = con.sql('''WITH positive AS (
            SELECT date,code,floor(score*1000000+.5) AS integer_score
            FROM scores WHERE formula_input_valid AND score>0
        ), ranked AS (
            SELECT *,rank() OVER(PARTITION BY date ORDER BY integer_score DESC) AS r FROM positive
        ), counts AS (
            SELECT date,count(*) AS positive_scores,count(*) FILTER(WHERE r<=5) AS before_guard
            FROM ranked GROUP BY date
        ), daily AS (
            SELECT date,count(*) FILTER(WHERE formula_input_valid) AS valid_inputs,
                count(*) FILTER(WHERE selected) AS actual_selected FROM scores GROUP BY date
        ) SELECT daily.date,valid_inputs,coalesce(positive_scores,0) AS positive_scores,
            coalesce(before_guard,0) AS before_guard,coalesce(before_guard,0)>5 AS suppressed,
            CASE WHEN before_guard<=5 THEN before_guard ELSE 0 END AS selected,
            actual_selected FROM daily LEFT JOIN counts USING(date) ORDER BY date''').df()
        sql['selected'] = sql.selected.fillna(0)
        pd.testing.assert_frame_equal(daily[['date', *columns]], sql[['date', *columns]], check_dtype=False)
        np.testing.assert_array_equal(sql.selected, sql.actual_selected)
        assert daily.selected.le(5).all()
        assert (daily.positive_scores.eq(0).sum() + daily.suppressed.sum()
                + daily.selected.gt(0).sum()) == len(daily)
        daily.insert(0, 'fold', fold)
        daily.insert(0, 'arm', arm)
        records.append(daily)
        summaries.append(dict(arm=arm, fold=fold, total_days=len(daily),
            no_positive_days=int(daily.positive_scores.eq(0).sum()),
            positive_days=int(daily.positive_scores.gt(0).sum()),
            suppressed_tie_days=int(daily.suppressed.sum()),
            selected_days=int(daily.selected.gt(0).sum()),
            before_guard_stock_days=int(daily.before_guard.sum()),
            selected_stock_days=int(daily.selected.sum())))
    con.close()
    pd.concat(records, ignore_index=True).to_parquet(ROOT / 'score_gate_daily.parquet', index=False)
    result = dict(passed=True, protocol_sha256=sha(PROTOCOL),
        implementation_sha256=sha(Path(__file__)), source_hashes=protocol['source_hashes'],
        daily_sha256=sha(ROOT / 'score_gate_daily.parquet'), summaries=summaries,
        all_frozen_selected_flags_and_daily_counts_rebuilt=True,
        new_fits=0, outcome_labels_or_prices_read=False, selection_changes=False,
        new_2026_prices_read=False)
    save_json(destination, result)
    return dict(summaries=summaries, sha256=sha(destination))


if __name__ == '__main__':
    print(json.dumps(review(), ensure_ascii=False))
