"""Describe already-frozen selections using visible prices only."""
import json
from pathlib import Path

import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research.corporate_cash import save_json, sha

ROOT = Path('data/research/tail_formula_selection_direction')
FEATURES = Path('data/research/tail_formula_float')
SOURCE = Path('data/research/next_day_winner/visible_base.parquet')
STEMS = ['tail_formula_before1000_model', 'tail_formula_cross_rotation', 'tail_formula_amount_concentration']


def run():
    assert not (ROOT / 'visible_portrait.json').exists()
    original = json.loads((FEATURES / 'feature_report.json').read_text())
    assert original['features_sha256'] == sha(FEATURES / 'features.parquet')
    f = pd.read_parquet(FEATURES / 'features.parquet', columns=['date', 'code', 'half', 'A01', 'A05'])
    c = base.conn()
    c.read_parquet(str(SOURCE)).create_view('visible')
    rows, sources = [], {str(FEATURES / 'feature_report.json'): sha(FEATURES / 'feature_report.json'), str(SOURCE): sha(SOURCE)}
    for stem in STEMS:
        path = Path('data/research') / (stem + '_2025')
        r = json.loads((path / 'selection_report.json').read_text())
        v = json.loads((path / 'selection_verification.json').read_text())
        assert v['passed'] and v['selection_report_sha256'] == sha(path / 'selection_report.json')
        assert r['selection_sha256'] == sha(path / 'selection.parquet')
        sources[str(path / 'selection_report.json')] = sha(path / 'selection_report.json')
        s = pd.read_parquet(path / 'selection.parquet', columns=['date', 'code', 'selected'])
        chosen = f.merge(s.loc[s.selected, ['date', 'code']], on=['date', 'code'], validate='one_to_one')
        got = []
        for half, group in chosen.groupby('half'):
            got.append(dict(half=half, rows=len(group), day_positive=int(group.A01.gt(0).sum()),
                tail_positive=int(group.A05.gt(0).sum()), both_positive=int((group.A01.gt(0) & group.A05.gt(0)).sum()),
                median_day_pct=float(group.A01.median()), median_tail_pct=float(group.A05.median())))
        c.register('selected', s)
        ex = c.sql('''WITH t AS(SELECT date,code,CASE WHEN date<'2025-07-01' THEN '2025H1' ELSE '2025H2' END AS half,
            100*(price_1449/preclose-1) AS d,100*(price_1449/price_1420-1) AS t
            FROM selected JOIN visible USING(date,code) WHERE selected)
            SELECT half,count(*) AS rows,count(*) FILTER(WHERE d>0) AS day_positive,
            count(*) FILTER(WHERE t>0) AS tail_positive,count(*) FILTER(WHERE d>0 AND t>0) AS both_positive,
            median(d) AS median_day_pct,median(t) AS median_tail_pct FROM t GROUP BY half ORDER BY half''').df()
        pd.testing.assert_frame_equal(pd.DataFrame(got), ex, check_dtype=False, rtol=0, atol=2e-10)
        rows.extend(dict(stem=stem, **row) for row in got)
    c.close()
    report = dict(passed=True, source_hashes=sources, portraits=rows,
        visible_prices_independently_rebuilt_from_original_source=True, new_economic_groups_computed=False,
        post_result_input_description_not_causal_explanation=True, new_2026_prices_read=False, no_exit_rules=True)
    ROOT.mkdir(parents=True, exist_ok=True)
    save_json(ROOT / 'visible_portrait.json', report)
    return dict(receipt_sha256=sha(ROOT / 'visible_portrait.json'), portraits=rows)


if __name__ == '__main__':
    print(json.dumps(run(), ensure_ascii=False, indent=2))
