"""Post-result attribution, never a new selection or execution rule."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha

ROOT = Path('data/research/touch_sequence')


def run():
    proof = json.loads((ROOT/'analysis_verification.json').read_text())
    assert proof['passed'] and proof['analysis_report_sha256'] == sha(ROOT/'analysis_report.json')
    c = duckdb.connect()
    c.read_parquet(str(ROOT/'features.parquet')).create_view('features')
    results = []
    for exit_name,filename in [('morning','labels.parquet'),('tail','old_tail_labels.parquet')]:
        path = ROOT/'morning'/filename
        c.read_parquet(str(path)).create_view('labels',replace=True)
        rows = c.sql('SELECT l.* FROM labels l JOIN features f USING(date,code) WHERE f.primary AND l.known_profit15').df()
        rows['entry_change'] = rows.entry_vwap/rows.price_1449-1
        rows['gross_holding'] = rows.exit_vwap/rows.entry_vwap-1
        rows['from_quote'] = rows.exit_vwap/rows.price_1449-1
        rows['cost_drag'] = rows.gross_holding-rows.net_return15
        columns = ['entry_change','gross_holding','from_quote','cost_drag','net_return15']
        daily = rows.groupby(['date','half'])[columns].mean().reset_index()
        result = daily.groupby('half')[columns].mean().reset_index()
        direct = c.sql('''WITH d AS (SELECT l.date,l.half,
            avg(l.entry_vwap/l.price_1449-1) AS entry_change,
            avg(l.exit_vwap/l.entry_vwap-1) AS gross_holding,
            avg(l.exit_vwap/l.price_1449-1) AS from_quote,
            avg(l.exit_vwap/l.entry_vwap-1-l.net_return15) AS cost_drag,
            avg(l.net_return15) AS net_return15
            FROM labels l JOIN features f USING(date,code) WHERE f.primary AND l.known_profit15 GROUP BY l.date,l.half)
            SELECT half,avg(entry_change) AS entry_change,avg(gross_holding) AS gross_holding,
            avg(from_quote) AS from_quote,avg(cost_drag) AS cost_drag,avg(net_return15) AS net_return15
            FROM d GROUP BY half ORDER BY half''').df()
        pd.testing.assert_frame_equal(result,direct,check_dtype=False,atol=2e-12,rtol=0)
        np.testing.assert_allclose(result.gross_holding-result.cost_drag,result.net_return15,atol=2e-12,rtol=0)
        result['exit'] = exit_name
        result['known_rows'] = result.half.map(rows.groupby('half').size())
        result['valid_net_dates'] = result.half.map(daily.groupby('half').size())
        results.extend(result.to_dict('records'))
    report = dict(analysis_verification_sha256=sha(ROOT/'analysis_verification.json'),
        post_result_diagnostic=True,changed_orders=False,independent_sql_verified=True,
        note='Only known cost15 primary rows; missing and no-trade cases not zero-filled. From-quote space is not an executable return.',
        components=results,new_2026_prices_read=False)
    save_json(ROOT/'loss_components.json',report)
    return report


if __name__=='__main__':
    print(json.dumps(run(),ensure_ascii=False,indent=2))
