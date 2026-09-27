"""Independent SQL reconstruction of portable past-only formula features."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha

ROOT=Path('data/research/tail_formula_1000')
BASE=Path('data/research/next_day_winner/visible_base.parquet')


def check():
    report=json.loads((ROOT/'feature_report.json').read_text())
    assert report['protocol_sha256']==sha(Path('config/tail_formula_1000_protocol.json'))
    assert report['base_sha256']==sha(BASE)
    for path,digest in report['source_sha256'].items():
        assert sha(Path(path))==digest
    for name,h in report['output_sha256'].items():
        assert sha(ROOT/name)==h
    c=duckdb.connect()
    c.execute('SET threads=4')
    c.execute("SET memory_limit='5GB'")
    c.read_parquet(str(BASE)).create_view('base')
    c.read_parquet(list(report['source_sha256'])).create_view('daily')
    c.execute('''CREATE VIEW active AS SELECT *,coalesce(adjustflag=3 AND isfinite(open) AND isfinite(high)
        AND isfinite(low) AND isfinite(close) AND least(open,high,low,close)>0 AND volume>0
        AND high>=greatest(open,close,low)-.0001 AND low<=least(open,close)+.0001
        AND abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
        AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001,false) AS good
        FROM daily WHERE date BETWEEN '2023-06-01' AND '2025-12-30' AND tradestatus=1''')
    c.execute('''CREATE VIEW history AS SELECT date,code,
        lag(close,1) OVER w AS p1,lag(close,6) OVER w AS p6,lag(close,21) OVER w AS p21,
        CASE WHEN count(*) OVER p5=5 THEN avg(close) OVER p5 END AS ma5,
        CASE WHEN count(*) OVER p20=20 THEN avg(close) OVER p20 END AS ma20,
        CASE WHEN count(*) OVER p5=5 THEN avg(volume) OVER p5 END AS v5,
        CASE WHEN count(*) OVER p20=20 THEN max(high) OVER p20 END AS h20,
        CASE WHEN count(*) OVER p20=20 THEN min(low) OVER p20 END AS l20,
        count(*) OVER p21=21 AND sum(good::INT) OVER p21=21 AS valid_history
        FROM active WINDOW w AS(PARTITION BY code ORDER BY date),
            p5 AS(PARTITION BY code ORDER BY date ROWS BETWEEN 5 PRECEDING AND 1 PRECEDING),
            p20 AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
            p21 AS(PARTITION BY code ORDER BY date ROWS BETWEEN 21 PRECEDING AND 1 PRECEDING)''')
    c.execute('''CREATE VIEW eligible AS SELECT b.*,h.* EXCLUDE(date,code) FROM base b LEFT JOIN history h USING(date,code)
        WHERE b.board='main' AND b.necessary_tradeable AND b.listing_age_sessions>=60 AND b.price_1449<=200''')
    expected=c.sql('''SELECT date,code,p1,p6,p21,ma5,ma20,v5,h20,l20,valid_history,
        coalesce(valid_history,false) AND abs(p1-preclose)<=.005 AS formula_input_valid,
        100*(price_1449/p1-1) AS F01,100*(daily_open/p1-1) AS F02,100*(price_1449/daily_open-1) AS F03,
        100*(price_1449-low_1449)/greatest(high_1449-low_1449,.01) AS F04,
        100*(high_1449-low_1449)/p1 AS F05,100*(high_1449-price_1449)/p1 AS F06,
        100*(least(daily_open,price_1449)-low_1449)/p1 AS F07,volume_1449/v5 AS F08,
        amount_1449/1e8 AS F09,price_1449 AS F10,100*(p1/p6-1) AS F11,100*(p1/p21-1) AS F12,
        100*(price_1449/ma5-1) AS F13,100*(price_1449/ma20-1) AS F14,
        100*(price_1449/h20-1) AS F15,100*(price_1449/l20-1) AS F16
        FROM eligible ORDER BY date,code''').df()
    f=pd.read_parquet(ROOT/'features.parquet')
    universe=pd.read_parquet(ROOT/'universe.parquet')
    keys=c.sql('SELECT date,code FROM eligible ORDER BY date,code').df()
    pd.testing.assert_frame_equal(keys,universe[['date','code']],check_exact=True)
    pd.testing.assert_frame_equal(f[universe.columns],universe,check_dtype=False,check_exact=True)
    names=[f'F{i:02d}' for i in range(1,17)]
    expected['formula_input_valid'] &= np.isfinite(expected[names]).all(axis=1)
    pd.testing.assert_frame_equal(f[expected.columns],expected,check_dtype=False,atol=2e-10,rtol=0)
    assert len(f)==report['rows'] and int(f.formula_input_valid.sum())==report['valid']
    assert not f.code.str[3:].str.startswith(('92','688','300','301')).any()
    assert f.isST.eq(0).all() and f.tradestatus.eq(1).all()
    assert f.board.eq('main').all()
    result=dict(passed=True,feature_report_sha256=sha(ROOT/'feature_report.json'),rows=len(f),
        fields=len(f)*16,history_fields=len(f)*10,valid=int(f.formula_input_valid.sum()),
        hard_exclusions_checked=True,chinext_excluded=True,outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_verification.json',result)
    return result


if __name__=='__main__':
    print(json.dumps(check(),ensure_ascii=False,indent=2))
