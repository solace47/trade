"""Independent full-window and historical-alignment checks for morning inputs."""
import argparse
import json

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as base
from trade_research import tail_formula_float as previous
from trade_research import tail_formula_morning_history as study
from trade_research.corporate_cash import save_json, sha

ROOT = study.ROOT
KEYS = ['code', 'date']
CLOCKS = [f'09{i:02d}' for i in range(31, 60)] + ['1000']


def windows():
    _, raw, _ = study.checked_raw()
    r = json.loads((ROOT / 'window_report.json').read_text())
    assert r['protocol_sha256'] == sha(study.PROTOCOL) and r['raw_report_sha256'] == sha(ROOT / 'raw_report.json')
    assert r['windows_sha256'] == sha(ROOT / 'windows.parquet')
    actual = pd.read_parquet(ROOT / 'windows.parquet').set_index(KEYS).sort_index()
    rows = minutes = usable = 0; visited = []
    for path in raw['parts_sha256']:
        f = pd.read_parquet(path)
        assert f.date.between('2023-06-01', '2025-12-30').all()
        assert (f.clock.isin(CLOCKS) | f.clock.eq('1449')).all()
        assert f.timestamp.dt.strftime('%Y-%m-%d').eq(f.date).all()
        assert f.timestamp.dt.strftime('%H%M').eq(f.clock).all()
        f['price'] = f.close.round(2)
        f['good'] = (f.timestamp.eq(f.timestamp.dt.floor('min')) & np.isfinite(f.close) & f.close.gt(0)
                     & f.close.sub(f.price).abs().le(.0001) & np.isfinite(f.volume)
                     & f.volume.ge(0) & f.volume.eq(np.floor(f.volume)))
        keys = f.groupby(KEYS).size().index
        morning = f.loc[f.clock.isin(CLOCKS)].copy()
        counts = morning.groupby(KEYS).agg(morning_bars=('clock', 'size'), morning_clocks=('clock', 'nunique'),
                                           morning_good=('good', 'sum')).reindex(keys, fill_value=0)
        counts['morning_valid'] = counts.eq(30).all(axis=1)
        tail = f.loc[f.clock.eq('1449')].groupby(KEYS).agg(tail_bars=('clock', 'size'), tail_good=('good', 'sum'))
        counts = counts.join(tail.reindex(keys, fill_value=0))
        counts['tail_valid'] = counts.tail_bars.eq(1) & counts.tail_good.eq(1)
        a = actual.loc[keys]
        pd.testing.assert_frame_equal(a[counts.columns], counts, check_dtype=False, check_exact=True)
        valid = counts.loc[counts.morning_valid].index
        m = morning.set_index(KEYS).loc[valid].reset_index().sort_values(KEYS + ['clock'])
        close = m.price.to_numpy().reshape(len(valid), 30)
        volume = m.volume.to_numpy().reshape(len(valid), 30)
        assert np.array_equal(m.clock.to_numpy().reshape(len(valid), 30), np.tile(CLOCKS, (len(valid), 1)))
        triple_prices = np.minimum.reduce([close[:, :-2], close[:, 1:-1], close[:, 2:]])
        positive_volume = (volume[:, :-2] > 0) & (volume[:, 1:-1] > 0) & (volume[:, 2:] > 0)
        sustained = np.max(np.where(positive_volume, triple_prices, 0), axis=1)
        np.testing.assert_array_equal(a.loc[valid, 'price_1000'], close[:, -1])
        np.testing.assert_array_equal(a.loc[valid, 'sustained_close'], sustained)
        valid_tail = counts.loc[counts.tail_valid].index
        prices = f.loc[f.clock.eq('1449')].set_index(KEYS).loc[valid_tail, 'price']
        np.testing.assert_array_equal(a.loc[valid_tail, 'price_1449'], prices)
        visited.extend(keys.tolist()); rows += len(keys); minutes += len(f); usable += len(valid)
    assert len(visited) == len(set(visited)) == len(actual) == r['rows']
    assert set(visited) == set(actual.index) and minutes == raw['raw_minutes'] == r['raw_minutes']
    report = dict(passed=True, window_report_sha256=sha(ROOT / 'window_report.json'), rows=rows,
                  raw_minutes=minutes, valid_morning_windows=usable,
                  all_window_counts_validity_and_usable_prices_independently_rebuilt=True,
                  all_twenty_eight_triplets_rebuilt_for_every_valid_morning=True,
                  invalid_windows_retained_without_imputation=True,
                  new_selection_outcomes_read=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'window_verification.json', report); return report


def features():
    _, _, _ = study.checked_raw(); old = study.checked_old()
    r = json.loads((ROOT / 'feature_report.json').read_text())
    assert r['protocol_sha256'] == sha(study.PROTOCOL)
    for key, name in [('features_sha256', 'features.parquet'), ('history_sha256', 'history.parquet'),
                      ('window_report_sha256', 'window_report.json'), ('window_verification_sha256', 'window_verification.json')]:
        assert r[key] == sha(ROOT / name)
    v = json.loads((ROOT / 'window_verification.json').read_text())
    assert v['passed'] and v['window_report_sha256'] == r['window_report_sha256']
    c = base.conn()
    calendar = pd.DataFrame(study.calendar_positions().items(), columns=['date', 'market_position'])
    c.register('calendar', calendar)
    c.execute(f'''CREATE VIEW lagged AS SELECT s.date,s.code,w.* EXCLUDE(date,code),c.market_position,
        lag(s.date) OVER b AS reference_date,lag(w.price_1449) OVER b AS reference_price,
        coalesce(lag(w.tail_valid) OVER b,false) AS reference_valid,
        coalesce(c.market_position-lag(c.market_position) OVER b>1,false)::INT AS stock_day_gap
        FROM read_parquet('{study.raw.SOURCE}/stock_days.parquet') s
        LEFT JOIN read_parquet('{ROOT}/windows.parquet') w USING(date,code)
        JOIN calendar c USING(date) WINDOW b AS(PARTITION BY s.code ORDER BY s.date)''')
    c.execute('''CREATE VIEW atoms AS SELECT *,coalesce(morning_valid AND reference_valid
        AND reference_date<date AND reference_price>0 AND isfinite(reference_price),false) AS atom_valid,
        CASE WHEN atom_valid THEN 100*(price_1000/reference_price-1) END AS atom_mark,
        CASE WHEN atom_valid THEN (sustained_close>reference_price)::DOUBLE END AS atom_positive FROM lagged''')
    c.execute('''CREATE VIEW history AS WITH h AS (
        SELECT *,count(*) OVER b AS n20,sum(atom_valid::INT) OVER b AS good20,
        first_value(reference_date) OVER b AS first_reference,min(date) OVER b AS first_morning,
        sum(stock_day_gap) OVER b AS gap20,avg(atom_mark) OVER b AS mean_mark,
        100*avg(atom_positive) OVER b AS positive_frequency
        FROM atoms WINDOW b AS(PARTITION BY code ORDER BY date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW))
        SELECT date,code,reference_date,reference_price,reference_valid,atom_valid,atom_mark,atom_positive,stock_day_gap,
        CASE WHEN n20=20 THEN first_reference END AS mh_first_reference,
        CASE WHEN n20=20 THEN first_morning END AS mh_first_morning,date AS mh_last_morning,
        CASE WHEN n20=20 THEN good20::DOUBLE END AS mh_valid_count,
        CASE WHEN n20=20 THEN gap20::DOUBLE END AS mh_gap_count,
        CASE WHEN good20=20 THEN mean_mark END AS mh_mean_mark,
        CASE WHEN good20=20 THEN positive_frequency END AS mh_positive_frequency FROM h''')
    history = c.sql('SELECT * FROM history ORDER BY date,code').df()
    actual_history = pd.read_parquet(ROOT / 'history.parquet')
    pd.testing.assert_frame_equal(actual_history[history.columns], history, check_dtype=False, atol=2e-12, rtol=0)
    c.register('old', old)
    expected = c.sql('''SELECT o.date,o.code,coalesce(h.mh_valid_count=20 AND h.mh_first_reference<h.mh_first_morning
        AND h.mh_last_morning=o.date AND o.V01>0 AND isfinite(h.mh_mean_mark)
        AND isfinite(h.mh_positive_frequency) AND isfinite(o.V01),false) AS good,
        CASE WHEN good THEN mh_mean_mark/V01 END AS HM01,
        CASE WHEN good THEN mh_positive_frequency END AS HM02,
        o.formula_input_valid AND good AND isfinite(HM01) AND isfinite(HM02) AS formula_input_valid
        FROM old o LEFT JOIN history h USING(date,code) ORDER BY date,code''').df()
    actual = pd.read_parquet(ROOT / 'features.parquet')
    pd.testing.assert_frame_equal(actual[old.columns].drop(columns='formula_input_valid'),
                                  old.drop(columns='formula_input_valid'), check_exact=True)
    pd.testing.assert_frame_equal(actual[['date','code','HM01','HM02','formula_input_valid']],
                                  expected.drop(columns='good'), check_dtype=False, atol=2e-12, rtol=0)
    np.testing.assert_array_equal(actual.morning_history_valid, expected.good)
    np.testing.assert_array_equal(actual.prior_formula_input_valid, old.formula_input_valid)
    c.register('expected', expected)
    integer = c.sql('SELECT date,code,' + ','.join(
        f'floor(least(greatest(100*{n}+10000+.000001,0),999999))::INT AS {n}' for n in ['HM01','HM02']) +
        ' FROM expected WHERE formula_input_valid ORDER BY date,code').df()
    valid = actual.loc[actual.formula_input_valid]
    np.testing.assert_array_equal(integer[['HM01','HM02']],
                                  np.floor(np.clip(100*valid[['HM01','HM02']].to_numpy()+10000+.000001,0,999999)).astype('int32'))
    c.close()
    assert len(actual) == r['rows'] and int(actual.formula_input_valid.sum()) == r['valid']
    assert int(old.formula_input_valid.sum()) == r['previous_valid']
    assert int((old.formula_input_valid & ~actual.formula_input_valid).sum()) == r['newly_invalid']
    assert r['expressions'] == study.EXPRESSIONS and r['native_header'] == study.HEADER
    report = dict(passed=True, feature_report_sha256=sha(ROOT / 'feature_report.json'), rows=len(actual),
                  valid=len(valid), history_rows=len(history), original_columns_unchanged=True,
                  all_stock_day_lags_twenty_morning_windows_scalars_encodings_and_validity_rebuilt=True,
                  current_morning_included_only_before_1449=True, new_selection_outcomes_read=False,
                  native_source_parity_verified=False, new_2026_prices_read=False, no_exit_rules=True)
    save_json(ROOT / 'feature_verification.json', report); return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage', choices=['windows', 'features'])
    a = p.parse_args(); print(json.dumps(globals()[a.stage](), ensure_ascii=False, indent=2))
