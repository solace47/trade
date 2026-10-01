"""Fixed external prefix rule with complete raw-input and economic checks."""
import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research.tail_formula_additive import conn
from trade_research.tail_formula_baseline import META
from trade_research.tail_formula_external_pattern import classify, BRANCHES
from trade_research.research_io import check_sources, save_json, sha
from trade_research import tail_formula_boundary_evaluation as evaluation
from tail_formula_reports import checked_selection, checked_analysis
import tail_formula_analysis_reuse as reuse
from verify_tail_formula_before1000 import analysis as verify_analysis
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared_compare
from audit_tail_formula_reference_coverage import audit


PROTOCOL = Path('config/tail_formula_external_pattern_protocol.json')
ROOT = Path('data/research/tail_formula_external_pattern')
RAW_CONTEXT = Path('data/research/tail_formula_minute_pressure/features.parquet')
ORIGINALS = ['data/research/tail_formula_stock_2024/morning2024',
             'data/research/tail_formula_morning_range/morning2025']


def checked():
    assert subprocess.check_output(['git', 'show', f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    p = json.loads(PROTOCOL.read_text())
    check_sources(p['source_hashes'])
    assert p['maximum_new_fits'] == 0 and not p['new_2026_prices_allowed']
    return p


def summarize(raw, con):
    """Pandas arithmetic and an independent SQL grouping of every prefix row."""
    assert raw.date.ge('2024-01-01').all() and raw.date.lt('2026-01-01').all()
    r = raw.copy()
    numeric = r[['open', 'high', 'low', 'close', 'volume', 'turnover']].to_numpy(float)
    good = np.isfinite(numeric).all(axis=1) & (numeric[:, :4] > 0).all(axis=1)
    good &= (numeric[:, 4:] >= 0).all(axis=1)
    good &= np.logical_and.reduce([r.high.ge(r[['open', 'low', 'close']].max(axis=1)),
        r.low.le(r[['open', 'high', 'close']].min(axis=1)),
        r.volume.eq(np.floor(r.volume)), r.volume.eq(0).eq(r.turnover.eq(0))])
    good &= (np.abs(numeric[:, :4] - np.round(numeric[:, :4], 2)) <= .0001).all(axis=1)
    unit = r.turnover / r.volume.where(r.volume.gt(0))
    good &= (~r.volume.gt(0) | (np.isfinite(unit) & unit.between(r.low - .0101, r.high + .0101))).to_numpy()
    r['good'] = good
    r['cents'] = np.round(r.close * 100)
    r['weighted'] = r.cents * r.volume
    r['morning'] = r.close.where(r.clock.lt('1130'))
    r['afternoon'] = r.close.where(r.clock.ge('1301'))
    r['tail_start'] = r.close.where(r.clock.eq('1430'))
    r['tail'] = r.close.where(r.clock.ge('1430'))
    r['tail_v'] = r.volume.where(r.clock.ge('1430'), 0)
    r['half_v'] = r.volume.where(r.clock.ge('1125'), 0)
    keys = ['date', 'code']
    result = r.groupby(keys, sort=True).agg(bars=('clock', 'size'), clocks=('clock', 'nunique'),
        good_bars=('good', 'sum'), first_close=('close', 'first'), last_close=('close', 'last'),
        morning_high=('morning', 'max'), afternoon_high=('afternoon', 'max'),
        day_high=('close', 'max'), day_low=('close', 'min'), tail_first=('tail_start', 'max'),
        tail_high=('tail', 'max'), tail_volume=('tail_v', 'sum'), half_volume=('half_v', 'sum'),
        volume=('volume', 'sum'), amount=('turnover', 'sum'), weighted=('weighted', 'sum'))
    result['vwap'] = result.weighted / result.volume.where(result.volume.gt(0)) / 100
    index = pd.MultiIndex.from_frame(r[keys])
    numerator, denominator = index.map(result.weighted), index.map(result.volume)
    assert np.nanmax(np.abs(r.cents * denominator)) < 2**53
    assert np.nanmax(np.abs(numerator)) < 2**53
    r['above'] = r.clock.ge('1430') & (r.cents * denominator > numerator)
    result['tail_above'] = r.groupby(keys).above.sum()
    result = result.reset_index()
    con.register('prefix', raw)
    sql = con.sql('''WITH valid AS(SELECT *,
        isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close)
        AND isfinite(volume) AND isfinite(turnover) AND least(open,high,low,close)>0
        AND volume>=0 AND turnover>=0 AND volume=floor(volume)
        AND high>=greatest(open,low,close) AND low<=least(open,high,close)
        AND abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
        AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001
        AND ((volume=0)=(turnover=0))
        AND (volume=0 OR (turnover/volume BETWEEN low-.0101 AND high+.0101)) AS good FROM prefix
    ), a AS(SELECT date,code,count(*) AS bars,count(DISTINCT clock) AS clocks,
        count(*) FILTER(WHERE good) AS good_bars,first(close ORDER BY clock) AS first_close,
        last(close ORDER BY clock) AS last_close,max(close) FILTER(WHERE clock<'1130') AS morning_high,
        max(close) FILTER(WHERE clock>='1301') AS afternoon_high,max(close) AS day_high,
        min(close) AS day_low,max(close) FILTER(WHERE clock='1430') AS tail_first,
        max(close) FILTER(WHERE clock>='1430') AS tail_high,
        sum(CASE WHEN clock>='1430' THEN volume ELSE 0 END) AS tail_volume,
        sum(CASE WHEN clock>='1125' THEN volume ELSE 0 END) AS half_volume,
        sum(volume) AS total_volume,sum(turnover) AS amount,
        sum(round(close*100)::HUGEINT*volume::HUGEINT) AS weighted,
        weighted/nullif(total_volume,0)/100 AS vwap FROM valid GROUP BY date,code
    ), above AS(SELECT prefix.date,prefix.code,count(*) FILTER(WHERE clock>='1430'
        AND round(close*100)::HUGEINT*a.total_volume::HUGEINT>a.weighted) AS tail_above
        FROM prefix JOIN a USING(date,code) GROUP BY prefix.date,prefix.code)
    SELECT a.* EXCLUDE(total_volume),a.total_volume AS volume,above.tail_above
        FROM a JOIN above USING(date,code) ORDER BY date,code''').df()
    pd.testing.assert_frame_equal(result[['date', 'code']], sql[['date', 'code']])
    for name in ['bars', 'clocks', 'good_bars', 'tail_above']:
        np.testing.assert_array_equal(result[name], sql[name])
    numeric_names = [x for x in result.columns if x not in ['date', 'code', 'bars', 'clocks', 'good_bars', 'tail_above']]
    np.testing.assert_allclose(result[numeric_names], sql[numeric_names], rtol=1e-12, atol=.001, equal_nan=True)
    quality = result.bars.eq(230) & result.clocks.eq(230) & result.good_bars.eq(230) & result.volume.gt(0)
    py_branch, py_buy = classify(result)
    py_branch[~quality] = -1
    py_buy[~quality] = False
    con.register('summary', sql)
    rebuilt = con.sql('''WITH d AS(SELECT *,100*(last_close-tail_first)/tail_first AS direction,
        CASE WHEN half_volume>0 THEN (tail_volume/20)/(half_volume/115) ELSE 1 END AS volume_ratio,
        tail_above/20 AS above FROM summary)
        SELECT CASE
        WHEN bars<>230 OR clocks<>230 OR good_bars<>230 OR NOT(volume>0) THEN -1
        WHEN morning_high>first_close*1.01 AND last_close<first_close*.99 THEN 0
        WHEN last_close>vwap*1.02 AND direction>1 AND volume_ratio>1.5 THEN 1
        WHEN above<.3 AND last_close<first_close*.98 THEN 2
        WHEN 100*(day_high-day_low)/day_low<2 AND abs(last_close-first_close)/first_close<.01 THEN 3
        WHEN afternoon_high>vwap*1.01 AND last_close>=vwap*.995 AND above>=.5 AND direction>-.3 THEN 4
        WHEN last_close/first_close-1>0 AND last_close/first_close-1<.03 AND above>=.7
            AND volume_ratio>=1 AND volume_ratio<=2 AND direction>=-.1 THEN 5
        WHEN last_close/first_close-1>.02 AND last_close/first_close-1<.06 AND above>=.6
            AND volume_ratio>=.9 AND abs(last_close-tail_high)/tail_high<.003 THEN 6
        WHEN above>=.4 AND last_close>first_close*.995 AND last_close>=vwap*.998 AND direction>-.2 THEN 7
        WHEN above>=.5 THEN 8 ELSE 9 END AS branch FROM d ORDER BY date,code''').df().branch.to_numpy()
    np.testing.assert_array_equal(py_branch, rebuilt)
    result['branch'] = py_branch
    result['buy'] = py_buy
    return result


def prepare():
    p = checked()
    assert not (ROOT / 'source_gate.json').exists(), 'Reuse completed source gates'
    f = pd.read_parquet(Path(p['features_root']) / 'features.parquet', columns=META,
        filters=[('date', '>=', '2024-01-01')]).sort_values(['date', 'code']).reset_index(drop=True)
    assert len(f) == 1258085 and f.date.lt('2026-01-01').all()
    target = f.loc[f.formula_input_valid, ['date', 'code']]
    manifest = json.loads(Path('data/research/next_day_winner/raw_report.json').read_text())
    sources = {}
    for name, digest in manifest['batch_manifests_sha256'].items():
        assert sha(Path(name)) == digest
        for file, value in json.loads(Path(name).read_text())['source_sha256'].items():
            assert file not in sources or sources[file] == value
            sources[file] = value
    codes = sorted(target.code.unique())
    parts = ROOT / 'inputs'
    parts.mkdir(exist_ok=True)
    con = conn()
    con.execute("SET memory_limit='2GB'")
    receipts, raw_rows = {}, 0
    for start in range(0, len(codes), 8):
        destination = parts / f'part_{start//8:04d}.parquet'
        receipt = destination.with_suffix('.json')
        if receipt.exists():
            v = json.loads(receipt.read_text())
            assert v['protocol_sha256'] == sha(PROTOCOL) and v['passed']
            assert v['part_sha256'] == sha(destination)
            check_sources(v['raw_file_sha256'])
        else:
            assert not destination.exists()
            subset = codes[start:start+8]
            files = [Path('data/hf/pilot/data/stock_1m') / code[:2].upper() / (code[3:] + '.parquet') for code in subset]
            hashes = {str(file): sources[str(file)] for file in files}
            check_sources(hashes)
            con.read_parquet([str(x) for x in files]).create_view('raw_source', replace=True)
            con.register('target', target.loc[target.code.isin(subset)])
            raw = con.sql('''SELECT t.date,t.code,strftime(timestamp,'%H%M') AS clock,
                open,high,low,close,volume,turnover FROM raw_source JOIN target t
                ON lower(exchange)||'.'||symbol=t.code AND strftime(timestamp,'%Y-%m-%d')=t.date
                WHERE timestamp>=TIMESTAMP '2024-01-01' AND timestamp<TIMESTAMP '2026-01-01'
                AND (strftime(timestamp,'%H%M') BETWEEN '0930' AND '1130'
                     OR strftime(timestamp,'%H%M') BETWEEN '1301' AND '1449') ORDER BY t.date,t.code,clock''').df()
            d = summarize(raw, con)
            d.to_parquet(destination, index=False, compression='zstd')
            v = dict(passed=True, protocol_sha256=sha(PROTOCOL), part_sha256=sha(destination),
                raw_file_sha256=hashes, rows=len(d), raw_rows=len(raw),
                all_prefix_summaries_quality_and_priority_branches_SQL_rebuilt=True,
                outcome_labels_read=False, new_2026_price_records_read=False)
            save_json(receipt, v)
            del raw, d
        raw_rows += v['raw_rows']
        receipts[str(destination)] = sha(destination)
        receipts[str(receipt)] = sha(receipt)
        print(json.dumps(dict(processed_codes=min(start+8, len(codes)), total_codes=len(codes), raw_rows=raw_rows)), flush=True)
    con.close()
    d = pd.concat([pd.read_parquet(Path(name)) for name in receipts if name.endswith('.parquet')], ignore_index=True)
    context = pd.read_parquet(RAW_CONTEXT, columns=['date', 'code', 'price_1449', 'volume_1449', 'amount_1449'],
        filters=[('date', '>=', '2024-01-01')])
    d = d.merge(context, on=['date', 'code'], validate='one_to_one')
    complete = d.bars.eq(230) & d.clocks.eq(230)
    assert d.loc[complete].last_close.sub(d.loc[complete].price_1449).abs().le(.0001).all()
    np.testing.assert_array_equal(d.loc[complete, 'volume'], d.loc[complete, 'volume_1449'])
    assert d.loc[complete].amount.sub(d.loc[complete].amount_1449).abs().le(.001+d.loc[complete].amount_1449.abs()*1e-12).all()
    d['prefix_quality_valid'] = complete & d.good_bars.eq(230) & d.volume.gt(0) & np.isfinite(d.vwap)
    result = f.merge(d[['date', 'code', 'prefix_quality_valid', 'branch', 'buy']], on=['date', 'code'], how='left', validate='one_to_one')
    result['prefix_quality_valid'] = result.prefix_quality_valid.fillna(False).astype(bool)
    result['literal_buy'] = result.formula_input_valid & result.prefix_quality_valid & result.buy.eq(True)
    result.to_parquet(ROOT / 'prefix_features.parquet', index=False, compression='zstd')
    d.to_parquet(ROOT / 'prefix_summary.parquet', index=False, compression='zstd')
    gate = dict(passed=True, protocol_sha256=sha(PROTOCOL), keys=len(f), original_valid=int(f.formula_input_valid.sum()),
        prefix_quality_valid=int(result.prefix_quality_valid.sum()), literal_buy=int(result.literal_buy.sum()),
        raw_rows=raw_rows, all_required_summaries_and_branches_SQL_rebuilt=True,
        features_sha256=sha(ROOT / 'prefix_features.parquet'), summary_sha256=sha(ROOT / 'prefix_summary.parquet'),
        input_receipts=receipts, raw_file_count=len(codes), newly_unknown_source=int((result.formula_input_valid & ~result.prefix_quality_valid).sum()),
        outcome_labels_read=False, new_fits=0, new_2026_price_records_read=False)
    save_json(ROOT / 'source_gate.json', gate)
    return {k: v for k, v in gate.items() if k != 'input_receipts'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['prepare'])
    args = parser.parse_args()
    print(json.dumps(prepare(), ensure_ascii=False))
