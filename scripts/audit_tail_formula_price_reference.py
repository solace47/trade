"""Audit prior reference-price resets without fitting or reading outcomes."""
import json
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.research_io import save_json, sha

PROTOCOL = Path('config/tail_formula_price_reference_audit.json')
ROOT = Path('data/research/tail_formula_price_reference')


def states(d):
    """Every active source row occupies its position, including a bad row."""
    d = d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True).copy()
    assert not d.date.duplicated().any()
    price = d[['open', 'high', 'low', 'close']].to_numpy(float)
    ref = d.preclose.to_numpy(float)
    good = (np.isfinite(price).all(axis=1) & (price > 0).all(axis=1)
            & d.adjustflag.eq(3).to_numpy() & (d.volume.to_numpy(float) > 0)
            & (price[:, 1] + .0001 >= price.max(axis=1))
            & (price[:, 2] - .0001 <= price.min(axis=1))
            & (np.abs(price - np.round(price, 2)) <= .0001).all(axis=1))
    prev = d.close.shift()
    reference_good = np.isfinite(ref) & (ref > 0) & good & pd.Series(good).shift(fill_value=False)
    d['reset'] = ((d.preclose-prev).abs() > .005).astype(float).where(reference_good)
    d['reference_good'] = reference_good
    d['prior_reference_count'] = d.reset.rolling(20, min_periods=0).count().shift(fill_value=0)
    d['prior_reset_count'] = d.reset.rolling(20, min_periods=20).sum().shift()
    tr = pd.concat([d.high-d.low, (d.high-prev).abs(), (d.low-prev).abs()], axis=1).max(axis=1)
    d['atr_raw'] = tr.rolling(20, min_periods=20).mean().shift()
    # This is a diagnostic alternative, not an adjusted-price input or factor.
    tr_ref = pd.concat([d.high-d.low, (d.high-d.preclose).abs(),
                        (d.low-d.preclose).abs()], axis=1).max(axis=1).where(reference_good)
    d['atr_reference'] = tr_ref.rolling(20, min_periods=20).mean().shift()
    d['history_price_good'] = pd.Series(good).rolling(21, min_periods=21).sum().shift().eq(21)
    d['prior_close'] = prev
    return d


def sql_states(c):
    return c.sql('''WITH a AS(SELECT *,lag(close) OVER w AS previous,
          lag(date) OVER w AS previous_date,
          isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close)
          AND least(open,high,low,close)>0 AND adjustflag=3 AND volume>0
          AND high+.0001>=greatest(open,high,low,close)
          AND low-.0001<=least(open,high,low,close)
          AND abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
          AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001 AS good
        FROM raw WHERE tradestatus=1 WINDOW w AS(PARTITION BY code ORDER BY date)),
      b AS(SELECT *,isfinite(preclose) AND preclose>0 AND good AND lag(good) OVER w AS rg,
          greatest(high-low,abs(high-previous),abs(low-previous)) AS tr_raw
        FROM a WINDOW w AS(PARTITION BY code ORDER BY date)),
      d AS(SELECT *,CASE WHEN rg THEN abs(preclose-previous)>.005 END AS reset,
          CASE WHEN rg THEN greatest(high-low,abs(high-preclose),abs(low-preclose)) END AS tr_ref
        FROM b),
      v AS(SELECT date,code,previous AS prior_close,
          count(reset) OVER w AS prior_reference_count,sum(reset::INT) OVER w AS prior_reset_count,
          avg(tr_raw) OVER w AS raw_avg,count(tr_raw) OVER w AS raw_n,
          avg(tr_ref) OVER w AS ref_avg,count(tr_ref) OVER w AS ref_n,
          count(good) OVER q=21 AND sum(good::INT) OVER q=21 AS history_price_good
        FROM d WINDOW w AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
          q AS(PARTITION BY code ORDER BY date ROWS BETWEEN 21 PRECEDING AND 1 PRECEDING))
      SELECT date,code,prior_close,prior_reference_count,
        CASE WHEN prior_reference_count=20 THEN prior_reset_count END AS prior_reset_count,
        CASE WHEN raw_n=20 THEN raw_avg END AS atr_raw,
        CASE WHEN ref_n=20 THEN ref_avg END AS atr_reference,history_price_good
      FROM v ORDER BY date,code''').df()


def run():
    p = json.loads(PROTOCOL.read_text())
    assert subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    assert p['implementation_sha256'] == sha(Path(__file__))
    assert not p['read_outcomes'] and not p['new_2026_values_allowed'] and p['maximum_fits'] == 0
    for file, digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest, file
    assert not (ROOT/'report.json').exists(), 'Never replace a completed audit'
    ROOT.mkdir(parents=True, exist_ok=True)
    feature_parts = []
    for source in p['features']:
        f = pd.read_parquet(source['file'], columns=['date','code','formula_input_valid','V01'],
                            filters=[('date','>=',source['first']),('date','<',source['end'])])
        feature_parts.append(f)
    keys = pd.concat(feature_parts).sort_values(['date','code']).reset_index(drop=True)
    assert len(keys) == p['expected_keys'] and int(keys.formula_input_valid.sum()) == p['expected_valid']
    assert not keys.duplicated(['date','code']).any() and keys.date.between('2023-01-01','2025-12-30').all()
    refs = []
    for source in p['references']:
        refs.append(pd.read_parquet(source['file'],columns=['date','code','preclose'],
                    filters=[('date','>=',source['first']),('date','<',source['end'])]))
    refs = pd.concat(refs).sort_values(['date','code']).reset_index(drop=True)
    keys = keys.merge(refs,on=['date','code'],how='left',validate='one_to_one')
    assert keys.preclose.notna().all()
    sources = {}
    for spec in p['daily_manifests']:
        for file,digest in json.loads(Path(spec['file']).read_text())[spec['field']].items():
            assert file not in sources or sources[file] == digest
            sources[file] = digest
    selected_sources = {p['daily_template'].format(code=code.replace('.','_')):code
                        for code in sorted(keys.code.unique())}
    raw, results = [], []
    for i,(file,code) in enumerate(selected_sources.items(),1):
        assert sha(Path(file)) == sources[file], file
        d = pd.read_parquet(file,columns=['date','code','open','high','low','close','preclose',
                           'volume','tradestatus','adjustflag'],
                           filters=[('date','>=',p['warmup_first']),('date','<','2026-01-01')])
        assert d.code.eq(code).all() and d.date.lt('2026-01-01').all()
        raw.append(d); s=states(d)
        results.append(s[['date','code','prior_close','prior_reference_count','prior_reset_count',
                          'atr_raw','atr_reference','history_price_good']])
        if i % 400 == 0: print(json.dumps({'verified_daily_files':i,'total':len(selected_sources)}),flush=True)
    raw = pd.concat(raw,ignore_index=True)
    rebuilt = pd.concat(results).sort_values(['date','code']).reset_index(drop=True)
    c=duckdb.connect();c.execute('SET threads=4');c.register('raw',raw)
    independent = sql_states(c); c.close()
    for name in ['prior_close','prior_reset_count','atr_raw','atr_reference']:
        np.testing.assert_allclose(rebuilt[name],independent[name],rtol=0,atol=2e-10,equal_nan=True)
    pd.testing.assert_frame_equal(rebuilt[['date','code','history_price_good']],
                                   independent[['date','code','history_price_good']],check_exact=True)
    np.testing.assert_array_equal(rebuilt.prior_reference_count.fillna(0), independent.prior_reference_count)
    out=keys.merge(rebuilt,on=['date','code'],how='left',validate='one_to_one')
    valid=out.formula_input_valid
    out['v01_raw_rebuilt']=100*out.atr_raw/out.preclose
    out['v01_reference_diagnostic']=100*out.atr_reference/out.preclose
    match=np.isfinite(out.v01_raw_rebuilt)&np.isfinite(out.V01)&(out.v01_raw_rebuilt-out.V01).abs().le(2e-10)
    out['reference_window_known']=out.history_price_good.fillna(False)&out.prior_reference_count.eq(20)
    out['reference_reset_in_prior20']=out.prior_reset_count.gt(0).where(out.reference_window_known)
    out.to_parquet(ROOT/'input_reference_audit.parquet',index=False,compression='zstd')
    records=[]
    for half,g in out.assign(half=out.date.str[:4]+'H'+np.where(out.date.str[5:7].astype(int).le(6),'1','2')).groupby('half'):
        v=g.loc[g.formula_input_valid]; known=v.reference_window_known; exposed=known&v.prior_reset_count.gt(0)
        records.append(dict(half=half,original_valid=len(v),reference_unknown=int((~known).sum()),
            prior20_reset_exposed=int(exposed.sum()),
            volatility_changed_over_tolerance=int((known&(v.v01_raw_rebuilt-v.v01_reference_diagnostic).abs().gt(2e-10)).sum()),
            reference_volatility_lower=int((known&v.v01_reference_diagnostic.lt(v.v01_raw_rebuilt-2e-10)).sum())))
    r=dict(passed=bool(match[valid].all()),protocol_sha256=sha(PROTOCOL),implementation_sha256=sha(Path(__file__)),
        source_hashes=p['source_hashes'],rows=len(out),original_valid=int(valid.sum()),
        daily_files=len(selected_sources),raw_rows=len(raw),daily_first=raw.date.min(),daily_last=raw.date.max(),
        original_valid_v01_mismatch=int((valid&~match).sum()),
        original_valid_reference_unknown=int((valid&~out.reference_window_known).sum()),
        source_reference_changes_not_verified_corporate_actions=True,
        raw_vs_reference_TR_is_diagnostic_not_asof_adjustment=True,
        independent_pandas_SQL_all_states_agree=True,by_half=records,
        output_sha256=sha(ROOT/'input_reference_audit.parquet'),maximum_fits=0,
        outcomes_read=False,selected_lists_or_economics_changed=False,new_2026_values_read=False)
    save_json(ROOT/'report.json',r)
    print(json.dumps({k:v for k,v in r.items() if k!='source_hashes'},ensure_ascii=False,indent=2),flush=True)
    return r


if __name__ == '__main__':
    run()
