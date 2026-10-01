"""Add the existing morning-extrema definition to 2023 training inputs only."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_morning_range as prior
from . import tail_formula_float as original
from .corporate_cash import save_json, sha

STEM = 'tail_formula_stock_2024'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
HISTORY = Path('data/research/tail_formula_quarter_history2024/inputs')
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
INTENT = Path('config') / (STEM + '_intent.json')
META = prior.META
EXPRESSIONS = prior.EXPRESSIONS
LABEL_FIELDS = ['date', 'code', 'next_date', 'known15', 'opportunity15', 'known_no_trade', 'adverse_return15']


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['intent_sha256'] == sha(INTENT) and p['expressions'] == EXPRESSIONS
    assert original.EXPRESSIONS == prior.ARMS['control']
    assert p['historical_first'] == '2023-01-01' and p['historical_end'] == '2024-01-01'
    assert p['expected_keys'] == 1144320 and not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items(): assert sha(Path(file)) == digest, file
    m = json.loads((ROOT / 'metadata_lookup_verification.json').read_text())
    assert m['passed'] and m['full_existing_2024_metadata_and_original_validity_exact']
    done = json.loads(Path(p['conditional_completion']).read_text())
    assert done['passed'] and len(done['comparisons']) == 6
    return p, m


def raw():
    p, metadata = checked(); INPUTS.mkdir(parents=True, exist_ok=True)
    assert not (INPUTS / 'raw_report.json').exists()
    keys = pd.read_parquet(HISTORY / 'features.parquet', columns=['date', 'code'],
                           filters=[('date', '>=', p['historical_first']), ('date', '<', p['historical_end'])])
    assert len(keys) == 557044
    folder = INPUTS / 'raw_parts'; folder.mkdir(exist_ok=True); parts, sources = {}, {}
    for start in range(0, len(metadata['files']), 64):
        items = metadata['files'][start:start+64]; codes = [s['code'] for s in items]
        out = folder / f'part_{start//64:03d}.parquet'; receipt_path = out.with_suffix('.json')
        identity = dict(protocol_sha256=sha(PROTOCOL), extractor_sha256=sha(Path(__file__)), codes=codes)
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            assert all(receipt[k] == v for k, v in identity.items()) and receipt['sha256'] == sha(out)
        else:
            assert not out.exists(), 'Unreceipted raw fragment must be investigated'
            for item in items:
                assert sha(Path(item['download_metadata'])) == item['download_metadata_sha256']
                assert sha(Path(item['file'])) == item['expected_sha256'], item['code']
            c = base.conn(); c.read_parquet([s['file'] for s in items]).create_view('original')
            c.register('keys', keys.loc[keys.code.isin(codes)])
            d = c.sql("""WITH b AS(SELECT lower(exchange)||'.'||symbol AS code,
                strftime(timestamp,'%Y-%m-%d') AS date,timestamp,open::DOUBLE AS open,high::DOUBLE AS high,
                low::DOUBLE AS low,close::DOUBLE AS close,volume::DOUBLE AS volume FROM original
                WHERE timestamp>=TIMESTAMP '2023-01-01' AND timestamp<TIMESTAMP '2024-01-01'
                AND strftime(timestamp,'%H%M') BETWEEN '0930' AND '1130')
                SELECT b.date,b.code,b.timestamp,b.open,b.high,b.low,b.close,b.volume
                FROM b JOIN keys USING(date,code) ORDER BY date,code,timestamp""").df(); c.close()
            assert not d.duplicated(['date', 'code', 'timestamp']).any()
            d.to_parquet(out, index=False, compression='zstd')
            receipt = dict(**identity, sha256=sha(out), rows=len(d)); save_json(receipt_path, receipt)
        parts[str(out)] = receipt['sha256']
        sources.update({s['file']: s['expected_sha256'] for s in items})
        print(json.dumps(dict(codes=start+len(items), total=len(metadata['files']), rows=receipt['rows'])), flush=True)
    r = dict(protocol_sha256=sha(PROTOCOL), metadata_sha256=sha(ROOT/'metadata_lookup_verification.json'),
        parts_sha256=parts, source_sha256=sources, historical_input_keys=len(keys),
        only_2023_morning_inputs_extracted=True, no_new_group_evaluation=True,
        new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'raw_report.json', r)
    return dict(parts=len(parts), codes=len(sources))


def checked_raw():
    checked(); r = json.loads((INPUTS / 'raw_report.json').read_text())
    assert r['protocol_sha256'] == sha(PROTOCOL)
    for file, digest in r['parts_sha256'].items(): assert sha(Path(file)) == digest, file
    return r


def aggregates():
    r = checked_raw(); c = base.conn(); c.read_parquet(list(r['parts_sha256'])).create_view('raw')
    d = c.sql('''WITH b AS(SELECT *,coalesce(timestamp=date_trunc('minute',timestamp)
        AND isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close) AND isfinite(volume)
        AND least(open,high,low,close)>0 AND volume>=0 AND high+.0001>=greatest(open,close,low)
        AND low-.0001<=least(open,close) AND abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
        AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001,false) AS good FROM raw)
        SELECT date,code,count(*) AS bars,count(DISTINCT strftime(timestamp,'%H%M')) AS clocks,
        count(*) FILTER(WHERE good) AS good_bars,count(*) FILTER(WHERE volume>0) AS active,
        max(floor(high*100+.5)) FILTER(WHERE volume>0) AS high_cents,
        min(floor(low*100+.5)) FILTER(WHERE volume>0) AS low_cents
        FROM b GROUP BY date,code ORDER BY date,code''').df(); c.close()
    return d


def add_extrema(frame, agg):
    f = frame.merge(agg, on=['date', 'code'], how='left', validate='one_to_one')
    valid = (f.bars.eq(121) & f.clocks.eq(121) & f.good_bars.eq(121) & f.active.gt(0)
             & f.high_cents.gt(0) & f.low_cents.gt(0) & f.V01.gt(0) & np.isfinite(f.V01) & f.A04.gt(0))
    q = np.floor(f.A04 * 100 + .5)
    with np.errstate(all='ignore'):
        for name, edge in [('AMHD', 'high_cents'), ('AMLD', 'low_cents')]:
            f[name] = (100 * (q / f[edge] - 1) / f.V01).where(valid)
    valid &= np.isfinite(f[list(prior.NEW_EXPRESSIONS)]).all(axis=1)
    f['morning_input_valid'] = valid
    f['formula_input_valid'] &= valid
    return f


def features():
    checked(); assert not (INPUTS / 'feature_report.json').exists()
    old = pd.read_parquet(HISTORY / 'features.parquet', filters=[('date', '<', '2024-01-01')])
    agg = aggregates(); agg.to_parquet(INPUTS / 'morning_aggregates.parquet', index=False, compression='zstd')
    early = add_extrema(old, agg)
    current = pd.read_parquet(prior.INPUTS / 'features.parquet', columns=[*META, *EXPRESSIONS],
        filters=[('date', '>=', '2024-01-01'), ('date', '<', '2025-01-01')])
    f = pd.concat([early[[*META, *EXPRESSIONS]], current], ignore_index=True).sort_values(['date', 'code']).reset_index(drop=True)
    assert len(f) == 1144320 and not f.duplicated(['date', 'code']).any()
    assert np.isfinite(f.loc[f.formula_input_valid, list(EXPRESSIONS)]).all().all()
    f.to_parquet(INPUTS / 'features.parquet', index=False, compression='zstd')
    r = dict(protocol_sha256=sha(PROTOCOL), features_sha256=sha(INPUTS/'features.parquet'), rows=len(f),
        valid=int(f.formula_input_valid.sum()), first=f.date.min(), last=f.date.max(), expressions=EXPRESSIONS,
        native_header=prior.HEADER, raw_report_sha256=sha(INPUTS/'raw_report.json'),
        morning_aggregates_sha256=sha(INPUTS/'morning_aggregates.parquet'),
        historical_newly_invalid=int((old.formula_input_valid & ~early.formula_input_valid).sum()),
        current_2024_values_and_validity_reused_exactly=True, historical_missing_not_imputed=True,
        software_compilation_verified=False, native_source_parity_verified=False,
        no_new_group_evaluation=True, new_2026_prices_read=False, no_exit_rules=True)
    save_json(INPUTS / 'feature_report.json', r)
    return {k:v for k,v in r.items() if k not in ['expressions', 'native_header']}


def pandas_aggregates(raw):
    d = raw.copy(); a = d[['open', 'high', 'low', 'close', 'volume']].to_numpy(float)
    o, h, l, cl, v = a.T
    good = (np.isfinite(a).all(axis=1) & (a[:,:4].min(axis=1)>0) & (v>=0)
        & (h+.0001>=np.maximum.reduce([o,cl,l])) & (l-.0001<=np.minimum(o,cl))
        & (np.abs(a[:,:4] - np.rint(a[:,:4]*100)/100)<=.0001).all(axis=1)
        & d.timestamp.eq(d.timestamp.dt.floor('min')))
    d['good'] = good; d['active'] = v>0; d['clock'] = d.timestamp.dt.strftime('%H%M')
    d['hc'] = np.floor(h*100+.5); d['lc'] = np.floor(l*100+.5)
    d.loc[~d.active, ['hc', 'lc']] = np.nan
    return d.groupby(['date','code']).agg(bars=('timestamp','size'), clocks=('clock','nunique'),
        good_bars=('good','sum'), active=('active','sum'), high_cents=('hc','max'), low_cents=('lc','min')).reset_index()


def verify():
    r = checked_raw(); fr = json.loads((INPUTS/'feature_report.json').read_text())
    assert fr['protocol_sha256'] == sha(PROTOCOL) and fr['features_sha256'] == sha(INPUTS/'features.parquet')
    parts = []
    for file in r['parts_sha256']:
        d = pd.read_parquet(file)
        assert d.date.eq(d.timestamp.dt.strftime('%Y-%m-%d')).all()
        assert d.date.ge('2023-01-01').all() and d.date.lt('2024-01-01').all()
        assert d.timestamp.dt.strftime('%H%M').between('0930','1130').all()
        assert not d.duplicated(['date','code','timestamp']).any()
        parts.append(pandas_aggregates(d))
    agg = pd.concat(parts, ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    assert not agg.duplicated(['date','code']).any()
    pd.testing.assert_frame_equal(pd.read_parquet(INPUTS/'morning_aggregates.parquet'), agg, check_dtype=False, check_exact=True)
    old = pd.read_parquet(HISTORY/'features.parquet', filters=[('date','<','2024-01-01')])
    a = old.merge(agg,on=['date','code'],how='left',validate='one_to_one')
    c = base.conn(); c.register('a',a)
    names = ','.join(prior.ARMS['control'])
    expected = c.sql(f'''WITH s AS(SELECT *,coalesce(bars=121 AND clocks=121 AND good_bars=121
        AND active>0 AND high_cents>0 AND low_cents>0 AND A04>0 AND V01>0 AND isfinite(V01),false) AS ready FROM a)
        SELECT date,code,half,board,decision_shares,formula_input_valid AND ready AS formula_input_valid,{names},
        CASE WHEN ready THEN 100*(floor(A04*100+.5)/high_cents-1)/V01 END AS AMHD,
        CASE WHEN ready THEN 100*(floor(A04*100+.5)/low_cents-1)/V01 END AS AMLD FROM s ORDER BY date,code''').df(); c.close()
    got = pd.read_parquet(INPUTS/'features.parquet')
    early = got.loc[got.date.lt('2024-01-01')].reset_index(drop=True)
    pd.testing.assert_frame_equal(early[META+list(prior.ARMS['control'])], expected[META+list(prior.ARMS['control'])], check_exact=True)
    np.testing.assert_allclose(early[list(prior.NEW_EXPRESSIONS)],expected[list(prior.NEW_EXPRESSIONS)],rtol=0,atol=2e-11,equal_nan=True)
    current = pd.read_parquet(prior.INPUTS/'features.parquet',columns=[*META,*EXPRESSIONS],
        filters=[('date','>=','2024-01-01'),('date','<','2025-01-01')])
    pd.testing.assert_frame_equal(got.loc[got.date.ge('2024-01-01')].reset_index(drop=True),current,check_exact=True)
    enc = lambda x: np.floor(np.clip(100*x+10000+.000001,0,999999)).astype('int32')
    good = early.formula_input_valid
    np.testing.assert_array_equal(enc(early.loc[good,list(EXPRESSIONS)].to_numpy()),enc(expected.loc[good,list(EXPRESSIONS)].to_numpy()))
    v = dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),rows=len(got),
        all_2023_raw_quality_extrema_and_50_value_encodings_independently_verified=True,
        all_original_2023_48_values_and_2024_50_values_preserved=True,
        current_2024_input_intersection_unchanged=True,
        historical_same_quality_control_required=True, no_new_group_evaluation=True,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'feature_verification.json',v); return v


def native():
    r = checked_raw(); v = json.loads((INPUTS/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(INPUTS/'feature_report.json')
    f = pd.read_parquet(INPUTS/'features.parquet',filters=[('date','<','2024-01-01')])
    agg = pd.read_parquet(INPUTS/'morning_aggregates.parquet')
    d = f[['date','code','A04','V01','AMHD','AMLD']].merge(agg,on=['date','code'],how='left',validate='one_to_one')
    ready = d.bars.eq(121)&d.clocks.eq(121)&d.good_bars.eq(121)&d.active.gt(0)&d.high_cents.gt(0)&d.low_cents.gt(0)&d.A04.gt(0)&d.V01.gt(0)&np.isfinite(d.V01)
    env = dict(AMREADY=ready.to_numpy(),AMQC=np.floor(d.A04.to_numpy()*100+.5),AMHC=d.high_cents.to_numpy(),
               AMLC=d.low_cents.to_numpy(),VP20=d.V01.to_numpy(),IF=np.where,DRAWNULL=np.nan)
    for name, expr in prior.NEW_EXPRESSIONS.items():
        with np.errstate(all='ignore'): actual=eval(expr,{'__builtins__':{}},env)
        np.testing.assert_allclose(actual,d[name],rtol=0,atol=2e-11,equal_nan=True)
    sample = f[['date','code']].copy(); sample['half']=sample.date.str[5:7].le('06')
    sample['key_hash']=[hashlib.sha256((day+code).encode()).hexdigest() for day,code in zip(sample.date,sample.code)]
    sample=sample.sort_values(['half','key_hash']).groupby('half').head(24)
    c=base.conn(); c.read_parquet(list(r['parts_sha256'])).create_view('raw'); c.register('keys',sample[['date','code']])
    bars=c.sql('SELECT r.* FROM raw r JOIN keys USING(date,code) ORDER BY date,code,timestamp').df(); c.close()
    index=f.set_index(['date','code']); probes=[]
    for key, b in bars.groupby(['date','code']):
        row=index.loc[key]; result=prior.native_values(b,row.A04,row.V01)
        np.testing.assert_allclose(result,row[list(prior.NEW_EXPRESSIONS)].to_numpy(float),rtol=0,atol=2e-11,equal_nan=True)
        probes.append(dict(date=key[0],code=key[1],bars=len(b)))
    assert len(probes)==48
    out=dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),
        feature_verification_sha256=sha(INPUTS/'feature_verification.json'),
        original_48_native_definitions_and_source_proofs_reused=True,
        all_new_literal_numeric_expressions_and_48_literal_header_samples_verified=True,probes=probes,
        software_compilation_verified=False,native_source_parity_verified=False,
        no_new_group_evaluation=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'native_input_verification.json',out);return dict(samples=len(probes),passed=True)


def labels():
    checked(); assert not (INPUTS/'full_label_report.json').exists()
    files=[HISTORY/'history_2023/full_labels.parquet',prior.INPUTS/'full_labels.parquet']
    parents=[file.parent for file in files]
    for parent in parents:
        r=json.loads((parent/'full_label_report.json').read_text());v=json.loads((parent/'full_label_verification.json').read_text())
        assert v['passed'] and v['label_report_sha256']==sha(parent/'full_label_report.json')
        assert r['labels_sha256']==sha(parent/'full_labels.parquet')
    frames=[pd.read_parquet(file,columns=LABEL_FIELDS,filters=[('date','>=',first),('date','<',end)])
        for file,first,end in [(files[0],'2023-01-01','2024-01-01'),(files[1],'2024-01-01','2025-01-01')]]
    f=pd.concat(frames,ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    c=base.conn();cols=','.join(LABEL_FIELDS)
    expected=c.sql(f"SELECT {cols} FROM read_parquet('{files[0]}') WHERE date>='2023-01-01' AND date<'2024-01-01' UNION ALL SELECT {cols} FROM read_parquet('{files[1]}') WHERE date>='2024-01-01' AND date<'2025-01-01' ORDER BY date,code").df();c.close()
    pd.testing.assert_frame_equal(f,expected,check_exact=True)
    keys=pd.read_parquet(INPUTS/'features.parquet',columns=['date','code'])
    pd.testing.assert_frame_equal(f[['date','code']],keys,check_exact=True)
    assert len(f)==1144320 and not f.duplicated(['date','code']).any()
    f.to_parquet(INPUTS/'full_labels.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),labels_sha256=sha(INPUTS/'full_labels.parquet'),rows=len(f),
        source_hashes={str(parent/file):sha(parent/file) for parent in parents for file in ['full_label_report.json','full_label_verification.json']},
        strict_0931_0959_source_labels_reused=True,historical_tax_and_original_unknowns_unchanged=True,
        training_labels_only=True,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'full_label_report.json',r)
    save_json(INPUTS/'full_label_verification.json',dict(passed=True,label_report_sha256=sha(INPUTS/'full_label_report.json'),
        all_original_seven_fields_and_full_keys_independently_sql_verified=True,
        no_new_group_evaluation=True,new_2026_prices_read=False,no_exit_rules=True))
    return dict(rows=len(f),passed=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['raw','features','verify','native','labels'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
