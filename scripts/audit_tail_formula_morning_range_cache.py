"""Inventory schemas and timestamp coverage without projecting cached prices."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from trade_research import tail_formula_additive as base
from trade_research.corporate_cash import save_json, sha

ROOT = Path('data/research/tail_formula_morning_range')
POOL = Path('data/research/tail_formula_dense_path/inputs/features.parquet')


def schemas():
    path = ROOT/'cache_schema_inventory_v2.json'
    assert not path.exists()
    records = []; candidates = []; errors = []
    # Reading all schemas is safe, including 2026 filenames. A substring filter
    # wrongly omitted stock codes such as 002026 in the exploratory v1 list.
    for file in sorted(Path('data/research').rglob('*.parquet')):
        try:
            columns = pq.read_schema(file).names
            records.append(dict(path=str(file), columns=columns))
            if {'timestamp','high','low','close'}.issubset(columns):
                candidates.append(str(file))
        except Exception as error:
            errors.append(dict(path=str(file), error=str(error)))
    old = json.loads((ROOT/'cache_schema_inventory.json').read_text())
    new = set(candidates)-set(old['timestamp_hlc_candidates'])
    out = dict(passed=not errors, records=records, timestamp_hlc_candidates=candidates,
        errors=errors, no_path_substring_exclusions=True,
        exploratory_v1_sha256=sha(ROOT/'cache_schema_inventory.json'),
        additional_candidates=sorted(new), no_row_values_or_row_group_statistics_read=True,
        no_price_fields_read=True, no_new_2026_prices_read=True,
        inventory_helper_sha256=sha(Path(__file__)))
    save_json(path,out)
    return dict(schema_inventory_sha256=sha(path), files=len(records),
        candidates=len(candidates), additional_candidates=len(new), errors=len(errors))


def identity(columns):
    if 'code' in columns:
        return 'lower(code::VARCHAR)'
    if {'exchange','symbol'}.issubset(columns):
        return "lower(exchange::VARCHAR)||'.'||lpad(symbol::VARCHAR,6,'0')"
    return None


def coverage():
    path = ROOT/'cache_timestamp_coverage_v2.json'; assert not path.exists()
    invpath = ROOT/'cache_schema_inventory_v2.json'
    inv = json.loads(invpath.read_text()); assert inv['passed']
    p = pd.read_parquet(POOL,columns=['date','code'])
    keys = p.loc[p.date.ge('2024-01-01') & p.date.lt('2026-01-01')].reset_index(drop=True)
    assert len(keys)==1258085 and not keys.duplicated().any()
    keys_sha = sha(POOL)
    candidates = set(inv['timestamp_hlc_candidates']); groups = defaultdict(list); unsupported = []
    for r in inv['records']:
        if r['path'] not in candidates:continue
        expr = identity(r['columns'])
        if expr is None:unsupported.append(r);continue
        groups[(str(Path(r['path']).parent),expr)].append(r['path'])
    folder = ROOT/'coverage_parts_v2'; folder.mkdir(exist_ok=True)
    c = base.conn()
    masks = c.sql('''SELECT bit_or(CASE WHEN i<64 THEN 1::UBIGINT<<i ELSE 0::UBIGINT END)::UBIGINT,
        bit_or(CASE WHEN i>=64 THEN 1::UBIGINT<<(i-64) ELSE 0::UBIGINT END)::UBIGINT
        FROM range(121) r(i)''').df();c.close()
    assert tuple(int(x) for x in masks.iloc[0])==(2**64-1,2**57-1)
    assert all(str(t)=='uint64' for t in masks.dtypes)
    records = []; outputs = []; source_hashes = {}
    for number,((family,expr),files) in enumerate(sorted(groups.items())):
        out = folder/f'part_{number:03d}.parquet'; meta = out.with_suffix('.json')
        inputs = {f:sha(Path(f)) for f in files}; source_hashes.update(inputs)
        if out.exists():
            receipt = json.loads(meta.read_text())
            assert receipt['source_hashes']==inputs and receipt['keys_source_sha256']==keys_sha
            assert receipt['output_sha256']==sha(out)
        else:
            c = base.conn(); c.register('pool',keys)
            c.read_parquet(files,union_by_name=True).create_view('raw')
            # Project only physical timestamps and security identity. Cached
            # signal dates can describe the previous day and must not be used.
            f = c.sql(f'''WITH t AS(SELECT {expr} AS code,try_cast(timestamp AS TIMESTAMP) AS ts
                FROM raw WHERE try_cast(timestamp AS TIMESTAMP)>=TIMESTAMP '2024-01-01'
                AND try_cast(timestamp AS TIMESTAMP)<TIMESTAMP '2026-01-01'),
                b AS(SELECT code,ts,strftime(ts,'%Y-%m-%d') AS date,
                    hour(ts)*60+minute(ts)-570 AS idx,ts=date_trunc('minute',ts) AS aligned
                FROM t WHERE strftime(ts,'%H%M') BETWEEN '0930' AND '1130')
                SELECT date,code,count(*) AS rows,count(DISTINCT ts) AS unique_timestamps,
                    count(*) FILTER(WHERE aligned) AS aligned_rows,
                    bit_or(CASE WHEN aligned AND idx<64 THEN 1::UBIGINT<<idx ELSE 0::UBIGINT END)::UBIGINT AS mask0,
                    bit_or(CASE WHEN aligned AND idx>=64 THEN 1::UBIGINT<<(idx-64) ELSE 0::UBIGINT END)::UBIGINT AS mask1
                FROM b JOIN pool USING(date,code) GROUP BY date,code ORDER BY date,code''').df();c.close()
            # Empty results use nullable UInt64 in pandas; nonempty masks are
            # exact UBIGINT values, never a float intermediary.
            assert f[['mask0','mask1']].notna().all().all()
            if len(f):assert all(str(f[n].dtype)=='uint64' for n in ['mask0','mask1'])
            for name in ['mask0','mask1']:f[name]=f[name].astype('uint64')
            f.to_parquet(out,index=False,compression='zstd')
            receipt = dict(source_family=family,identity_expression=expr,source_hashes=inputs,
                keys_source_sha256=keys_sha,output_sha256=sha(out),rows=len(f),
                full_121_label_keys=int((f.mask0.eq(2**64-1)&f.mask1.eq(2**57-1)).sum()),
                cached_rows=int(f.rows.sum()),duplicate_timestamp_rows=int((f.rows-f.unique_timestamps).sum()),
                no_price_fields_projected=True,no_strategy_returns_or_selections_read=True,
                no_new_2026_prices_read=True)
            save_json(meta,receipt)
        records.append(receipt);outputs.append(str(out))
        print(json.dumps(dict(family=family,keys=receipt['rows'],full=receipt['full_121_label_keys'])),flush=True)
    c = base.conn();c.read_parquet(outputs,union_by_name=True).create_view('coverage')
    f = c.sql('SELECT date,code,bit_or(mask0)::UBIGINT AS mask0,bit_or(mask1)::UBIGINT AS mask1 FROM coverage GROUP BY date,code ORDER BY date,code').df();c.close()
    assert all(str(f[n].dtype)=='uint64' for n in ['mask0','mask1'])
    out = ROOT/'cached_timestamp_union_v2.parquet';f.to_parquet(out,index=False,compression='zstd')
    r = dict(passed=True,schema_inventory_sha256=sha(invpath),keys_source_sha256=keys_sha,
        source_hashes=source_hashes,records=records,unsupported_identity=unsupported,
        complete_keys=1258085,cached_keys=len(f),full_121_label_keys=int((f.mask0.eq(2**64-1)&f.mask1.eq(2**57-1)).sum()),
        cached_timestamp_union_sha256=sha(out),helper_sha256=sha(Path(__file__)),
        timestamps_are_actual_current_day_not_signal_dates=True,no_price_fields_projected=True,
        no_new_2026_prices_read=True,no_strategy_returns_or_selections_read=True,
        timestamp_coverage_not_quality_or_source_provenance=True,
        prior_draft_coverage_sha256=sha(ROOT/'cache_timestamp_coverage.json'),
        hugeint_to_pandas_float_precision_repaired_before_input_protocol=True,
        all_mask_results_explicit_uint64_without_float_intermediate=True,
        all_121_labels_synthetic_mask_checked=True)
    save_json(path,r)
    return {k:r[k] for k in ['cached_keys','full_121_label_keys','complete_keys','unsupported_identity']}


def alternate():
    """The single OHLC cache without a timestamp uses date + HHMM labels."""
    path=ROOT/'alternate_clock_coverage.json';assert not path.exists()
    raw=Path('data/research/external_tail_combo/raw_prefix.parquet')
    report=raw.parent/'input_report.json';r=json.loads(report.read_text())
    assert sha(raw)==r['outputs_sha256']['raw_prefix']
    hashes=json.loads(Path('data/research/economic_winner/input_manifest.json').read_text())['source_sha256']
    for file,digest in r['minute_files_sha256'].items():assert hashes[file]==digest
    keys=pd.read_parquet(POOL,columns=['date','code']);keys=keys[keys.date.ge('2024-01-01')].reset_index(drop=True)
    c=base.conn();c.register('pool',keys)
    labels=c.sql(f'''SELECT r.date,r.code,label FROM read_parquet('{raw}') r JOIN pool USING(date,code)
        WHERE r.date>='2024-01-01' AND r.date<'2026-01-01' AND label BETWEEN '0930' AND '1130'
        ORDER BY date,code,label''').df();c.close()
    frames=[];sources={};codes=sorted(labels.code.unique())
    for start in range(0,len(codes),64):
        subset=codes[start:start+64];group=labels[labels.code.isin(subset)]
        files=[file for file in r['minute_files_sha256'] if Path(file).stem in [code[3:] for code in subset]
            and Path(file).parent.name.lower() in [code[:2] for code in subset]]
        for file in files:assert sha(Path(file))==hashes[file];sources[file]=hashes[file]
        c=base.conn();c.register('wanted',group[['date','code']].drop_duplicates());c.read_parquet(files).create_view('original')
        physical=c.sql('''WITH t AS(SELECT lower(exchange)||'.'||symbol AS code,timestamp,
            strftime(timestamp,'%Y-%m-%d') AS date,strftime(timestamp,'%H%M') AS label
            FROM original WHERE timestamp>=TIMESTAMP '2024-01-01' AND timestamp<TIMESTAMP '2026-01-01'
            AND strftime(timestamp,'%H%M') BETWEEN '0930' AND '1130')
            SELECT date,code,label,timestamp=date_trunc('minute',timestamp) AS aligned
            FROM t JOIN wanted USING(date,code) ORDER BY date,code,label''').df();c.close()
        pd.testing.assert_frame_equal(group.reset_index(drop=True),physical[['date','code','label']],check_exact=True)
        assert physical.aligned.all()
        frames.append(physical)
    f=pd.concat(frames,ignore_index=True);g=f.groupby(['date','code']).agg(rows=('label','size'),labels=('label','nunique')).reset_index()
    assert g.rows.eq(121).all() and g.labels.eq(121).all()
    out=ROOT/'alternate_full_keys.parquet';g[['date','code']].to_parquet(out,index=False,compression='zstd')
    result=dict(passed=True,keys=len(g),keys_sha256=sha(out),raw_cache_sha256=sha(raw),
        input_report_sha256=sha(report),source_sha256=sources,helper_sha256=sha(Path(__file__)),
        all_full_labels_and_actual_source_timestamp_alignment_equal=True,
        no_price_fields_projected=True,no_new_2026_prices_read=True,no_strategy_returns_or_selections_read=True)
    save_json(path,result);return dict(keys=len(g),alternate_coverage_sha256=sha(path))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['schemas','coverage','alternate'])
    print(json.dumps(globals()[parser.parse_args().stage](),ensure_ascii=False,indent=2))
