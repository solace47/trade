"""Count profitable observable minutes only inside the fixed training windows."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from .corporate_cash import MINUTES,save_json,sha

ROOT=Path('data/research/tail_formula_duration')
PROTOCOL=Path('config/tail_formula_duration_labels_protocol.json')
MANIFEST=Path('data/research/economic_winner/input_manifest.json')
START='2024-01-01'
END='2025-07-01'


def source_keys():
    config=json.loads(PROTOCOL.read_text())
    report=json.loads((base.SOURCE/'full_label_report.json').read_text())
    proof=json.loads((base.SOURCE/'full_label_verification.json').read_text())
    assert sha(base.SOURCE/'full_label_report.json')==config['full_label_report_sha256']
    assert sha(base.SOURCE/'full_label_verification.json')==config['full_label_verification_sha256']
    assert proof['passed'] and proof['label_report_sha256']==sha(base.SOURCE/'full_label_report.json')
    assert report['labels_sha256']==sha(base.SOURCE/'full_labels.parquet')
    assert sha(MANIFEST)==config['raw_manifest_sha256']
    c=base.conn()
    keys=c.execute('''SELECT date,code,next_date,half,board,decision_shares,buy_cash15,
        known15,opportunity15,any_opportunity15,mark_1000_return15
        FROM read_parquet(?) WHERE date>=? AND next_date<? AND known15 ORDER BY date,code''',
        [str(base.SOURCE/'full_labels.parquet'),START,END]).df()
    c.close()
    assert not keys.duplicated(['date','code']).any() and keys.date.ge(START).all() and keys.next_date.lt(END).all()
    assert keys.next_date.gt(keys.date).all() and keys.board.eq('main').all()
    assert keys.buy_cash15.gt(0).all() and np.isfinite(keys.buy_cash15).all()
    return keys,json.loads(MANIFEST.read_text())['source_sha256']


def labels():
    if (ROOT/'duration_report.json').exists():
        raise ValueError('Do not replace frozen training duration labels')
    keys,hashes=source_keys();codes=sorted(keys.code.unique())
    folder=ROOT/'duration_parts';folder.mkdir(parents=True,exist_ok=True)
    parts={};extractor=sha(Path(__file__))
    for offset in range(0,len(codes),64):
        subset=codes[offset:offset+64];path=folder/f'part_{offset//64:03d}.parquet';receipt=path.with_suffix('.json')
        if receipt.exists():
            meta=json.loads(receipt.read_text())
            assert meta['protocol_sha256']==sha(PROTOCOL) and meta['manifest_sha256']==sha(MANIFEST)
            assert meta['codes']==subset and meta['extractor_sha256']==extractor and meta['sha256']==sha(path)
        else:
            files=[MINUTES/code[:2].upper()/(code[3:]+'.parquet') for code in subset]
            for file in files:
                assert sha(file)==hashes[str(file)]
            c=base.conn();c.read_parquet([str(x) for x in files]).create_view('raw')
            sub=keys.loc[keys.code.isin(subset)]
            c.register('keys',sub)
            rows=c.sql('''WITH raw_window AS (
                SELECT lower(exchange)||'.'||symbol AS code,strftime(timestamp,'%Y-%m-%d') AS next_date,
                    timestamp,(extract(hour FROM timestamp)*60+extract(minute FROM timestamp)-571)::INTEGER AS pos,
                    close::DOUBLE AS close,volume::DOUBLE AS volume FROM raw
                WHERE timestamp>=TIMESTAMP '2024-01-01' AND timestamp<TIMESTAMP '2025-07-01'
                    AND strftime(timestamp,'%H:%M') BETWEEN '09:31' AND '10:00'),
                joined AS(SELECT k.date,k.code,k.next_date,k.buy_cash15,s.timestamp,s.pos,s.close,s.volume,
                    k.decision_shares*(s.close-greatest(s.close*.0015,.005)) AS mark_value
                    FROM raw_window s JOIN keys k USING(code,next_date)),
                values AS(SELECT *,
                    (mark_value-greatest(mark_value*.0003,5)-mark_value*.00051)/buy_cash15-1 AS net_return
                    FROM joined),
                flags AS(SELECT *,coalesce(volume>0 AND net_return>0 AND isfinite(net_return),false) AS positive
                    FROM values)
                SELECT date,code,next_date,count(*) AS bars,count(DISTINCT pos) AS labels,
                    min(pos) AS first_pos,max(pos) AS last_pos,
                    count(*) FILTER(WHERE timestamp=date_trunc('minute',timestamp)) AS minute_labels,
                    count(*) FILTER(WHERE volume>0) AS active_minutes,
                    count(*) FILTER(WHERE positive) AS profitable_minutes15,
                    bit_or(CASE WHEN positive THEN 1::BIGINT << pos ELSE 0::BIGINT END) AS profitable_mask15,
                    max(net_return) FILTER(WHERE pos=29 AND volume>0) AS mark_1000_return15_rebuilt
                FROM flags GROUP BY date,code,next_date ORDER BY date,code''').df()
            c.close()
            pd.testing.assert_frame_equal(rows[['date','code','next_date']],sub[['date','code','next_date']].reset_index(drop=True),check_exact=True)
            assert rows.bars.eq(30).all() and rows.labels.eq(30).all() and rows.minute_labels.eq(30).all()
            assert rows.first_pos.eq(0).all() and rows.last_pos.eq(29).all()
            rows.to_parquet(path,index=False,compression='zstd')
            meta=dict(protocol_sha256=sha(PROTOCOL),manifest_sha256=sha(MANIFEST),codes=subset,
                extractor_sha256=extractor,sha256=sha(path),rows=len(rows))
            save_json(receipt,meta)
        parts[str(path)]=meta['sha256']
        print(json.dumps(dict(codes=offset+len(subset),total_codes=len(codes)),ensure_ascii=False),flush=True)
    c=base.conn();c.read_parquet(list(parts)).create_view('parts')
    rows=c.sql('SELECT * FROM parts ORDER BY date,code').df();c.close()
    pd.testing.assert_frame_equal(rows[['date','code','next_date']],keys[['date','code','next_date']],check_exact=True)
    rows['profitable_fraction15']=rows.profitable_minutes15/30
    rows.to_parquet(ROOT/'duration_labels.parquet',index=False,compression='zstd')
    report=dict(protocol_sha256=sha(PROTOCOL),full_label_report_sha256=sha(base.SOURCE/'full_label_report.json'),
        full_label_verification_sha256=sha(base.SOURCE/'full_label_verification.json'),manifest_sha256=sha(MANIFEST),
        extractor_sha256=extractor,parts_sha256=parts,labels_sha256=sha(ROOT/'duration_labels.parquet'),
        rows=len(rows),raw_minutes=int(rows.bars.sum()),signal_start=rows.date.min(),signal_end=rows.date.max(),
        observation_start=rows.next_date.min(),observation_end=rows.next_date.max(),
        original_unknowns_never_filled=True,year_2025_is_exploratory=True,new_2025H2_duration_read=False,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'duration_report.json',report)
    return {k:v for k,v in report.items() if k!='parts_sha256'}


def checked_report():
    r=json.loads((ROOT/'duration_report.json').read_text())
    p=json.loads((ROOT/'duration_verification.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['labels_sha256']==sha(ROOT/'duration_labels.parquet')
    assert p['passed'] and p['duration_report_sha256']==sha(ROOT/'duration_report.json')
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['labels']);a=p.parse_args()
    print(json.dumps(labels(),ensure_ascii=False,indent=2))
