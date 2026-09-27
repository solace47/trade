"""Independent wire replay and historical day/stock-minute reconciliation."""
import json
from pathlib import Path
import struct
import zlib

import numpy as np
import pandas as pd

from trade_research.corporate_cash import sha,save_json

ROOT=Path('data/research/index_minute_probe')
PROTOCOL=Path('config/index_minute_probe_protocol.json')


def replay(raw):
    packed,expanded=struct.unpack_from('<HH',raw,12)
    assert len(raw)==16+packed
    payload=raw[16:] if packed==expanded else zlib.decompress(raw[16:])
    assert len(payload)==expanded
    count=int.from_bytes(payload[:2],'little')
    if not count:
        assert len(payload) in (2,6)
        return []
    pos=6;total=0;result=[]
    for sequence in range(count):
        values=[]
        for _ in range(3):
            encoded=[payload[pos]];pos+=1
            while encoded[-1]>=128:
                encoded.append(payload[pos]);pos+=1
                assert len(encoded)<=9
            magnitude=encoded[0]%64+sum((n%128)*2**(6+7*i) for i,n in enumerate(encoded[1:]))
            values.append(-magnitude if encoded[0]&64 else magnitude)
        total+=values[0]
        result.append(dict(sequence=sequence,price_raw=total,reserved_raw=values[1],volume_raw=values[2]))
    assert pos==len(payload)
    return result


def main():
    r=json.loads((ROOT/'source_report.json').read_text());cfg=json.loads(PROTOCOL.read_text())
    assert r['protocol_sha256']==sha(PROTOCOL)
    assert {(s['symbol'],s['date']) for s in r['sessions']}=={(s,d) for s in cfg['symbols'] for d in cfg['dates']}
    indices=pd.read_parquet('data/research/tail_formula_market/indices.parquet').set_index(['code','date'])
    results=[];wires={}
    for item in r['sessions']:
        code,date=item['symbol'],item['date'];prefix=ROOT/'wire'/f'{code}_{date}'
        request=prefix.with_name(prefix.name+'.request.bin').read_bytes()
        assert request[:12]==bytes.fromhex('0c01300001010d000d00b40f')
        day,market,symbol=struct.unpack('<IB6s',request[12:])
        assert (day,market,symbol.decode())==(int(date.replace('-','')),int(code.startswith('sh.')),code[3:])
        response=prefix.with_name(prefix.name+'.response.bin');decoded=replay(response.read_bytes())
        path=Path(item['path']);assert sha(path)==item['sha256'] and decoded==json.loads(path.read_text())
        assert len(decoded)==item['rows']==240
        values=np.array([q['price_raw'] for q in decoded])/100
        assert np.isfinite(values).all() and (values>0).all()
        check=dict(code=code,date=date,rows=len(values))
        if code=='sh.600000':
            minute_path=Path('data/hf/pilot/data/stock_1m/SH/600000.parquet')
            f=pd.read_parquet(minute_path,filters=[('timestamp','>=',pd.Timestamp(date)),('timestamp','<',pd.Timestamp(date)+pd.Timedelta(days=1))]).sort_values('timestamp')
            f=f.loc[f.timestamp.dt.strftime('%H:%M').ne('09:30')]
            clocks=list(range(571,691))+list(range(781,901))
            assert (f.timestamp.dt.hour*60+f.timestamp.dt.minute).tolist()==clocks
            assert (values>=f.low.to_numpy()-1e-6).all() and (values<=f.high.to_numpy()+1e-6).all()
            check.update(exact_minute_close_matches=int((abs(values-f.close.to_numpy())<1e-6).sum()),
                max_close_difference=float(abs(values-f.close.to_numpy()).max()),all_within_same_minute_envelope=True)
        else:
            d=indices.loc[(code,date)]
            assert (values>=d.low-.0051).all() and (values<=d.high+.0051).all()
            assert abs(values[-1]-d.close)<=.0051
            check.update(close_difference=float(values[-1]-d.close),all_within_daily_envelope=True)
        results.append(check);wires[str(response)]=sha(response)
    proof=dict(passed=True,source_report_sha256=sha(ROOT/'source_report.json'),wire_sha256=wires,details=results,
        historical_data_nonempty=True,clock_grid_inferred_from_order_and_stock_control=True,
        all_stock_control_points_inside_same_minute=True,exact_minute_close_semantics_verified=False,
        native_INDEXC_parity_verified=False,conservative_research_cutoff='14:48',
        limitation='分时代表价与另一分钟源收盘不完全一致；09:31至15:00的240点位置在股票对照内成立，仍非交易所时间戳或原生INDEXC逐点一致证明。后续只用至14:48，保留源口径局限。全日包络仅为事后来源核验，不作为交易过滤。',
        outcomes_read=False,new_2026_prices_read=False)
    save_json(ROOT/'verification.json',proof)
    print(json.dumps(proof,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
