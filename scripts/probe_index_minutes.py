"""Bounded historical index source probe, independent of strategy outcomes."""
import json
from pathlib import Path
import struct

from probe_historical_ticks import connect,exchange,save_json
from trade_research.corporate_cash import sha

ROOT=Path('data/research/index_minute_probe')
PROTOCOL=Path('config/index_minute_probe_protocol.json')


def decode(body):
    if len(body)<2:
        raise ValueError('Missing count')
    count=struct.unpack_from('<H',body)[0]
    if count==0:
        assert len(body) in (2,6)
        return []
    assert count<=242 and len(body)>=6
    cursor=6;price=0;rows=[]
    def integer():
        nonlocal cursor
        first=body[cursor];cursor+=1
        value=first&63;shift=6;byte=first
        while byte&128:
            assert shift<=55
            byte=body[cursor];cursor+=1
            value+=(byte&127)<<shift;shift+=7
        return -value if first&64 else value
    for sequence in range(count):
        price+=integer();reserved=integer();volume=integer()
        rows.append(dict(sequence=sequence,price_raw=price,reserved_raw=reserved,volume_raw=volume))
    assert cursor==len(body)
    return rows


def request(sock,symbol,date,prefix):
    assert date[:4] in ['2024','2025']
    market,code=symbol.split('.')
    packet=bytes.fromhex('0c01300001010d000d00b40f')+struct.pack('<IB6s',int(date.replace('-','')),int(market=='sh'),code.encode())
    return decode(exchange(sock,packet,prefix))


def main():
    if (ROOT/'source_report.json').exists():
        raise ValueError('Do not replace the fixed probe')
    cfg=json.loads(PROTOCOL.read_text());report=dict(protocol_sha256=sha(PROTOCOL),sessions=[],outcomes_read=False,new_2026_prices_read=False)
    with connect(cfg['host'],cfg,ROOT/'wire') as sock:
        for symbol in cfg['symbols']:
            for date in cfg['dates']:
                prefix=ROOT/'wire'/f'{symbol}_{date}'
                item=dict(symbol=symbol,date=date)
                try:
                    rows=request(sock,symbol,date,prefix)
                    path=ROOT/f'{symbol}_{date}.json';save_json(path,rows)
                    item.update(rows=len(rows),path=str(path),sha256=sha(path),status='nonempty_unverified' if rows else 'empty')
                except Exception as exc:
                    item.update(status='error',error=str(exc),error_type=type(exc).__name__)
                report['sessions'].append(item)
                save_json(ROOT/'source_report.json',report)
                print(json.dumps(item,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
