"""Collect the frozen earlier index grid; retain every response and failure."""
import argparse
import json
from pathlib import Path
import struct
import time

import numpy as np
import pandas as pd

from probe_historical_ticks import connect, exchange
from probe_index_minutes import decode
from verify_index_minute_probe import replay
from trade_research import tail_formula_context as context
from trade_research.corporate_cash import save_json, sha
from trade_research.tail_formula_long48_inputs import ROOT, PROTOCOL, PROBE, policy
from trade_research.turnover_reference import CALENDAR

OUT = ROOT/'indices'


def sessions():
    p = policy(); cal = pd.read_parquet(CALENDAR)
    days = sorted(cal.loc[cal.is_trading_day.eq('1') & cal.calendar_date.between(p['signal_first'],p['signal_last']),'calendar_date'])
    assert len(days)==484
    return p,[(symbol,day) for symbol in p['index_symbols'] for day in days]


def collect():
    p,grid = sessions(); OUT.mkdir(exist_ok=True); (OUT/'raw').mkdir(exist_ok=True)
    if (OUT/'source_report.json').exists():
        raise ValueError('Do not replace frozen index responses')
    probe = json.loads((PROBE/'minute_report.json').read_text())
    reused = {(s['symbol'],s['date']):s for s in probe['sessions'] if s['symbol'] in p['index_symbols']}
    results = []; sock = None
    try:
        for i,(symbol,day) in enumerate(grid,1):
            meta = OUT/'raw'/f'{symbol}_{day}.meta.json'; path = OUT/'raw'/f'{symbol}_{day}.json'
            if meta.exists():
                item = json.loads(meta.read_text()); assert item['protocol_sha256']==sha(PROTOCOL)
                if 'path' in item:
                    assert sha(Path(item['path']))==item['sha256']
            elif (symbol,day) in reused:
                old = reused[(symbol,day)]
                assert sha(Path(old['path']))==old['sha256']
                item = dict(symbol=symbol,date=day,status=old['status'],rows=old['rows'],path=old['path'],sha256=old['sha256'],
                    request=old['request'],response=old['response'],protocol_sha256=sha(PROTOCOL),reused_probe=True)
                save_json(meta,item)
            else:
                item = dict(symbol=symbol,date=day,protocol_sha256=sha(PROTOCOL),attempts=[],reused_probe=False)
                for attempt in range(p['attempts']):
                    folder = OUT/'wire'/f'{symbol}_{day}'/str(attempt)
                    try:
                        if sock is None:
                            sock = connect(p['host'],p,folder)
                        packet = bytes.fromhex('0c01300001010d000d00b40f')+struct.pack('<IB6s',int(day.replace('-','')),int(symbol.startswith('sh.')),symbol[3:].encode())
                        rows = decode(exchange(sock,packet,folder/'history'))
                        item.update(rows=len(rows),status='nonempty' if rows else 'empty',
                            request=str(folder/'history.request.bin'),response=str(folder/'history.response.bin'))
                        item['attempts'].append(dict(attempt=attempt,status=item['status'])); break
                    except Exception as exc:
                        item['attempts'].append(dict(attempt=attempt,status='error',error_type=type(exc).__name__,error=str(exc)))
                        if sock is not None:
                            sock.close(); sock = None
                if 'status' in item:
                    save_json(path,rows); item.update(path=str(path),sha256=sha(path))
                else:
                    item['status'] = 'unavailable'
                save_json(meta,item); time.sleep(.08)
            results.append(item)
            if i%40==0:
                print(json.dumps(dict(sessions=i,total=len(grid),nonempty=sum(x['status']=='nonempty' for x in results))),flush=True)
    finally:
        if sock is not None:
            sock.close()
    r = dict(protocol_sha256=sha(PROTOCOL),probe_verification_sha256=sha(PROBE/'verification.json'),
        sessions=results,new_2026_prices_read=False,stock_outcomes_read=False)
    save_json(OUT/'source_report.json',r)
    return dict(sessions=len(results),reused=sum(x['reused_probe'] for x in results))


def verify():
    p,grid = sessions(); r = json.loads((OUT/'source_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and [(x['symbol'],x['date']) for x in r['sessions']]==grid
    daily = pd.read_parquet(PROBE/'index_daily.parquet').set_index(['code','date'])
    points = []; audits = []; sources = {}
    for item in r['sessions']:
        rows = []; symbol,day = item['symbol'],item['date']
        if 'path' in item:
            req,resp = Path(item['request']),Path(item['response']); raw = Path(item['path'])
            assert sha(raw)==item['sha256']
            packet = req.read_bytes(); assert packet[:12]==bytes.fromhex('0c01300001010d000d00b40f')
            assert struct.unpack('<IB6s',packet[12:])==(int(day.replace('-','')),int(symbol.startswith('sh.')),symbol[3:].encode())
            rows = replay(resp.read_bytes()); assert rows==json.loads(raw.read_text()) and len(rows)==item['rows']
            for path in [req,resp,raw]:
                sources[str(path)] = sha(path)
        values = np.array([x['price_raw']/100 for x in rows]); d = daily.loc[(symbol,day)]
        complete = (len(rows)==240 and [x['sequence'] for x in rows]==list(range(240)) and (values>0).all())
        good = complete and (values>=d.low-.0051).all() and (values<=d.high+.0051).all() and abs(values[-1]-d.close)<=.0051
        audits.append(dict(date=day,index_code=symbol,rows=len(rows),full_day_source_valid=bool(good)))
        a = dict(date=day,index_code=symbol,**context.points(rows))
        # Independently check only the visible prefix used by the model.
        prefix = rows[:228]
        valid = len(prefix)==228 and [x['sequence'] for x in prefix]==list(range(228)) and all(x['price_raw']>0 for x in prefix)
        assert a['prefix_valid']==valid
        for key,position in [('ip48',227),('ip20',199),('ip35',214),('ip01',120)]:
            expected = prefix[position]['price_raw']/100 if valid else np.nan
            np.testing.assert_allclose(a[key],expected,equal_nan=True,rtol=0,atol=0)
        points.append(a)
    f = pd.DataFrame(points).sort_values(['date','index_code']).reset_index(drop=True)
    audit = pd.DataFrame(audits).sort_values(['date','index_code']).reset_index(drop=True)
    f.to_parquet(OUT/'points.parquet',index=False,compression='zstd'); audit.to_parquet(OUT/'audit.parquet',index=False,compression='zstd')
    proof = dict(passed=True,protocol_sha256=sha(PROTOCOL),source_report_sha256=sha(OUT/'source_report.json'),
        index_daily_sha256=sha(PROBE/'index_daily.parquet'),sources_sha256=sources,
        output_sha256={name:sha(OUT/name) for name in ['points.parquet','audit.parquet']},
        sessions=len(grid),valid_prefixes=int(f.prefix_valid.sum()),full_day_conflicts=int((~audit.full_day_source_valid).sum()),
        native_INDEXC_parity_verified=False,new_2026_prices_read=False,stock_outcomes_read=False)
    save_json(OUT/'verification.json',proof)
    return {k:v for k,v in proof.items() if k!='sources_sha256'}


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['collect','verify'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
