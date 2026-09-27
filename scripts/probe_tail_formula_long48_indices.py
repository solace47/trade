"""Bounded earlier-input availability probe; no stock outcome labels."""
import argparse
import json
import socket
import struct
from pathlib import Path

import baostock as bs
import numpy as np
import pandas as pd

from probe_historical_ticks import connect, exchange
from probe_index_minutes import decode
from verify_index_minute_probe import replay
from trade_research.corporate_cash import MINUTES, save_json, sha
from trade_research.ingest import _login, _rows
from trade_research.turnover_reference import CALENDAR

ROOT = Path('data/research/tail_formula_long48_probe')
PROTOCOL = Path('config/tail_formula_long48_probe_protocol.json')


def policy():
    p = json.loads(PROTOCOL.read_text())
    assert p['calendar_sha256'] == sha(CALENDAR)
    assert p['dates'] == ['2022-01-04','2022-07-01','2023-01-03','2023-07-03']
    assert p['daily_first'] == '2021-06-01' and p['daily_last'] == '2023-12-29'
    assert p['index_symbols'] == ['sh.000001','sz.399001'] and p['stock_control'] == 'sh.600000'
    assert p['attempts'] == 2 and not p['new_2026_prices_allowed']
    ROOT.mkdir(parents=True,exist_ok=True)
    return p


def daily():
    p = policy()
    if (ROOT/'daily_report.json').exists():
        raise ValueError('Do not replace the earlier daily source probe')
    socket.setdefaulttimeout(20)
    report = dict(protocol_sha256=sha(PROTOCOL),files=[],new_2026_prices_read=False,stock_outcomes_read=False)
    try:
        _login()
        for code in p['index_symbols']:
            path = ROOT/(code+'_daily.json')
            item = dict(code=code)
            try:
                fields = 'date,code,open,high,low,close'
                data = _rows(bs.query_history_k_data_plus(code,fields,start_date=p['daily_first'],
                    end_date=p['daily_last'],frequency='d',adjustflag='3'))
                save_json(path,dict(code=code,first=p['daily_first'],last=p['daily_last'],records=data.to_dict('records')))
                item.update(path=str(path),sha256=sha(path),rows=len(data),status='received')
            except Exception as exc:
                item.update(status='error',error_type=type(exc).__name__,error=str(exc))
            report['files'].append(item); print(json.dumps(item,ensure_ascii=False),flush=True)
    except Exception as exc:
        report['connection_error'] = dict(error_type=type(exc).__name__,error=str(exc))
    finally:
        bs.logout()
    save_json(ROOT/'daily_report.json',report); return report


def minutes():
    p = policy()
    if (ROOT/'minute_report.json').exists():
        raise ValueError('Do not replace the earlier intraday source probe')
    report = dict(protocol_sha256=sha(PROTOCOL),sessions=[],new_2026_prices_read=False,stock_outcomes_read=False)
    (ROOT/'raw').mkdir(exist_ok=True)
    sock = None
    try:
        for symbol in p['index_symbols']+[p['stock_control']]:
            for date in p['dates']:
                item = dict(symbol=symbol,date=date,attempts=[])
                path = ROOT/'raw'/f'{symbol}_{date}.json'
                for attempt in range(p['attempts']):
                    folder = ROOT/'wire'/f'{symbol}_{date}'/str(attempt)
                    try:
                        if sock is None:
                            sock = connect(p['host'],p,folder)
                        packet = bytes.fromhex('0c01300001010d000d00b40f')+struct.pack('<IB6s',
                            int(date.replace('-','')),int(symbol.startswith('sh.')),symbol[3:].encode())
                        rows = decode(exchange(sock,packet,folder/'history'))
                        item.update(rows=len(rows),status='nonempty' if rows else 'empty',
                            request=str(folder/'history.request.bin'),response=str(folder/'history.response.bin'))
                        item['attempts'].append(dict(attempt=attempt,status=item['status'])); break
                    except Exception as exc:
                        item['attempts'].append(dict(attempt=attempt,status='error',error_type=type(exc).__name__,error=str(exc)))
                        if sock is not None:
                            sock.close(); sock = None
                if 'status' not in item:
                    item['status'] = 'unavailable'
                else:
                    # Local output failures must not cause a second network request.
                    save_json(path,rows)
                    item.update(path=str(path),sha256=sha(path))
                report['sessions'].append(item)
                save_json(ROOT/'minute_progress.json',report)
                print(json.dumps(item,ensure_ascii=False),flush=True)
    finally:
        if sock is not None:
            sock.close()
    save_json(ROOT/'minute_report.json',report); return report


def recover():
    """Recover already received replies after the first local raw-directory error."""
    p = policy(); report = json.loads((ROOT/'minute_report.json').read_text())
    assert report['protocol_sha256'] == sha(PROTOCOL)
    archived = ROOT/'superseded_local_write_error'
    assert not archived.exists()
    recovered = []; wire_hashes = {}
    (ROOT/'raw').mkdir(exist_ok=True)
    for item in report['sessions']:
        assert item['status'] == 'unavailable'
        assert len(item['attempts']) == 2 and all(x['error_type']=='FileNotFoundError' for x in item['attempts'])
        versions = []
        for i in range(2):
            folder = ROOT/'wire'/f"{item['symbol']}_{item['date']}"/str(i)
            req,resp = folder/'history.request.bin',folder/'history.response.bin'
            packet = req.read_bytes()
            assert packet[:12] == bytes.fromhex('0c01300001010d000d00b40f')
            assert struct.unpack('<IB6s',packet[12:]) == (int(item['date'].replace('-','')),int(item['symbol'].startswith('sh.')),item['symbol'][3:].encode())
            versions.append(replay(resp.read_bytes()))
            wire_hashes[str(resp)] = sha(resp); wire_hashes[str(req)] = sha(req)
        assert versions[0] == versions[1], 'Do not silently choose conflicting archived responses'
        path = ROOT/'raw'/f"{item['symbol']}_{item['date']}.json"
        save_json(path,versions[0])
        first = ROOT/'wire'/f"{item['symbol']}_{item['date']}"/'0'
        recovered.append(dict(item,path=str(path),sha256=sha(path),rows=len(versions[0]),
            status='nonempty' if versions[0] else 'empty',request=str(first/'history.request.bin'),
            response=str(first/'history.response.bin'),recovered_from_archived_wire=True,both_attempts_identical=True))
    archived.mkdir()
    archived_hashes = {}
    for name in ['minute_report.json','minute_progress.json','verification.json']:
        file = ROOT/name
        if file.exists():
            archived_hashes[name] = sha(file); file.rename(archived/name)
    result = dict(protocol_sha256=sha(PROTOCOL),sessions=recovered,local_output_error_recovered=True,
        archived_reports_sha256=archived_hashes,all_attempts_wire_sha256=wire_hashes,
        additional_network_requests=0,new_2026_prices_read=False,stock_outcomes_read=False)
    save_json(ROOT/'minute_report.json',result)
    return {k:v for k,v in result.items() if k not in ['sessions','all_attempts_wire_sha256']}


def verify():
    p = policy(); dr = json.loads((ROOT/'daily_report.json').read_text()); mr = json.loads((ROOT/'minute_report.json').read_text())
    assert dr['protocol_sha256'] == mr['protocol_sha256'] == sha(PROTOCOL)
    cal = pd.read_parquet(CALENDAR)
    expected_days = cal.loc[cal.is_trading_day.eq('1')&cal.calendar_date.between(p['daily_first'],p['daily_last']),'calendar_date'].sort_values().tolist()
    daily_rows = []; daily_sources = {}; daily_checks = []
    for item in dr['files']:
        if item['status'] != 'received':
            daily_checks.append(dict(code=item['code'],valid=False)); continue
        path = Path(item['path']); assert sha(path) == item['sha256']
        d = json.loads(path.read_text()); f = pd.DataFrame(d['records'])
        assert d['first'] == p['daily_first'] and d['last'] == p['daily_last']
        for col in ['open','high','low','close']:
            f[col] = pd.to_numeric(f[col],errors='raise')
        prices = f[['open','high','low','close']]
        good = (f.date.tolist()==expected_days and f.code.eq(item['code']).all() and np.isfinite(prices).all().all()
            and prices.gt(0).all().all() and f.high.add(.0001).ge(prices.max(axis=1)).all()
            and f.low.sub(.0001).le(prices.min(axis=1)).all())
        daily_checks.append(dict(code=item['code'],rows=len(f),valid=bool(good)))
        if good:
            daily_rows.append(f); daily_sources[str(path)] = sha(path)
    daily = pd.concat(daily_rows,ignore_index=True).set_index(['code','date']) if daily_rows else pd.DataFrame()
    expected = {(s,d) for s in p['index_symbols']+[p['stock_control']] for d in p['dates']}
    assert {(s['symbol'],s['date']) for s in mr['sessions']} == expected and len(mr['sessions']) == len(expected)
    checks = []; sources = {}
    for item in mr['sessions']:
        symbol,date = item['symbol'],item['date']; result = dict(symbol=symbol,date=date,status=item['status'],valid=False)
        if item['status'] not in ['nonempty','empty']:
            checks.append(result); continue
        path = Path(item['path']); assert sha(path) == item['sha256']
        req = Path(item['request']); resp = Path(item['response'])
        packet = req.read_bytes(); assert packet[:12] == bytes.fromhex('0c01300001010d000d00b40f')
        assert struct.unpack('<IB6s',packet[12:]) == (int(date.replace('-','')),int(symbol.startswith('sh.')),symbol[3:].encode())
        rows = replay(resp.read_bytes()); assert rows == json.loads(path.read_text()) and len(rows) == item['rows']
        for file in [path,req,resp]:
            sources[str(file)] = sha(file)
        values = np.array([x['price_raw'] for x in rows],dtype=float)/100
        good = len(values)==240 and np.isfinite(values).all() and (values>0).all()
        if good and symbol in p['index_symbols'] and not daily.empty and (symbol,date) in daily.index:
            d = daily.loc[(symbol,date)]
            result.update(rows=len(values),last_close_difference=float(values[-1]-d.close),
                within_daily_envelope=bool(((values>=d.low-.0051)&(values<=d.high+.0051)).all()))
            result['valid'] = bool(result['within_daily_envelope'] and abs(values[-1]-d.close)<=.0051)
        elif good and symbol == p['stock_control']:
            file = MINUTES/symbol[:2].upper()/(symbol[3:]+'.parquet')
            f = pd.read_parquet(file,columns=['timestamp','high','low','close'],
                filters=[('timestamp','>=',pd.Timestamp(date)),('timestamp','<',pd.Timestamp(date)+pd.Timedelta(days=1))]).sort_values('timestamp')
            f = f.loc[f.timestamp.dt.strftime('%H:%M').ne('09:30')]
            clocks = list(range(571,691))+list(range(781,901))
            aligned = (f.timestamp.dt.hour*60+f.timestamp.dt.minute).tolist() == clocks
            envelope = aligned and ((values>=f.low.to_numpy()-1e-6)&(values<=f.high.to_numpy()+1e-6)).all()
            result.update(rows=len(values),clocks_aligned=bool(aligned),within_minute_envelope=bool(envelope),valid=bool(envelope))
            sources[str(file)] = sha(file)
        checks.append(result)
    complete = len(daily_checks)==2 and all(x['valid'] for x in daily_checks) and all(x['valid'] for x in checks)
    result = dict(passed=True,source_available_and_consistent=bool(complete),protocol_sha256=sha(PROTOCOL),
        daily_report_sha256=sha(ROOT/'daily_report.json'),minute_report_sha256=sha(ROOT/'minute_report.json'),
        daily_checks=daily_checks,session_checks=checks,sources_sha256={**daily_sources,**sources},
        independent_wire_replay_complete=all(x['status'] in ['nonempty','empty'] for x in mr['sessions']),
        native_INDEXC_parity_verified=False,
        stock_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    if daily_rows:
        pd.concat(daily_rows,ignore_index=True).sort_values(['code','date']).to_parquet(ROOT/'index_daily.parquet',index=False,compression='zstd')
        result['index_daily_sha256'] = sha(ROOT/'index_daily.parquet')
    save_json(ROOT/'verification.json',result)
    return {k:v for k,v in result.items() if k!='sources_sha256'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['daily','minutes','recover','verify'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
