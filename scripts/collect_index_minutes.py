"""Collect the fixed historical index-date grid with one transparent retry."""
import json
from pathlib import Path
import shutil

import pandas as pd

from probe_index_minutes import request
from probe_historical_ticks import connect
from trade_research.corporate_cash import save_json,sha

ROOT=Path('data/research/tail_formula_context')
PROTOCOL=Path('config/tail_formula_context_protocol.json')


def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'source_report.json').exists():
        raise ValueError('Do not replace frozen index sources')
    cfg=json.loads(Path('config/index_minute_probe_protocol.json').read_text())
    proof=json.loads(Path('data/research/index_minute_probe/verification.json').read_text())
    assert proof['passed'] and proof['historical_data_nonempty']
    source=Path('data/research/tail_formula_market/features.parquet')
    days=sorted(pd.read_parquet(source,columns=['date']).date.unique())
    assert days[0]>='2024-01-01' and days[-1]<='2025-12-30'
    report=dict(protocol_sha256=sha(PROTOCOL),source_feature_sha256=sha(source),
        probe_verification_sha256=sha(Path('data/research/index_minute_probe/verification.json')),
        sessions=[],outcomes_read=False,new_2026_prices_read=False)
    sock=None
    try:
        for symbol in ['sh.000001','sz.399001']:
            for date in days:
                path=ROOT/'raw'/f'{symbol}_{date}.json';attempts=[]
                if not path.exists():
                    for attempt in range(2):
                        folder=ROOT/'wire'/f'{symbol}_{date}'/str(attempt)
                        try:
                            if sock is None:
                                sock=connect(cfg['host'],cfg,folder)
                            rows=request(sock,symbol,date,folder/'history')
                            save_json(path,rows)
                            attempts.append(dict(attempt=attempt,status='nonempty' if rows else 'empty',rows=len(rows)))
                            break
                        except Exception as exc:
                            attempts.append(dict(attempt=attempt,status='error',error_type=type(exc).__name__,error=str(exc)))
                            if sock is not None:
                                sock.close();sock=None
                    save_json(ROOT/'attempts'/f'{symbol}_{date}.json',attempts)
                item=dict(symbol=symbol,date=date,status='cached' if path.exists() else 'unavailable')
                if path.exists():
                    item.update(path=str(path),sha256=sha(path),rows=len(json.loads(path.read_text())))
                report['sessions'].append(item)
                if len(report['sessions'])%100==0:
                    print(json.dumps(dict(completed=len(report['sessions']),expected=2*len(days))),flush=True)
        save_json(ROOT/'source_report.json',report)
    finally:
        if sock is not None:
            sock.close()
    print(json.dumps(dict(completed=len(report['sessions']),days=len(days),report_sha256=sha(ROOT/'source_report.json'))),flush=True)


if __name__=='__main__':
    main()
