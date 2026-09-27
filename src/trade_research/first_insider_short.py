"""Complete disclosed first insider purchases, with fixed short exits."""
import argparse
import bisect
import hashlib
import json
from pathlib import Path
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pdfplumber

from .corporate_cash import save_json, sha
from .first_insider_buy_source import confirm_original, strict_first_title
from .touch_sequence import BASE, BALANCE, prepare_exits
from .touch_sequence_analysis import analyze as evaluate
from .turnover_reference import CALENDAR

ROOT = Path('data/research/first_insider_short')
SOURCE = Path('data/research/first_insider_buy')
PROTOCOL = Path('config/first_insider_short_protocol.json')
GROUPS = ['direct_actor', 'related_actor']
RELATED = re.compile(r'一致行动人|控制的企业|间接控股股东')


def sources():
    if (ROOT/'source_report.json').exists():
        raise ValueError('Do not replace frozen original evidence')
    ROOT.mkdir(parents=True,exist_ok=True)
    hashes = {str(PROTOCOL):sha(PROTOCOL)}
    records = []
    for year in [2024,2025]:
        index_path = SOURCE/f'title_candidates_{year}.parquet'
        audit_path = SOURCE/f'pdf_audit_{year}.jsonl'
        for path in [index_path,audit_path]:
            hashes[str(path)] = sha(path)
        index = pd.read_parquet(index_path)
        audit = pd.read_json(audit_path,lines=True)
        assert len(index)==len(audit) and not index.pdf_url.duplicated().any()
        audit = audit.set_index('pdf_url')
        for r in index.sort_values(['notice_date','code']).itertuples(index=False):
            old = audit.loc[r.pdf_url]
            assert all(getattr(r,n)==old[n] for n in ['code','notice_date','title','announcement_time_ms'])
            assert strict_first_title(r.title)
            file = SOURCE/'pdfs'/str(year)/Path(urlparse(r.pdf_url).path).name
            assert file.read_bytes()[:4]==b'%PDF'
            hashes[str(file)] = sha(file)
            with pdfplumber.open(file) as doc:
                text = '\n'.join(page.extract_text() or '' for page in doc.pages[:3])
                pages = len(doc.pages)
            compact = re.sub(r'\s+','',text)
            assert compact==re.sub(r'\s+','',old.text_first_three_pages) and pages==old.pages
            current = confirm_original(text,r.code,r.notice_date)
            assert all(current.get(n)==old[n] for n in ['status','buy_date','shares','trade_evidence'])
            assert current['status']=='ok' and current['shares']>0
            assert current['buy_date']<=r.notice_date and current['trade_evidence'] in compact
            stamp = pd.to_datetime(r.announcement_time_ms,unit='ms',utc=True).tz_convert('Asia/Shanghai')
            assert stamp.strftime('%Y-%m-%d')==r.notice_date
            records.append(dict(r._asdict(),**current,pdf_path=str(file),pdf_sha256=hashes[str(file)],
                normalized_text_sha256=hashlib.sha256(compact.encode()).hexdigest(),
                related_actor_title=bool(RELATED.search(r.title)),pages=pages))
    table = pd.DataFrame(records).sort_values(['notice_date','code','pdf_url']).reset_index(drop=True)
    assert not table.pdf_url.duplicated().any()
    table.to_parquet(ROOT/'source_events.parquet',index=False,compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL),source_sha256=hashes,rows=len(table),
        original_texts_reproduced=len(table),source_events_sha256=sha(ROOT/'source_events.parquet'),
        outcomes_read=False,new_2026_prices_read=False,precise_first_publication_verified=False)
    save_json(ROOT/'source_report.json',report)
    return {k:v for k,v in report.items() if k!='source_sha256'}


def inputs():
    if (ROOT/'input_report.json').exists():
        raise ValueError('Do not replace frozen event orders')
    source = json.loads((ROOT/'source_report.json').read_text())
    assert source['protocol_sha256']==sha(PROTOCOL)
    assert source['source_events_sha256']==sha(ROOT/'source_events.parquet')
    for path,h in source['source_sha256'].items():
        assert sha(Path(path))==h
    assert sha(BASE)==json.loads(BASE.with_name('base_report.json').read_text())['base_sha256']
    schedule = pd.read_parquet(CALENDAR)
    days = sorted(schedule.loc[schedule.is_trading_day.eq('1') & schedule.calendar_date.between('2024-01-01','2025-12-31'),'calendar_date'])
    events = pd.read_parquet(ROOT/'source_events.parquet')
    def next_day(day):
        i=bisect.bisect_right(days,day)
        return days[i] if i<len(days) else None
    events['date'] = events.notice_date.map(next_day)
    events['in_signal_range'] = events.date.between('2024-01-01','2025-12-30').fillna(False)
    base = pd.read_parquet(BASE)
    events = events.merge(base[['date','code','necessary_tradeable']],on=['date','code'],how='left',validate='many_to_one')
    events['base_available'] = events.necessary_tradeable.notna()
    events['eligible'] = events.in_signal_range & events.necessary_tradeable.fillna(False)
    events.to_parquet(ROOT/'all_events.parquet',index=False,compression='zstd')
    selected = events.loc[events.eligible].groupby(['date','code']).agg(
        notice_date=('notice_date','max'),buy_date=('buy_date','max'),
        related_actor_title=('related_actor_title','any'),source_documents=('pdf_url','size')).reset_index()
    assert selected.date.gt(selected.notice_date).all()
    event_keys = pd.MultiIndex.from_frame(events.loc[events.in_signal_range,['date','code']].drop_duplicates())
    base['event'] = pd.MultiIndex.from_frame(base[['date','code']]).isin(event_keys)
    necessary = base.loc[base.necessary_tradeable].copy()
    treated = selected.merge(necessary,on=['date','code'],validate='one_to_one')
    assert treated.event.all()
    controls = {d:p.sort_values('code') for d,p in necessary.loc[~necessary.event].groupby('date')}
    pairs = []
    for r in treated.itertuples():
        choices = controls[r.date]
        choices = choices.loc[choices.board.eq(r.board) & choices.code.str[:2].eq(r.code[:2])
            & (choices.return_1449-r.return_1449).abs().le(.01)
            & (choices.return20_prior_adjusted-r.return20_prior_adjusted).abs().le(.05)
            & (choices.price_1449/r.price_1449).between(.5,2)
            & (choices.amount_1449/r.amount_1449).between(.5,2)]
        item = dict(date=r.date,code=r.code,half=r.half,control_code=None,distance=np.nan,
                    **{k+'_difference':np.nan for k in BALANCE})
        if not choices.empty:
            dist = ((choices.return_1449-r.return_1449).abs()/.01
                +(choices.return20_prior_adjusted-r.return20_prior_adjusted).abs()/.05
                +np.abs(np.log2(choices.price_1449/r.price_1449))
                +np.abs(np.log2(choices.amount_1449/r.amount_1449)))
            peer = choices.loc[dist.idxmin()]
            item.update(control_code=peer.code,distance=float(dist.min()),
                **{k+'_difference':float(getattr(r,k)-peer[k]) for k in BALANCE})
        pairs.append(item)
    pairs = pd.DataFrame(pairs).sort_values(['date','code']).reset_index(drop=True)
    control_keys = pairs.loc[pairs.control_code.notna(),['date','control_code']].rename(columns={'control_code':'code'}).drop_duplicates()
    reference = control_keys.merge(necessary,on=['date','code'],validate='one_to_one')
    treated['group'] = np.where(treated.related_actor_title,'related_actor','direct_actor')
    reference['group'] = 'non_event_control'
    frame = pd.concat([treated,reference],ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    frame['primary'] = frame.event
    frame['source_valid'] = True
    assert not frame.duplicated(['date','code']).any() and len(pairs)==frame.primary.sum()
    frame.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    pairs.to_parquet(ROOT/'primary_pairs.parquet',index=False,compression='zstd')
    report = dict(protocol_sha256=sha(PROTOCOL),source_report_sha256=sha(ROOT/'source_report.json'),
        base_sha256=sha(BASE),calendar_sha256=sha(CALENDAR),original_documents=len(events),
        base_available=int(events.base_available.sum()),eligible_documents=int(events.eligible.sum()),
        primary=int(frame.primary.sum()),paired=int(pairs.control_code.notna().sum()),rows=len(frame),
        counts=frame.groupby(['half','board','group']).size().rename('rows').reset_index().to_dict('records'),
        balance={k:pairs[k+'_difference'].describe().to_dict() for k in BALANCE},
        output_sha256={n:sha(ROOT/n) for n in ['all_events.parquet','features.parquet','primary_pairs.parquet']},
        outcomes_read=False,new_2026_prices_read=False)
    save_json(ROOT/'input_report.json',report)
    return report


def exits():
    return prepare_exits(ROOT=ROOT,PROTOCOL=PROTOCOL)


def analyze():
    return evaluate(ROOT=ROOT,GROUPS=GROUPS,event_column='event',
        interpretation='Exploratory disclosed actual purchases followed after notice date; verified actor evidence is not a claim of continuing buying or a profitable portfolio')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['sources','inputs','exits','analyze'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
