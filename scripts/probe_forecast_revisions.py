"""Freeze an issuer-revision catalog and input-only original-PDF pilot."""
import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
import re
import subprocess
from urllib.parse import urlparse

import pandas as pd
import pdfplumber

import collect_buyback_index as cninfo
from trade_research.corporate_cash import save_json, sha

ROOT=Path('data/research/forecast_revision_probe')
PROTOCOL=Path('config/forecast_revision_source_protocol.json')
KEYWORDS=['业绩预告修正','业绩预告更正']
EXCLUDE=re.compile(r'回复|致歉|监管|风险提示|关注函|问询')


def catalog():
    if (ROOT/'catalog_report.json').exists():
        raise ValueError('Do not replace frozen forecast-revision catalog')
    ROOT.mkdir(parents=True,exist_ok=True)
    cninfo.REQUEST_DELAY_SEC=.25
    frames=[]
    searches=[]
    for year in [2024,2025]:
        for keyword in KEYWORDS:
            cache=ROOT/'index'/str(year)
            records=cninfo._range_rows(date(year,1,1),date(year,12,31),cache,searchkey=keyword)
            if not records:
                searches.append(dict(year=year,keyword=keyword,raw_rows=0,a_share_rows=0))
                continue
            frame=cninfo._candidates(records)
            assert frame.notice_date.str.startswith(str(year)).all()
            frame['search_keyword']=keyword
            frames.append(frame)
            searches.append(dict(year=year,keyword=keyword,raw_rows=len(records),a_share_rows=len(frame)))
            print(json.dumps(searches[-1],ensure_ascii=False),flush=True)
    combined=pd.concat(frames,ignore_index=True)
    combined.to_parquet(ROOT/'catalog_search_rows.parquet',index=False,compression='zstd')
    identity=['code','notice_date','title','pdf_url','announcement_time_ms']
    unique=combined[identity].drop_duplicates().sort_values(['notice_date','code','pdf_url']).reset_index(drop=True)
    assert not unique.pdf_url.duplicated().any()
    unique['title_candidate']=unique.title.str.contains(r'业绩预告.{0,4}(?:修正|更正)',regex=True)&~unique.title.str.contains(EXCLUDE)
    unique['half']=unique.notice_date.str[:4]+unique.notice_date.str[5:7].map(lambda m:'H1' if m<='06' else 'H2')
    unique['selection_hash']=[hashlib.sha256(('forecast-revision|'+url).encode()).hexdigest() for url in unique.pdf_url]
    unique.to_parquet(ROOT/'catalog.parquet',index=False,compression='zstd')
    sample=unique.loc[unique.title_candidate].sort_values(['half','selection_hash']).groupby('half').head(2).reset_index(drop=True)
    sample.to_parquet(ROOT/'pilot.parquet',index=False,compression='zstd')
    result=dict(protocol_sha256=sha(PROTOCOL),searches=searches,unique_documents=len(unique),
        title_candidates=int(unique.title_candidate.sum()),pilot_documents=len(sample),
        candidate_halves=unique.loc[unique.title_candidate].half.value_counts().sort_index().to_dict(),
        raw_sha256={str(p):sha(p) for p in sorted((ROOT/'index').rglob('*.json'))},
        output_sha256={n:sha(ROOT/n) for n in ['catalog_search_rows.parquet','catalog.parquet','pilot.parquet']},
        coverage='Complete pages of the two specified title searches, not proof of all issuer revisions',
        outcomes_read=False,new_2026_prices_read=False)
    save_json(ROOT/'catalog_report.json',result)
    return {k:v for k,v in result.items() if k not in ['raw_sha256','output_sha256']}


def pilot():
    if (ROOT/'pilot_report.json').exists():
        raise ValueError('Do not replace inspected forecast original pilot')
    report=json.loads((ROOT/'catalog_report.json').read_text())
    assert report['protocol_sha256']==sha(PROTOCOL)
    for name,h in report['output_sha256'].items():
        assert sha(ROOT/name)==h
    for name,h in report['raw_sha256'].items():
        assert sha(Path(name))==h
    sample=pd.read_parquet(ROOT/'pilot.parquet')
    pdfs=ROOT/'pdfs'
    pdfs.mkdir(exist_ok=True)
    extracted=[]
    for r in sample.itertuples():
        parsed=urlparse(r.pdf_url)
        assert parsed.scheme=='https' and parsed.netloc=='static.cninfo.com.cn'
        file=pdfs/Path(parsed.path).name
        receipt=file.with_suffix('.receipt.json')
        if not receipt.exists():
            if file.exists():
                raise ValueError('Unreceipted source PDF must be inspected before retry')
            result=subprocess.run(['curl','-fLsS','--max-time','30','--retry','2',r.pdf_url,'-o',str(file)],capture_output=True,text=True)
            assert result.returncode==0,(r.pdf_url,result.stderr)
            assert file.read_bytes()[:4]==b'%PDF'
            save_json(receipt,dict(url=r.pdf_url,sha256=sha(file),download_succeeded=True))
        source=json.loads(receipt.read_text())
        assert source['url']==r.pdf_url and source['sha256']==sha(file)
        with pdfplumber.open(file) as doc:
            pages=[dict(page=i+1,text=p.extract_text() or '',tables=p.extract_tables()) for i,p in enumerate(doc.pages)]
        texts='\n'.join(p['text'] for p in pages)
        row=dict(r._asdict(),pdf_path=str(file),pdf_sha256=sha(file),pages=pages,
            code_in_text=r.code.split('.')[1] in texts,
            text_sha256=hashlib.sha256(texts.encode()).hexdigest())
        extracted.append(row)
        save_json(file.with_suffix('.text.json'),row)
        print(json.dumps(dict(code=r.code,date=r.notice_date,pages=len(pages),pdf=str(file)),ensure_ascii=False),flush=True)
    result=dict(catalog_report_sha256=sha(ROOT/'catalog_report.json'),documents=len(extracted),
        all_codes_in_text=all(x['code_in_text'] for x in extracted),
        output_sha256={str(p):sha(p) for p in sorted(pdfs.iterdir())},
        financial_ranges_not_yet_interpreted=True,outcomes_read=False,new_2026_prices_read=False)
    save_json(ROOT/'pilot_report.json',result)
    return {k:v for k,v in result.items() if k!='output_sha256'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['catalog','pilot'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
