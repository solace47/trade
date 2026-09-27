"""Independent calendar joins, actual-share text evidence and nearest controls."""
from decimal import Decimal
import json
from pathlib import Path
import re

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha
from trade_research.turnover_reference import CALENDAR

ROOT = Path('data/research/first_insider_short')
BASE = Path('data/research/next_day_winner/visible_base.parquet')


def check():
    inputs = json.loads((ROOT/'input_report.json').read_text())
    source = json.loads((ROOT/'source_report.json').read_text())
    assert inputs['source_report_sha256']==sha(ROOT/'source_report.json')
    assert inputs['base_sha256']==sha(BASE) and inputs['calendar_sha256']==sha(CALENDAR)
    assert inputs['protocol_sha256']==sha(Path('config/first_insider_short_protocol.json'))
    assert source['source_events_sha256']==sha(ROOT/'source_events.parquet')
    for path,h in source['source_sha256'].items():
        assert sha(Path(path))==h
    for name,h in inputs['output_sha256'].items():
        assert sha(ROOT/name)==h
    records = pd.read_parquet(ROOT/'source_events.parquet')
    for r in records.itertuples():
        assert r.status=='ok' and r.notice_date[:4] in ['2024','2025'] and r.buy_date<=r.notice_date
        assert r.pdf_sha256==sha(Path(r.pdf_path))
        # Independent Decimal check of the completed-trade quotation; planned
        # amount is never used as a ranking feature or executed purchase.
        tokens = re.findall(r'(\d[\d,]*(?:\.\d+)?)([万亿]?)股',r.trade_evidence)
        quantities = [Decimal(x.replace(',',''))*{'':1,'万':10000,'亿':100000000}[unit] for x,unit in tokens]
        assert Decimal(str(r.shares)) in quantities and r.shares>0
        assert not any(word in r.trade_evidence for word in ['拟','计划','不低于','不超过','预计'])
        y,m,d = r.buy_date.split('-')
        old = pd.read_json(Path(r.pdf_path).parents[2]/('pdf_audit_'+r.notice_date[:4]+'.jsonl'),lines=True)
        text = re.sub(r'\s+','',old.loc[old.pdf_url.eq(r.pdf_url),'text_first_three_pages'].item())
        assert r.code[-6:] in text and r.trade_evidence in text
        spans = [text[max(0,x.start()-110):x.start()] for x in re.finditer(re.escape(r.trade_evidence),text)]
        day_pattern = rf'{int(y)}年0?{int(m)}月0?{int(d)}日'
        assert any(re.search(day_pattern,p) for p in spans)
    c = duckdb.connect()
    c.read_parquet(str(BASE)).create_view('base')
    c.read_parquet(str(CALENDAR)).create_view('calendar')
    c.read_parquet(str(ROOT/'source_events.parquet')).create_view('source')
    expected = c.sql("""WITH dates AS (SELECT s.*,
        (SELECT min(calendar_date) FROM calendar d WHERE d.is_trading_day='1'
         AND d.calendar_date>s.notice_date AND d.calendar_date BETWEEN '2024-01-01' AND '2025-12-31') AS date
        FROM source s)
        SELECT d.*,coalesce(d.date BETWEEN '2024-01-01' AND '2025-12-30',false) AS in_signal_range,
        b.necessary_tradeable,b.code IS NOT NULL AS base_available,
        coalesce(d.date BETWEEN '2024-01-01' AND '2025-12-30' AND b.necessary_tradeable,false) AS eligible
        FROM dates d LEFT JOIN base b USING(date,code) ORDER BY notice_date,code,pdf_url""").df()
    actual = pd.read_parquet(ROOT/'all_events.parquet').sort_values(['notice_date','code','pdf_url']).reset_index(drop=True)
    # Normalize absent Boolean metadata only; absence is not made eligible.
    for name in ['necessary_tradeable']:
        actual[name] = actual[name].astype('boolean')
        expected[name] = expected[name].astype('boolean')
    pd.testing.assert_frame_equal(actual[expected.columns],expected,check_dtype=False,check_exact=True)
    c.register('mapped',expected)
    c.execute("""CREATE VIEW treated AS SELECT b.*,
        bool_or(e.related_actor_title) AS related_actor_title,count(*) AS source_documents,
        max(e.notice_date) AS notice_date,max(e.buy_date) AS buy_date
        FROM base b JOIN mapped e USING(date,code) WHERE e.eligible GROUP BY ALL""")
    keys = c.sql('SELECT date,code,half,related_actor_title,source_documents,notice_date,buy_date FROM treated ORDER BY date,code').df()
    f = pd.read_parquet(ROOT/'features.parquet')
    saved = f.loc[f.primary,keys.columns].sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(saved,keys,check_dtype=False,check_exact=True)
    c.execute("""CREATE VIEW choices AS SELECT a.date,a.code,b.code AS control_code,
        abs(a.return_1449-b.return_1449)/.01+abs(a.return20_prior_adjusted-b.return20_prior_adjusted)/.05
        +abs(log2(b.price_1449/a.price_1449))+abs(log2(b.amount_1449/a.amount_1449)) AS distance,
        a.return_1449-b.return_1449 AS return_1449_difference,
        a.return20_prior_adjusted-b.return20_prior_adjusted AS return20_prior_adjusted_difference,
        a.price_1449-b.price_1449 AS price_1449_difference,a.amount_1449-b.amount_1449 AS amount_1449_difference
        FROM treated a JOIN base b ON a.date=b.date AND a.board=b.board AND substr(a.code,1,2)=substr(b.code,1,2)
        AND b.necessary_tradeable AND abs(a.return_1449-b.return_1449)<=.01
        AND abs(a.return20_prior_adjusted-b.return20_prior_adjusted)<=.05
        AND b.price_1449/a.price_1449 BETWEEN .5 AND 2 AND b.amount_1449/a.amount_1449 BETWEEN .5 AND 2
        WHERE NOT EXISTS(SELECT 1 FROM mapped m WHERE m.in_signal_range AND m.date=b.date AND m.code=b.code)""")
    pairs = c.sql("""WITH nearest AS (SELECT * FROM choices QUALIFY row_number() OVER(PARTITION BY date,code ORDER BY distance,control_code)=1)
        SELECT t.date,t.code,t.half,n.* EXCLUDE(date,code) FROM treated t LEFT JOIN nearest n USING(date,code) ORDER BY date,code""").df()
    saved = pd.read_parquet(ROOT/'primary_pairs.parquet')
    pd.testing.assert_frame_equal(saved[pairs.columns],pairs,check_dtype=False,atol=2e-12,rtol=0)
    c.register('features',f)
    frozen_base = c.sql('SELECT b.* FROM base b JOIN features f USING(date,code) ORDER BY date,code').df()
    pd.testing.assert_frame_equal(f[frozen_base.columns],frozen_base,check_dtype=False,check_exact=True)
    wanted_keys = pd.concat([pairs[['date','code']],pairs.loc[pairs.control_code.notna(),['date','control_code']].rename(columns={'control_code':'code'})]).drop_duplicates().sort_values(['date','code']).reset_index(drop=True)
    pd.testing.assert_frame_equal(f[['date','code']],wanted_keys,check_dtype=False,check_exact=True)
    expected_event = pd.MultiIndex.from_frame(f[['date','code']]).isin(pd.MultiIndex.from_frame(keys[['date','code']]))
    np.testing.assert_array_equal(f.event,expected_event)
    np.testing.assert_array_equal(f.primary,expected_event)
    assert f.source_valid.all()
    want_group = np.where(~f.event,'non_event_control',np.where(f.related_actor_title.fillna(False),'related_actor','direct_actor'))
    np.testing.assert_array_equal(f.group,want_group)
    assert len(keys)==inputs['primary'] and len(f)==inputs['rows']
    result = dict(passed=True,input_report_sha256=sha(ROOT/'input_report.json'),original_documents=len(records),
        decimal_share_and_date_evidence=len(records),primary=len(keys),rows=len(f),paired=int(pairs.control_code.notna().sum()),
        outcomes_read=False,new_2026_prices_read=False)
    save_json(ROOT/'input_verification.json',result)
    return result


if __name__=='__main__':
    print(json.dumps(check(),ensure_ascii=False,indent=2))
