"""Rebuild fixed case hashes and all greedy control edges before fetching notices."""
import json
from pathlib import Path

import duckdb
import pandas as pd

from trade_research.corporate_cash import save_json,sha

root=Path("data/research/winner_event_timing");report=json.loads((root/"input_report.json").read_text())
assert sha(root/"members.parquet")==report["members_sha256"]
assert sha(Path("config/winner_event_timing_protocol.json"))==report["protocol_sha256"]
assert sha(Path("data/research/next_day_winner/cohort.parquet"))==report["source_cohort_sha256"]
assert sha(Path("data/research/corporate_cash/source/stock_list.json"))==report["stock_mapping_sha256"]
c=duckdb.connect();c.execute("SET threads=4")
c.read_parquet("data/research/next_day_winner/cohort.parquet").create_view("source")
c.execute("""CREATE TABLE pool AS SELECT date,code,board,half,cohort,next_date,known_label,winner,
    return_1449,return20_prior_adjusted,price_1449,amount_1449 FROM source
    WHERE necessary_tradeable AND date BETWEEN '2024-01-09' AND '2025-12-30' AND next_date<='2025-12-31'""")
chosen=c.sql("""WITH hashes AS(SELECT *,sha256('winner-event-timing-v1|'||date||'|'||code) AS identity_hash
    FROM pool WHERE known_label AND winner),one_day AS(SELECT * FROM hashes
    QUALIFY row_number() OVER(PARTITION BY half,board,cohort,date ORDER BY identity_hash,code)=1),
    cases AS(SELECT * FROM one_day QUALIFY row_number() OVER(PARTITION BY half,board,cohort ORDER BY identity_hash,date,code)<=5)
    SELECT *,row_number() OVER(PARTITION BY date ORDER BY identity_hash,code) daily_rank FROM cases""").df()
members=pd.read_parquet(root/"members.parquet");case=members.loc[members.arm.eq("case")]
pd.testing.assert_frame_equal(chosen.sort_values(["date","code"]).reset_index(drop=True),
    case[chosen.columns].sort_values(["date","code"]).reset_index(drop=True),check_dtype=False,check_exact=True)
assert len(case)==120 and case.groupby(["half","board","cohort"]).size().eq(5).all()
assert not case.duplicated(["half","board","cohort","date"]).any()
c.register("cases",chosen)
edges=c.sql("""SELECT a.date,a.code AS case_code,a.daily_rank,b.code AS peer_code,
    abs(b.return20_prior_adjusted-a.return20_prior_adjusted)/.05+abs(b.return_1449-a.return_1449)/.02
    +abs(ln(b.amount_1449/a.amount_1449))/ln(2)+abs(ln(b.price_1449/a.price_1449))/ln(2) distance
    FROM cases a JOIN pool b ON a.date=b.date AND a.board=b.board AND b.known_label AND NOT b.winner
    WHERE abs(b.return20_prior_adjusted-a.return20_prior_adjusted)<=.05 AND abs(b.return_1449-a.return_1449)<=.02
    AND b.price_1449/a.price_1449 BETWEEN .5 AND 2 AND b.amount_1449/a.amount_1449 BETWEEN .5 AND 2
    ORDER BY a.date,a.daily_rank,distance,b.code""").df()
matched=[];used=set()
for (date,code),g in edges.groupby(["date","case_code"],sort=False):
    for row in g.itertuples():
        if (date,row.peer_code) in used:continue
        matched.append({"date":date,"code":row.peer_code,"pair_id":date+":"+code,"daily_rank":row.daily_rank,"distance":row.distance})
        used.add((date,row.peer_code));break
control=members.loc[members.arm.eq("control")]
expected=pd.DataFrame(matched).sort_values(["date","code"]).reset_index(drop=True)
pd.testing.assert_frame_equal(expected,control[expected.columns].sort_values(["date","code"]).reset_index(drop=True),
    check_dtype=False,atol=1e-12,rtol=0)
c.read_parquet("data/baostock/market_2020_2026/metadata/calendar.parquet").create_view("calendar")
dates=c.sql("SELECT calendar_date FROM calendar WHERE is_trading_day='1' AND calendar_date BETWEEN '2024-01-01' AND '2025-12-31' ORDER BY calendar_date").df().calendar_date.tolist()
positions={date:i for i,date in enumerate(dates)}
for row in members.itertuples():
    assert row.source_first==dates[positions[row.date]-5]
    assert row.source_last==dates[positions[row.date]+1]==row.next_date
result={"cases":len(case),"controls":len(control),"independent_hash_strata":24,"all_matching_edges":len(edges),
    "all_source_windows_verified":len(members),"members_sha256":sha(root/"members.parquet"),
    "input_report_sha256":sha(root/"input_report.json"),"announcements_read_before_selection":False}
save_json(root/"independent_input_checks.json",result);print(json.dumps(result,indent=2))
