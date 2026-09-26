"""Describe next-day winners using prior-day inputs, without selecting on outcomes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from .corporate_cash import DAILY, MINUTES, save_json, sha
from .sector_resilience_1449 import INDUSTRIES, DELIST_NOTICES
from .turnover_reference import CALENDAR

ROOT = Path("data/research/next_day_winner")
PROTOCOL = Path("config/next_day_winner_protocol.json")
RULE_COMMIT = "89f9299"


def connection() -> duckdb.DuckDBPyConnection:
    c = duckdb.connect()
    c.execute("SET threads=4")
    c.execute("SET memory_limit='6GB'")
    c.execute("SET preserve_insertion_order=false")
    return c


def calendar() -> list[str]:
    table = pd.read_parquet(CALENDAR)
    return sorted(table.loc[table.is_trading_day.eq("1") &
        table.calendar_date.between("2024-01-01", "2025-12-31"), "calendar_date"])


def base(output: Path = ROOT) -> dict:
    if (output / "base_report.json").exists():
        raise ValueError("Frozen cohort inputs cannot be replaced")
    output.mkdir(parents=True, exist_ok=True)
    days = calendar()
    sources = [PROTOCOL, CALENDAR, INDUSTRIES, DELIST_NOTICES,
        *sorted(Path("data/research/minute_prefix_1449").glob("202[45]/*.parquet")),
        *sorted(Path("data/research/market_snapshots_ci").glob("*.parquet")),
        *sorted(DAILY.glob("*.parquet"))]
    c = connection()
    c.register("schedule", pd.DataFrame({"date": days[:-1], "next_date": days[1:]}))
    c.read_parquet("data/research/minute_prefix_1449/202[45]/*.parquet").create_view("prefix")
    c.read_parquet("data/research/market_snapshots_ci/*.parquet").create_view("snapshots")
    c.read_parquet(str(INDUSTRIES)).create_view("industries")
    c.read_parquet(str(DELIST_NOTICES)).create_view("notices")
    c.read_parquet(str(DAILY / "*.parquet")).create_view("daily")
    frame = c.sql("""SELECT p.*, s.preclose, s.open_1450 AS daily_open,
        s.isST,s.tradestatus,s.listing_age_sessions,s.reference_gap,
        s.return5_prior_adjusted,s.return20_prior_adjusted,s.ma20_prior_adjusted,s.volume5_prior,
        CASE WHEN p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%' THEN 'main'
             WHEN p.code LIKE 'sz.30%' THEN 'chinext' ELSE 'star' END AS board,
        CASE WHEN h.industry<>'' AND h.asof<=p.date AND h.updateDate<p.date
             AND date_diff('day',h.updateDate::DATE,p.date::DATE)<=370 THEN h.industry END AS industry,
        EXISTS(SELECT 1 FROM notices n WHERE n.code=p.code AND n.notice_date<p.date
          AND regexp_matches(n.title,'进入退市整理|退市整理期交易')) AS known_delisting
        FROM prefix p JOIN schedule d USING(date) JOIN snapshots s USING(date,code)
        LEFT JOIN industries h ON p.code=h.code AND p.date>=h.effective_date
          AND(h.next_effective_date IS NULL OR p.date<h.next_effective_date)
        WHERE s.isST=0 AND s.tradestatus=1 AND s.listing_age_sessions>=20
          AND(p.code LIKE 'sh.60%' OR p.code LIKE 'sz.00%' OR p.code LIKE 'sz.30%' OR p.code LIKE 'sh.68%')
          AND NOT p.quote_outside_traded_range AND p.amount_1449>0 AND p.volume_1449>0
          AND s.preclose>0 AND s.open_1450>0 AND p.price_1449>0
          AND abs(p.price_1449-round(p.price_1449,2))<=.0001
        ORDER BY p.date,p.code""").df()
    if frame.duplicated(["date", "code"]).any():
        raise ValueError("Overlapping input or industry identities")
    for col in ("price_1449", "price_1420", "price_1435", "high_1449", "low_1449"):
        frame[col] = frame[col].round(2)
    cents = np.rint(frame.price_1449*100).astype("int64")
    affordable = 2_000_000//cents
    frame["decision_shares"] = np.where(frame.board.eq("star"),
        np.where(affordable.ge(200), affordable, 0), affordable//100*100)
    rates = np.where(frame.board.eq("main"), 10, 20)
    prior_cents = np.rint(frame.preclose*100).astype("int64")
    frame["upper_limit"] = ((prior_cents*(100+rates)+50)//100)/100
    frame["sealed_quote_1449"] = frame.price_1449.ge(frame.upper_limit-.005)
    frame["necessary_tradeable"] = (frame.price_1449+np.maximum(frame.price_1449*.0015, .005)
        < frame.upper_limit-.005) & frame.decision_shares.gt(0) & frame.amount_1449.ge(3e7) \
        & ~frame.reference_gap & ~frame.known_delisting
    frame["return_1449"] = frame.price_1449/frame.preclose-1
    frame["gap_open"] = frame.daily_open/frame.preclose-1
    frame["return_tail29"] = frame.price_1449/frame.price_1420-1
    frame["return_tail14"] = frame.price_1449/frame.price_1435-1
    span = (frame.high_1449-frame.low_1449).where(frame.high_1449.gt(frame.low_1449))
    frame["position_1449"] = (frame.price_1449-frame.low_1449)/span
    frame["drawdown_1449"] = frame.price_1449/frame.high_1449-1
    frame["rebound_1449"] = frame.price_1449/frame.low_1449-1
    frame["vwap_premium"] = frame.price_1449/(frame.amount_1449/frame.volume_1449)-1
    frame["tail_vwap_premium"] = frame.price_1449/(frame.amount_last29/frame.volume_last29.where(frame.volume_last29.gt(0)))-1
    frame["volume_ratio_prior5"] = frame.volume_1449/frame.volume5_prior.where(frame.volume5_prior.gt(0))*241/230
    frame["tail_amount_fraction"] = frame.amount_last29/frame.amount_1449
    frame["tail_volume_intensity"] = (frame.volume_last29/29)/((frame.volume_1449-frame.volume_last29)/201).replace(0,np.nan)
    frame["ma20_distance"] = frame.price_1449/frame.ma20_prior_adjusted-1
    frame["half"] = frame.date.str[:4]+np.where(frame.date.str[5:7].le("06"),"H1","H2")
    clipped = frame.return_1449.clip(-.2,.2)
    frame["peer_clip"] = clipped
    frame["peer_rising"] = frame.return_1449.gt(0).astype(int)
    for keys, stem in ((["date","board"],"market"),(["date","industry"],"industry")):
        group = frame.groupby(keys)
        n = group.code.transform("size")-1
        frame[stem+"_peers"] = n
        frame[stem+"_return"] = (group.peer_clip.transform("sum")-clipped)/n.where(n.gt(0))
        frame[stem+"_rising"] = (group.peer_rising.transform("sum")-frame.peer_rising)/n.where(n.gt(0))
    frame["industry_excess"] = frame.industry_return-frame.market_return
    frame = frame.drop(columns=["peer_clip","peer_rising"])
    frame.to_parquet(output/"visible_base.parquet", index=False, compression="zstd")
    c.register("base_keys",frame[["date","code"]])
    labels = c.sql("""SELECT b.date,b.code,d.next_date,
        n.open AS next_open,n.high AS next_high,n.low AS next_low,n.close AS next_close,
        n.preclose AS next_preclose,n.tradestatus AS next_trade_status,n.isST AS next_isST,
        n.adjustflag AS next_adjustflag,p.close AS day_close,
        n.tradestatus=1 AND n.adjustflag=3 AND n.open>0 AND n.close>0 AND n.low>0
         AND n.preclose>0 AND n.high>=greatest(n.open,n.low,n.close)
         AND n.low<=least(n.open,n.high,n.close) AS known_label
        FROM base_keys b JOIN schedule d USING(date)
        LEFT JOIN daily n ON n.code=b.code AND n.date=d.next_date
        LEFT JOIN daily p ON p.code=b.code AND p.date=b.date
        ORDER BY b.date,b.code""").df()
    c.close()
    if labels.duplicated(["date","code"]).any() or len(labels)!=len(frame):
        raise ValueError("Daily source lost or duplicated cohort identities")
    labels["known_label"] = labels.known_label.fillna(False)
    for col,source in (("next_gain","next_close"),("next_gap","next_open"),("next_peak","next_high")):
        labels[col] = (labels[source]/labels.next_preclose-1).where(labels.known_label)
    # Decimal cent inequalities avoid classifying a rounded exact threshold inconsistently.
    pcent = np.rint(labels.next_preclose.fillna(0)*100).astype("int64")
    winner = labels.known_label & (np.rint(labels.next_close.fillna(0)*100).astype("int64")*100 >= pcent*105)
    gap = labels.known_label & (np.rint(labels.next_open.fillna(0)*100).astype("int64")*100 >= pcent*103)
    peak = labels.known_label & (np.rint(labels.next_high.fillna(0)*100).astype("int64")*100 >= pcent*108)
    labels["winner"] = winner.where(labels.known_label).astype("boolean")
    labels["cohort"] = np.select([~labels.known_label,winner & gap,winner & ~gap,peak & ~winner],
        ["unknown","gap_and_hold","intraday_hold","spike_fade"],default="other")
    labels["next_reference_gap"] = (labels.next_preclose-labels.day_close).abs().gt(.005)
    labels.to_parquet(output/"labels.parquet",index=False,compression="zstd")
    joined = frame.merge(labels[["date","code","cohort","known_label","winner"]],on=["date","code"],validate="one_to_one")
    overview = joined.groupby(["half","board","cohort"]).agg(rows=("code","size"),
        days=("date","nunique"), necessary_tradeable=("necessary_tradeable","sum"),
        sealed_quote_1449=("sealed_quote_1449","sum")).reset_index()
    save_json(output/"source_manifest.json",{"rule_commit":RULE_COMMIT,
        "sha256":{str(p):sha(p) for p in sources},"new_2026_prices_read":False})
    result={"rule_commit":RULE_COMMIT,"rows":len(frame),"codes":frame.code.nunique(),
        "first":frame.date.min(),"last":frame.date.max(),"known_labels":int(labels.known_label.sum()),
        "winners":int(winner.sum()),"unknown_labels":int((~labels.known_label).sum()),
        "by_half_board_cohort":overview.to_dict("records"),
        "base_sha256":sha(output/"visible_base.parquet"),"labels_sha256":sha(output/"labels.parquet"),
        "source_manifest_sha256":sha(output/"source_manifest.json"),"strategy_returns_computed":False}
    save_json(output/"base_report.json",result)
    return {k:v for k,v in result.items() if k!="by_half_board_cohort"}


def raw(output: Path = ROOT, batch_size: int = 64) -> dict:
    report=json.loads((output/"base_report.json").read_text())
    if sha(output/"visible_base.parquet")!=report["base_sha256"]:
        raise ValueError("Frozen inputs changed")
    frame=pd.read_parquet(output/"visible_base.parquet",columns=["code"])
    codes=sorted(frame.code.unique())
    folder=output/"raw_features";folder.mkdir(exist_ok=True)
    batches=[]
    for start in range(0,len(codes),batch_size):
        subset=codes[start:start+batch_size]; part=folder/f"part_{start//batch_size:03d}.parquet"
        provenance=part.with_suffix(".json")
        paths=[MINUTES/code[:2].upper()/(code[3:]+".parquet") for code in subset]
        if part.exists() and provenance.exists():
            saved=json.loads(provenance.read_text())
            if saved.get("extractor_version")!=2 or saved["codes"]!=subset or saved["sha256"]!=sha(part):
                raise ValueError("A raw feature batch changed")
            batches.append(saved);continue
        c=connection();c.read_parquet([str(p) for p in paths]).create_view("minutes")
        # Entry bars are aggregated separately; every path feature is based only on the prefix.
        features=c.sql("""WITH src AS (
          SELECT lower(exchange)||'.'||symbol AS code,strftime(timestamp,'%Y-%m-%d') AS date,
            strftime(timestamp,'%H%M') AS label,round(close,2) AS close,close AS raw_close,open,high,low,volume,turnover
          FROM minutes WHERE timestamp>=TIMESTAMP '2024-01-01' AND timestamp<TIMESTAMP '2026-01-01'
          AND(strftime(timestamp,'%H%M') BETWEEN '0930' AND '1130'
              OR strftime(timestamp,'%H%M') BETWEEN '1301' AND '1455')
        ), pref AS (
          SELECT *,lag(close) OVER w AS previous_close,
             sum(volume) OVER w AS cumulative_volume,sum(turnover) OVER w AS cumulative_amount
          FROM src WHERE label<='1449'
          WINDOW w AS(PARTITION BY date,code ORDER BY label ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        ), agg AS (
          SELECT date,code,count(*) AS prefix_bars,count(DISTINCT label) AS prefix_labels,
            max(close) FILTER(WHERE label='1000') AS price_1000,
            max(close) FILTER(WHERE label='1130') AS price_1130,
            avg(CASE WHEN cumulative_volume>0 THEN (close>=cumulative_amount/cumulative_volume)::DOUBLE END) AS above_vwap_fraction,
            avg(CASE WHEN cumulative_volume>0 AND label>='1301' THEN (close>=cumulative_amount/cumulative_volume)::DOUBLE END) AS afternoon_above_vwap,
            avg((close>previous_close)::DOUBLE) FILTER(WHERE label>='1421') AS tail_up_fraction,
            sum(sign(close-previous_close)*turnover) FILTER(WHERE label>='1421')/
                nullif(sum(turnover) FILTER(WHERE label>='1421'),0) AS tail_signed_amount,
            (max(close) FILTER(WHERE label='1449')-max(close) FILTER(WHERE label='1420'))/
                nullif(sum(abs(close-previous_close)) FILTER(WHERE label>='1421'),0) AS tail_path_efficiency,
            max(close/nullif(previous_close,0)-1) FILTER(WHERE label>='1421') AS tail_max_jump,
            (max(close) FILTER(WHERE label='1449'))/nullif(max(close) FILTER(WHERE label>='1421'),0)-1 AS tail_giveback,
            sum(turnover) FILTER(WHERE label<='1000')/nullif(sum(turnover),0) AS early_amount_fraction,
            count(*) FILTER(WHERE volume>0 AND (turnover/volume<low-.0001 OR turnover/volume>high+.0001)) AS prefix_amount_bad_bars,
            sum(volume) AS prefix_volume,sum(turnover) AS prefix_amount,
            max(close) FILTER(WHERE label='1449') AS prefix_price_1449
          FROM pref GROUP BY date,code
        ), ent AS (
          SELECT date,code,count(*) AS entry_bars,count(DISTINCT label) AS entry_labels,
            sum(volume) AS entry_volume,sum(turnover)/nullif(sum(volume),0) AS entry_vwap,
            max(high) FILTER(WHERE volume>0) AS entry_high,min(low) FILTER(WHERE volume>0) AS entry_low,
            count(*) FILTER(WHERE volume>0 AND (turnover/volume<low-.0001 OR turnover/volume>high+.0001)) AS entry_amount_bad_bars,
            count(*) FILTER(WHERE open>0 AND raw_close>0 AND low>0 AND high>=greatest(open,low,raw_close)
              AND low<=least(open,high,raw_close) AND volume>=0 AND turnover>=0) AS entry_valid_bars
          FROM src WHERE label BETWEEN '1452' AND '1455' GROUP BY date,code
        ) SELECT * FROM agg LEFT JOIN ent USING(date,code) ORDER BY date,code""").df()
        c.close()
        features.to_parquet(part,index=False,compression="zstd")
        saved={"extractor_version":2,"codes":subset,"rows":len(features),"sha256":sha(part),
            "source_sha256":{str(p):sha(p) for p in paths}}
        save_json(provenance,saved);batches.append(saved)
        print(json.dumps({"raw_codes_completed":start+len(subset),"total_codes":len(codes),"part":part.name},ensure_ascii=False),flush=True)
    result={"codes":len(codes),"batches":len(batches),"rows":sum(b["rows"] for b in batches),
        "batch_manifests_sha256":{str(p):sha(p) for p in sorted(folder.glob("*.json"))},
        "last_feature_label":"1449","entry_window_separate":True,"new_2026_prices_read":False}
    save_json(output/"raw_report.json",result)
    return result


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["base","raw"])
    args=parser.parse_args()
    print(json.dumps(globals()[args.stage](),ensure_ascii=False,indent=2))
