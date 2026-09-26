"""Independent daily labels, raw path samples, and descriptive result checks."""
import argparse
import hashlib
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha
from trade_research.quote_precision import quote_cents, fixed_quote_shares

root=Path("data/research/next_day_winner")
parser=argparse.ArgumentParser()
parser.add_argument("stage",choices=["base","paths"])
stage=parser.parse_args().stage
report=json.loads((root/"base_report.json").read_text())
base=pd.read_parquet(root/"visible_base.parquet")
labels=pd.read_parquet(root/"labels.parquet")
assert sha(root/"visible_base.parquet")==report["base_sha256"]
assert sha(root/"labels.parquet")==report["labels_sha256"]
c=duckdb.connect();c.execute("SET threads=4")
c.register("base",base);c.register("labels",labels)
c.read_parquet("data/baostock/market_2020_2026/daily/*.parquet").create_view("daily")
calendar=pd.read_parquet("data/baostock/market_2020_2026/metadata/calendar.parquet")
days=sorted(calendar.loc[calendar.is_trading_day.eq("1") & calendar.calendar_date.between("2024-01-01","2025-12-31"),"calendar_date"])
nextday=dict(zip(days[:-1],days[1:]))
assert (labels.next_date==labels.date.map(nextday)).all()
assert len(base)==len(labels)==len(base.drop_duplicates(["date","code"]))
assert (base.isST==0).all() and (base.tradestatus==1).all()
assert base.listing_age_sessions.ge(20).all()
assert base.date.between("2024-01-01","2025-12-30").all()
audit=c.sql("""SELECT count(*) FILTER(WHERE abs(b.daily_open-d.open)>.0001
       OR abs(b.preclose-d.preclose)>.0001 OR d.code IS NULL) AS bad_current_known_quotes,
     count(*) FILTER(WHERE b.isST<>d.isST OR b.tradestatus<>d.tradestatus) AS bad_current_status
     FROM base b LEFT JOIN daily d USING(date,code)""").df().iloc[0].to_dict()
assert not any(audit.values()),audit
expected=c.sql("""SELECT l.date,l.code,
     coalesce(d.tradestatus=1 AND d.adjustflag=3 AND d.open>0 AND d.close>0 AND d.low>0
       AND d.preclose>0 AND d.high>=greatest(d.open,d.low,d.close)
       AND d.low<=least(d.open,d.high,d.close),false) AS known,
     cast(round(d.close*100) AS BIGINT)*100>=cast(round(d.preclose*100) AS BIGINT)*105 AS win,
     cast(round(d.open*100) AS BIGINT)*100>=cast(round(d.preclose*100) AS BIGINT)*103 AS gap,
     cast(round(d.high*100) AS BIGINT)*100>=cast(round(d.preclose*100) AS BIGINT)*108 AS peak,
     d.close/d.preclose-1 AS gain,d.open/d.preclose-1 AS gap_return,d.high/d.preclose-1 AS peak_return
     FROM labels l LEFT JOIN daily d ON l.next_date=d.date AND l.code=d.code
     ORDER BY l.date,l.code""").df()
labels=labels.sort_values(["date","code"]).reset_index(drop=True)
assert np.array_equal(labels.known_label,expected.known)
assert labels.loc[expected.known,"winner"].astype(bool).equals(expected.loc[expected.known,"win"].astype(bool))
groups=np.select([~expected.known,expected.win.fillna(False)&expected.gap.fillna(False),expected.win.fillna(False),expected.peak.fillna(False)],
    ["unknown","gap_and_hold","intraday_hold","spike_fade"],default="other")
assert np.array_equal(groups,labels.cohort)
for actual,target in (("next_gain","gain"),("next_gap","gap_return"),("next_peak","peak_return")):
    assert np.allclose(labels.loc[expected.known,actual],expected.loc[expected.known,target],rtol=0,atol=1e-12)
cent=np.rint(base.price_1449*100).astype("int64")
prior=np.rint(base.preclose*100).astype("int64")
lim=((prior*np.where(base.board.eq("main"),110,120)+50)//100)/100
assert np.allclose(lim,base.upper_limit,rtol=0,atol=0)
shares=np.where(base.board.eq("star"),np.where(2_000_000//cent>=200,2_000_000//cent,0),2_000_000//cent//100*100)
assert np.array_equal(shares,base.decision_shares)
necessary=(base.price_1449+np.maximum(.005,base.price_1449*.0015)<lim-.005)&(shares>0)&base.amount_1449.ge(3e7)&~base.reference_gap&~base.known_delisting
assert np.array_equal(necessary,base.necessary_tradeable)
result={"daily_labels_rebuilt":len(labels),"winners":int(labels.winner.sum()),"unknown":int((~labels.known_label).sum()),
    "daily_open_and_reference_errors":audit,"sizing_and_necessary_conditions_checked":len(base),"new_2026_prices_read":False}

if stage=="paths":
    from trade_research.next_day_winner_analysis import FEATURES
    frame=pd.read_parquet(root/"cohort.parquet")
    joined=base[["date","code","board","half"]].merge(labels[["date","code","cohort"]],on=["date","code"])
    joined["order"]=[hashlib.sha256((d+code).encode()).hexdigest() for d,code in zip(joined.date,joined.code)]
    samples=joined.sort_values("order").groupby(["board","half","cohort"]).head(4)
    actual=frame.set_index(["date","code"])
    max_error=0.;checked=0
    for code,selection in samples.groupby("code"):
        path=Path("data/hf/pilot/data/stock_1m")/code[:2].upper()/(code[3:]+".parquet")
        m=pd.read_parquet(path,filters=[("timestamp",">=",pd.Timestamp(selection.date.min())),
            ("timestamp","<",pd.Timestamp(selection.date.max())+pd.Timedelta(days=1))])
        m["date"]=m.timestamp.dt.strftime("%Y-%m-%d");m["label"]=m.timestamp.dt.strftime("%H%M")
        for key in selection.itertuples():
            d=m.loc[m.date.eq(key.date)].sort_values("timestamp")
            p=d.loc[d.label.between("0930","1130")|d.label.between("1301","1449")].copy()
            assert len(p)==p.label.nunique()==230
            close=p.close.round(2);previous=close.shift();vol=p.volume;amount=p.turnover
            above=close.ge(amount.cumsum()/vol.cumsum().replace(0,np.nan)).astype(float).where(vol.cumsum().gt(0))
            tail=p.label.ge("1421");change=close-previous
            def price(label):return float(close.loc[p.label.eq(label)].iloc[0])
            denominator=change.loc[tail].abs().sum()
            expected_features={"price_1000":price("1000"),"price_1130":price("1130"),
                "above_vwap_fraction":above.mean(),"afternoon_above_vwap":above.loc[p.label.ge("1301")].mean(),
                "tail_up_fraction":change.loc[tail].gt(0).mean(),
                "tail_signed_amount":(np.sign(change.loc[tail])*amount.loc[tail]).sum()/amount.loc[tail].sum() if amount.loc[tail].sum()>0 else np.nan,
                "tail_path_efficiency":(price("1449")-price("1420"))/denominator if denominator else np.nan,
                "tail_max_jump":(close/previous-1).loc[tail].max(),
                "tail_giveback":price("1449")/close.loc[tail].max()-1,
                "early_amount_fraction":amount.loc[p.label.le("1000")].sum()/amount.sum()}
            row=actual.loc[(key.date,code)]
            for col,val in expected_features.items():
                assert np.isclose(row[col],val,rtol=1e-10,atol=1e-11,equal_nan=True),(key.date,code,col,row[col],val)
                if np.isfinite(val):max_error=max(max_error,abs(row[col]-val))
            assert row.decision_shares==fixed_quote_shares(code,row.price_1449,20000)
            e=d.loc[d.label.between("1452","1455")]
            assert len(e)==row.entry_bars
            assert e.volume.sum()==row.entry_volume
            if e.volume.sum()>0:
                assert np.isclose(e.turnover.sum()/e.volume.sum(),row.entry_vwap,rtol=1e-12,atol=1e-12)
            checked+=1
    # Rebuild every feature-band probability with an independent SQL percentile rank.
    probabilities=pd.read_parquet(root/"feature_probabilities.parquet")
    c.register("cohort",frame)
    stats_checked=0
    for feature in FEATURES:
        rebuilt=c.sql(f"""WITH ranked AS(SELECT date,half,board,known_label,winner_float,necessary_tradeable,
            CASE WHEN \"{feature}\" IS NULL THEN NULL ELSE
            (rank() OVER(PARTITION BY date,board ORDER BY \"{feature}\" NULLS LAST)
              +(count(*) OVER(PARTITION BY date,board,\"{feature}\")-1)/2.)
                /nullif(count(\"{feature}\") OVER(PARTITION BY date,board),0) END AS r FROM cohort),
          tagged AS(SELECT *,CASE WHEN r IS NULL THEN 'missing' WHEN r<=.2 THEN 'low20'
              WHEN r>=.8 THEN 'high20' ELSE 'middle60' END AS band FROM ranked WHERE necessary_tradeable)
          SELECT half,board,band,count(*) AS rows,sum(known_label::INT) AS known,
            sum(winner_float) AS winners,avg(winner_float) AS probability
          FROM tagged GROUP BY ALL ORDER BY half,board,band""").df()
        saved=probabilities.loc[probabilities.feature.eq(feature)].sort_values(["half","board","band"]).reset_index(drop=True)
        assert len(rebuilt)==len(saved),feature
        for col in ("half","board","band"):
            assert np.array_equal(rebuilt[col],saved[col]),(feature,col)
        for col in ("rows","known","winners","probability"):
            assert np.allclose(rebuilt[col],saved[col],rtol=0,atol=1e-12,equal_nan=True),(feature,col)
        stats_checked+=len(saved)
    result.update({"raw_sample_stock_days":checked,"raw_feature_max_error":max_error,"probability_cells_rebuilt":stats_checked})
save_json(root/("independent_"+stage+"_checks.json"),result)
print(json.dumps(result,ensure_ascii=False,indent=2))
