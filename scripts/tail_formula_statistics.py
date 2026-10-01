"""Date-block uncertainty and independent scalar comparisons."""
from datetime import date
import numpy as np
import pandas as pd

checked = 0

def weekly_interval(daily: pd.Series) -> list[float] | None:
    if daily.isna().any() or not len(daily):
        return None
    x = daily.to_frame("value")
    x["week"] = pd.to_datetime(x.index).to_period("W-SUN").astype(str)
    blocks = x.groupby("week").value.agg(["sum", "count"])
    if len(blocks) < 2:
        return None  # One observed week cannot estimate between-week uncertainty.
    rng = np.random.default_rng(20260926)
    indices = rng.integers(len(blocks), size=(10000, len(blocks)))
    means = blocks["sum"].to_numpy()[indices].sum(axis=1) / blocks["count"].to_numpy()[indices].sum(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def eq(actual,expected,where):
    global checked
    checked+=1
    if isinstance(expected,list):
        assert actual is not None,where
        np.testing.assert_allclose(actual,expected,atol=2e-12,rtol=0,err_msg=str(where))
    elif expected is None:
        assert actual is None,(where,actual,expected)
    elif isinstance(expected,(float,np.floating)):
        assert actual is not None and abs(actual-expected)<2e-12,(where,actual,expected)
    else:
        assert actual==expected,(where,actual,expected)


def interval(frame,column):
    if len(frame)==0 or frame[column].isna().any():return None
    weeks={}
    for day,value in frame[["date",column]].itertuples(index=False,name=None):
        iso=date.fromisoformat(day).isocalendar()
        key=(iso.year,iso.week)
        total,count=weeks.get(key,(0.,0))
        weeks[key]=(total+value,count+1)
    if len(weeks)<2:return None
    blocks=np.array([weeks[k] for k in sorted(weeks)])
    choices=np.random.default_rng(20260926).choice(len(blocks),size=(10000,len(blocks)),replace=True)
    numerator=np.take(blocks[:,0],choices).sum(axis=1)
    denominator=np.take(blocks[:,1],choices).sum(axis=1)
    return np.percentile(numerator/denominator,[2.5,97.5]).tolist()
