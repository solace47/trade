import numpy as np
import pandas as pd

from trade_research.tail_formula_intraday_scale import CONTROL,CHANGED,EXPRESSIONS,daily_ranges,transform


def raw():
    n=30
    return pd.DataFrame(dict(date=pd.date_range('2023-01-01',periods=n).strftime('%Y-%m-%d'),
        code=['sh.600000']*n,open=np.full(n,10.),close=np.full(n,10.),high=np.full(n,10.1),
        low=np.full(n,9.9),volume=np.full(n,1000),tradestatus=np.ones(n),adjustflag=np.full(n,3)))


def test_prior_twenty_ranges_ignore_overnight_drop_and_current_range():
    d=raw();d.loc[25:,'open']=8.;d.loc[25:,'close']=8.;d.loc[25:,'high']=8.1;d.loc[25:,'low']=7.9
    a=daily_ranges(d)
    np.testing.assert_allclose(a.loc[25:,'ir_mean'],.2,atol=1e-12)
    np.testing.assert_allclose(a.ir_mean,a.ir_literal_mean,atol=1e-12,equal_nan=True)
    d.loc[26,'high']=9.;b=daily_ranges(d)
    assert a.loc[26,'ir_mean']==b.loc[26,'ir_mean']
    assert b.loc[27,'ir_mean']>a.loc[27,'ir_mean']


def test_bad_active_history_is_kept_and_suspension_is_not_a_completed_day():
    d=raw();d.loc[10,'high']=np.nan;d.loc[8,'tradestatus']=0
    a=daily_ranges(d)
    assert d.loc[8,'date'] not in a.date.tolist()
    assert not a.loc[a.date.eq(d.loc[28,'date']),'ir_good'].item()


def test_same_width_transform_preserves_all_original_values_and_zero_is_invalid():
    old=pd.DataFrame({'date':['2024-01-02'],'code':['sh.600000'],'formula_input_valid':[True],
                      **{n:[1.] for n in CONTROL}});old['V01']=2.
    p=old[['date','code']].assign(ir_mean=.1,ir_prior_close=10.,ir_good=True)
    f,good=transform(old,p)
    pd.testing.assert_frame_equal(f[old.columns],old,check_exact=True)
    assert len(CONTROL)==len(EXPRESSIONS)==50 and len(CHANGED)==26 and good.all()
    for n,new in CHANGED.items():assert f[new].item()==(1. if n=='V01' else 2.)
    p['ir_mean']=0.;f,good=transform(old,p)
    assert not good.any() and f[list(CHANGED.values())].isna().all().all()
