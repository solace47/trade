"""Independently rebuild lagged denominators with strict backward ASOF joins."""
import json

import numpy as np
import pandas as pd

from trade_research import tail_formula_float as study
from trade_research.corporate_cash import save_json,sha


def main():
    root=study.ROOT;r=json.loads((root/'feature_report.json').read_text())
    for key,path in [('protocol_sha256',study.PROTOCOL),
        ('previous_feature_report_sha256',study.context.ROOT/'feature_report.json'),
        ('daily_report_sha256',study.base.SOURCE/'feature_report.json'),
        ('calendar_sha256',study.CALENDAR),('features_sha256',root/'features.parquet')]:
        assert r[key]==sha(path)
    old=pd.read_parquet(study.context.ROOT/'features.parquet')
    f=pd.read_parquet(root/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns.drop('formula_input_valid')],old.drop(columns='formula_input_valid'),check_exact=True)
    c=study.base.conn();c.register('keys',old[['date','code']])
    c.read_parquet(study.context.market.stock.source_files()).create_view('daily')
    a=c.sql('''WITH prior AS(SELECT date,code,volume,turn,adjustflag FROM daily
        WHERE tradestatus=1 AND date BETWEEN '2023-06-01' AND '2025-12-30')
        SELECT k.date,k.code,d.date AS float_source_date,d.volume::DOUBLE AS float_prior_volume,
        d.turn::DOUBLE AS float_prior_turn,d.adjustflag::DOUBLE AS float_prior_adjustflag
        FROM keys k ASOF LEFT JOIN prior d ON k.code=d.code AND k.date>d.date ORDER BY k.date,k.code''').df()
    c.close()
    columns=['float_source_date','float_prior_volume','float_prior_turn','float_prior_adjustflag']
    pd.testing.assert_frame_equal(f[columns],a[columns],check_dtype=False,check_exact=True)
    cal=pd.read_parquet(study.CALENDAR)
    days=sorted(cal.loc[cal.is_trading_day.eq('1')&cal.calendar_date.between('2023-06-01','2025-12-30'),'calendar_date'])
    indices=pd.Index(days)
    gaps=indices.get_indexer(a.date)-indices.get_indexer(a.float_source_date)
    assert np.array_equal(f.float_source_gap.to_numpy(),gaps)
    valid=(a.float_source_date.lt(a.date)&a.float_prior_adjustflag.eq(3)
        &a.float_prior_volume.gt(0)&a.float_prior_turn.gt(0)
        &np.isfinite(a[['float_prior_volume','float_prior_turn']]).all(axis=1))
    shares=(a.float_prior_volume/(a.float_prior_turn/100)).where(valid)
    expected={'float_shares_proxy':shares,'S01':np.log(1+old.price_1449*shares/100000000),
        'S02':old.volume_1449/shares*100,'S03':old.v29/shares*100}
    for name,values in expected.items():
        np.testing.assert_allclose(f[name],values,rtol=2e-14,atol=1e-12,equal_nan=True)
    assert f.float_source_valid.equals(valid)
    good=old.formula_input_valid&valid&np.isfinite(f[list(study.EXPRESSIONS)]).all(axis=1)
    assert f.formula_input_valid.equals(good) and int(good.sum())==r['valid']
    assert len(f)==r['rows'] and int((old.formula_input_valid&~good).sum())==r['newly_invalid']
    assert int(f.float_source_gap.gt(1).sum())==r['prior_stock_day_gap_above_one']
    proof=dict(passed=True,feature_report_sha256=sha(root/'feature_report.json'),rows=len(f),valid=int(good.sum()),
        all_denominators_and_three_inputs_rebuilt=True,strict_backward_asof_verified=True,
        all_previous_45_values_unchanged=True,native_FINANCE7_parity_verified=False,
        outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(root/'feature_verification.json',proof)
    print(json.dumps(proof,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
