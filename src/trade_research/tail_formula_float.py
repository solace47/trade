"""Lagged float-size and turnover inputs for the two next-morning time folds."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_context as context
from . import tail_formula_context_2024 as linkage
from . import tail_formula_recent as study
from . import tail_formula_relative as relative
from .corporate_cash import save_json,sha
from .turnover_reference import CALENDAR

ROOT=Path('data/research/tail_formula_float')
PROTOCOL=Path('config/tail_formula_float_protocol.json')
NEW_EXPRESSIONS={'S01':'LN(1+PSH*Q/100000000)',
    'S02':'10000*DV/PSH','S03':'10000*VALUEWHEN(TIME=1449,SUM(V,29))/PSH'}
EXPRESSIONS={**context.EXPRESSIONS,**NEW_EXPRESSIONS}
HEADER=context.HEADER+'PSH:=REF(FINANCE(7),B0);\n'


def features():
    if (ROOT/'feature_report.json').exists():
        raise ValueError('Do not replace frozen float inputs')
    ROOT.mkdir(parents=True,exist_ok=True)
    p=json.loads((context.ROOT/'feature_verification.json').read_text())
    r=json.loads((context.ROOT/'feature_report.json').read_text())
    assert p['passed'] and p['feature_report_sha256']==sha(context.ROOT/'feature_report.json')
    assert r['features_sha256']==sha(context.ROOT/'features.parquet')
    files=context.market.stock.source_files()
    c=base.conn();c.read_parquet(files).create_view('daily')
    prior=c.sql('''WITH a AS(SELECT date,code,volume::DOUBLE AS volume,turn::DOUBLE AS turn,
        adjustflag::DOUBLE AS adjustflag FROM daily WHERE tradestatus=1
        AND date BETWEEN '2023-06-01' AND '2025-12-30'),
        b AS(SELECT date,code,lag(date) OVER w AS float_source_date,
        lag(volume) OVER w AS float_prior_volume,lag(turn) OVER w AS float_prior_turn,
        lag(adjustflag) OVER w AS float_prior_adjustflag FROM a WINDOW w AS(PARTITION BY code ORDER BY date))
        SELECT * FROM b WHERE date>='2024-01-01' ORDER BY date,code''').df()
    c.close()
    old=pd.read_parquet(context.ROOT/'features.parquet')
    f=old.merge(prior,on=['date','code'],how='left',validate='one_to_one')
    cal=pd.read_parquet(CALENDAR)
    days=sorted(cal.loc[cal.is_trading_day.eq('1')&cal.calendar_date.between('2023-06-01','2025-12-30'),'calendar_date'])
    rank={d:i for i,d in enumerate(days)}
    f['float_source_gap']=f.date.map(rank)-f.float_source_date.map(rank)
    f['float_source_valid']=(f.float_source_date.lt(f.date)&f.float_prior_adjustflag.eq(3)
        &np.isfinite(f.float_prior_volume)&f.float_prior_volume.gt(0)
        &np.isfinite(f.float_prior_turn)&f.float_prior_turn.gt(0))
    f['float_shares_proxy']=(100*f.float_prior_volume/f.float_prior_turn).where(f.float_source_valid)
    f['S01']=np.log1p(f.float_shares_proxy*f.price_1449/1e8)
    f['S02']=100*f.volume_1449/f.float_shares_proxy
    f['S03']=100*f.v29/f.float_shares_proxy
    f['formula_input_valid'] &= f.float_source_valid&np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    f=f.sort_values(['date','code']).reset_index(drop=True)
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    report=dict(protocol_sha256=sha(PROTOCOL),previous_feature_report_sha256=sha(context.ROOT/'feature_report.json'),
        daily_report_sha256=sha(base.SOURCE/'feature_report.json'),calendar_sha256=sha(CALENDAR),
        features_sha256=sha(ROOT/'features.parquet'),rows=len(f),valid=int(f.formula_input_valid.sum()),
        previous_valid=int(old.formula_input_valid.sum()),newly_invalid=int((old.formula_input_valid&~f.formula_input_valid).sum()),
        invalid_denominators=int((~f.float_source_valid).sum()),
        prior_stock_day_gap_above_one=int(f.float_source_gap.gt(1).sum()),
        valid_with_prior_gap_above_one=int((f.formula_input_valid&f.float_source_gap.gt(1)).sum()),
        maximum_prior_market_session_gap=float(f.float_source_gap.max()),
        first_source_date=f.float_source_date.min(),last_source_date=f.float_source_date.max(),
        expressions=EXPRESSIONS,native_header=HEADER,native_FINANCE7_parity_verified=False,
        software_compilation_verified=False,outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',report)
    return {k:v for k,v in report.items() if k not in ['expressions','native_header']}


def setup(fold):
    if fold=='combined':
        linkage.ROOT=Path('data/research/tail_formula_float_2024')
        linkage.H2=Path('data/research/tail_formula_float_recent')
        linkage.COMBINED=Path('data/research/tail_formula_float_2025')
        linkage.PROTOCOL=Path('config/tail_formula_float_combined_protocol.json')
        linkage.H2_SELECTION_SHA=None;linkage.H2_MODEL_SHA=None;linkage.H2_OUTCOMES_PREVIOUSLY_SEEN=False
    else:
        context.setup('relative')
        root=Path('data/research/tail_formula_float_'+fold)
        protocol=Path('config/tail_formula_float_'+fold+'_protocol.json')
        base.ROOT=root;base.PROTOCOL=protocol;relative.PROTOCOL=protocol
        study.ROOT=root;study.PROTOCOL=protocol
        base.FEATURES=ROOT;base.EXPRESSIONS=EXPRESSIONS;base.HEADER=HEADER


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['features','model','verify_model','scores','freeze','verify','analyze','diagnose'])
    p.add_argument('--fold',choices=['2024','recent','combined'],default='2024')
    a=p.parse_args()
    if a.stage=='features':
        r=features()
    else:
        setup(a.fold)
        if a.fold=='combined':
            assert a.stage in ['freeze','verify','analyze']
            r=(linkage.common_analysis(linkage.COMBINED,linkage.PROTOCOL) if a.stage=='analyze'
                else getattr(linkage,'combine' if a.stage=='freeze' else 'verify_combined')())
        elif a.stage in ['model','verify_model']:
            r=getattr(relative,a.stage)('relative')
        elif a.stage in ['freeze','verify']:
            r=getattr(study,a.stage)()
        else:
            r=getattr(base,a.stage)()
    print(json.dumps(r,ensure_ascii=False,indent=2))
