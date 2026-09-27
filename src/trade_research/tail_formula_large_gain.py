"""Previous completed-day large raw-close advances: count and recency."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_float as previous
from . import tail_formula_prior_day as adapter
from .corporate_cash import save_json, sha

STEM = 'tail_formula_large_gain'
ROOT = Path('data/research') / STEM
PROTOCOL = Path('config') / (STEM + '_protocol.json')
HEADER = previous.HEADER
for i in range(1,21):
    HEADER += f'HGF{i}:=IF(100*INTPART(DCP{i}*100+0.5)>=109*INTPART(DCP{i+1}*100+0.5),1,0);\n'
age = '21'
for i in reversed(range(1,21)):
    age = f'MIN(IF(HGF{i}>0,{i},21),{age})'
NEW_EXPRESSIONS = {'HG01': '+'.join(f'HGF{i}' for i in range(1,21)), 'HG02': age}
EXPRESSIONS = {**previous.EXPRESSIONS, **NEW_EXPRESSIONS}


def checked_sources():
    p = json.loads(PROTOCOL.read_text())
    assert p['previous_feature_report_sha256'] == sha(previous.ROOT/'feature_report.json')
    assert p['daily_feature_report_sha256'] == sha(base.SOURCE/'feature_report.json')
    v = json.loads((previous.ROOT/'feature_verification.json').read_text())
    r = json.loads((previous.ROOT/'feature_report.json').read_text())
    assert v['passed'] and v['feature_report_sha256']==sha(previous.ROOT/'feature_report.json')
    assert r['features_sha256']==sha(previous.ROOT/'features.parquet')
    assert p['history_first']=='2023-06-01' and p['history_last']=='2025-12-30' and p['history_stock_days']==20
    return previous.context.market.stock.source_files()


def features():
    assert not (ROOT/'feature_report.json').exists()
    files=checked_sources();c=base.conn();c.read_parquet(files).create_view('daily')
    hist=c.sql('''WITH a AS (SELECT date,code,close::DOUBLE AS raw_close,preclose::DOUBLE AS preclose,
        adjustflag::DOUBLE AS adjustflag,TRY_CAST(floor(close::DOUBLE*100+.5) AS BIGINT) AS cents
        FROM daily WHERE tradestatus=1 AND date BETWEEN '2023-06-01' AND '2025-12-30'),
        b AS (SELECT *,lag(date) OVER w AS prev_date,lag(cents) OVER w AS prev_cents,
            lag(raw_close) OVER w AS prev_close,lag(adjustflag) OVER w AS prev_adjust,
            row_number() OVER w AS rn FROM a WINDOW w AS(PARTITION BY code ORDER BY date)),
        t AS (SELECT *,coalesce(cents>0 AND prev_cents>0 AND isfinite(raw_close) AND isfinite(prev_close)
            AND abs(raw_close-cents/100.)<=.0001 AND abs(prev_close-prev_cents/100.)<=.0001
            AND adjustflag=3 AND prev_adjust=3,false) AS good,
            coalesce(100*cents>=109*prev_cents,false) AS event,
            coalesce(abs(preclose-prev_close)>.005,false) AS ref_break FROM b)
        SELECT date,code,count(*) OVER h AS hg_rows,sum(good::INT) OVER h AS hg_good,
            min(date) OVER h AS hg_first_date,max(date) OVER h AS hg_last_date,
            first_value(prev_date) OVER h AS hg_reference_date,sum(event::INT) OVER h AS hg_count,
            coalesce(rn-max(CASE WHEN event THEN rn END) OVER h,21) AS hg_age,
            sum(ref_break::INT) OVER h AS hg_reference_breaks
        FROM t WINDOW h AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING)
        ORDER BY date,code''').df();c.close()
    old=pd.read_parquet(previous.ROOT/'features.parquet')
    f=old.merge(hist,on=['date','code'],how='left',validate='one_to_one').sort_values(['date','code']).reset_index(drop=True)
    valid=f.hg_rows.eq(20)&f.hg_good.eq(20)&f.hg_reference_date.lt(f.hg_first_date)&f.hg_last_date.lt(f.date)
    f['HG01']=f.hg_count.where(valid);f['HG02']=f.hg_age.where(valid)
    f['large_gain_history_valid']=valid;f['prior_formula_input_valid']=f.formula_input_valid
    f['formula_input_valid'] &= valid & np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    ROOT.mkdir(parents=True,exist_ok=True);hist.to_parquet(ROOT/'history.parquet',index=False,compression='zstd')
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),previous_feature_report_sha256=sha(previous.ROOT/'feature_report.json'),
        daily_feature_report_sha256=sha(base.SOURCE/'feature_report.json'),features_sha256=sha(ROOT/'features.parquet'),
        history_sha256=sha(ROOT/'history.parquet'),rows=len(f),valid=int(f.formula_input_valid.sum()),
        previous_valid=int(f.prior_formula_input_valid.sum()),newly_invalid=int((f.prior_formula_input_valid&~f.formula_input_valid).sum()),
        valid_with_historical_reference_breaks=int((f.formula_input_valid&f.hg_reference_breaks.gt(0)).sum()),
        first_reference_date=f.hg_reference_date.dropna().min(),last_history_date=f.hg_last_date.dropna().max(),
        expressions=EXPRESSIONS,native_header=HEADER,original_48_inputs_unchanged=True,
        event_is_not_verified_limit_up_or_adjusted_return=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'feature_report.json',r);return {k:v for k,v in r.items() if k not in ['expressions','native_header']}


def configure():
    adapter.STEM=STEM;adapter.ROOT=ROOT;adapter.PROTOCOL=PROTOCOL
    adapter.COMBINED_PROTOCOL=Path('config')/(STEM+'_combined_protocol.json')
    adapter.CONTROL=Path('data/research')/(STEM+'_control')
    adapter.EXPRESSIONS=EXPRESSIONS;adapter.HEADER=HEADER
    r=json.loads((ROOT/'feature_verification.json').read_text())
    assert r['passed'] and r['feature_report_sha256']==sha(ROOT/'feature_report.json')
    assert r['native_20_event_and_distance_expressions_rebuilt']
    for fold in ['2024','recent','combined']:
        p=json.loads((Path('config')/(STEM+'_'+fold+'_protocol.json')).read_text())
        assert p['inputs_protocol_sha256']==sha(PROTOCOL)


if __name__=='__main__':
    from . import tail_formula_context_2024 as linkage
    from . import tail_formula_recent as study
    from . import tail_formula_relative as relative
    from .tail_formula_offset_logit48 import verify_scores
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['features','model','verify_model','scores','verify_scores','freeze','verify','analyze'])
    p.add_argument('--fold',choices=['2024','recent','combined','control'],default='2024');a=p.parse_args()
    if a.stage=='features':
        result=features()
    else:
        configure()
        if a.fold=='control':
            assert a.stage in ['freeze','verify','analyze']
            result=linkage.common_analysis(adapter.CONTROL,adapter.COMBINED_PROTOCOL) if a.stage=='analyze' else adapter.control(a.stage+'_control')
        else:
            adapter.setup(a.fold)
            if a.stage=='analyze':
                assert (Path('data/research')/(STEM+'_2025')/'selection_verification.json').exists()
                assert (adapter.CONTROL/'selection_verification.json').exists()
            if a.fold=='combined':
                assert a.stage in ['freeze','verify','analyze']
                result=linkage.common_analysis(linkage.COMBINED,linkage.PROTOCOL) if a.stage=='analyze' else getattr(linkage,'combine' if a.stage=='freeze' else 'verify_combined')()
            elif a.stage in ['model','verify_model']:
                result=getattr(relative,a.stage)('relative')
            elif a.stage=='verify_scores':
                result=verify_scores()
            elif a.stage in ['freeze','verify']:
                result=getattr(study,a.stage)()
            else:
                result=getattr(base,a.stage)()
    print(json.dumps(result,ensure_ascii=False,indent=2))
