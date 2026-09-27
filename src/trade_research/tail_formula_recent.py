"""A fixed recent-12-month update, evaluated only after its training cutoff."""
import argparse
import json
from pathlib import Path

import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_relative as relative
from . import tail_formula_volatility as inputs
from .corporate_cash import save_json,sha

ROOT=Path('data/research/tail_formula_recent')
PROTOCOL=Path('config/tail_formula_recent_protocol.json')


def setup():
    base.ROOT=ROOT;base.FEATURES=inputs.ROOT;base.EXPRESSIONS=inputs.EXPRESSIONS;base.HEADER=inputs.HEADER
    base.PROTOCOL=PROTOCOL;relative.PROTOCOL=PROTOCOL


def freeze():
    if (ROOT/'selection_report.json').exists():
        raise ValueError('Do not replace the fixed recent-model selection')
    proof=json.loads((ROOT/'score_verification.json').read_text())
    assert proof['passed'] and proof['score_report_sha256']==sha(ROOT/'score_report.json')
    old=json.loads((inputs.ROOT/'selection_report.json').read_text())
    assert sha(inputs.ROOT/'selection_report.json')=='341c0d40b5c28de3513fdda2968499fb66adef937ae693365de0f23478c90d35'
    assert old['chosen_threshold']['id']==3 and old['chosen_threshold']['training_quantile']==.995
    m=json.loads((ROOT/'model_report.json').read_text());cut=m['thresholds'][3]
    config=json.loads(PROTOCOL.read_text())
    start,end=config.get('evaluation_start','2025-07-01'),config.get('evaluation_end')
    assert start>=m['training_end'] and (end is None or end>start)
    assert cut['training_quantile']==.995 and m['last_observation']<start
    f=pd.read_parquet(ROOT/'scores.parquet')
    out=f[['date','code','half','board','decision_shares']].copy()
    out['selected']=f.date.ge(start)&f.formula_input_valid&f.score.gt(cut['threshold'])
    if end is not None:
        out['selected'] &= f.date.lt(end)
    out.to_parquet(ROOT/'selection.parquet',index=False,compression='zstd')
    (ROOT/'frozen_numeric_core.tdx').write_text(base.native_core(m,cut['threshold'],base.EXPRESSIONS,base.HEADER))
    r=dict(protocol_sha256=sha(PROTOCOL),model_report_sha256=sha(ROOT/'model_report.json'),
        score_report_sha256=sha(ROOT/'score_report.json'),prior_calibration_report_sha256=sha(inputs.ROOT/'selection_report.json'),
        selection_sha256=sha(ROOT/'selection.parquet'),core_sha256=sha(ROOT/'frozen_numeric_core.tdx'),
        chosen_threshold=cut,selected=int(out.selected.sum()),
        by_half=out.groupby('half').selected.agg(['size','sum']).reset_index().to_dict('records'),
        evaluation_start=start,training_includes_2025H1=m.get('new_2025_score_groups_read',False),year_2025_is_exploratory=True,
        new_2025H2_score_groups_read=False,new_2026_prices_read=False,no_exit_rules=True,software_compilation_verified=False)
    if end is not None:
        r['evaluation_end']=end
    save_json(ROOT/'selection_report.json',r)
    return r


def verify():
    r=json.loads((ROOT/'selection_report.json').read_text())
    for key,path in [('protocol_sha256',PROTOCOL),('model_report_sha256',ROOT/'model_report.json'),
                     ('score_report_sha256',ROOT/'score_report.json'),('selection_sha256',ROOT/'selection.parquet'),
                     ('core_sha256',ROOT/'frozen_numeric_core.tdx'),('prior_calibration_report_sha256',inputs.ROOT/'selection_report.json')]:
        assert r[key]==sha(path)
    m=json.loads((ROOT/'model_report.json').read_text())
    config=json.loads(PROTOCOL.read_text())
    assert m['training_start']==config['training_start'] and m['training_end']==config['training_end']
    start,end=config.get('evaluation_start','2025-07-01'),config.get('evaluation_end')
    assert r['evaluation_start']==start and r.get('evaluation_end')==end and start>=m['training_end']
    cut=m['thresholds'][3];assert cut==r['chosen_threshold'] and cut['training_quantile']==.995
    c=base.conn();c.read_parquet(str(ROOT/'scores.parquet')).create_view('scores')
    time_condition=f"date>='{start}'"+(f" AND date<'{end}'" if end else '')
    expected=c.sql('SELECT date,code,half,board,decision_shares,'+time_condition+' AND formula_input_valid AND score>'+
        format(cut['threshold'],'.17e')+' AS selected FROM scores ORDER BY date,code').df()
    pd.testing.assert_frame_equal(pd.read_parquet(ROOT/'selection.parquet'),expected,check_exact=True,check_dtype=False)
    assert expected.loc[expected.selected,'date'].ge(m['training_end']).all()
    proof=dict(passed=True,selection_report_sha256=sha(ROOT/'selection_report.json'),rows=len(expected),
        all_selection_flags_rebuilt=True,no_training_period_selection=True,new_2025H2_score_groups_read=False,
        new_2026_prices_read=False,no_exit_rules=True)
    save_json(ROOT/'selection_verification.json',proof)
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['model','verify_model','scores','freeze','verify','analyze','diagnose'])
    a=p.parse_args();setup()
    if a.stage in ['model','verify_model']:
        r=getattr(relative,a.stage)('relative')
    elif a.stage in ['freeze','verify']:
        r=globals()[a.stage]()
    else:
        r=getattr(base,a.stage)()
    print(json.dumps(r,ensure_ascii=False,indent=2))
