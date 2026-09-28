"""Completed daily-bar helper after the recorded first-minute-open mismatch."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from . import tail_formula_reversal_change as inputs
from .corporate_cash import DAILY, save_json, sha

PROTOCOL = Path('config/tail_formula_reversal_change_daily_native_protocol.json')
HELPER = ('RN:=IF(O>REF(C,1) AND C<O,1,0);\n'
    'RP:=IF(O<REF(C,1) AND C>O,1,0);\n'
    'RG:=O>0 AND C>0 AND REF(C,1)>0;\n'
    'READY:=BARSCOUNT(C)>=262 AND REF(COUNT(RG,260),1)=260;\n'
    'RA01:IF(READY,100*(REF(SUM(RN,20),1)/20-REF(SUM(RN,240),21)/240),DRAWNULL);\n'
    'RA02:IF(READY,100*(REF(SUM(RP,20),1)/20-REF(SUM(RP,240),21)/240),DRAWNULL);\n')
NEW_EXPRESSIONS = {'RA01':'YJRC01.RA01#DAY','RA02':'YJRC01.RA02#DAY'}
EXPRESSIONS = {**inputs.previous.EXPRESSIONS,**NEW_EXPRESSIONS}
HEADER = inputs.previous.HEADER


def checked():
    p = json.loads(PROTOCOL.read_text())
    assert p['inputs_protocol_sha256'] == sha(inputs.PROTOCOL)
    assert p['helper_source'] == HELPER and p['expressions'] == NEW_EXPRESSIONS
    for file,digest in p['evidence'].items():
        assert sha(Path(file)) == digest
    assert p['all_features_and_validity_unchanged'] and not p['new_2026_prices_allowed']
    return p


def expression_to_python(expr):
    if expr.startswith('IF(') and expr.endswith(')'):
        args=[];start=3;depth=0
        for i in range(3,len(expr)-1):
            if expr[i]=='(':depth+=1
            elif expr[i]==')':depth-=1
            elif expr[i]==',' and depth==0:
                args.append(expr[start:i]);start=i+1
        args.append(expr[start:-1]);assert len(args)==3
        return 'IF('+','.join(expression_to_python(a) for a in args)+')'
    if ' AND ' in expr:
        return '('+') & ('.join(expression_to_python(a) for a in expr.split(' AND '))+')'
    return re.sub(r'(?<![<>=!])=(?!=)','==',expr)


def replay_helper(opens,closes):
    opens,closes = np.asarray(opens,float),np.asarray(closes,float)
    assert len(opens)==len(closes)
    env = dict(O=opens,C=closes,DRAWNULL=np.nan,IF=np.where,
        REF=lambda a,n:pd.Series(a).shift(int(n)).to_numpy(),
        SUM=lambda a,n:pd.Series(a).rolling(int(n),min_periods=int(n)).sum().to_numpy(),
        COUNT=lambda a,n:pd.Series(np.asarray(a,float)).rolling(int(n),min_periods=int(n)).sum().to_numpy(),
        BARSCOUNT=lambda a:np.arange(1,len(a)+1))
    for line in HELPER.splitlines():
        name,expr = line.rstrip(';').split(':=') if ':=' in line else line.rstrip(';').split(':')
        env[name] = eval(expression_to_python(expr),{'__builtins__':{}},env)
    return np.column_stack([env['RA01'],env['RA02']])


def native():
    checked();_,sources = inputs.checked_sources()
    v = json.loads((inputs.ROOT/'feature_verification.json').read_text())
    assert v['passed'] and v['feature_report_sha256'] == sha(inputs.ROOT/'feature_report.json')
    out = inputs.ROOT/'native_input_verification.json';assert not out.exists()
    f = pd.read_parquet(inputs.ROOT/'features.parquet');f = f.loc[f.formula_input_valid].copy()
    f['identity'] = [hashlib.sha256((d+'|'+s+'|reversal-change-native-v1').encode()).hexdigest() for d,s in zip(f.date,f.code)]
    samples = f.sort_values('identity').groupby('half',sort=True).head(8).sort_values(['date','code'])
    receipts=[]
    for row in samples.itertuples():
        path = DAILY/(row.code.replace('.','_')+'.parquet');assert sha(path)==sources[str(path)]
        d = pd.read_parquet(path,columns=['date','open','close','tradestatus','adjustflag'],
            filters=[('date','>=',row.rc_reference_date),('date','<',row.date)])
        d = d.loc[d.tradestatus.eq(1)].sort_values('date')
        assert len(d)==261 and d.adjustflag.eq(3).all() and d.date.iloc[-1]==row.rc_last_date
        o,c = np.r_[d.open.to_numpy(float),row.price_1449],np.r_[d.close.to_numpy(float),row.price_1449]
        values = replay_helper(o,c)
        assert np.isnan(values[:-1]).all(), 'Need 261 completed price days, not 260'
        np.testing.assert_allclose(values[-1],[row.RA01,row.RA02],rtol=0,atol=2e-12)
        enc=lambda a:np.floor(np.clip(100*np.asarray(a)+10000+.000001,0,999999))
        np.testing.assert_array_equal(enc(values[-1]),enc([row.RA01,row.RA02]))
        o[-1]=1000000;c[-1]=.01
        np.testing.assert_array_equal(replay_helper(o,c)[-1],values[-1])
        o[-1]=.01;c[-1]=1000000
        np.testing.assert_array_equal(replay_helper(o,c)[-1],values[-1])
        # Appending later bars must not change the value attached to today.
        extended=replay_helper(np.r_[o,17,19],np.r_[c,4,2])
        np.testing.assert_array_equal(extended[261],values[-1])
        receipts.append(dict(date=row.date,code=row.code,first_history_date=row.rc_reference_date,
            last_history_date=row.rc_last_date,prior_daily_rows=261,daily_source_sha256=sources[str(path)]))
    assert len(receipts)==32
    helper_path=inputs.ROOT/'YJRC01.tdx';assert not helper_path.exists();helper_path.write_text(HELPER)
    proof=dict(passed=True,protocol_sha256=sha(inputs.PROTOCOL),native_protocol_sha256=sha(PROTOCOL),
        feature_report_sha256=sha(inputs.ROOT/'feature_report.json'),feature_verification_sha256=sha(inputs.ROOT/'feature_verification.json'),
        helper_sha256=sha(helper_path),deployed_expressions=EXPRESSIONS,deployed_header=HEADER,
        samples=receipts,raw_prior_daily_rows=8352,actual_daily_ref_sum_count_and_readiness_rebuilt=True,
        current_and_future_daily_prices_do_not_change_current_output=True,minute_open_parity_failed_and_not_claimed=True,
        requires_aligned_active_daily_history=True,software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(out,proof)
    return {k:v for k,v in proof.items() if k not in ['samples','deployed_expressions','deployed_header']}


if __name__=='__main__':
    print(json.dumps(native(),ensure_ascii=False,indent=2))
