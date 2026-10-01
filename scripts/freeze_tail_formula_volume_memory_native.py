"""Correct ROUND arity without replacing pinned input/model receipts."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from trade_research import tail_formula_volume_memory as study
from trade_research.corporate_cash import save_json,sha
import run_tail_formula_volume_memory as runner

PROTOCOL=Path('config/tail_formula_volume_memory_native_erratum.json')
ROOT=study.ROOT/'native'
HELPER=study.HELPER.replace('ROUND(C*100,0)','ROUND(C*100)')


def replay(closes,volumes):
    def shift(x,n): return pd.Series(x).shift(int(n)).to_numpy()
    def count(x,n): return pd.Series(np.asarray(x,float)).rolling(int(n),min_periods=int(n)).sum().to_numpy()
    def product(x,n):
        out=np.full(len(x),np.nan)
        if len(x)>=n: out[n-1:]=np.prod(sliding_window_view(x,n),axis=1)
        return out
    def dma(x,a):
        out=np.zeros(len(x)); previous=x[0]
        for i in range(len(x)):
            previous=a[i]*x[i]+(1-a[i])*previous; out[i]=previous
        return out
    env=dict(C=np.asarray(closes,float),V=np.asarray(volumes,float),DRAWNULL=np.nan,
        ROUND=lambda x:np.floor(x+.5),ABS=np.abs,IF=np.where,REF=shift,COUNT=count,SUM=count,
        MULAR=product,DMA=dma,BARSCOUNT=lambda x:np.arange(1,len(x)+1))
    with np.errstate(all='ignore'):
        for line in HELPER.splitlines():
            name,expr=line.rstrip(';').split(':=') if ':=' in line else line.rstrip(';').split(':')
            if name=='PG':
                assert expr=='C>0 AND ABS(C*100-ROUND(C*100))<=0.01'
                expr='(C>0) & (ABS(C*100-ROUND(C*100))<=0.01)'
            if name=='READY':
                assert expr=='BARSCOUNT(C)>=80 AND REF(COUNT(PG AND VG,79),1)=79'
                expr='(BARSCOUNT(C)>=80) & (REF(COUNT(PG & VG,79),1)==79)'
            if name=='TA':
                assert expr=='IF(COUNT(VG,20)=20,V/SUM(V,20),0.5)'
                expr='IF(COUNT(VG,20)==20,V/SUM(V,20),0.5)'
            env[name]=eval(expr,{'__builtins__':{}},env)
    return env['AN60']


def checked():
    p=json.loads(PROTOCOL.read_text())
    assert p['input_protocol_sha256']==sha(study.PROTOCOL) and p['new_helper']==HELPER
    assert study.HELPER.count('ROUND(C*100,0)')==2 and 'ROUND(C*100,0)' not in HELPER
    for file,digest in p['source_hashes'].items(): assert sha(Path(file))==digest,file
    return p


def verify():
    checked(); ROOT.mkdir(parents=True,exist_ok=True); assert not (ROOT/'verification.json').exists()
    p=study.checked(); d=study.daily(p)
    states=pd.read_parquet(study.INPUTS/'states.parquet')
    pieces=[]; future_checks=0
    for code,g in d.groupby('code',sort=False):
        closes=g.close.to_numpy(); volumes=g.volume.to_numpy()
        x=np.r_[closes,1e6]; v=np.r_[volumes,1e12]
        actual=replay(x,v)[1:]
        # Current partial/dummy bar changes cannot enter the prior output.
        if len(g)>=79 and g.good.iloc[-79:].all():
            second=replay(np.r_[closes,.01],np.r_[volumes,1.])[1:]
            np.testing.assert_array_equal(actual,second)
            future_checks+=1
        pieces.append(pd.DataFrame(dict(code=code,history_date=g.date,export_anchor=actual)))
    literal=pd.concat(pieces,ignore_index=True)
    pd.testing.assert_frame_equal(states[['code','history_date']],literal[['code','history_date']],check_exact=True)
    eligible=states.history_input_valid
    np.testing.assert_allclose(literal.loc[eligible,'export_anchor'],states.loc[eligible,'anchor'],rtol=0,atol=2e-9)
    h=study.map_states(pd.read_parquet(study.INPUTS/'history.parquet',columns=['date','code']),
        literal.rename(columns={'export_anchor':'anchor'}).assign(history_input_valid=eligible.to_numpy()))
    f=pd.read_parquet(study.INPUTS/'features.parquet',columns=['date','code','A04','V01','VM01','history_input_valid'])
    expected=(100*(f.A04/h.anchor-1)/f.V01).where(f.history_input_valid)
    np.testing.assert_allclose(f.VM01,expected,rtol=0,atol=2e-8,equal_nan=True)
    enc=lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999))
    finite=np.isfinite(expected)
    np.testing.assert_array_equal(enc(f.loc[finite,'VM01']),enc(expected[finite]))
    helper=ROOT/'YJVMA01.tdx'; helper.write_text(HELPER)
    save_json(ROOT/'verification.json',dict(passed=True,protocol_sha256=sha(PROTOCOL),
        corrected_helper_sha256=sha(helper),original_feature_verification_sha256=sha(study.INPUTS/'feature_verification.json'),
        source_states=len(states),eligible_states=int(eligible.sum()),feature_rows=len(f),
        literal_corrected_export_all_source_eligible_states_and_integer_encodings_equal=True,
        final_current_bar_perturbation_source_stocks=future_checks,only_round_arity_changed=True,
        native_source_quality_guard_equivalence_not_claimed=True,
        source_adjustflag_and_share_integrality_remain_provider_quality_requirements=True,
        zero_new_fits_scores_or_economic_group_aggregations=True,new_2026_prices_read=False,
        software_compilation_verified=False,native_source_parity_verified=False,no_exit_rules=True))
    return dict(proof_sha256=sha(ROOT/'verification.json'),eligible_states=int(eligible.sum()),source_stocks=future_checks)


def freeze():
    checked(); v=json.loads((ROOT/'verification.json').read_text())
    assert v['passed'] and v['protocol_sha256']==sha(PROTOCOL)
    assert v['corrected_helper_sha256']==sha(ROOT/'YJVMA01.tdx')
    result=runner.freeze(); path=study.ROOT/'joint_selection_freeze.json'; joint=json.loads(path.read_text())
    for file in [PROTOCOL,ROOT/'verification.json',ROOT/'YJVMA01.tdx',Path(__file__)]:
        joint['source_hashes'][str(file)]=sha(file)
    joint.update(native_round_arity_corrected_before_first_group_evaluation=True,
        current_native_daily_helper=str(ROOT/'YJVMA01.tdx'),all_original_input_and_model_receipts_unchanged=True)
    save_json(path,joint); result['joint_sha256']=sha(path)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['verify','freeze'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2),flush=True)
