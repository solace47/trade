"""Replace only the scale of the original fifty inputs with prior daily ranges."""
import json
from pathlib import Path
import re
import subprocess

import numpy as np
import pandas as pd

from . import tail_formula_baseline as prior
from . import tail_formula_additive as base
from .research_io import save_json, sha, check_sources, check_runtime

STEM='tail_formula_intraday_scale'
ROOT=Path('data/research')/STEM
INPUTS=ROOT/'inputs'
PROTOCOL=Path('config')/(STEM+'_input_protocol.json')
INTENT=Path('config')/(STEM+'_intent.json')
META,CONTROL=prior.META,prior.CONTROL
CHANGED={n:'IR'+n for n,e in CONTROL.items() if '/VP20' in e or '/V01' in e or n=='V01'}
assert len(CHANGED)==26
EXPRESSIONS={CHANGED.get(n,n):('IV20' if n=='V01' else e.replace('/VP20','/IV20').replace('/V01','/IV20'))
             for n,e in CONTROL.items()}
EXTRA_HEADER='IRDAY:=HHV(H,B0)-LLV(L,B0);\nIRMEAN:=('+ '+'.join(f'REF(IRDAY,B{i})' for i in range(20))+')/20;\n'
EXTRA_HEADER+='IV20:=100*IRMEAN/DYNAINFO(3);\nIRREADY:=IRMEAN>0 AND IV20>0 AND BARSCOUNT(C)>B19;\n'
HEADER=prior.HEADER+EXTRA_HEADER
NATIVE_GATE='IRREADY'


def checked():
    check_runtime()
    p=json.loads(PROTOCOL.read_text())
    assert subprocess.check_output(['git','show',f'HEAD:{PROTOCOL}'])==PROTOCOL.read_bytes()
    assert p['intent_sha256']==sha(INTENT) and p['arms']=={'control':CONTROL,'memory':EXPRESSIONS}
    assert p['native_header']==HEADER and p['changed_fields']==CHANGED
    assert p['expected_keys']==1815129 and p['expected_original_valid']==1602413
    assert p['newly_invalid_allowed']==0 and p['maximum_new_fits']==4
    assert not p['new_2026_prices_allowed']
    check_sources(p['source_hashes'])
    a=json.loads(Path(p['conditional_audit']).read_text())
    assert a['passed'] and a['original_valid_v01_mismatch']==a['original_valid_reference_unknown']==0
    v=json.loads(Path(p['reference_verification']).read_text())
    assert v['passed'] and v['all_1602413_valid_current_preclose_equals_prior_actual_close']
    return p


def daily_ranges(d):
    """Keep bad active rows; use strictly prior twenty completed stock days."""
    d=d.loc[d.tradestatus.eq(1)].sort_values('date').reset_index(drop=True).copy()
    assert not d.date.duplicated().any()
    prices=d[['open','high','low','close']].to_numpy(float)
    good=(np.isfinite(prices).all(axis=1)&(prices>0).all(axis=1)&d.adjustflag.eq(3)
          &d.volume.gt(0)&(d.high+.0001>=prices.max(axis=1))&(d.low-.0001<=prices.min(axis=1))
          &(np.abs(prices-np.round(prices,2))<=.0001).all(axis=1))
    d['ir_mean']=(d.high-d.low).rolling(20,min_periods=20).mean().shift()
    d['ir_literal_mean']=sum((d.high-d.low).shift(i) for i in range(1,21))/20
    d['ir_good']=good.astype(int).rolling(21,min_periods=21).sum().shift().eq(21)
    d['ir_first_date']=d.date.shift(20)
    d['ir_last_date']=d.date.shift()
    d['ir_prior_close']=d.close.shift()
    return d


def transform(old, primitive):
    d=old[['date','code','formula_input_valid','V01']].merge(primitive,on=['date','code'],how='left',validate='one_to_one')
    iv20=100*d.ir_mean/d.ir_prior_close
    good=d.ir_good.fillna(False)&np.isfinite(iv20)&iv20.gt(0)&np.isfinite(d.V01)&d.V01.gt(0)
    f=old.copy()
    for n,new in CHANGED.items():
        f[new]=(iv20 if n=='V01' else old[n]*d.V01/iv20).where(good)
    good&=np.isfinite(f[list(EXPRESSIONS)]).all(axis=1)
    return f,good


def prepare():
    p=checked();assert not (INPUTS/'feature_report.json').exists()
    INPUTS.mkdir(parents=True,exist_ok=True)
    old=prior.original();sources={}
    for spec in p['daily_manifests']:
        for file,digest in json.loads(Path(spec['file']).read_text())[spec['field']].items():
            assert file not in sources or sources[file]==digest
            sources[file]=digest
    raw=[];primitives=[];receipts={}
    for i,code in enumerate(sorted(old.code.unique()),1):
        file=Path(p['daily_template'].format(code=code.replace('.','_')))
        assert sha(file)==sources[str(file)];receipts[str(file)]=sources[str(file)]
        d=pd.read_parquet(file,columns=['date','code','open','high','low','close','volume','tradestatus','adjustflag'],
            filters=[('date','>=',p['warmup_first']),('date','<','2026-01-01')])
        assert d.code.eq(code).all() and d.date.lt('2026-01-01').all()
        raw.append(d);v=daily_ranges(d)
        primitives.append(v[['date','code','ir_mean','ir_literal_mean','ir_good','ir_first_date','ir_last_date','ir_prior_close']])
        if i%400==0:print(json.dumps(dict(daily_files=i,total=len(old.code.unique()))),flush=True)
    raw=pd.concat(raw,ignore_index=True)
    direct=pd.concat(primitives).sort_values(['date','code']).reset_index(drop=True)
    c=base.conn();c.register('raw_daily',raw)
    rebuilt=c.sql('''WITH d AS(SELECT *,
        isfinite(open) AND isfinite(high) AND isfinite(low) AND isfinite(close)
        AND least(open,high,low,close)>0 AND adjustflag=3 AND volume>0
        AND high+.0001>=greatest(open,high,low,close) AND low-.0001<=least(open,high,low,close)
        AND abs(open-round(open,2))<=.0001 AND abs(high-round(high,2))<=.0001
        AND abs(low-round(low,2))<=.0001 AND abs(close-round(close,2))<=.0001 AS good
        FROM raw_daily WHERE tradestatus=1),v AS(SELECT date,code,
        count(*) OVER w AS n,avg(high-low) OVER w AS mean,
        count(*) OVER q=21 AND sum(good::INT) OVER q=21 AS ir_good,
        lag(date,20) OVER a AS ir_first_date,lag(date) OVER a AS ir_last_date,
        lag(close) OVER a AS ir_prior_close FROM d
        WINDOW w AS(PARTITION BY code ORDER BY date ROWS BETWEEN 20 PRECEDING AND 1 PRECEDING),
        q AS(PARTITION BY code ORDER BY date ROWS BETWEEN 21 PRECEDING AND 1 PRECEDING),
        a AS(PARTITION BY code ORDER BY date))
        SELECT date,code,CASE WHEN n=20 THEN mean END AS ir_mean,ir_good,
        ir_first_date,ir_last_date,ir_prior_close FROM v ORDER BY date,code''').df()
    pd.testing.assert_frame_equal(direct[['date','code','ir_good']],rebuilt[['date','code','ir_good']],check_exact=True)
    for n in ['ir_mean','ir_prior_close']:
        np.testing.assert_allclose(direct[n],rebuilt[n],rtol=0,atol=2e-10,equal_nan=True)
    np.testing.assert_allclose(direct.ir_mean,direct.ir_literal_mean,rtol=0,atol=2e-10,equal_nan=True)
    for n in ['ir_first_date','ir_last_date']:
        assert direct[n].fillna('').equals(rebuilt[n].fillna(''))
    primitive=old[['date','code']].merge(direct,on=['date','code'],how='left',validate='one_to_one')
    f,good=transform(old,primitive)
    invalid=int((old.formula_input_valid&~good).sum())
    audit=dict(passed=invalid==0,rows=len(f),original_valid=int(old.formula_input_valid.sum()),
        new_valid=int((old.formula_input_valid&good).sum()),newly_invalid=invalid,stop_fit_if_domain_changes=True,
        no_new_economics_read=True,new_2026_prices_read=False)
    save_json(INPUTS/'input_domain_audit.json',audit)
    assert invalid==0,'Freeze a matched-domain protocol before any fits'
    assert primitive.loc[old.formula_input_valid,'ir_last_date'].lt(old.loc[old.formula_input_valid,'date']).all()
    f['prior_formula_input_valid']=old.formula_input_valid
    f['formula_input_valid']&=good
    pd.testing.assert_frame_equal(f[[*META,*CONTROL]],old,check_exact=True)
    # Source-domain verification before any encoded or model result.
    c.register('f',f);c.register('primitives',primitive)
    names=list(EXPRESSIONS)
    terms={CHANGED.get(n,n):('100*ir_mean/ir_prior_close' if n=='V01' else
        f'f.{n}*f.V01/(100*ir_mean/ir_prior_close)' if n in CHANGED else f'f.{n}') for n in CONTROL}
    columns=','.join(f'floor(least(greatest(100*({terms[n]})+10000+.000001,0),999999))::INT AS {n}' for n in names)
    independent=c.sql('SELECT date,code,'+columns+' FROM f JOIN primitives USING(date,code) WHERE formula_input_valid ORDER BY date,code').df()
    actual=np.floor(np.clip(100*f.loc[f.formula_input_valid,names].to_numpy(float)+10000+.000001,0,999999)).astype('int32')
    np.testing.assert_array_equal(actual,independent[names].to_numpy(dtype='int32'));c.close()
    literal=primitive.copy();literal['ir_mean']=literal.ir_literal_mean
    native,native_good=transform(old,literal)
    np.testing.assert_array_equal(good,native_good)
    native_encoded=np.floor(np.clip(100*native.loc[f.formula_input_valid,names].to_numpy(float)+10000+.000001,0,999999)).astype('int32')
    np.testing.assert_array_equal(actual,native_encoded)
    declarations=re.findall(r'(?m)^([A-Z][A-Z0-9]*):=',HEADER)+list(EXPRESSIONS)
    assert len(declarations)==len({n.casefold() for n in declarations})
    f.to_parquet(INPUTS/'features.parquet',index=False,compression='zstd')
    primitive.to_parquet(INPUTS/'primitives.parquet',index=False,compression='zstd')
    for name in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        assert p['label_projection_root']=='data/research/tail_formula_retained_interval/inputs'
        target=INPUTS/name;assert not target.exists();target.symlink_to(Path('../../tail_formula_retained_interval/inputs')/name)
    adapter=INPUTS/'intraday_scale_inputs.tdx';adapter.write_text(EXTRA_HEADER+''.join(f'{n}:={e};\n' for n,e in EXPRESSIONS.items()))
    report=dict(protocol_sha256=sha(PROTOCOL),implementation_sha256=sha(Path(__file__)),
        features_sha256=sha(INPUTS/'features.parquet'),primitives_sha256=sha(INPUTS/'primitives.parquet'),
        rows=len(f),valid=int(f.formula_input_valid.sum()),newly_invalid=0,arms_width=[50,50],changed_fields=CHANGED,
        expressions=EXPRESSIONS,native_header=HEADER,source_hashes=p['source_hashes'],daily_source_sha256=receipts,
        adapter_sha256=sha(adapter),all_original_fifty_metadata_and_validity_unchanged=True,
        independent_pandas_SQL_daily_windows_and_all_encodings_equal=True,
        source_values_from_prior_completed_stock_days_only=True,new_group_outcomes_read=False,
        new_2026_prices_read=False,no_exit_rules=True,software_compilation_verified=False,native_source_parity_verified=False)
    save_json(INPUTS/'feature_report.json',report)
    proof=dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),
        original_keys_values_metadata_validity_equal=True,changed_fields=26,both_arms_width_50=True,
        independent_source_windows_and_encoding_equal=True,no_new_group_economics=True,new_2026_prices_read=False)
    save_json(INPUTS/'feature_verification.json',proof)
    save_json(INPUTS/'native_input_verification.json',dict(proof,all_literal_twenty_prior_day_terms_equal=True,
        mathematical_replay_only=True,software_compilation_verified=False,native_source_parity_verified=False))
    return dict(feature_report_sha256=sha(INPUTS/'feature_report.json'),**audit)
