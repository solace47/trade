"""Late index position relative to the completed 09:31--11:30 quote range."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import tail_formula_additive as base
from . import tail_formula_morning_range as prior
from . import tail_formula_index_morning as early
from .corporate_cash import save_json, sha

STEM = 'tail_formula_index_morning_range'
ROOT = Path('data/research') / STEM
INPUTS = ROOT / 'inputs'
PROTOCOL = Path('config') / (STEM + '_input_protocol.json')
INTENT = Path('config') / (STEM + '_intent.json')
META = prior.META
EXTRA_HEADER = '''IRDT:=VALUEWHEN(TIME=1130,DATE);
IRCLK:=VALUEWHEN(TIME=1130,COUNT(TIME>=931 AND TIME<=1130,120));
IRGD:=VALUEWHEN(TIME=1130,COUNT(INDEXC>0,120));
IRHI:=VALUEWHEN(TIME=1130,HHV(INDEXC,120));
IRLO:=VALUEWHEN(TIME=1130,LLV(INDEXC,120));
IRTD:=VALUEWHEN(TIME=1448,DATE);
IRREADY:=AMREADY AND RTREADY AND IRDT=DATE AND IRTD=DATE AND IRCLK=120 AND IRGD=120 AND IRHI>=IRLO AND IRLO>0 AND IX1448>0;
'''
HEADER = prior.HEADER + EXTRA_HEADER
CORE_GATE = 'IRREADY'
NEW_EXPRESSIONS = {'IMH': 'IF(IRREADY,100*(IX1448/IRHI-1),DRAWNULL)',
                   'IML': 'IF(IRREADY,100*(IX1448/IRLO-1),DRAWNULL)'}
ARMS = {'control': prior.EXPRESSIONS, 'range': {**prior.EXPRESSIONS, **NEW_EXPRESSIONS}}


def checked():
    early.checked(); p = json.loads(PROTOCOL.read_text())
    assert p['intent_sha256'] == sha(INTENT) and p['arms'] == ARMS and p['native_header'] == HEADER
    assert p['expected_keys'] == 1258085 and p['expected_valid'] == 1117397
    assert p['morning_quote_sequences'] == [0,119] and p['tail_quote_sequence'] == 227
    assert p['no_new_raw_extraction'] and not p['new_2026_prices_allowed']
    for file, digest in p['source_hashes'].items(): assert sha(Path(file)) == digest, file
    complete = json.loads(Path(p['conditional_completion']).read_text())
    gate = json.loads(Path(p['conditional_gate']).read_text())
    assert complete['passed'] and gate['passed'] and not gate['supports_2024_extension']
    return p


def quote_range(rows):
    prefix = rows[:228]
    good = early.source.points(prefix)['prefix_valid']
    good &= all(np.isfinite(row['price_raw']) for row in prefix)
    morning = [row['price_raw']/100 for row in prefix[:120]] if good else [np.nan]
    return dict(ir_prefix_valid=good, ir_high=max(morning), ir_low=min(morning),
                ir_last=prefix[227]['price_raw']/100 if good else np.nan)


def measure(last, high, low, parent_valid):
    assert last.shape == high.shape == low.shape == parent_valid.shape
    good = parent_valid.copy() & np.isfinite(last) & np.isfinite(high) & np.isfinite(low)
    good &= (last > 0) & (low > 0) & (high >= low)
    with np.errstate(all='ignore'):
        return dict(index_morning_range_valid=good,
            IMH=np.where(good,100*(last/high-1),np.nan),
            IML=np.where(good,100*(last/low-1),np.nan))


def features():
    checked(); INPUTS.mkdir(parents=True,exist_ok=True)
    assert not (INPUTS/'feature_report.json').exists()
    old = pd.read_parquet(prior.INPUTS/'features.parquet')
    context = pd.read_parquet(early.INPUTS/'features.parquet',columns=[
        'date','code','index_code','index_prior_close','prefix_valid','iq_original_J01'])
    points = [dict(date=item['date'],index_code=item['symbol'],**quote_range(rows))
              for item,rows in early.raw_sessions()]
    f=old.merge(context,on=['date','code'],how='left',validate='one_to_one')
    f=f.merge(pd.DataFrame(points),on=['date','index_code'],how='left',validate='many_to_one',sort=False)
    pd.testing.assert_frame_equal(f[old.columns],old,check_exact=True)
    np.testing.assert_array_equal(f.J01,f.iq_original_J01)
    np.testing.assert_array_equal(f.index_code,np.where(f.code.str.startswith('sh.'),'sh.000001','sz.399001'))
    np.testing.assert_array_equal(f.prefix_valid,f.ir_prefix_valid)
    result=measure(f.ir_last.to_numpy(float),f.ir_high.to_numpy(float),f.ir_low.to_numpy(float),f.ir_prefix_valid.eq(True).to_numpy())
    for name,value in result.items(): f[name]=value
    f['formula_input_valid'] &= f.index_morning_range_valid
    np.testing.assert_array_equal(f.formula_input_valid,old.formula_input_valid)
    assert len(f)==1258085 and f.formula_input_valid.sum()==1117397
    f.to_parquet(INPUTS/'features.parquet',index=False,compression='zstd')
    r=dict(protocol_sha256=sha(PROTOCOL),features_sha256=sha(INPUTS/'features.parquet'),rows=len(f),
        valid=int(f.formula_input_valid.sum()),parent_feature_report_sha256=sha(prior.INPUTS/'feature_report.json'),
        all_original_50_values_metadata_keys_and_effective_domain_unchanged=True,
        morning_quote_count=120,expressions=ARMS['range'],native_header=HEADER,
        no_new_raw_extraction=True,no_future_index_suffix_interpreted=True,new_group_outcomes_read=False,
        new_2026_prices_read=False,no_exit_rules=True,software_compilation_verified=False,native_source_parity_verified=False)
    save_json(INPUTS/'feature_report.json',r)
    for name in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        (INPUTS/name).symlink_to((prior.INPUTS/name).resolve())
    return {k:r[k] for k in ['rows','valid','all_original_50_values_metadata_keys_and_effective_domain_unchanged']}


def verify():
    checked(); r=json.loads((INPUTS/'feature_report.json').read_text())
    assert r['protocol_sha256']==sha(PROTOCOL) and r['features_sha256']==sha(INPUTS/'features.parquet')
    f=pd.read_parquet(INPUTS/'features.parquet'); old=pd.read_parquet(prior.INPUTS/'features.parquet')
    pd.testing.assert_frame_equal(f[old.columns],old,check_exact=True)
    contexts=pd.read_parquet(early.INPUTS/'features.parquet',columns=[
        'date','code','index_code','index_prior_close','prefix_valid','iq_original_J01'])
    atoms,sessions=[],[]
    for item,rows in early.raw_sessions():
        sessions.append(dict(date=item['date'],index_code=item['symbol']))
        atoms.extend(dict(date=item['date'],index_code=item['symbol'],ordinal=i,
            sequence=row['sequence'],price_raw=row['price_raw']) for i,row in enumerate(rows))
    c=base.conn();c.register('atoms',pd.DataFrame(atoms));c.register('sessions',pd.DataFrame(sessions))
    c.register('parent',old);c.register('contexts',contexts)
    c.sql('''WITH p AS(SELECT date,index_code,count(*)=228 AND
        bool_and(ordinal=sequence AND isfinite(price_raw) AND price_raw>0) AS good,
        max(price_raw) FILTER(WHERE ordinal<120)/100. AS high,
        min(price_raw) FILTER(WHERE ordinal<120)/100. AS low,
        max(price_raw) FILTER(WHERE ordinal=227)/100. AS last
        FROM atoms GROUP BY date,index_code) SELECT s.*,coalesce(good,false) AS good,high,low,last
        FROM sessions s LEFT JOIN p USING(date,index_code)''').create_view('points')
    d=c.sql('''WITH z AS(SELECT f.date,f.code,c.index_prior_close,c.iq_original_J01,
        c.prefix_valid,p.good,c.index_code,p.high,p.low,p.last,
        coalesce(p.good AND isfinite(high) AND isfinite(low) AND isfinite(last)
            AND high>=low AND low>0 AND last>0,false) AS ok
        FROM parent f LEFT JOIN contexts c USING(date,code)
        LEFT JOIN points p ON f.date=p.date AND c.index_code=p.index_code)
        SELECT *,CASE WHEN ok THEN 100.*(last/high-1) END AS IMH,
        CASE WHEN ok THEN 100.*(last/low-1) END AS IML FROM z ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(f[['date','code']],d[['date','code']],check_exact=True)
    np.testing.assert_array_equal(f.index_code,d.index_code)
    np.testing.assert_array_equal(f.index_prior_close,d.index_prior_close)
    np.testing.assert_array_equal(f.J01,d.iq_original_J01)
    np.testing.assert_array_equal(d.prefix_valid,d.good)
    np.testing.assert_array_equal(f.index_morning_range_valid,d.ok)
    maximum=0.;encode=lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999))
    for a,b in [('ir_last','last'),('ir_high','high'),('ir_low','low')]:
        np.testing.assert_allclose(f[a],d[b].where(d.good),atol=2e-12,rtol=0,equal_nan=True)
    for name in NEW_EXPRESSIONS:
        np.testing.assert_allclose(f[name],d[name],atol=2e-12,rtol=0,equal_nan=True)
        np.testing.assert_array_equal(encode(f.loc[d.ok,name]),encode(d.loc[d.ok,name]))
        maximum=max(maximum,float((f[name]-d[name]).abs().max()))
    np.testing.assert_array_equal(f.formula_input_valid,old.formula_input_valid & d.ok)
    out=dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),rows=len(f),
        valid=int(f.formula_input_valid.sum()),max_difference=maximum,
        all_original_50_values_metadata_and_effective_domain_unchanged=True,
        all_prefix_quality_morning_extrema_tail_points_references_values_and_codes_sql_rebuilt=True,
        effective_input_intersection_unchanged=True,domain_reference=str(prior.INPUTS),
        no_future_index_suffix_interpreted=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'feature_verification.json',out);return out


def native():
    checked();proof=json.loads((INPUTS/'feature_verification.json').read_text())
    assert proof['passed'] and proof['feature_report_sha256']==sha(INPUTS/'feature_report.json')
    f=pd.read_parquet(INPUTS/'features.parquet');good=f.index_morning_range_valid
    for name,divisor in [('IMH',f.ir_high),('IML',f.ir_low)]:
        expected=(100*(f.ir_last/divisor-1)).where(good)
        np.testing.assert_allclose(f[name],expected,atol=2e-12,rtol=0,equal_nan=True)
        enc=lambda x:np.floor(np.clip(100*x+10000+.000001,0,999999))
        np.testing.assert_array_equal(enc(f.loc[good,name]),enc(expected.loc[good]))
    assert 'HHV(INDEXC,120)' in EXTRA_HEADER and 'LLV(INDEXC,120)' in EXTRA_HEADER
    identity=json.loads((prior.INPUTS/'native_input_verification.json').read_text())['source_samples']
    keys=list(dict.fromkeys((x['date'],x['code']) for x in identity));indexed=f.set_index(['date','code'])
    sessions={(item['date'],item['symbol']):(item,rows) for item,rows in early.raw_sessions()};cases=[]
    for day,code in keys:
        row=indexed.loc[(day,code)];item,rows=sessions[(day,row.index_code)]
        # A stock has 121 morning bars including 09:30; this index window omits that first bar.
        clock=pd.date_range(day+' 09:31',day+' 11:30',freq='min').strftime('%H%M').astype(int)
        assert len(clock)==120 and clock[0]==931 and clock[-1]==1130
        ready=early.source.points(rows)['prefix_valid'] and all(np.isfinite(q['price_raw']) for q in rows)
        assert ready==row.index_morning_range_valid
        prices=[q['price_raw']/100 for q in rows[:120]] if ready else [np.nan]
        high,low,last=(max(prices),min(prices),rows[227]['price_raw']/100) if ready else (np.nan,np.nan,np.nan)
        got=[100*(last/high-1),100*(last/low-1)] if ready else [np.nan,np.nan]
        np.testing.assert_allclose(got,row[list(NEW_EXPRESSIONS)].to_numpy(float),atol=2e-12,rtol=0)
        cases.append(dict(date=day,code=code,index_code=row.index_code,index_source_sha256=item['sha256'],
            existing_stock_source_identity_reused=True,literal_120_quote_range_and_tail_point_replayed=True))
    helper=INPUTS/'native_inputs.tdx';helper.write_text(HEADER+'\n'.join(f'{k}:={v};' for k,v in NEW_EXPRESSIONS.items())+'\n')
    out=dict(passed=True,feature_report_sha256=sha(INPUTS/'feature_report.json'),helper_sha256=sha(helper),
        feature_verification_sha256=sha(INPUTS/'feature_verification.json'),samples=cases,
        all_new_full_literal_values_and_codes_checked=True,fixed_old_case_identities_only=True,
        no_future_index_suffix_interpreted=True,no_new_raw_extraction=True,
        software_compilation_verified=False,native_source_parity_verified=False,
        new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(INPUTS/'native_input_verification.json',out);return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['features','verify','native']);args=p.parse_args()
    print(json.dumps(globals()[args.stage](),ensure_ascii=False,indent=2))
