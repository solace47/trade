"""Source-fixed trend/pullback conditions, with finite history and 14:49 inputs."""
import argparse
import json
from functools import lru_cache
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pandas as pd

from trade_research import tail_formula_baseline as baseline
from trade_research import tail_formula_additive as numeric
from trade_research.research_io import check_runtime, check_sources, save_json, sha
from tail_formula_reports import checked_selection, checked_analysis
import finish_tail_formula_rule_search as shared
import finish_tail_formula_profit_rule_search as comparisons

ROOT = Path('data/research/tail_formula_trend_reversion_literal')
PROTOCOL = Path('config/tail_formula_trend_reversion_literal_protocol.json')
KEYS = shared.KEYS
RAW = ['price_1449', 'daily_open', 'high_1449', 'low_1449', 'volume_1449', 'amount_1449']
ATOMS = ['prior_date', 'day60_date', 'nprior', 'close_good60', 'volume_good19',
         'sum19c', 'sum59c', 'p1c', 'p5c', 'p60c', 'sum19v']


@lru_cache(maxsize=1)
def checked():
    check_runtime()
    assert subprocess.check_output(['git', 'show', f'HEAD:{PROTOCOL}']) == PROTOCOL.read_bytes()
    assert sha(PROTOCOL) in subprocess.check_output(['git', 'show', 'HEAD:docs/selection-formula.md']).decode()
    p = json.loads(PROTOCOL.read_text()); check_sources(p['source_hashes'])
    assert p['selector_fits'] == p['new_tree_fits'] == 0 and not p['new_2026_prices_allowed']
    registry = json.loads(Path(p['prior_registry']).read_text())
    e = dict(p, input_protocol_sha256=sha(PROTOCOL),
             prior_analysis_roots=registry['prior_analysis_roots'],
             prior_completion_manifests=registry['prior_completion_manifests'])
    return p, e


def visible():
    p, _ = checked()
    f = baseline.original().loc[lambda x: x.date.ge('2024-01-01'), [*KEYS, 'formula_input_valid']].reset_index(drop=True)
    a = pd.read_parquet(p['visible_cache'], columns=[*KEYS, 'formula_input_valid', *RAW])
    pd.testing.assert_frame_equal(f, a[[*KEYS, 'formula_input_valid']], check_exact=True)
    assert len(f)==1258085 and f.date.lt('2026-01-01').all()
    return a.rename(columns={'formula_input_valid':'original_input_valid'})


def prepare():
    p, _ = checked(); assert not (ROOT/'input_report.json').exists()
    f = visible()
    d = pd.read_parquet(p['daily_cache'], columns=['date','code','close','volume','adjustflag','tradestatus'],
                        filters=[('date','>=','2023-01-01'),('date','<','2026-01-01')])
    assert not d.duplicated(['date','code']).any() and d.date.lt('2026-01-01').all()
    d = d.loc[d.tradestatus.eq(1)].sort_values(['code','date']).reset_index(drop=True)
    groups = {code:g.reset_index(drop=True) for code,g in d.groupby('code',sort=False)}
    parts = []
    for code,k in f[['date','code']].groupby('code',sort=False):
        g = groups[code]; n = np.searchsorted(g.date.to_numpy(),k.date.to_numpy(),side='left')
        assert (n>0).all()
        safe_close=g.close.where(np.isfinite(g.close),0)
        cc = np.floor(safe_close.to_numpy()*100+.5).astype('int64')
        cg = np.isfinite(g.close)&g.close.gt(0)&g.adjustflag.eq(3)&(abs(g.close*100-cc)<=.01)
        vg = np.isfinite(g.volume)&g.volume.ge(0)&g.volume.eq(np.floor(g.volume))&g.adjustflag.eq(3)
        cs = np.r_[0,np.cumsum(cc)]; vs = np.r_[0,np.cumsum(g.volume.where(np.isfinite(g.volume),0).to_numpy())]
        cs_good = np.r_[0,np.cumsum(cg.to_numpy())]; vs_good = np.r_[0,np.cumsum(vg.to_numpy())]
        out=k.copy(); out['nprior']=n
        out['prior_date']=g.date.iloc[n-1].to_numpy()
        out['day60_date']=np.where(n>=60,g.date.iloc[np.maximum(n-60,0)].to_numpy(),None)
        out['close_good60']=cs_good[n]-cs_good[np.maximum(n-60,0)]
        out['volume_good19']=vs_good[n]-vs_good[np.maximum(n-19,0)]
        for m in [19,59]:out['sum'+str(m)+'c']=cs[n]-cs[np.maximum(n-m,0)]
        for m in [1,5,60]:out['p'+str(m)+'c']=cc[np.maximum(n-m,0)]
        out['sum19v']=vs[n]-vs[np.maximum(n-19,0)]
        parts.append(out)
    atoms=pd.concat(parts,ignore_index=True).sort_values(['date','code']).reset_index(drop=True)
    c=numeric.conn();c.register('daily',d);c.register('keys',f[['date','code']])
    expected=c.sql('''WITH clean AS(SELECT *,CASE WHEN isfinite(close) THEN round(close*100)::BIGINT ELSE 0 END AS cc,
        CASE WHEN isfinite(volume) THEN volume ELSE 0 END AS vv FROM daily),
        a AS(SELECT date AS prior_date,code,cc AS p1c,
        row_number() OVER w AS nprior,lag(date,59) OVER w AS day60_date,
        lag(cc,4) OVER w AS p5c,lag(cc,59) OVER w AS p60c,
        sum(cc) OVER(PARTITION BY code ORDER BY date ROWS BETWEEN 18 PRECEDING AND CURRENT ROW) AS sum19c,
        sum(cc) OVER(PARTITION BY code ORDER BY date ROWS BETWEEN 58 PRECEDING AND CURRENT ROW) AS sum59c,
        sum(vv) OVER(PARTITION BY code ORDER BY date ROWS BETWEEN 18 PRECEDING AND CURRENT ROW) AS sum19v,
        count(*) FILTER(WHERE isfinite(close) AND close>0 AND adjustflag=3
            AND abs(close*100-round(close*100))<=.01)
            OVER(PARTITION BY code ORDER BY date ROWS BETWEEN 59 PRECEDING AND CURRENT ROW) AS close_good60,
        count(*) FILTER(WHERE isfinite(volume) AND volume>=0 AND volume=floor(volume) AND adjustflag=3)
            OVER(PARTITION BY code ORDER BY date ROWS BETWEEN 18 PRECEDING AND CURRENT ROW) AS volume_good19
        FROM clean WINDOW w AS(PARTITION BY code ORDER BY date))
        SELECT k.date,k.code,a.* EXCLUDE(code) FROM keys k ASOF LEFT JOIN a
        ON k.code=a.code AND k.date>a.prior_date ORDER BY k.date,k.code''').df();c.close()
    # Early incomplete histories are not inputs; never shift past a bad row.
    ready=atoms.nprior.ge(60)
    pd.testing.assert_frame_equal(atoms[['date','code']],expected[['date','code']],check_exact=True)
    pd.testing.assert_frame_equal(atoms.loc[ready,['prior_date','day60_date']],expected.loc[ready,['prior_date','day60_date']],check_exact=True)
    for name in ATOMS[2:]:np.testing.assert_allclose(atoms.loc[ready,name],expected.loc[ready,name],rtol=0,atol=1e-6 if name=='sum19v' else 0)
    f=f.merge(atoms,on=['date','code'],validate='one_to_one')
    for old,new in [('price_1449','qc'),('daily_open','oc'),('high_1449','hc'),('low_1449','lc')]:
        f[new]=np.floor(f[old].fillna(0)*100+.5).astype('int64')
    f['sum20c']=f.sum19c+f.qc;f['sum60c']=f.sum59c+f.qc
    current=np.isfinite(f[RAW]).all(axis=1)&f[['price_1449','daily_open','high_1449','low_1449']].gt(0).all(axis=1)
    current&=f.low_1449.le(f.price_1449)&f.price_1449.le(f.high_1449)&f.volume_1449.gt(0)&f.amount_1449.gt(0)
    for old,new in [('price_1449','qc'),('daily_open','oc'),('high_1449','hc'),('low_1449','lc')]:current&=(abs(f[old]*100-f[new])<=.01)
    f['formula_input_valid']=f.original_input_valid&current&f.nprior.ge(60)&f.close_good60.eq(60)&f.volume_good19.eq(19)
    f['formula_input_valid']&=f.prior_date.lt(f.date)&f.day60_date.lt(f.prior_date)&(f.sum19v+f.volume_1449).gt(0)
    atoms.to_parquet(ROOT/'history_atoms.parquet',index=False,compression='zstd')
    f.to_parquet(ROOT/'features.parquet',index=False,compression='zstd')
    report=dict(protocol_sha256=sha(PROTOCOL),rows=len(f),previous_valid=int(f.original_input_valid.sum()),
        valid=int(f.formula_input_valid.sum()),newly_invalid=int((f.original_input_valid&~f.formula_input_valid).sum()),
        history_atoms_sha256=sha(ROOT/'history_atoms.parquet'),features_sha256=sha(ROOT/'features.parquet'),
        all_prior_60_dates_close_prices_and_19_volumes_SQL_rebuilt=True,
        source_original_domain_and_all_metadata_unchanged=True,no_future_full_day_inputs=True,
        no_outcomes_read=True,new_2026_prices_read=False,native_client_parity_verified=False)
    save_json(ROOT/'input_report.json',report)
    return report


def freeze():
    p,e=checked(); destination=ROOT/'joint_selection_freeze.json';assert not destination.exists()
    report=json.loads((ROOT/'input_report.json').read_text())
    assert report['protocol_sha256']==sha(PROTOCOL) and report['features_sha256']==sha(ROOT/'features.parquet')
    assert report['history_atoms_sha256']==sha(ROOT/'history_atoms.parquet') and report['no_outcomes_read']
    f=pd.read_parquet(ROOT/'features.parquet');old=visible()
    pd.testing.assert_frame_equal(f[[*KEYS,'original_input_valid',*RAW]],old,check_exact=True)
    valid=f.formula_input_valid
    flags=valid & (3*f.sum20c>f.sum60c) & (60*f.qc>f.sum60c) & (f.qc>f.p60c)
    flags&=(100*(f.sum20c-20*f.qc)>=3*f.sum20c)&(100*(f.sum20c-20*f.qc)<=10*f.sum20c)
    flags&=(100*f.qc>82*f.p5c)&(100*f.qc<=94*f.p5c)&f.amount_1449.gt(100000000)&(1000*f.qc>905*f.p1c)
    stop=(f.qc>f.oc)|(f.qc>f.p1c)|((f.hc>f.lc)&(100*(np.minimum(f.oc,f.qc)-f.lc)>=35*(f.hc-f.lc))&(100*f.qc>102*f.lc))
    flags&=stop
    c=numeric.conn();c.register('f',f)
    expected=c.sql('''SELECT date,code,coalesce(formula_input_valid
        AND 3*sum20c>sum60c AND 60*qc>sum60c AND qc>p60c
        AND 100*(sum20c-20*qc)>=3*sum20c AND 100*(sum20c-20*qc)<=10*sum20c
        AND 100*qc>82*p5c AND 100*qc<=94*p5c AND amount_1449>100000000 AND 1000*qc>905*p1c
        AND (qc>oc OR qc>p1c OR (hc>lc AND 100*(least(oc,qc)-lc)>=35*(hc-lc) AND 100*qc>102*lc)),false) AS selected,
        coalesce(original_input_valid AND nprior>=60 AND close_good60=60 AND volume_good19=19
        AND prior_date<date AND day60_date<prior_date AND sum19v+volume_1449>0
        AND isfinite(price_1449) AND isfinite(daily_open) AND isfinite(high_1449) AND isfinite(low_1449)
        AND isfinite(volume_1449) AND isfinite(amount_1449)
        AND price_1449>0 AND daily_open>0 AND high_1449>0 AND low_1449>0
        AND low_1449<=price_1449 AND price_1449<=high_1449 AND volume_1449>0 AND amount_1449>0
        AND abs(price_1449*100-round(price_1449*100))<=.01
        AND abs(daily_open*100-round(daily_open*100))<=.01
        AND abs(high_1449*100-round(high_1449*100))<=.01
        AND abs(low_1449*100-round(low_1449*100))<=.01,false) AS input_valid
        FROM f ORDER BY date,code''').df();c.close()
    pd.testing.assert_frame_equal(f[['date','code']],expected[['date','code']],check_exact=True)
    np.testing.assert_array_equal(flags,expected.selected);np.testing.assert_array_equal(valid,expected.input_valid)
    receipts=dict(p['source_hashes'],**{str(PROTOCOL):sha(PROTOCOL)})
    receipts[str(Path(__file__).relative_to(Path.cwd()))]=sha(Path(__file__))
    for name in ['input_report.json','features.parquet','history_atoms.parquet']:receipts[str(ROOT/name)]=sha(ROOT/name)
    lists,lookup=[],[];keys=f[KEYS]
    for year in ['2024','2025']:
        control=Path(e['controls'][year]);original=checked_selection(control)
        pd.testing.assert_frame_equal(keys,original[KEYS],check_exact=True)
        lists.append(dict(group='control'+year,root=str(control),original_unchanged=True))
        for name in ['selection.parquet','selection_report.json','selection_verification.json']:receipts[str(control/name)]=sha(control/name)
        for family,selected in [('rule',flags&f.date.str.startswith(year)),('matched',original.selected&valid)]:
            root=ROOT/(family+year);assert not root.exists();root.mkdir()
            out=keys.copy();out['selected']=selected.to_numpy();out.to_parquet(root/'selection.parquet',index=False,compression='zstd')
            chosen=out.loc[out.selected];sizes=chosen.groupby('date').size();equivalent=[]
            for previous in e['prior_analysis_roots']:
                previous=Path(previous);record=json.loads((previous/'selection_report.json').read_text())
                if record['selected']==len(chosen) and checked_selection(previous).equals(out):equivalent.append(str(previous))
            lookup.append(dict(group=family+year,prior_complete_lists=len(e['prior_analysis_roots']),exact_equivalents=equivalent))
            save_json(root/'selection_report.json',dict(protocol_sha256=sha(PROTOCOL),selection_sha256=sha(root/'selection.parquet'),
                rows=len(out),selected=len(chosen),days=len(sizes),median_daily=float(sizes.median()) if len(sizes) else None,
                max_daily=int(sizes.max()) if len(sizes) else None,
                half_counts=chosen.groupby('half').agg(rows=('code','size'),days=('date','nunique')).reset_index().to_dict('records'),
                no_outcome_or_fill_filter=True,source_fixed_conditions=p['conditions'] if family=='rule' else 'original selected AND identical historical input quality',
                new_2026_prices_read=False,no_exit_rules=True))
            save_json(root/'selection_verification.json',dict(passed=True,selection_report_sha256=sha(root/'selection_report.json'),
                all_conditions_qualification_scopes_and_metadata_SQL_rebuilt=True,original_control_projection_exact=True,
                input_report_sha256=sha(ROOT/'input_report.json'),native_client_parity_verified=False))
            lists.append(dict(group=family+year,root=str(root),selected=len(chosen),days=len(sizes)))
            for name in ['selection.parquet','selection_report.json','selection_verification.json']:receipts[str(root/name)]=sha(root/name)
    save_json(ROOT/'selection_equivalence_lookup.json',dict(passed=True,records=lookup,keyword_absence_not_novelty_proof=True,no_outcomes_read=True))
    receipts[str(ROOT/'selection_equivalence_lookup.json')]=sha(ROOT/'selection_equivalence_lookup.json')
    save_json(destination,dict(passed=True,input_protocol_sha256=sha(PROTOCOL),execution_protocol_sha256=sha(PROTOCOL),
        source_hashes=receipts,selections=lists,fits_completed=0,all_two_year_rule_and_matched_complete_lists_fixed_together_before_economics=True,
        new_economic_outcomes_read=False,new_2026_prices_read=False))
    return dict(joint_sha256=sha(destination),selections=lists,equivalence=lookup)


def finish():
    p,e=checked()
    identical=True
    for year in ['2024','2025']:
        a,b=checked_analysis(ROOT/('matched'+year)),checked_analysis(Path(e['controls'][year]))
        identical &= a[0].equals(b[0]) and all(a[1][k]==b[1][k] for k in ['summaries','daily_summary_sha256','label_report_sha256'])
    if identical:
        shared.finish()
        path=ROOT/'complete_results_manifest.json';manifest=json.loads(path.read_text())
        for year in ['2024','2025']:
            for name in ['same_dates','shared_unknowns']:
                src=ROOT/(name+'_'+year+'.json');dst=ROOT/('matched_'+name+'_'+year+'.json')
                assert not dst.exists();dst.symlink_to(src.resolve())
                manifest['source_hashes'][str(dst)]=sha(dst)
            pair=next(r for r in manifest['comparisons'] if r['left']=='rule'+year)
            manifest['comparisons'].append(dict(pair,right='matched'+year,reused_pair=True,
                exact_control_frame_label_and_all_statistics_equal=True))
        save_json(path,manifest)
    else:
        comparisons.finish()
    gate_path=ROOT/'selection_gate.json';gate=json.loads(gate_path.read_text())
    quality,bounds=[],[]
    for year in ['2024','2025']:
        def main(root):
            _,r=checked_analysis(root)
            return next(s for s in r['summaries'] if s['arm']=='formula' and s['period']==year and s['bps']==15 and not s['sensitive'])
        a,b=main(ROOT/('rule'+year)),main(ROOT/('matched'+year))
        quality.append(a['rate'] is not None and b['rate'] is not None and a['bad3'] is not None and b['bad3'] is not None
            and a['rate']>b['rate'] and a['bad3']<=b['bad3'])
        r=json.loads((ROOT/('matched_shared_unknowns_'+year+'.json')).read_text())
        bounds.append(next(s for s in r['summaries'] if s['period']==year and s['bps']==15 and not s['sensitive']))
    gate['criteria']['both_year_opportunity_above_matched_and_bad3_not_above']=all(quality)
    gate['criteria']['shared_unknown_lower_ci_both_years_above_matched']=all(s['lower_ci'] is not None and s['lower_ci'][0]>0 for s in bounds)
    gate['supports_further_validation']=all(gate['criteria'].values());gate['matched_control_increment']=bounds
    save_json(gate_path,gate)
    path=ROOT/'complete_results_manifest.json';r=json.loads(path.read_text())
    r['source_hashes'][str(gate_path)]=sha(gate_path);r['criteria']=gate['criteria']
    r.pop('complete_two_year_four_half_comparisons_with_all_three_controls',None)
    r.update(complete_two_year_four_half_comparisons_with_both_controls=True,selector_fits=0,new_tree_fits=0,
        source_fixed_conditions=True,no_parameter_or_window_search=True)
    check_sources(r['source_hashes']);save_json(path,r)
    return dict(complete_sha256=sha(path),criteria=r['criteria'],fingerprints=len(r['source_hashes']),comparisons=len(r['comparisons']))


shared.ROOT=comparisons.ROOT=ROOT
shared.EXECUTION=comparisons.EXECUTION=PROTOCOL
shared.fit=SimpleNamespace(PROTOCOL=PROTOCOL,EXECUTION=PROTOCOL)
shared.checked=comparisons.checked=checked

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','freeze','analyze','finish'])
    stage=parser.parse_args().stage
    print(json.dumps(shared.analyze() if stage=='analyze' else globals()[stage](),ensure_ascii=False),flush=True)
