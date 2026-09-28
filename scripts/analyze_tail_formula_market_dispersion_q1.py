"""Evaluate the fixed direct-dispersion Q1 list and reuse all four original controls."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trade_research import tail_formula_boundary_evaluation as evaluation
from trade_research import tail_formula_market_dispersion_q1 as study
from trade_research import tail_formula_path_relative_q1 as previous
from trade_research.corporate_cash import save_json, sha
from verify_tail_formula_before1000 import analysis as verify_analysis
from reuse_tail_formula_selected_analysis import reuse
from compare_tail_formula_same_dates import compare
from compare_tail_formula_shared_unknowns import compare as shared

LABELS = study.ROOT/'before1000'
ROOT = study.ROOT/'market_dispersion'


def setup():
    p, gate = study.checked_models()
    coverage = json.loads((study.ROOT/'label_coverage_verification.json').read_text())
    assert coverage['selection_verification_sha256'] == sha(study.ROOT/'selection_verification.json')
    evaluation.source.ROOT = LABELS; evaluation.PERIODS = p['periods']
    labels = pd.read_parquet(LABELS/'full_labels.parquet',columns=['date','code'])
    sr = json.loads((study.ROOT/'selection_report.json').read_text())
    from trade_research import tail_formula_q1_candidate_inputs as inputs
    universe = pd.read_parquet(inputs.OLD/'universe.parquet',columns=['date','code'])
    needed = universe.loc[universe.date.isin(sr['union_selected_dates'])]
    matched = needed.merge(labels,on=['date','code'],validate='one_to_one')
    assert len(matched) == len(needed), 'Every selected date needs its full base pool'
    return p, gate


def analyze():
    setup();return evaluation.analyze(ROOT,study.PROTOCOL)


def verify():
    setup();return verify_analysis(ROOT)


def reuse_controls():
    p, gate = setup(); records = {}
    full = pd.read_parquet(LABELS/'full_labels.parquet')
    old = pd.read_parquet(previous.ROOT/'before1000/full_labels.parquet')
    overlap = old[['date','code']].merge(full,on=['date','code'],validate='one_to_one')
    pd.testing.assert_frame_equal(old,overlap[old.columns],check_exact=True)
    for variant in ['control','path','equal_weight','path_relative']:
        root = study.ROOT/variant; source = previous.ROOT/variant
        reuse(root,source)
        evaluation.attach_labels(root)
        report = json.loads((source/'analysis_report.json').read_text())
        report.update(protocol_sha256=sha(study.PROTOCOL),selection_report_sha256=sha(root/'selection_report.json'),
            label_report_sha256=sha(root/'full_label_report.json'),reused_analysis_report_sha256=sha(source/'analysis_report.json'),
            original_control_statistics_unchanged=True)
        (root/'daily_summary.parquet').symlink_to((source/'daily_summary.parquet').resolve())
        save_json(root/'analysis_report.json',report)
        verify_analysis(root)
        records[variant] = dict(analysis_report_sha256=sha(root/'analysis_report.json'),
            analysis_verification_sha256=sha(root/'analysis_verification.json'),
            old_base_labels_exactly_reused=len(overlap))
    for variant in ['path','equal_weight','path_relative','market_dispersion']:
        root = study.ROOT/('control_'+variant)
        assert json.loads((root/'selection_report.json').read_text())['identical_to_full_control']
        reuse(root,study.ROOT/'control')
    r = dict(passed=True,records=records,all_four_selections_and_statistics_unchanged=True,
        all_identical_quality_controls_reuse_full_control=True,
        independent_checks_on_current_label_source=True,strict_blind=False,no_exit_rules=True)
    save_json(study.ROOT/'control_reuse_verification.json',r);return r


def reference_audit():
    p, gate = setup(); out = ROOT/'reference_coverage_verification.json';assert not out.exists()
    proof = json.loads((ROOT/'analysis_verification.json').read_text())
    assert proof['passed'] and proof['analysis_report_sha256'] == sha(ROOT/'analysis_report.json')
    selected = pd.read_parquet(ROOT/'selection.parquet')
    selected = selected.loc[selected.selected,['date','code']]
    rows = selected.merge(pd.read_parquet(ROOT/'full_labels.parquet'),on=['date','code'],validate='one_to_one')
    assert len(rows) == len(selected)
    old_daily = pd.read_parquet(ROOT/'daily_summary.parquet')
    c = study.base.conn();c.register('rows',rows)
    summaries=[];daily=[];missing=[]
    for bps in [5,15]:
        for sensitive in [False,True]:
            known = f'sensitive_known{bps}' if sensitive else f'known{bps}'
            mark = f'mark_0959_return{bps}'
            r=rows[['date','code','half']].copy();r['known']=rows[known].astype(int)
            valid=rows[known] & np.isfinite(rows[mark])
            r['available']=valid.astype(int);r['missing']=(rows[known]&~valid).astype(int)
            r['positive']=(valid&rows[mark].gt(0)).astype(int);r['mean_reference']=rows[mark].where(valid)
            got=r.groupby(['date','half'],sort=True).agg(selected=('code','size'),known=('known','sum'),
                available=('available','sum'),missing=('missing','sum'),positive=('positive','sum'),
                mean_reference=('mean_reference','mean')).reset_index()
            expected=c.sql(f'''SELECT date,half,count(*) AS selected,count(*) FILTER(WHERE {known}) AS known,
                count(*) FILTER(WHERE {known} AND isfinite({mark})) AS available,
                count(*) FILTER(WHERE {known} AND NOT coalesce(isfinite({mark}),false)) AS missing,
                count(*) FILTER(WHERE {known} AND isfinite({mark}) AND {mark}>0) AS positive,
                avg({mark}) FILTER(WHERE {known} AND isfinite({mark})) AS mean_reference
                FROM rows GROUP BY date,half ORDER BY date,half''').df()
            pd.testing.assert_frame_equal(got,expected,check_dtype=False,rtol=0,atol=2e-12)
            orig=old_daily.loc[old_daily.arm.eq('formula')&old_daily.bps.eq(bps)&old_daily.sensitive.eq(sensitive)].sort_values('date')
            assert orig.date.tolist()==got.date.tolist()
            np.testing.assert_allclose(orig.mean_reference,got.mean_reference,rtol=0,atol=2e-12)
            assert (got.known==got.available+got.missing).all()
            got['positive_rate']=got.positive/got.available.replace(0,np.nan)
            for period in p['periods']:
                g=evaluation.period(got,period)
                summaries.append(dict(bps=bps,sensitive=sensitive,period=period,days=len(g),
                    reference_days=int(g.available.gt(0).sum()),known=int(g.known.sum()),available=int(g.available.sum()),
                    missing=int(g.missing.sum()),positive_rate=evaluation.number(g.positive_rate.mean()),
                    mean_reference=evaluation.number(g.mean_reference.mean())))
            got['bps']=bps;got['sensitive']=sensitive
            daily.extend(got.astype(object).where(got.notna(),None).to_dict('records'))
            missing.extend(dict(bps=bps,sensitive=sensitive,**x) for x in rows.loc[rows[known]&~valid,['date','code']].to_dict('records'))
    c.close()
    report=dict(passed=True,analysis_report_sha256=sha(ROOT/'analysis_report.json'),summaries=summaries,daily=daily,
        missing_reference_cases=missing,counts_and_means_sql_verified=True,known_binary_outcomes_retained=True,
        all_unknown_dates_retained=True,reference_rates_conditional_on_available_marks=True,
        no_selection_changes_or_zero_imputation=True,strict_blind=False,no_exit_rules=True)
    save_json(out,report);return {k:v for k,v in report.items() if k not in ['daily','missing_reference_cases']}


def comparisons():
    p,gate=setup(); output=study.ROOT/'comparisons';output.mkdir(exist_ok=True)
    control_proof=json.loads((study.ROOT/'control_reuse_verification.json').read_text());assert control_proof['passed']
    specs=[];reports={}
    for variant in ['control','path','equal_weight','path_relative']:
        right=study.ROOT/variant
        specs.append(dict(left=str(ROOT),right=str(right),output=str(output/(variant+'_shared_unknowns.json')),
            left_analysis_sha256=sha(ROOT/'analysis_report.json'),right_analysis_sha256=sha(right/'analysis_report.json')))
    policy=dict(protocol_sha256=sha(study.PROTOCOL),signal_range=[p['signal_first'],p['signal_last']],periods=p['periods'],
        labels_sha256=sha(LABELS/'full_labels.parquet'),comparisons=specs,diagnostic_only=True,does_not_change_selected_lists=True)
    save_json(output/'shared_unknowns_protocol.json',policy)
    for variant,spec in zip(['control','path','equal_weight','path_relative'],specs):
        right=Path(spec['right']); same=output/(variant+'_same_dates.json')
        compare(ROOT,right,same,p['periods'],intersection_only=True);shared(spec,policy)
        for file in [same,Path(spec['output'])]:
            assert json.loads(file.read_text())['passed'];reports[str(file)]=sha(file)
    for variant in gate['active_variants']:
        folder=study.ROOT/variant
        for file in ['analysis_report.json','analysis_verification.json']:
            reports[str(folder/file)]=sha(folder/file)
    for file in [study.ROOT/'control_reuse_verification.json',ROOT/'reference_coverage_verification.json']:
        assert json.loads(file.read_text())['passed'];reports[str(file)]=sha(file)
    result=dict(passed=True,protocol_sha256=sha(study.PROTOCOL),reports=reports,
        all_four_controls_and_all_four_periods_reported=True,strict_blind=False,no_q2_signal_prices_read=True,no_exit_rules=True)
    save_json(study.ROOT/'complete_results_manifest.json',result);return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['analyze','verify','reuse_controls','reference_audit','comparisons'])
    print(json.dumps(globals()[p.parse_args().stage](),ensure_ascii=False,indent=2))
