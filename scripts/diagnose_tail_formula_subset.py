"""Describe kept and removed candidates without changing either frozen rule."""
import argparse
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha
from trade_research.tail_formula_1000_analysis import daily_summary, number, period as select_period
from trade_research.reference_gain_accounting import weekly_interval
from verify_tick_flow_analysis import eq, interval


def main(selected, original, output, protocol=None):
    if output.exists():
        raise ValueError('Do not replace an existing subset diagnosis')
    sources = {}; reference_labels = []
    for name, root in [('selected',selected),('original',original)]:
        for kind in ['selection','analysis']:
            report = root/f'{kind}_report.json'
            proof = json.loads((root/f'{kind}_verification.json').read_text())
            assert proof['passed'] and proof[f'{kind}_report_sha256'] == sha(report)
            sources[f'{name}_{kind}_report_sha256'] = sha(report)
        report = json.loads((root/'selection_report.json').read_text())
        assert report['selection_sha256'] == sha(root/'selection.parquet')
        reference_labels.append(json.loads((root/'analysis_report.json').read_text()).get('reference_label', '10:00'))
    assert len(set(reference_labels)) == 1, 'Both groups must use the same observation window'
    reference_label = reference_labels[0]
    assert reference_label in ['10:00', '09:59']
    mark = 'mark_0959_return' if reference_label == '09:59' else 'mark_1000_return'
    mean = 'mean_reference' if reference_label == '09:59' else 'mean1000'
    negative = 'negative_reference' if reference_label == '09:59' else 'negative1000'
    assert sha(selected/'full_labels.parquet') == sha(original/'full_labels.parquet')
    source = pd.read_parquet(selected/'full_labels.parquet')
    frames = [pd.read_parquet(root/'selection.parquet',columns=['date','code','selected'])
              for root in [selected,original]]
    pd.testing.assert_frame_equal(frames[0][['date','code']],frames[1][['date','code']],check_exact=True)
    flags = frames[0].rename(columns={'selected':'kept'})
    flags['prior'] = frames[1].selected
    assert (~flags.kept | flags.prior).all(), 'Expected a subset, not a changed ranking'
    signal_dates = flags.loc[flags.kept,'date'].unique()
    r = source.merge(flags,on=['date','code'],validate='one_to_one')
    periods = ['2025H1','2025H2','2025']
    if protocol is None:
        assert r.date.lt('2026-01-01').all(), 'Default diagnosis is confined to exposed 2024-2025'
    else:
        # Explicitly bounded extension; the default historical diagnosis is unchanged.
        from trade_research import tail_formula_size_agreement_q1 as q1
        assert protocol.resolve() == q1.PROTOCOL.resolve()
        p = q1.policy(); q1.checked_model()
        assert selected.resolve() == (q1.ROOT/'all').resolve()
        assert original.resolve() == (q1.OLD/'all').resolve()
        assert not p['new_2026_stock_prices_allowed'] and p['prior_q1_original_outcomes_exposed']
        assert r.date.between(p['signal_first'],p['signal_last']).all()
        periods = p['periods']; sources['protocol_sha256'] = sha(protocol)
    r['retained_date'] = r.date.isin(signal_dates)
    groups = dict(kept=r.kept, removed=r.prior & ~r.kept,
                  original_on_retained_dates=r.prior & r.retained_date,
                  original_on_removed_dates=r.prior & ~r.retained_date)
    clauses = dict(kept='kept', removed='prior AND NOT kept',
                   original_on_retained_dates='prior AND retained_date',
                   original_on_removed_dates='prior AND NOT retained_date')
    c = duckdb.connect(); c.execute('SET threads=4'); c.register('r',r)
    rows=[]; paired=[]; checks=0; daily_rows=0
    for bps in [5,15]:
        for sensitive in [False,True]:
            tables = {}
            known = f'sensitive_known{bps}' if sensitive else f'known{bps}'
            for arm,mask in groups.items():
                if reference_label == '09:59':
                    from trade_research.tail_formula_boundary_evaluation import daily_summary as boundary_summary
                    d = boundary_summary(r.loc[mask], bps, sensitive)
                else:
                    d = daily_summary(r.loc[mask],bps,sensitive).reset_index()
                expected = c.sql(f'''WITH d AS (SELECT date,count(*) AS rows,
                    count(*) FILTER(WHERE {known}) AS known,
                    count(*) FILTER(WHERE {known} AND opportunity{bps}=1) AS success,
                    count(*) FILTER(WHERE NOT {known} AND NOT known_no_trade) AS unknown,
                    count(*) FILTER(WHERE known_no_trade) AS no_trade,
                    avg(one_percent{bps}) FILTER(WHERE {known}) AS one_percent_rate,
                    avg({mark}{bps}) FILTER(WHERE {known}) AS {mean},
                    avg(({mark}{bps}<0)::INT) FILTER(WHERE {known}) AS {negative},
                    avg((adverse_return{bps}<=-.03)::INT) FILTER(WHERE {known}) AS bad3
                    FROM r WHERE {clauses[arm]} GROUP BY date)
                    SELECT *,success/nullif(known,0) AS rate,success/rows AS lower,
                    (success+unknown)/rows AS upper FROM d ORDER BY date''').df()
                pd.testing.assert_frame_equal(d[expected.columns],expected,check_dtype=False,rtol=0,atol=2e-12)
                d['half'] = d.date.str[:4]+np.where(d.date.str[5:7].le('06'),'H1','H2')
                tables[arm] = d
                daily_rows += len(d)
                for period in periods:
                    q = select_period(d,period)
                    s = dict(arm=arm,bps=bps,sensitive=sensitive,period=period,days=len(q),
                        **{k:int(q[k].sum()) for k in ['rows','known','success','unknown','no_trade']})
                    for metric in ['rate','lower','upper','one_percent_rate',mean,negative,'bad3']:
                        s[metric] = number(q[metric].mean())
                        s[metric+'_ci'] = weekly_interval(q.set_index('date')[metric])
                        eq(s[metric+'_ci'],interval(q,metric),metric)
                        values = q[metric].dropna().to_numpy()
                        eq(s[metric],float(sum(values)/len(values)) if len(values) else None,metric)
                        checks += 2
                    rows.append(s)
            a,b=tables['kept'],tables['original_on_retained_dates']
            pd.testing.assert_series_equal(a.date,b.date,check_exact=True)
            delta = a[['date','half']].copy()
            for metric in ['rate','one_percent_rate',mean,negative,'bad3']:
                delta[metric+'_delta'] = a[metric]-b[metric]
            delta['lower_delta'] = a.lower-b.upper
            delta['upper_delta'] = a.upper-b.lower
            for period in periods:
                q=select_period(delta,period)
                s=dict(bps=bps,sensitive=sensitive,period=period,days=len(q))
                for metric in delta.columns[2:]:
                    s[metric]=number(q[metric].mean())
                    s[metric+'_ci']=weekly_interval(q.set_index('date')[metric])
                    eq(s[metric+'_ci'],interval(q,metric),metric); checks+=1
                paired.append(s)
    c.close()
    report = dict(sources=sources,full_labels_sha256=sha(selected/'full_labels.parquet'),
        summaries=rows,kept_minus_original_on_same_dates=paired,passed=True,daily_rows=daily_rows,
        independently_rebuilt_daily_groups=True,summary_checks=checks,
        subset_asserted_for_every_key=True,no_posthoc_subgroup_promoted=True,
        diagnostic_only=True,year_2025_is_exploratory=True,new_2026_prices_read=bool(r.date.ge('2026-01-01').any()),
        new_source_price_reads=False,existing_labels_only=True,no_exit_rules=True)
    if reference_label == '09:59':
        report['reference_label'] = reference_label
    save_json(output,report)
    return {k:v for k,v in report.items() if k not in ['summaries','kept_minus_original_on_same_dates']}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selected',type=Path,required=True)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--protocol',type=Path,help='Only the explicitly frozen cached Q1 intersection extension is supported')
    a=p.parse_args()
    print(json.dumps(main(a.selected,a.original,a.output,a.protocol),ensure_ascii=False,indent=2))
