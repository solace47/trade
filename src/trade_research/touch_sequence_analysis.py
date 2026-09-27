"""Predeclared exits and reverse portraits of current-day touch sequences."""
import json

import pandas as pd

from .corporate_cash import save_json, sha
from .economic_winner_analysis import number
from .reference_gain_accounting import weekly_interval
from .tick_flow_winner_analysis import aggregate, in_period, summary, PERIODS
from .touch_sequence import ROOT, GROUPS

PAIR_COUNTS = ['no_match', 'both_known', 'selected_only', 'control_only', 'both_no_trade', 'neither_known']
PAIR_VALUES = ['selected_paired_mean', 'control_paired_mean', 'net_difference']


def analyze():
    if (ROOT / 'analysis_report.json').exists():
        raise ValueError('Do not replace inspected touch results')
    inputs = json.loads((ROOT / 'input_report.json').read_text())
    gate = json.loads((ROOT / 'input_verification.json').read_text())
    assert gate['passed'] and gate['input_report_sha256'] == sha(ROOT / 'input_report.json')
    for name, h in inputs['output_sha256'].items():
        assert sha(ROOT / name) == h
    morning = ROOT / 'morning'
    proof = json.loads((morning / 'label_verification.json').read_text())
    assert proof['passed'] and proof['label_report_sha256'] == sha(morning / 'label_report.json')
    assert sha(morning / 'labels.parquet') == json.loads((morning / 'label_report.json').read_text())['labels_sha256']
    old_inputs = json.loads((morning / 'input_report.json').read_text())
    assert old_inputs['source_input_report_sha256'] == sha(ROOT / 'input_report.json')
    assert sha(morning / 'old_tail_labels.parquet') == old_inputs['output_sha256']['old_tail_labels.parquet']
    features = pd.read_parquet(ROOT / 'features.parquet')
    pairs = pd.read_parquet(ROOT / 'primary_pairs.parquet')
    scenarios, days, pair_rows, pair_days = [], [], [], []
    groups, pair_groups, reverse = [], [], []
    for exit_name, filename in [('morning', 'labels.parquet'), ('tail', 'old_tail_labels.parquet')]:
        labels = pd.read_parquet(morning / filename,
            columns=['date','code','necessary_tradeable','decision_shares','label5','label15','net_return5','net_return15'])
        assert labels.necessary_tradeable.all()
        f = features.merge(labels, on=['date','code'],validate='one_to_one', suffixes=('','_label'))
        assert len(f) == len(features) and f.decision_shares.eq(f.decision_shares_label).all()
        for cost in [5,15]:
            p = f[['date','code','half','group','primary','touched','source_valid']].copy()
            p['exit'] = exit_name
            p['cost_bps'] = cost
            p['original_label'] = f[f'label{cost}']
            p['original_net_return'] = f[f'net_return{cost}']
            # The prefix input must be usable on both sides; never re-match
            # after a control fails its prefix proof or has a missing exit.
            p['label'] = p.original_label.where(p.source_valid | p.original_label.eq('no_trade'),'unknown')
            p['net_return'] = p.original_net_return.where(p.source_valid)
            p['known'] = p.net_return.notna()
            p['unknown'] = p.label.eq('unknown')
            p['no_trade'] = p.label.eq('no_trade')
            p['winner'] = p.net_return.ge(.01)
            p['loser'] = p.net_return.le(-.01)
            p['positive'] = p.net_return.gt(0)
            assert (p.known.astype(int)+p.unknown.astype(int)+p.no_trade.astype(int)).eq(1).all()
            scenarios.append(p)
            touches = p.loc[p.touched]
            for group in ['all',*GROUPS]:
                rows = touches if group=='all' else touches.loc[touches.group.eq(group)]
                day = aggregate(rows,['date','half'])
                day['exit'] = exit_name
                day['cost_bps'] = cost
                day['group'] = group
                days.append(day)
                for period in PERIODS:
                    groups.append(summary(in_period(day,period),in_period(rows,period),
                        dict(exit=exit_name,cost_bps=cost,group=group,period=period)))
            q = pairs.merge(p[['date','code','label','net_return']], on=['date','code'],validate='one_to_one')
            q = q.merge(p[['date','code','label','net_return']].rename(columns={'code':'control_code','label':'control_label',
                'net_return':'control_return'}),on=['date','control_code'],how='left',validate='many_to_one')
            q['no_match'] = q.control_code.isna()
            q['both_known'] = q.net_return.notna() & q.control_return.notna()
            q['selected_only'] = ~q.no_match & q.net_return.notna() & q.control_return.isna()
            q['control_only'] = ~q.no_match & q.net_return.isna() & q.control_return.notna()
            q['both_no_trade'] = ~q.no_match & q.label.eq('no_trade') & q.control_label.eq('no_trade')
            q['neither_known'] = ~q.no_match & q.net_return.isna() & q.control_return.isna() & ~q.both_no_trade
            assert q[PAIR_COUNTS].sum(axis=1).eq(1).all()
            q['selected_paired_mean'] = q.net_return.where(q.both_known)
            q['control_paired_mean'] = q.control_return.where(q.both_known)
            q['net_difference'] = q.selected_paired_mean-q.control_paired_mean
            q['exit'] = exit_name
            q['cost_bps'] = cost
            pair_rows.append(q)
            day = q.groupby(['date','half']).agg(n=('code','size'), **{x:(x,'sum') for x in PAIR_COUNTS},
                **{x:(x,'mean') for x in PAIR_VALUES}).reset_index()
            day['exit'] = exit_name
            day['cost_bps'] = cost
            pair_days.append(day)
            for period in PERIODS:
                d = in_period(day,period)
                item = dict(exit=exit_name,cost_bps=cost,period=period,stock_days=int(d.n.sum()),dates=len(d),
                    paired_dates=int(d.net_difference.notna().sum()),**{x:int(d[x].sum()) for x in PAIR_COUNTS})
                for metric in PAIR_VALUES:
                    values = d.set_index('date')[metric].sort_index()
                    item[metric] = number(values.mean())
                    item[metric+'_week_interval'] = weekly_interval(values)
                pair_groups.append(item)
            if cost==15:
                classes = {label:touches.label.eq(label) for label in ['economic_winner','economic_loser','middle','unknown','no_trade']}
                classes.update(big_winner=touches.net_return.ge(.03),big_loser=touches.net_return.le(-.03))
                for period in PERIODS:
                    for label,mask in classes.items():
                        rows = in_period(touches.loc[mask],period)
                        count = rows.groupby(['date','group']).size().unstack(fill_value=0).reindex(columns=GROUPS,fill_value=0)
                        fractions = count.div(count.sum(axis=1),axis=0)
                        reverse.append(dict(exit=exit_name,period=period,label=label,rows=len(rows),dates=len(count),
                            counts=rows.group.value_counts().to_dict(),mean_daily_group_fraction=fractions.mean().to_dict()))
            print(json.dumps(dict(exit=exit_name,cost_bps=cost,groups=len(groups))),flush=True)
    outputs = {'scenarios':scenarios,'groups_daily':days,'pairs':pair_rows,'pairs_daily':pair_days}
    for name,tables in outputs.items():
        pd.concat(tables,ignore_index=True).to_parquet(ROOT / (name+'.parquet'),index=False,compression='zstd')
    report = dict(input_report_sha256=sha(ROOT/'input_report.json'),input_verification_sha256=sha(ROOT/'input_verification.json'),
        morning_label_verification_sha256=sha(morning/'label_verification.json'),
        morning_labels_sha256=sha(morning/'labels.parquet'),tail_labels_sha256=sha(morning/'old_tail_labels.parquet'),
        rows=len(features),touched=int(features.touched.sum()),primary=int(features.primary.sum()),
        interpretation='Exploratory 2024/2025; source-invalid prefix profits unknown; independent cash scenarios, not a portfolio or investor identity',
        groups=groups,pair_groups=pair_groups,reverse=reverse,
        output_sha256={n+'.parquet':sha(ROOT/(n+'.parquet')) for n in outputs},new_2026_prices_read=False)
    save_json(ROOT/'analysis_report.json',report)
    return dict(groups=len(groups),pair_groups=len(pair_groups),reverse=len(reverse))


if __name__=='__main__':
    print(json.dumps(analyze(),ensure_ascii=False,indent=2))
