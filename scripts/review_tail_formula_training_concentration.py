"""Describe training-day concentration without changing any selected rule."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

from trade_research import tail_formula_additive as numeric
from trade_research.research_io import check_sources,save_json,sha

ROOT=Path('data/research/tail_formula_profit_rule_search')
PROTOCOL=ROOT/'training_concentration_protocol.json'
EXECUTION=ROOT/'training_concentration_execution.json'
OUTPUT=ROOT/'training_concentration.json'
PROGRAM=Path('scripts/review_tail_formula_training_concentration.py')


def number(x):
    return float(x) if pd.notna(x) else None


def review():
    assert not OUTPUT.exists()
    p,e=json.loads(PROTOCOL.read_text()),json.loads(EXECUTION.read_text())
    document=subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    assert sha(PROTOCOL) in document and sha(EXECUTION) in document
    check_sources(p['source_hashes'])
    check_sources(e['source_hashes'])
    source=json.loads((ROOT/'training_market_decomposition.json').read_text())
    assert source['passed']
    check_sources(source['source_hashes'])
    records,memo=[],{}
    for group in source['groups']:
        model=Path('data/research')/group['study']/group['fold']/'model_report.json'
        assert sha(model)==group['model_report_sha256']
        split=json.loads(model.read_text())['fold']['split']
        key=(group['daily_source'],split,group['half'])
        if key not in memo:
            d=pd.read_parquet(group['daily_source'])
            q=d.loc[(d.date.ge(split) if group['half'] else d.date.lt(split)) & d.reference.notna() & d.base_reference.notna()]
            values=q.sort_values(['reference','date'],ascending=[False,True]).reference
            excess=q.sort_values(['excess','date'],ascending=[False,True]).excess
            stats=dict(known_reference_days=len(q),positive_reference_day_fraction=number(values.gt(0).mean()) if len(q) else None,
                median_date_reference=number(values.median()),minimum_date_reference=number(values.min()),
                maximum_date_reference=number(values.max()),mean_reference=number(values.mean()),
                mean_after_removing_highest_one_date=number(values.iloc[1:].mean()),
                mean_after_removing_highest_two_dates=number(values.iloc[2:].mean()),
                clipped_mean_reference_minus3_to_plus3_percent=number(values.clip(-.03,.03).mean()),
                median_date_excess=number(excess.median()),mean_date_excess=number(excess.mean()),
                excess_after_removing_highest_one_date=number(excess.iloc[1:].mean()),
                excess_after_removing_highest_two_dates=number(excess.iloc[2:].mean()))
            con=numeric.conn()
            con.register('daily',d)
            op='>=' if group['half'] else '<'
            independent=con.sql(f'''WITH ranked AS(SELECT date,reference,excess,
                row_number() OVER(ORDER BY reference DESC,date ASC) AS ref_rank,
                row_number() OVER(ORDER BY excess DESC,date ASC) AS excess_rank FROM daily
                WHERE CAST(date AS VARCHAR) {op} '{split}' AND reference IS NOT NULL AND base_reference IS NOT NULL)
                SELECT count(*) AS known_reference_days,avg((reference>0)::INT) AS positive_reference_day_fraction,
                median(reference) AS median_date_reference,min(reference) AS minimum_date_reference,
                max(reference) AS maximum_date_reference,avg(reference) AS mean_reference,
                avg(reference) FILTER(WHERE ref_rank>1) AS mean_after_removing_highest_one_date,
                avg(reference) FILTER(WHERE ref_rank>2) AS mean_after_removing_highest_two_dates,
                avg(least(greatest(reference,-.03),.03)) AS clipped_mean_reference_minus3_to_plus3_percent,
                median(excess) AS median_date_excess,avg(excess) AS mean_date_excess,
                avg(excess) FILTER(WHERE excess_rank>1) AS excess_after_removing_highest_one_date,
                avg(excess) FILTER(WHERE excess_rank>2) AS excess_after_removing_highest_two_dates FROM ranked''').df().iloc[0]
            con.close()
            assert stats['known_reference_days']==int(independent.known_reference_days)
            for name in stats.keys()-{'known_reference_days'}:
                value=number(independent[name])
                assert (value is None)==(stats[name] is None)
                if value is not None:
                    assert abs(value-stats[name])<=2e-12
            memo[key]=stats
        records.append(dict(study=group['study'],fold=group['fold'],half=group['half'],
                            empty_rule=group['empty_rule'],daily_source=group['daily_source'],**memo[key]))
    assert len(records)==p['groups']==24
    save_json(OUTPUT,dict(passed=True,protocol_sha256=sha(PROTOCOL),execution_sha256=sha(EXECUTION),
        groups=records,source_daily_hashes=source['source_hashes'],all_24_groups_SQL_verified=True,
        no_effective_post_selection_confidence_claim=True,new_fits=0,changed_rules=0,
        new_selections=0,new_2026_prices_read=False,no_exit_rules=True))
    return dict(output_sha256=sha(OUTPUT),groups=len(records),unique_calculations=len(memo))


if __name__=='__main__':
    print(json.dumps(review(),ensure_ascii=False))
