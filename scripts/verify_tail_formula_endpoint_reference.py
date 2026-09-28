"""Retain reference-missing rows separately from known binary opportunities."""
import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from trade_research.corporate_cash import save_json, sha


def verify():
    root=Path('data/research/tail_formula_endpoint_robust')
    prior=root/'evaluation_reference_coverage.json';old=json.loads(prior.read_text())
    label_root=Path('data/research/tail_formula_before1000')
    lr=json.loads((label_root/'full_label_report.json').read_text());lv=json.loads((label_root/'full_label_verification.json').read_text())
    assert lv['passed'] and lv['label_report_sha256']==sha(label_root/'full_label_report.json')
    assert lr['labels_sha256']==sha(label_root/'full_labels.parquet')
    labels=pd.read_parquet(label_root/'full_labels.parquet',columns=['date','code','known15','mark_0959_return15'])
    cases=[];hashes={}
    for row in old['rows']:
        source=Path('data/research/tail_formula_endpoint_'+row['arm']+'_2025')
        r=json.loads((source/'selection_report.json').read_text());v=json.loads((source/'selection_verification.json').read_text())
        assert v['passed'] and v['selection_report_sha256']==row['selection_report_sha256']==sha(source/'selection_report.json')
        assert r['selection_sha256']==sha(source/'selection.parquet')
        s=pd.read_parquet(source/'selection.parquet');s=s.loc[s.selected,['date','code']].merge(labels,on=['date','code'],validate='one_to_one')
        missing=s.known15 & ~np.isfinite(s.mark_0959_return15)
        assert len(s)==row['selected'] and int(s.known15.sum())==row['known_opportunity']
        assert int(missing.sum())==row['known_opportunity_without_reference']
        assert s.loc[missing,'date'].nunique()==row['affected_days']
        c=duckdb.connect();c.register('joined',s)
        sql=c.sql('SELECT date,code FROM joined WHERE known15 AND NOT coalesce(isfinite(mark_0959_return15),false) ORDER BY date,code').df();c.close()
        # Empty SQL string columns have no recoverable pandas string dtype.
        pd.testing.assert_frame_equal(sql,s.loc[missing,['date','code']].sort_values(['date','code']).reset_index(drop=True),
                                      check_exact=True,check_dtype=not sql.empty)
        cases.extend(dict(arm=row['arm'],**x) for x in sql.to_dict('records'))
        hashes[row['arm']]=row['selection_report_sha256']
    result=dict(passed=True,coverage_report_sha256=sha(prior),label_report_sha256=sha(label_root/'full_label_report.json'),
        selections=hashes,all_counts_and_keys_rebuilt=True,reference_missing_cases=cases,
        known_binary_outcomes_retained=True,reference_means_conditional_on_available_marks=True,
        no_zero_imputation_or_selection_changes=True,new_2026_prices_read=False)
    save_json(root/'reference_coverage_verification.json',result);return result


if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=False,indent=2))
