"""Bind existing rules chronologically; an empty update keeps the last rule."""
import argparse
import json
from pathlib import Path
import subprocess

import pandas as pd

from trade_research import tail_formula_additive as numeric
from trade_research.research_io import check_sources, save_json, sha
from run_tail_formula_rule_search import condition_key

ROOT=Path('data/research/tail_formula_rule_persistence')
PROTOCOL=Path('config/tail_formula_rule_persistence_protocol.json')
EXECUTION=Path('config/tail_formula_rule_persistence_execution.json')


def checked():
    p,e=json.loads(PROTOCOL.read_text()),json.loads(EXECUTION.read_text())
    assert e['input_protocol_sha256']==sha(PROTOCOL)
    check_sources(p['source_hashes'])
    check_sources(e['source_hashes'])
    text=subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    assert sha(PROTOCOL) in text and sha(EXECUTION) in text
    assert p['selector_fits']==0 and len(p['folds'])==4
    return p,e


def prepare():
    p,_=checked()
    assert not (ROOT/'all_models_verified.json').exists()
    source=Path(p['original_bank'])
    previous=None
    observations,records=[],[]
    for index,fold in enumerate(p['folds']):
        path=source/fold['id']
        report=json.loads((path/'model_report.json').read_text())
        verification=json.loads((path/'model_verification.json').read_text())
        assert verification['passed'] and verification['model_report_sha256']==sha(path/'model_report.json')
        assert report['fold']==fold and report['feature_names']==p['feature_names']
        assert report['search_trace_sha256']==sha(path/'search_trace.json')
        observations.append(dict(index=index,nonempty=report['chosen_rule'] is not None))
        if report['chosen_rule'] is not None:
            previous=index
        records.append(dict(index=index,fold=fold['id'],source_index=previous,
                            carried_forward=report['chosen_rule'] is None and previous is not None))
    con=numeric.conn()
    con.register('updates',pd.DataFrame(observations))
    independent=con.sql('''SELECT "index", max("index") FILTER(WHERE nonempty)
        OVER(ORDER BY "index" ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
        AS source_index FROM updates ORDER BY "index"''').df()
    con.close()
    receipt=[]
    for record,(_,row) in zip(records,independent.iterrows(),strict=True):
        expected=None if pd.isna(row.source_index) else int(row.source_index)
        assert expected==record['source_index'] and int(row['index'])==record['index']
        fold=p['folds'][record['index']]
        current=source/fold['id']
        origin=None if expected is None else source/p['folds'][expected]['id']
        source_report=None if origin is None else json.loads((origin/'model_report.json').read_text())
        if source_report is not None:
            assert source_report['chosen_rule'] is not None
            assert source_report['fold']['training_end']<=fold['evaluation_start']
            assert source_report['fold']['evaluation_start']<=fold['evaluation_start']
        destination=ROOT/fold['id']
        assert not destination.exists()
        destination.mkdir(parents=True)
        trace=dict(policy=p['policy'],all_update_bindings=records,
                   current_update_source_sha256=sha(current/'model_report.json'),
                   rule_source_report_sha256=None if origin is None else sha(origin/'model_report.json'),
                   fits_completed=0,searches_completed=0,new_economics_read=False)
        save_json(destination/'search_trace.json',trace)
        save_json(destination/'model_report.json',dict(protocol_sha256=sha(PROTOCOL),
            execution_protocol_sha256=sha(EXECUTION),variant='last_nonempty_existing_rule',
            fold=fold,source_fold=None if source_report is None else source_report['fold'],
            feature_names=p['feature_names'],source_model_report=None if origin is None else str(origin/'model_report.json'),
            source_model_report_sha256=None if origin is None else sha(origin/'model_report.json'),
            chosen_rule=None if source_report is None else source_report['chosen_rule'],
            carried_forward=record['carried_forward'],search_trace_sha256=sha(destination/'search_trace.json'),
            fits_completed=0,new_atomic_conditions=0,new_input_features=0,new_2026_prices_read=False,
            no_exit_rules=True,source_training_metrics_not_current_training_fit=True))
        save_json(destination/'model_verification.json',dict(passed=True,
            model_report_sha256=sha(destination/'model_report.json'),
            chronological_last_nonempty_SQL_verified=True,source_conditions_and_training_scope_unchanged=True,
            source_future_update_never_applied_early=True,new_fits=0))
        receipt.append(dict(fold=fold['id'],model_report_sha256=sha(destination/'model_report.json'),
                            carried_forward=record['carried_forward']))
    save_json(ROOT/'all_models_verified.json',dict(passed=True,folds=receipt,
        chronological_bindings=records,independent_SQL_verified=True,fits_completed=0,
        not_four_new_models=True,no_new_economics=True,new_2026_prices_read=False))
    return dict(binding_sha256=sha(ROOT/'all_models_verified.json'),bindings=records,new_fits=0)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare'])
    parser.parse_args()
    print(json.dumps(prepare(),ensure_ascii=False))
