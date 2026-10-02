"""Fit the eight fixed matched models without reading selection economics."""
import json
from pathlib import Path
import subprocess

from trade_research import tail_formula_additive as numeric
from trade_research import tail_formula_baseline as original
from trade_research import tail_formula_relative as relative
from trade_research.research_io import check_runtime, check_sources, sha, save_json

ROOT=Path('data/research/tail_formula_order_target')
INPUT=Path('config/tail_formula_order_target_input.json')
EXECUTION=Path('config/tail_formula_order_target_execution.json')


def checked():
    check_runtime()
    p=json.loads(INPUT.read_text());e=json.loads(EXECUTION.read_text())
    assert e['input_protocol_sha256']==sha(INPUT) and e['maximum_new_fits']==8
    text=subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    for path in [INPUT,EXECUTION]:
        assert subprocess.check_output(['git','show',f'HEAD:{path}'])==path.read_bytes() and sha(path) in text
    check_sources(p['source_hashes']);check_sources(e['source_hashes'])
    v=json.loads((ROOT/'source_gate.json').read_text())
    assert v['passed'] and v['targets_sha256']==sha(ROOT/'targets.parquet')
    assert sha(ROOT/'source_gate.json') in text
    assert len(e['models'])==8
    equivalence=json.loads((ROOT/'training_target_equivalence.json').read_text())
    assert equivalence['passed'] and len(equivalence['records'])==4
    assert all(v['same_X_keys_and_date_weights'] and not v['centered_targets_equal'] for v in equivalence['records'])
    return e


def setup(item):
    config=Path(item['config']);q=json.loads(config.read_text())
    assert q['master_input_protocol_sha256']==sha(INPUT)
    assert q['target']=='relative' and q['model_max_depth']==3
    numeric.ROOT=ROOT/'models'/item['arm']/item['fold']
    numeric.FEATURES=ROOT;numeric.SOURCE=ROOT/(item['arm']+'_labels')
    numeric.PROTOCOL=relative.PROTOCOL=config
    numeric.EXPRESSIONS=original.EXPRESSIONS;numeric.QUANTILES=[.995]
    return q


def main():
    e=checked();assert not (ROOT/'all_models_verified.json').exists()
    records=[];receipts=dict(e['source_hashes']);newly_fitted=0
    for item in e['models']:
        q=setup(item);root=numeric.ROOT
        if (root/'model_report.json').exists():
            m=json.loads((root/'model_report.json').read_text())
            assert m['protocol_sha256']==sha(Path(item['config']))
            assert m['feature_report_sha256']==sha(ROOT/'feature_report.json')
            assert m['label_report_sha256']==sha(numeric.SOURCE/'full_label_report.json')
            print(json.dumps(dict(resuming_saved_model=item['fold'],arm=item['arm'],no_refit=True)),flush=True)
        else:
            assert not root.exists(), 'Inspect an interrupted folder; never silently refit'
            print(json.dumps(dict(fitting=item['fold'],arm=item['arm'],new_fits_completed=newly_fitted)),flush=True)
            relative.model('relative');newly_fitted+=1
        relative.verify_model('relative')
        m=json.loads((root/'model_report.json').read_text())
        assert m['feature_names']==list(original.EXPRESSIONS) and len(m['thresholds'])==1
        assert m['thresholds'][0]['training_quantile']==.995
        assert m['last_observation']<item['evaluation_start']
        for name in ['model_report.json','model_verification.json']:receipts[str(root/name)]=sha(root/name)
        records.append(dict(arm=item['arm'],fold=item['fold'],root=str(root),
            model_report_sha256=sha(root/'model_report.json'),model_verification_sha256=sha(root/'model_verification.json')))
        print(json.dumps(dict(model_verified=item['fold'],arm=item['arm'],training_rows=m['rows'],last_observation=m['last_observation'])),flush=True)
    assert len(records)==8
    check_sources(receipts)
    save_json(ROOT/'all_models_verified.json',dict(passed=True,models=records,source_hashes=receipts,
        input_protocol_sha256=sha(INPUT),execution_protocol_sha256=sha(EXECUTION),fits_completed=8,
        new_fits_this_invocation=newly_fitted,all_models_before_selection_economics=True,
        all_targets_integer_inputs_date_weights_nodes_and_thresholds_independently_verified=True,
        new_2026_prices_read=False,no_exit_rules=True))
    print(json.dumps(dict(models=8,new_fits_this_invocation=newly_fitted,models_sha256=sha(ROOT/'all_models_verified.json'))),flush=True)


if __name__=='__main__':
    main()
