"""Fit four component models and replay four exactly equivalent controls."""
import argparse
import copy
import json
from pathlib import Path
import subprocess

from trade_research import tail_formula_additive as numeric
from trade_research import tail_formula_relative as relative
from trade_research.research_io import check_runtime,check_sources,sha,save_json

ROOT=Path('data/research/tail_formula_day_night_matched')
PROTOCOL=Path('config/tail_formula_day_night_matched_protocol.json')
EXECUTION=Path('config/tail_formula_day_night_matched_execution.json')


def checked():
    check_runtime()
    p=json.loads(PROTOCOL.read_text());e=json.loads(EXECUTION.read_text())
    assert e['input_protocol_sha256']==sha(PROTOCOL)
    for path in [PROTOCOL,EXECUTION]:
        assert subprocess.check_output(['git','show',f'HEAD:{path}'])==path.read_bytes()
        assert sha(path) in subprocess.check_output(['git','show','HEAD:docs/selection-formula.md']).decode()
    check_sources(p['source_hashes']);check_sources(e['source_hashes'])
    assert e['maximum_new_fits']==4 and e['exact_controls_to_reuse']==4
    v=json.loads((ROOT/'input_verification.json').read_text())
    assert v['passed'] and v['newly_invalid']==0
    proof=json.loads((ROOT/'control_training_equivalence.json').read_text())
    assert proof['passed'] and len(proof['records'])==4
    return p,e,proof


def setup(item):
    config=Path(item['config']);q=json.loads(config.read_text())
    assert q['master_protocol_sha256']==sha(PROTOCOL)
    assert q['target']=='relative' and q['model_max_depth']==3
    numeric.ROOT=ROOT/item['arm']/item['fold']
    numeric.FEATURES=numeric.SOURCE=ROOT
    numeric.PROTOCOL=relative.PROTOCOL=config
    report=json.loads((ROOT/'feature_report.json').read_text())
    numeric.EXPRESSIONS={n:report['native_expressions'][n] for n in q['feature_names']}
    numeric.QUANTILES=[.995]
    return q


def fit():
    p,e,proof=checked();assert not (ROOT/'all_models_verified.json').exists()
    records=[];receipts=dict(e['source_hashes']);new_fits=0
    for item in e['models']:
        q=setup(item);root=numeric.ROOT
        assert not root.exists(), 'Never repeat a fitted or reused model'
        if item['arm']=='control50':
            record=next(v for v in proof['records'] if v['fold']==item['fold'])
            assert record['all_training_values_targets_weights_parameters_equal']
            source=Path(record['root']);old=json.loads((source/'model_report.json').read_text())
            assert old['rows']==record['rows'] and old['training_start']==q['training_start']
            assert old['training_end']==q['training_end'] and old['feature_names']==q['feature_names']
            m=copy.deepcopy(old)
            m.update(protocol_sha256=sha(Path(item['config'])),feature_report_sha256=sha(ROOT/'feature_report.json'),
                label_report_sha256=sha(ROOT/'full_label_report.json'),
                reused_model_report_sha256=sha(source/'model_report.json'),
                control_training_equivalence_sha256=sha(ROOT/'control_training_equivalence.json'))
            # Select the one prespecified old quantile; do not search the other cuts.
            m['thresholds']=[t for t in old['thresholds'] if t['training_quantile']==.995]
            assert len(m['thresholds'])==1
            root.mkdir(parents=True);save_json(root/'model_report.json',m)
            relative.verify_model('relative')
            v=json.loads((root/'model_verification.json').read_text())
            v.update(exact_all_training_values_targets_weights_parameters_before_reuse=True,
                original_model_root=str(source),original_model_report_sha256=sha(source/'model_report.json'),
                original_model_verification_sha256=sha(source/'model_verification.json'),
                control_training_equivalence_sha256=sha(ROOT/'control_training_equivalence.json'),new_fit=False)
            save_json(root/'model_verification.json',v)
            reused=True
        else:
            print(json.dumps(dict(fitting=item['fold'],arm=item['arm'],fits_completed=new_fits)),flush=True)
            relative.model('relative');relative.verify_model('relative')
            new_fits+=1;reused=False
        for name in ['model_report.json','model_verification.json']:
            receipts[str(root/name)]=sha(root/name)
        records.append(dict(arm=item['arm'],fold=item['fold'],root=str(root),reused=reused,
            model_report_sha256=sha(root/'model_report.json'),model_verification_sha256=sha(root/'model_verification.json')))
        print(json.dumps(dict(model_verified=item['fold'],arm=item['arm'],reused=reused)),flush=True)
    assert new_fits==4 and sum(x['reused'] for x in records)==4
    check_sources(receipts)
    save_json(ROOT/'all_models_verified.json',dict(passed=True,models=records,source_hashes=receipts,
        input_protocol_sha256=sha(PROTOCOL),execution_protocol_sha256=sha(EXECUTION),
        fits_completed=new_fits,exact_controls_reused=4,all_models_before_new_economics=True,
        new_2026_prices_read=False,no_exit_rules=True))
    return dict(models=len(records),new_fits=new_fits,exact_controls_reused=4,
        models_sha256=sha(ROOT/'all_models_verified.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['fit'])
    parser.parse_args()
    print(json.dumps(fit(),ensure_ascii=False),flush=True)
