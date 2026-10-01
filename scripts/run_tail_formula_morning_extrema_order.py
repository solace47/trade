"""Reuse fixed-model machinery for morning order with exact original controls."""
import argparse
import json
from pathlib import Path

from trade_research import tail_formula_morning_extrema_order as study
from trade_research.corporate_cash import save_json, sha
import run_tail_formula_volume_memory as common


def bind():
    common.study = study
    common.EXECUTION = Path('config') / (study.STEM + '_model_protocol.json')
    common.ARMS = {'control': study.CONTROL, 'memory': study.EXPRESSIONS}
    # The shared runner's legacy internal arm name is metadata only. Its
    # inputs are the explicitly bound 52 morning-order expressions above.


def protocols():
    bind(); result = common.protocols()
    path = study.ROOT / 'prefit_lookup_verification.json'; r = json.loads(path.read_text())
    p = study.checked()
    for item in r['records']:
        if item['arm'] == 'control':
            assert any(candidate['root'] == p['old_model_roots'][item['fold']] for candidate in item['lookup']['matches'])
    r['maximum_new_fits'] = 4
    r['all_original_controls_must_reuse_exact_training_values_targets_and_weights'] = True
    save_json(path, r); result['prefit_sha256'] = sha(path)
    return result


def fit(arm, fold):
    bind(); reuse = common.reuse_control
    def require_reuse(p, q, record):
        result = reuse(p, q, record)
        assert result, 'No additional control fit is authorized by this fixed four-fit protocol'
        return result
    common.reuse_control = require_reuse
    return common.fit(arm, fold)


def freeze():
    bind(); common.freeze()
    path = study.ROOT / 'joint_selection_freeze.json'; r = json.loads(path.read_text())
    for item in r['models']:
        root = study.ROOT / item['arm'] / item['fold']
        source = root / 'frozen_numeric_core.tdx'; text = source.read_text()
        # Normalize the common runner's old volume-reference predicate before
        # the final joint receipt exists in Git. No order input is named VMREF.
        assert text.count('CORE:VMREF>0 AND VP20>0 AND SC>') == 1
        text = text.replace('CORE:VMREF>0 AND VP20>0 AND SC>', 'CORE:SC>')
        source.write_text(text); r['source_hashes'][str(source)] = sha(source)
        model = json.loads((root / 'model_report.json').read_text())
        item['am_order_split_nodes'] = sum(t['feature'].count(50) for t in model['trees']) if item['arm'] == 'memory' else 0
        item['am_low_age_split_nodes'] = sum(t['feature'].count(51) for t in model['trees']) if item['arm'] == 'memory' else 0
        del item['new_feature_nodes']
    assert sum(item['new_fit'] for item in r['models']) == 4
    r.update(arm_display_names={'control': '原50', 'memory': '上午极值时序52'},
             no_new_control_fits=True, source_specific_native_predicates_normalized_before_final_joint=True,
             native_input_verification_sha256=sha(study.INPUTS / 'native_input_verification.json'))
    save_json(path, r)
    return dict(joint_sha256=sha(path), selections=r['selections'], models=r['models'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['prepare', 'protocols', 'fit', 'freeze', 'analyze', 'finish'])
    parser.add_argument('--arm', choices=['control', 'memory'])
    parser.add_argument('--fold', choices=['2024h1', '2024h2', '2025h1', '2025h2']); a = parser.parse_args()
    if a.stage == 'prepare': result = study.prepare()
    elif a.stage == 'protocols': result = protocols()
    elif a.stage == 'fit': result = fit(a.arm, a.fold)
    elif a.stage == 'freeze': result = freeze()
    else:
        bind(); result = getattr(common, a.stage)()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
