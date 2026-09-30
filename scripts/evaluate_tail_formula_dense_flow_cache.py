"""Extend only verified comparison caches, without changing the frozen study.

The flow lists equal the already evaluated price lists. Frozen evaluator
checks both complete lists, labels and daily summaries before each reuse.
"""
import json
from pathlib import Path

import evaluate_tail_formula_dense_flow as evaluate
from trade_research.corporate_cash import save_json, sha

PROTOCOL = Path('config/tail_formula_dense_flow_comparison_cache_protocol.json')
original_checked = evaluate.study.checked


def checked():
    master = original_checked(); p = json.loads(PROTOCOL.read_text())
    assert p['master_protocol_sha256'] == sha(evaluate.study.PROTOCOL)
    assert p['joint_selection_sha256'] == sha(evaluate.ROOT/'joint_selection_freeze.json')
    for file,digest in p['source_hashes'].items():
        assert sha(Path(file)) == digest
    for item in p['comparison_cache']:
        a = json.loads(Path(item['same_dates']).read_text())
        b = json.loads(Path(item['shared_unknowns']).read_text())
        assert a['passed'] and b['passed']
        for side in ['left','right']:
            root = Path(item[side])
            assert a['inputs'][side]['root'] == b['comparison'][side] == str(root)
            assert a['inputs'][side]['analysis_report_sha256'] == b['comparison'][side+'_analysis_sha256'] == sha(root/'analysis_report.json')
    return {**master,'comparison_cache':[*master['comparison_cache'],*p['comparison_cache']]}


if __name__ == '__main__':
    evaluate.study.checked = checked
    result = evaluate.finish()
    path = evaluate.ROOT/'complete_results_manifest.json'; receipt = json.loads(path.read_text())
    receipt['reports'][str(PROTOCOL)] = sha(PROTOCOL)
    receipt['additional_verified_comparison_caches_only'] = True
    save_json(path,receipt); result['completion_sha256'] = sha(path)
    print(json.dumps(result,ensure_ascii=False,indent=2))
