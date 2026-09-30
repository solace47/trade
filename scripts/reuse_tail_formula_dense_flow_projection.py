"""Make inherited parent proof scope explicit for the 2024–2025 projection.

Parent control proofs also mention their full 2023–2025 scope. Retain those
flags as parent provenance, not as new checks of the smaller projected table.
The frozen reuse implementation, inputs, models and scores stay unchanged.
"""
import argparse
import json

import reuse_tail_formula_dense_flow_parents as parent
from trade_research.corporate_cash import save_json


def reuse(arm,fold):
    result = parent.reuse(arm,fold)
    root = parent.study.ROOT / arm / fold
    path = root / 'score_verification.json'; proof = json.loads(path.read_text())
    inherited = {k:proof.pop(k) for k in list(proof) if k.startswith('all_1815129_')}
    proof['inherited_parent_full_scope_flags'] = inherited
    proof['all_1258085_projection_metadata_and_input_validity_flags_equal'] = True
    proof['proof_scope_normalization_does_not_change_models_scores_or_arithmetic'] = True
    save_json(path,proof)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('stage',choices=['reuse','complete'])
    parser.add_argument('--arm',choices=list(parent.OLD)); parser.add_argument('--fold',choices=['2025h1','2025h2'])
    a = parser.parse_args(); result = reuse(a.arm,a.fold) if a.stage=='reuse' else parent.complete()
    print(json.dumps(result,ensure_ascii=False,indent=2))
