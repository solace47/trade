"""Export the native arity correction before the existing joint evaluation."""
import argparse
import json
from pathlib import Path

import evaluate_tail_formula_daily_regression_reused as evaluation
from trade_research import tail_formula_daily_regression_native as native
from trade_research.corporate_cash import save_json,sha


def freeze():
    native.checked_proof();evaluation.base.native_core=native.native_core
    result=evaluation.freeze()
    path=evaluation.ROOT/'joint_selection_freeze.json';r=json.loads(path.read_text())
    for file in [native.PROTOCOL,native.ROOT/'verification.json',Path(__file__)]:
        r['source_hashes'][str(file)]=sha(file)
    r.update(native_deployment_correction_verified_before_group_results=True,
        numerical_features_models_scores_and_selections_unchanged_by_correction=True)
    save_json(path,r);result['joint_sha256']=sha(path);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['freeze','analyze','finish'])
    a=p.parse_args();native.checked_proof()
    result=freeze() if a.stage=='freeze' else getattr(evaluation,a.stage)()
    print(json.dumps(result,ensure_ascii=False,indent=2))
