"""Reuse the fixed four-fold tree and score verifier for dense paths."""
import argparse
import json

from . import tail_formula_daily_regression_model as common
from . import tail_formula_dense_path as study

common.study = study
protocols = common.protocols
setup = common.setup
verify_model = common.verify_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['protocols','model','verify_model','scores','verify_scores'])
    parser.add_argument('--arm',choices=list(study.ARMS))
    parser.add_argument('--fold',choices=['2024h1','2024h2','2025h1','2025h2'])
    a = parser.parse_args()
    if a.stage == 'protocols': result = protocols()
    else:
        assert a.arm and a.fold; spec = setup(a.arm,a.fold)
        if a.stage == 'model': result = common.relative.model('relative')
        elif a.stage == 'verify_model': result = verify_model(spec)
        elif a.stage == 'scores': result = common.base.scores()
        else: result = common.verify_scores(expected_expressions=common.base.EXPRESSIONS)
    print(json.dumps(result,ensure_ascii=False,indent=2))
