"""Use the shared paired runner and the already proven evaluation-only projection."""
import argparse
import json

import run_tail_formula_paired_study as shared
from run_tail_formula_minute_vwap import reuse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['protocols', 'reuse', 'model', 'verify_model', 'scores', 'verify_scores', 'freeze', 'analyze', 'finish'])
    parser.add_argument('--fold', choices=['2025h1', '2025h2']); args = parser.parse_args()
    shared.configure('tail_formula_minute_vwap_range')
    if args.stage in ['protocols', 'freeze', 'analyze', 'finish']:
        result = getattr(shared, args.stage)()
    elif args.stage == 'reuse':
        assert args.fold; result = reuse(args.fold)
    else:
        assert args.fold; shared.model.setup('range', args.fold)
        if args.stage == 'model': result = shared.relative.model('relative')
        elif args.stage == 'verify_model': result = shared.relative.verify_model('relative')
        elif args.stage == 'scores': result = shared.base.scores()
        else: result = shared.verify_scores(expected_expressions=shared.base.EXPRESSIONS)
    print(json.dumps(result, ensure_ascii=False, indent=2))
