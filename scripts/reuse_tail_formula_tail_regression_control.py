"""Bind the exact existing-control auditor to the minute-regression study."""
import argparse
import json
from pathlib import Path

import reuse_tail_formula_daily_regression_control as common
from trade_research import tail_formula_tail_regression as study
from trade_research import tail_formula_tail_regression_model as model

PROTOCOL = Path('config/tail_formula_tail_regression_reuse_protocol.json')
common.study = study
common.model = model
common.PROTOCOL = PROTOCOL
OLD = common.OLD
checked_complete = common.checked_complete


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','reuse','scores','complete'])
    parser.add_argument('--fold',choices=list(OLD));a = parser.parse_args()
    result = getattr(common,a.stage)(a.fold) if a.stage in ['reuse','scores'] else getattr(common,a.stage)()
    print(json.dumps(result,ensure_ascii=False,indent=2))
