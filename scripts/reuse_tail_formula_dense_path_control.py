"""Bind the exact existing-control auditor to the dense-path study."""
import argparse
import json
from pathlib import Path

import reuse_tail_formula_daily_regression_control as common
from trade_research import tail_formula_dense_path as study
from trade_research import tail_formula_dense_path_model as model

PROTOCOL = Path('config/tail_formula_dense_path_reuse_protocol.json')
common.study = study
common.model = model
common.PROTOCOL = PROTOCOL
OLD = common.OLD
checked_complete = common.checked_complete

def prepare():
    common.checked()
    try:
        model.protocols()
    except AssertionError as error:
        assert str(error) == "Audit and reuse existing fits before proceeding"
    report = json.loads((study.ROOT/"prefit_lookup_verification.json").read_text())
    assert report["passed"]
    assert not any(x["lookup"]["matches"] for x in report["records"] if x["arm"] != "control")
    return dict(counts=report["counts"],new_path_fits=4,canonical_controls_to_reuse=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','reuse','scores','complete'])
    parser.add_argument('--fold',choices=list(OLD));a = parser.parse_args()
    result = prepare() if a.stage == "prepare" else (getattr(common,a.stage)(a.fold) if a.stage in ["reuse","scores"] else getattr(common,a.stage)())
    print(json.dumps(result,ensure_ascii=False,indent=2))
