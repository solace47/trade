"""Apply the unchanged opportunity statistics to both frozen Q1 selection arms."""
import argparse
import json

from verify_tail_formula_1000_analysis import analysis_check
from trade_research.corporate_cash import sha
from trade_research.tail_formula_1000_analysis import analyze
from trade_research.tail_formula_forward import ROOT, PROTOCOL, checked_model
from trade_research.tail_formula_forward_observations import LABELS


def main(arm):
    checked_model()
    freeze=json.loads((ROOT/'selection_verification.json').read_text())
    assert freeze['passed'] and freeze['selection_report_sha256']==sha(ROOT/'selection_report.json')
    path=ROOT/arm
    proof=json.loads((path/'selection_verification.json').read_text())
    assert proof['passed'] and proof['selection_report_sha256']==sha(path/'selection_report.json')
    for name in ['full_labels.parquet','full_label_report.json','full_label_verification.json']:
        target=path/name
        if not target.exists():
            target.symlink_to((LABELS/name).resolve())
        assert sha(target)==sha(LABELS/name)
    analyze(path,PROTOCOL)
    return analysis_check(path)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('arm',choices=['all','shortlist'])
    print(json.dumps(main(parser.parse_args().arm),ensure_ascii=False,indent=2))
