"""Run frozen input arithmetic with a smaller number of database workers."""
import argparse
import json
from pathlib import Path

from trade_research import tail_formula_morning_range as study
from trade_research.corporate_cash import save_json,sha

PROTOCOL=Path('config/tail_formula_morning_range_execution_protocol.json')


def run(stage):
    p=json.loads(PROTOCOL.read_text())
    assert p['input_protocol_sha256']==sha(study.PROTOCOL)
    assert p['helper_sha256']==sha(Path(__file__))
    for f,h in p['source_hashes'].items():assert sha(Path(f))==h
    original_conn=study.base.conn
    def tuned_conn():
        c=original_conn()
        c.execute('SET threads=1')
        c.execute("SET memory_limit='6GB'")
        c.execute('SET preserve_insertion_order=false')
        return c
    study.base.conn=tuned_conn
    result=getattr(study,stage)()
    assert sha(study.INPUTS/'raw_report.json')==p['raw_report_sha256']
    file={'features':'feature_report.json','verify':'feature_verification.json','native':'native_input_verification.json'}[stage]
    receipt=dict(completed=True,execution_protocol_sha256=sha(PROTOCOL),input_protocol_sha256=sha(study.PROTOCOL),
        output_receipt_sha256=sha(study.INPUTS/file),raw_report_sha256=p['raw_report_sha256'],
        unchanged_frozen_source_and_arithmetic=True,no_new_raw_extraction=True,
        resources=p['resources'],new_2026_prices_read=False,no_exit_rules=True)
    save_json(study.INPUTS/(stage+'_execution_receipt.json'),receipt)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['features','verify','native'])
    print(json.dumps(run(parser.parse_args().stage),ensure_ascii=False,indent=2))
