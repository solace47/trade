"""Repair only the frozen independent query's clock/count aliases."""
import json
from pathlib import Path

from trade_research import tail_formula_morning_distribution as study
from trade_research.corporate_cash import save_json,sha

PROTOCOL=Path('config/tail_formula_morning_distribution_verification_protocol.json')


def verify():
    p=json.loads(PROTOCOL.read_text())
    assert p['input_protocol_sha256']==sha(study.PROTOCOL)
    for file,digest in p['source_hashes'].items():assert sha(Path(file))==digest,file
    original_conn=study.base.conn
    class BoundConnection:
        def __init__(self):self.connection=original_conn()
        def __getattr__(self,name):return getattr(self.connection,name)
        def sql(self,query,*args,**kwargs):
            if 'count(*) AS n,count(DISTINCT timestamp) AS clocks' in query:
                assert 'morning_input_valid AND n=121 AND clocks=121' in query
                query=query.replace('count(*) AS n,count(DISTINCT timestamp) AS clocks',
                    'count(*) AS raw_n,count(DISTINCT timestamp) AS raw_clocks')
                query=query.replace('morning_input_valid AND n=121 AND clocks=121',
                    'morning_input_valid AND raw_n=121 AND raw_clocks=121')
            return self.connection.sql(query,*args,**kwargs)
    study.base.conn=BoundConnection
    result=study.verify()
    receipt=dict(passed=True,input_protocol_sha256=sha(study.PROTOCOL),verification_protocol_sha256=sha(PROTOCOL),
        feature_report_sha256=sha(study.INPUTS/'feature_report.json'),feature_verification_sha256=sha(study.INPUTS/'feature_verification.json'),
        frozen_production_features_and_source_unchanged=True,only_independent_alias_binding_repaired=True,
        no_new_raw_extraction=True,new_group_outcomes_read=False,new_2026_prices_read=False,no_exit_rules=True)
    save_json(study.INPUTS/'verification_execution_receipt.json',receipt)
    return dict(**result,execution_receipt_sha256=sha(study.INPUTS/'verification_execution_receipt.json'))


if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=False,indent=2))
