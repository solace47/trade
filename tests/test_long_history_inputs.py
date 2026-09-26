import json
from pathlib import Path

import baostock as bs
import pandas as pd

from trade_research.long_history_inputs import _catalog_job
from trade_research import ingest


def test_cached_explicit_empty_catalogue_does_not_query(tmp_path,monkeypatch):
    (tmp_path/'vendor').mkdir();(tmp_path/'source_conflicts').mkdir()
    p=tmp_path/'vendor/sh.600000_2022.json';p.write_text('[]')
    def unexpected(*args,**kwargs):
        raise AssertionError('An existing checked response must be reused')
    monkeypatch.setattr(bs,'query_dividend_data',unexpected)
    assert _catalog_job(('sh.600000','2022',str(tmp_path),None))=={'cached':True}


def test_wrong_security_response_is_not_cached_as_empty(tmp_path,monkeypatch):
    (tmp_path/'vendor').mkdir();(tmp_path/'source_conflicts').mkdir()
    monkeypatch.setattr(bs,'query_dividend_data',lambda *a,**k:object())
    monkeypatch.setattr(ingest,'_rows',lambda _:pd.DataFrame([{'code':'sh.600001','dividOperateDate':'2022-06-01'}]))
    result=_catalog_job(('sh.600000','2022',str(tmp_path),None))
    assert 'error' in result
    assert not (tmp_path/'vendor/sh.600000_2022.json').exists()
    assert (tmp_path/'vendor/sh.600000_2022.error.json').exists()
    assert json.loads((tmp_path/'source_conflicts/sh.600000_2022_raw.json').read_text())[0]['code']=='sh.600001'
