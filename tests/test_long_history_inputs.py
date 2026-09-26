import json
from pathlib import Path

import baostock as bs
import pandas as pd
import pytest

from trade_research.long_history_inputs import _catalog_job, supplement_catalog
from trade_research import ingest, cash_dividend_catalog
from trade_research.corporate_cash import sha


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


def historical_holding_across_year(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    output=tmp_path/'study';(output/'historical_catalog').mkdir(parents=True)
    pd.DataFrame({'date':['2022-12-30','2023-01-03'],'code':['sh.600000']*2}).to_parquet(
        output/'historical_training_windows.parquet',index=False)
    pd.DataFrame({'code':['sh.600000'],'year':['2022']}).to_parquet(
        output/'historical_catalog/query_coverage.parquet',index=False)
    recent=tmp_path/'data/research/cash_dividend_catalog';recent.mkdir(parents=True)
    pd.DataFrame({'code':['sh.600000'],'year':['2024']}).to_parquet(
        recent/'query_coverage.parquet',index=False)
    return output


def test_holding_next_year_requires_query_even_without_next_year_signal(tmp_path,monkeypatch):
    output=historical_holding_across_year(tmp_path,monkeypatch)
    def incomplete(folder):
        jobs=pd.read_parquet(folder/'jobs.parquet')
        assert jobs.to_dict('records')==[{'code':'sh.600000','year':'2023'}]
        return {'complete':False}
    monkeypatch.setattr(cash_dividend_catalog,'fetch',incomplete)
    with pytest.raises(ValueError,match='Unqueried cross-year'):
        supplement_catalog(output)
    assert not (output/'holding_year_catalog_coverage.json').exists()


def test_checked_cross_year_catalogue_can_be_reused_without_network(tmp_path,monkeypatch):
    output=historical_holding_across_year(tmp_path,monkeypatch)
    folder=output/'cross_year_catalog';folder.mkdir()
    jobs=pd.DataFrame({'code':['sh.600000'],'year':['2023']})
    jobs.to_parquet(folder/'jobs.parquet',index=False)
    jobs.to_parquet(folder/'query_coverage.parquet',index=False)
    pd.DataFrame({'code':pd.Series(dtype=str)}).to_parquet(folder/'events.parquet',index=False)
    (folder/'catalog_report.json').write_text(json.dumps({'events_sha256':sha(folder/'events.parquet')}))
    def unexpected(*args,**kwargs):
        raise AssertionError('A completed catalogue should be reused')
    monkeypatch.setattr(cash_dividend_catalog,'fetch',unexpected)
    report=supplement_catalog(output)
    assert report['required_code_years']==2
    assert report['supplementary_code_years']==1
    assert report['complete_holding_year_coverage']
